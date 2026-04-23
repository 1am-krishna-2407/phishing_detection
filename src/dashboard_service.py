from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import pandas as pd

# Disable hf_transfer/Xet-backed downloads so the app uses the standard
# Hugging Face client path, which is more reliable in constrained runtimes.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import tqdm as hf_tqdm
from src.text_data_utils import prepare_text_for_model

if TYPE_CHECKING:
    import torch
    from transformers import AutoTokenizer

    from src.image_model import ImagePhishingModel
    from src.text_model import TextPhishingModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
URL_LOG_PATH = LOGS_DIR / "url_prediction_log.csv"

TEXT_MODEL_PATH = MODELS_DIR / "text_model_phase1.pt"
OCR_MODEL_PATH = MODELS_DIR / "ocr_text_model_phase2_5.pt"
TEXT_MODEL_QUANTIZED_PATH = MODELS_DIR / "text_model_phase1_dynamic_int8.pt"
OCR_MODEL_QUANTIZED_PATH = MODELS_DIR / "ocr_text_model_phase2_5_dynamic_int8.pt"
IMAGE_MODEL_PATH = MODELS_DIR / "image_model_phase2.pt"
THRESHOLD_CONFIG_PATH = MODELS_DIR / "branch_thresholds.json"

HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "Krishna787/phishing-detection-models").strip()
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "").strip() or None
HF_MODEL_TOKEN = os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip() or None
HF_CACHE_DIR = os.getenv("HF_HOME", "").strip() or os.getenv("HUGGINGFACE_HUB_CACHE", "").strip() or None
HF_TOKENIZER_REPO_ID = os.getenv("HF_TOKENIZER_REPO_ID", "distilbert-base-uncased").strip()
HF_TOKENIZER_REVISION = os.getenv("HF_TOKENIZER_REVISION", "").strip() or None
HF_TOKENIZER_DIR = MODELS_DIR / "tokenizers" / HF_TOKENIZER_REPO_ID.replace("/", "__")
HF_TOKENIZER_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
)

LOG_COLUMNS = [
    "timestamp_utc",
    "url",
    "prediction",
    "phishing_probability",
    "url_probability",
    "image_probability",
    "ocr_probability",
    "ocr_text_excerpt",
    "image_name",
]

TRUSTED_DOMAINS = {
    "google.com",
    "youtube.com",
    "gmail.com",
    "microsoft.com",
    "live.com",
    "apple.com",
    "amazon.com",
    "github.com",
    "openai.com",
    "wikipedia.org",
    "paypal.com",
    "linkedin.com",
    "reddit.com",
    "yahoo.com",
    "netflix.com",
}

RUNTIME_PROFILE = os.getenv("MODEL_PROFILE", "full").strip().lower() or "full"
if RUNTIME_PROFILE == "lightweight":
    # Lightweight mode is intentionally disabled for local runtime stability.
    RUNTIME_PROFILE = "full"
if RUNTIME_PROFILE not in {"instant", "balanced", "full"}:
    RUNTIME_PROFILE = "full"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


PROFILE_DEFAULTS = {
    "instant": {
        "hf_auto_download": False,
        "preload_url_model": False,
        "use_quantized_text_models": False,
        "enable_url_model": False,
        "enable_ocr_model": False,
        "enable_image_model": False,
        "enable_ocr_extraction": True,
    },
    "lightweight": {
        "hf_auto_download": True,
        "preload_url_model": True,
        "use_quantized_text_models": False,
        "enable_url_model": True,
        "enable_ocr_model": False,
        "enable_image_model": False,
        "enable_ocr_extraction": True,
    },
    "balanced": {
        "hf_auto_download": True,
        "preload_url_model": True,
        "use_quantized_text_models": False,
        "enable_url_model": True,
        "enable_ocr_model": True,
        "enable_image_model": False,
        "enable_ocr_extraction": True,
    },
    "full": {
        "hf_auto_download": True,
        "preload_url_model": True,
        "use_quantized_text_models": False,
        "enable_url_model": True,
        "enable_ocr_model": True,
        "enable_image_model": True,
        "enable_ocr_extraction": True,
    },
}[RUNTIME_PROFILE]

HF_AUTO_DOWNLOAD = _env_flag("HF_AUTO_DOWNLOAD", PROFILE_DEFAULTS["hf_auto_download"])
PRELOAD_URL_MODEL = _env_flag("PRELOAD_URL_MODEL", PROFILE_DEFAULTS["preload_url_model"])
USE_QUANTIZED_TEXT_MODELS = _env_flag("USE_QUANTIZED_TEXT_MODELS", PROFILE_DEFAULTS["use_quantized_text_models"])
ENABLE_URL_MODEL = _env_flag("ENABLE_URL_MODEL", PROFILE_DEFAULTS["enable_url_model"])
ENABLE_OCR_MODEL = _env_flag("ENABLE_OCR_MODEL", PROFILE_DEFAULTS["enable_ocr_model"])
ENABLE_IMAGE_MODEL = _env_flag("ENABLE_IMAGE_MODEL", PROFILE_DEFAULTS["enable_image_model"])
ENABLE_OCR_EXTRACTION = _env_flag("ENABLE_OCR_EXTRACTION", PROFILE_DEFAULTS["enable_ocr_extraction"])

SUSPICIOUS_TEXT_TERMS = (
    "login",
    "sign in",
    "verify",
    "verification",
    "password",
    "account",
    "confirm",
    "security alert",
    "suspended",
    "unusual activity",
    "bank",
    "wallet",
    "gift card",
    "crypto",
    "limited time",
    "urgent",
    "immediately",
    "click below",
    "reset your password",
    "microsoft 365",
    "outlook",
    "paypal",
)


class ServiceConfigurationError(RuntimeError):
    """Raised when required runtime assets are unavailable."""


logger = logging.getLogger(__name__)
_download_log_lock = threading.Lock()
_tokenizer_lock = threading.Lock()
_url_bundle_lock = threading.Lock()
_ocr_bundle_lock = threading.Lock()
_image_transform_lock = threading.Lock()
_image_bundle_lock = threading.Lock()
_warmup_lock = threading.Lock()
_warmup_thread: threading.Thread | None = None
_warmup_status = {
    "state": "idle",
    "message": "URL model has not started warming yet.",
}
_tokenizer_cache: Any | None = None
_url_bundle_cache: Any | None = None
_ocr_bundle_cache: Any | None = None
_image_transform_cache: Any | None = None
_image_bundle_cache: Any | None = None


def _dependency_error(feature_name: str, package_name: str) -> ServiceConfigurationError:
    return ServiceConfigurationError(
        f"{feature_name} requires the optional `{package_name}` package, which is not installed "
        f"for the current runtime profile `{RUNTIME_PROFILE}`."
    )


@dataclass(frozen=True)
class PredictionResult:
    prediction: str
    probability: float
    url_probability: float | None
    image_probability: float | None
    ocr_probability: float | None
    ocr_text: str | None


@dataclass(frozen=True)
class BranchConfig:
    name: str
    trigger_threshold: float
    decision_threshold: float


DEFAULT_BRANCHES: tuple[BranchConfig, ...] = (
    BranchConfig(name="url", trigger_threshold=0.90, decision_threshold=0.60),
    BranchConfig(name="image", trigger_threshold=0.90, decision_threshold=0.60),
    BranchConfig(name="ocr", trigger_threshold=0.80, decision_threshold=0.60),
)


def _emit_runtime_log(message: str) -> None:
    with _download_log_lock:
        logger.info(message)
        print(message, flush=True)


class _LoggingTqdm(hf_tqdm):
    """Logs download progress to Streamlit/runtime logs every 5%."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_logged_percent = -5
        self._label = self.desc or "Hugging Face download"

    def _format_progress(self) -> str | None:
        if not self.total:
            return None

        percent = int((self.n / self.total) * 100)
        if percent < 100 and percent < self._last_logged_percent + 5:
            return None

        self._last_logged_percent = 100 if percent >= 100 else percent - (percent % 5)
        downloaded_mb = self.n / (1024 * 1024)
        total_mb = self.total / (1024 * 1024)
        return f"{self._label}: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)"

    def update(self, n: int = 1) -> None:
        super().update(n)
        message = self._format_progress()
        if message:
            _emit_runtime_log(message)

    def close(self) -> None:
        if self.total and self.n >= self.total and self._last_logged_percent < 100:
            downloaded_mb = self.n / (1024 * 1024)
            total_mb = self.total / (1024 * 1024)
            _emit_runtime_log(f"{self._label}: 100% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
            self._last_logged_percent = 100
        super().close()


def get_device() -> torch.device:
    try:
        import torch
    except ImportError as exc:
        raise _dependency_error("Model inference", "torch") from exc

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _download_from_hugging_face(target_path: Path) -> Path:
    if target_path.exists():
        return target_path

    if not HF_MODEL_REPO_ID:
        raise ServiceConfigurationError(
            f"Missing Hugging Face repo for `{target_path.name}`"
        )

    try:
        _emit_runtime_log(
            f"Downloading `{target_path.name}` from `{HF_MODEL_REPO_ID}`..."
        )

        downloaded_path = hf_hub_download(
            repo_id=HF_MODEL_REPO_ID,
            filename=target_path.name,
            revision=HF_MODEL_REVISION,
            token=HF_MODEL_TOKEN,
            cache_dir=HF_CACHE_DIR,
            local_files_only=False,
            force_download=True,
        )

    except Exception as exc:
        raise ServiceConfigurationError(
            f"""
❌ Failed to download `{target_path.name}` from Hugging Face.

Repo: {HF_MODEL_REPO_ID}

Possible causes:
- File name mismatch
- Private repo without HF_TOKEN
- Network timeout on Streamlit
- Corrupted upload

Actual error:
{str(exc)}
"""
        ) from exc

    _emit_runtime_log(f"✅ Downloaded `{target_path.name}` successfully.")
    return Path(downloaded_path)


def _ensure_model_file(target_path: Path) -> Path:
    if target_path.exists():
        return target_path
    if not HF_AUTO_DOWNLOAD:
        raise ServiceConfigurationError(
            f"Missing model checkpoint: `{target_path.relative_to(PROJECT_ROOT)}`. "
            "Add the file locally or set `HF_AUTO_DOWNLOAD=1` to download it from Hugging Face."
        )
    return _download_from_hugging_face(target_path)


def _quantized_text_path(checkpoint_path: Path) -> Path | None:
    if checkpoint_path == TEXT_MODEL_PATH:
        return TEXT_MODEL_QUANTIZED_PATH
    if checkpoint_path == OCR_MODEL_PATH:
        return OCR_MODEL_QUANTIZED_PATH
    return None


def _resolve_text_checkpoint_path(checkpoint_path: Path, device: "torch.device") -> tuple[Path, bool]:
    quantized_path = _quantized_text_path(checkpoint_path)
    if (
        USE_QUANTIZED_TEXT_MODELS
        and device.type == "cpu"
        and quantized_path is not None
    ):
        return _ensure_model_file(quantized_path), True

    return _ensure_model_file(checkpoint_path), False


def _load_text_checkpoint(
    model: "TextPhishingModel",
    checkpoint_path: Path,
    device: "torch.device",
) -> tuple["TextPhishingModel", int]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise _dependency_error("Text model loading", "torch") from exc

    checkpoint_path, is_quantized = _resolve_text_checkpoint_path(checkpoint_path, device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    max_len = 128

    if isinstance(checkpoint, dict) and "max_len" in checkpoint:
        max_len = int(checkpoint["max_len"])

    if is_quantized:
        model = torch.quantization.quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)
        device = torch.device("cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.bert.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    if is_quantized:
        _emit_runtime_log(f"Loaded quantized CPU text model `{checkpoint_path.name}`.")
    return model, max_len


def get_runtime_profile() -> dict[str, Any]:
    active_branches: list[str] = ["url model" if ENABLE_URL_MODEL else "url heuristic"]
    if ENABLE_IMAGE_MODEL:
        active_branches.append("image cnn")
    elif ENABLE_OCR_EXTRACTION:
        active_branches.append("image heuristic")
    if ENABLE_OCR_EXTRACTION:
        active_branches.append("ocr extraction")
    if ENABLE_OCR_MODEL:
        active_branches.append("ocr text model")
    elif ENABLE_OCR_EXTRACTION:
        active_branches.append("ocr heuristic")

    return {
        "name": RUNTIME_PROFILE,
        "uses_quantized_text_models": USE_QUANTIZED_TEXT_MODELS,
        "active_branches": active_branches,
        "heavy_models_enabled": ENABLE_IMAGE_MODEL or ENABLE_OCR_MODEL,
        "downloads_enabled": HF_AUTO_DOWNLOAD,
    }


def get_distilbert_tokenizer() -> "AutoTokenizer":
    global _tokenizer_cache

    if _tokenizer_cache is not None:
        return _tokenizer_cache

    with _tokenizer_lock:
        if _tokenizer_cache is not None:
            return _tokenizer_cache

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise _dependency_error("Tokenizer loading", "transformers") from exc

        try:
            _tokenizer_cache = AutoTokenizer.from_pretrained(HF_TOKENIZER_DIR, local_files_only=True)
            return _tokenizer_cache
        except Exception:
            if not HF_AUTO_DOWNLOAD:
                raise ServiceConfigurationError(
                    "Missing DistilBERT tokenizer files. Add the tokenizer under "
                    f"`{HF_TOKENIZER_DIR.relative_to(PROJECT_ROOT)}` or set `HF_AUTO_DOWNLOAD=1` "
                    "to download it from Hugging Face."
                )

            _emit_runtime_log(
                f"Starting Hugging Face tokenizer download for `{HF_TOKENIZER_REPO_ID}`."
            )
            try:
                _tokenizer_cache = AutoTokenizer.from_pretrained(
                    HF_TOKENIZER_REPO_ID,
                    revision=HF_TOKENIZER_REVISION,
                    token=HF_MODEL_TOKEN,
                    cache_dir=HF_CACHE_DIR,
                    local_files_only=False,
                    use_fast=True,
                )
            except Exception as exc:
                raise ServiceConfigurationError(
                    f"Failed to download tokenizer `{HF_TOKENIZER_REPO_ID}` from Hugging Face. "
                    "Check the internet connection or add the tokenizer files locally."
                ) from exc
            _emit_runtime_log(
                f"Finished Hugging Face tokenizer download for `{HF_TOKENIZER_REPO_ID}`."
            )
            return _tokenizer_cache


def get_url_bundle() -> tuple["TextPhishingModel", "AutoTokenizer", int, "torch.device"]:
    global _url_bundle_cache

    if not ENABLE_URL_MODEL:
        raise ServiceConfigurationError("URL model is disabled by runtime profile.")

    if _url_bundle_cache is not None:
        return _url_bundle_cache

    with _url_bundle_lock:
        if _url_bundle_cache is not None:
            return _url_bundle_cache

        tokenizer = get_distilbert_tokenizer()

        try:
            from src.text_model import TextPhishingModel
        except ImportError as exc:
            missing_package = "transformers" if "transformers" in str(exc) else "torch"
            raise _dependency_error("URL model", missing_package) from exc

        device = get_device()
        model = TextPhishingModel().to(device)
        model, max_len = _load_text_checkpoint(model, TEXT_MODEL_PATH, device)
        _url_bundle_cache = (model, tokenizer, max_len, device)
        return _url_bundle_cache


def get_ocr_bundle() -> tuple["TextPhishingModel", "AutoTokenizer", int, "torch.device"]:
    global _ocr_bundle_cache

    if not ENABLE_OCR_MODEL:
        raise ServiceConfigurationError("OCR text model is disabled by runtime profile.")

    if _ocr_bundle_cache is not None:
        return _ocr_bundle_cache

    with _ocr_bundle_lock:
        if _ocr_bundle_cache is not None:
            return _ocr_bundle_cache

        tokenizer = get_distilbert_tokenizer()

        try:
            from src.text_model import TextPhishingModel
        except ImportError as exc:
            missing_package = "transformers" if "transformers" in str(exc) else "torch"
            raise _dependency_error("OCR text model", missing_package) from exc

        device = get_device()
        model = TextPhishingModel().to(device)
        model, max_len = _load_text_checkpoint(model, OCR_MODEL_PATH, device)
        _ocr_bundle_cache = (model, tokenizer, max_len, device)
        return _ocr_bundle_cache


def get_image_transform() -> Any:
    global _image_transform_cache

    if _image_transform_cache is not None:
        return _image_transform_cache

    with _image_transform_lock:
        if _image_transform_cache is not None:
            return _image_transform_cache

        try:
            from torchvision import transforms
        except ImportError as exc:
            raise _dependency_error("Image preprocessing", "torchvision") from exc

        _image_transform_cache = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        return _image_transform_cache


def get_image_bundle() -> tuple["ImagePhishingModel", "torch.device"]:
    global _image_bundle_cache

    if not ENABLE_IMAGE_MODEL:
        raise ServiceConfigurationError("Image model is disabled by runtime profile.")

    if _image_bundle_cache is not None:
        return _image_bundle_cache

    with _image_bundle_lock:
        if _image_bundle_cache is not None:
            return _image_bundle_cache

        try:
            import torch
        except ImportError as exc:
            raise _dependency_error("Image model", "torch") from exc

        try:
            from src.image_model import ImagePhishingModel
        except ImportError as exc:
            missing_package = "torchvision" if "torchvision" in str(exc) else "torch"
            raise _dependency_error("Image model", missing_package) from exc

        device = get_device()
        model = ImagePhishingModel().to(device)
        image_checkpoint_path = _ensure_model_file(IMAGE_MODEL_PATH)
        model.load_state_dict(torch.load(image_checkpoint_path, map_location=device))
        model.eval()
        _image_bundle_cache = (model, device)
        return _image_bundle_cache


def get_runtime_diagnostics() -> list[str]:
    issues: list[str] = []

    required_paths: list[Path] = []
    if ENABLE_URL_MODEL:
        required_paths.append(TEXT_MODEL_QUANTIZED_PATH if USE_QUANTIZED_TEXT_MODELS else TEXT_MODEL_PATH)
    if ENABLE_OCR_MODEL:
        required_paths.append(OCR_MODEL_QUANTIZED_PATH if USE_QUANTIZED_TEXT_MODELS else OCR_MODEL_PATH)
    if ENABLE_IMAGE_MODEL:
        required_paths.append(IMAGE_MODEL_PATH)

    for required_path in required_paths:
        if not required_path.exists() and (not HF_MODEL_REPO_ID or not HF_AUTO_DOWNLOAD):
            issues.append(f"Missing model checkpoint: `{required_path.relative_to(PROJECT_ROOT)}`")
    if (
        HF_AUTO_DOWNLOAD
        and HF_MODEL_REPO_ID
        and required_paths
        and not all(path.exists() for path in required_paths)
    ):
        issues.append(
            "Enabled model checkpoints are not stored locally yet. They will be downloaded from "
            f"Hugging Face repo `{HF_MODEL_REPO_ID}` on first use."
        )

    missing_tokenizer_files = [
        filename for filename in HF_TOKENIZER_PATTERNS if not (HF_TOKENIZER_DIR / filename).exists()
    ]
    if missing_tokenizer_files and ENABLE_URL_MODEL and not HF_AUTO_DOWNLOAD:
        issues.append(
            "DistilBERT tokenizer files are missing locally. URL predictions will use the "
            "fast heuristic fallback until the tokenizer is added."
        )
    elif missing_tokenizer_files and ENABLE_URL_MODEL and HF_AUTO_DOWNLOAD:
        issues.append(
            "DistilBERT tokenizer files are missing locally. They will be downloaded from "
            f"Hugging Face repo `{HF_TOKENIZER_REPO_ID}` on first text prediction."
        )

    if not ENABLE_URL_MODEL:
        issues.append("Instant mode disables the URL transformer checkpoint; URLs are scored heuristically.")
    if not ENABLE_IMAGE_MODEL:
        issues.append("Lightweight mode disables the CNN image checkpoint; screenshots use OCR-assisted heuristics.")
    if ENABLE_OCR_EXTRACTION and not ENABLE_OCR_MODEL:
        issues.append("Lightweight mode disables the OCR text checkpoint; extracted text is scored heuristically.")
    if ENABLE_OCR_EXTRACTION and shutil.which("tesseract") is None:
        issues.append(
            "Tesseract binary not found. Add `tesseract-ocr` to `packages.txt` or install it locally."
        )

    return issues


def _set_warmup_status(state: str, message: str) -> None:
    with _warmup_lock:
        _warmup_status["state"] = state
        _warmup_status["message"] = message


def _warm_url_model() -> None:
    _set_warmup_status("running", "Preparing URL model in the background.")
    try:
        get_url_bundle()
    except Exception as exc:
        _set_warmup_status("error", str(exc))
        _emit_runtime_log(f"URL model warmup failed: {exc}")
    else:
        _set_warmup_status("ready", "URL model is ready.")
        _emit_runtime_log("URL model warmup completed.")


def start_background_warmup() -> None:
    global _warmup_thread

    if not PRELOAD_URL_MODEL or not ENABLE_URL_MODEL:
        return

    with _warmup_lock:
        if _warmup_thread and _warmup_thread.is_alive():
            return
        if _warmup_status["state"] in {"running", "ready"}:
            return

        _warmup_thread = threading.Thread(
            target=_warm_url_model,
            name="url-model-warmup",
            daemon=True,
        )
        _warmup_thread.start()


def get_warmup_status() -> dict[str, str]:
    with _warmup_lock:
        return dict(_warmup_status)


def _resolve_branches() -> tuple[BranchConfig, ...]:
    overrides: dict[str, float] = {}
    if THRESHOLD_CONFIG_PATH.exists():
        with open(THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        overrides = {
            str(name): float(value)
            for name, value in raw.items()
            if isinstance(value, (int, float))
        }

    return tuple(
        BranchConfig(
            name=branch.name,
            trigger_threshold=branch.trigger_threshold,
            decision_threshold=overrides.get(branch.name, branch.decision_threshold),
        )
        for branch in DEFAULT_BRANCHES
    )


def _predict_text_probability(text: str, bundle_loader: Any) -> float:
    try:
        import torch
    except ImportError as exc:
        raise _dependency_error("Text scoring", "torch") from exc

    cleaned = text.strip() or "[NO_TEXT]"
    model, tokenizer, max_len, device = bundle_loader()
    encoded = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask).view(-1)
        return float(torch.sigmoid(logits)[0].cpu())


def heuristic_text_risk(text: str, source: str = "text") -> float:
    cleaned = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if not cleaned:
        return 0.0

    score = 0.05
    matches = sum(term in cleaned for term in SUSPICIOUS_TEXT_TERMS)
    score += min(matches * 0.08, 0.40)

    if "http://" in cleaned or "https://" in cleaned:
        score += 0.08
    if re.search(r"(?:\d{1,3}\.){3}\d{1,3}", cleaned):
        score += 0.12
    if any(token in cleaned for token in ("bit.ly", "tinyurl", "t.co", "rb.gy")):
        score += 0.18
    if any(token in cleaned for token in ("otp", "one time password", "gift card", "seed phrase")):
        score += 0.18
    if source == "ocr":
        if len(cleaned) > 200:
            score += 0.05
        if any(token in cleaned for token in ("sign in", "log in", "keep me signed in")):
            score += 0.10

    return max(0.0, min(score, 0.98))


def heuristic_image_risk(image_bytes: bytes, ocr_text: str | None = None) -> float:
    import numpy as np
    from PIL import Image

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    contrast = float(grayscale.std())

    score = 0.12
    if width >= 1000 and height >= 500:
        score += 0.05
    if contrast < 28:
        score += 0.06
    if ocr_text:
        score = max(score, 0.20 + (0.80 * heuristic_text_risk(ocr_text, source="ocr")))

    return max(0.0, min(score, 0.98))


def predict_url_probability(url: str) -> float:
    normalized_url = normalize_url(url)
    prepared_url = prepare_text_for_model(normalized_url, "urls_phase1_csv")
    heuristic_probability = heuristic_url_risk(normalized_url)

    if not ENABLE_URL_MODEL:
        return heuristic_probability

    try:
        model_probability = _predict_text_probability(prepared_url, get_url_bundle)
    except ServiceConfigurationError as exc:
        _emit_runtime_log(f"Using URL heuristic fallback: {exc}")
        return heuristic_probability

    if is_trusted_domain(normalized_url) and heuristic_probability <= 0.10:
        return min(model_probability, 0.05)

    return (0.35 * model_probability) + (0.65 * heuristic_probability)


def extract_ocr_text_from_bytes(image_bytes: bytes) -> str:
    import numpy as np
    import pytesseract
    from PIL import Image

    if not ENABLE_OCR_EXTRACTION:
        return ""

    from src.ocr_extractor import preprocess_image

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_array = preprocess_image(np.array(image)[:, :, ::-1])
    text = pytesseract.image_to_string(image_array, config="--psm 6")
    return text.strip()


def predict_ocr_probability(text: str) -> float:
    heuristic_probability = heuristic_text_risk(text, source="ocr")
    if not ENABLE_OCR_MODEL:
        return heuristic_probability

    try:
        model_probability = _predict_text_probability(text, get_ocr_bundle)
    except ServiceConfigurationError as exc:
        _emit_runtime_log(f"Using OCR heuristic fallback: {exc}")
        return heuristic_probability

    return (0.65 * model_probability) + (0.35 * heuristic_probability)


def try_predict_ocr_probability(text: str) -> float | None:
    try:
        return predict_ocr_probability(text)
    except ServiceConfigurationError as exc:
        _emit_runtime_log(f"Skipping OCR text model: {exc}")
        return None


def predict_image_probability(image_bytes: bytes, ocr_text: str | None = None) -> float:
    try:
        import torch
    except ImportError as exc:
        raise _dependency_error("Image scoring", "torch") from exc
    from PIL import Image

    heuristic_probability = heuristic_image_risk(image_bytes, ocr_text=ocr_text)

    if not ENABLE_IMAGE_MODEL:
        return heuristic_probability

    try:
        model, device = get_image_bundle()
    except ServiceConfigurationError as exc:
        _emit_runtime_log(f"Using image heuristic fallback: {exc}")
        return heuristic_probability

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = get_image_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor).view(-1)
        model_probability = float(torch.sigmoid(logits)[0].cpu())
        return (0.70 * model_probability) + (0.30 * heuristic_probability)


def _decision_threshold(name: str, fallback: float = 0.60) -> float:
    for branch in _resolve_branches():
        if branch.name == name:
            if name == "url":
                return max(branch.decision_threshold, 0.60)
            return branch.decision_threshold
    return fallback


def normalize_url(url: str) -> str:
    cleaned = (url or "").strip()
    cleaned = cleaned.strip("`'\" ")
    if cleaned and "://" not in cleaned:
        cleaned = f"https://{cleaned}"
    return cleaned


def extract_hostname(url: str) -> str:
    parsed = urlparse(normalize_url(url))
    hostname = (parsed.hostname or "").lower().strip(".")
    return hostname


def is_trusted_domain(url: str) -> bool:
    hostname = extract_hostname(url)
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in TRUSTED_DOMAINS)


def heuristic_url_risk(url: str) -> float:
    normalized = normalize_url(url)
    hostname = extract_hostname(normalized)
    parsed = urlparse(normalized)

    if not hostname:
        return 0.85

    score = 0.05
    suspicious_terms = (
        "login",
        "verify",
        "secure",
        "update",
        "account",
        "signin",
        "confirm",
        "password",
        "bank",
        "wallet",
    )

    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", hostname):
        score += 0.45

    if hostname.count(".") >= 3:
        score += 0.12
    if "-" in hostname:
        score += 0.08
    if "@" in normalized:
        score += 0.35
    if len(normalized) > 90:
        score += 0.10
    if parsed.query:
        score += 0.05
    if parsed.fragment:
        score += 0.03

    text_to_scan = f"{hostname}{parsed.path}{parsed.query}".lower()
    score += min(sum(term in text_to_scan for term in suspicious_terms) * 0.08, 0.32)

    if is_trusted_domain(normalized):
        score = min(score, 0.03)

    clean_homepage = parsed.path in ("", "/") and not parsed.query and not parsed.fragment
    if clean_homepage and hostname.count(".") <= 2 and "-" not in hostname and not is_trusted_domain(normalized):
        score = min(score, 0.18)

    return max(0.0, min(score, 0.99))


def _weighted_fusion(
    url_probability: float | None,
    image_probability: float | None,
    ocr_probability: float | None,
) -> float | None:
    weights = {"url": 0.35, "image": 0.45, "ocr": 0.20}
    weighted_sum = 0.0
    total = 0.0

    if url_probability is not None:
        weighted_sum += weights["url"] * url_probability
        total += weights["url"]
    if image_probability is not None:
        weighted_sum += weights["image"] * image_probability
        total += weights["image"]
    if ocr_probability is not None:
        weighted_sum += weights["ocr"] * ocr_probability
        total += weights["ocr"]

    return (weighted_sum / total) if total else None


def predict_phishing(url: str | None = None, image_bytes: bytes | None = None) -> PredictionResult:
    url_probability = predict_url_probability(url) if url else None
    ocr_text = extract_ocr_text_from_bytes(image_bytes) if image_bytes else None
    image_probability = predict_image_probability(image_bytes, ocr_text=ocr_text) if image_bytes else None
    ocr_probability = try_predict_ocr_probability(ocr_text) if ocr_text is not None else None

    final_probability = _weighted_fusion(url_probability, image_probability, ocr_probability)
    if final_probability is None:
        final_probability = 0.0

    branch_scores = {
        "url": url_probability,
        "image": image_probability,
        "ocr": ocr_probability,
    }
    prediction = "Legitimate"
    available_scores = {name: score for name, score in branch_scores.items() if score is not None}

    for branch in _resolve_branches():
        score = branch_scores.get(branch.name)
        if score is not None and score >= branch.trigger_threshold:
            prediction = "Phishing"
            final_probability = 1.0
            break
    else:
        # OCR often captures phishing intent directly from call-to-action text.
        # If OCR branch crosses its decision threshold, let it drive final verdict.
        if ocr_probability is not None and ocr_probability >= _decision_threshold("ocr"):
            prediction = "Phishing"
            final_probability = max(final_probability, ocr_probability)
        elif len(available_scores) == 1:
            branch_name, score = next(iter(available_scores.items()))
            threshold = _decision_threshold(branch_name)
            prediction = "Phishing" if score >= threshold else "Legitimate"
            final_probability = score
        elif len(available_scores) > 1:
            prediction = "Phishing" if final_probability >= 0.60 else "Legitimate"

    return PredictionResult(
        prediction=prediction,
        probability=final_probability,
        url_probability=url_probability,
        image_probability=image_probability,
        ocr_probability=ocr_probability,
        ocr_text=ocr_text,
    )


def append_url_log(url: str, result: PredictionResult, image_name: str | None = None) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    new_file = not URL_LOG_PATH.exists()
    excerpt = (result.ocr_text or "").replace("\n", " ").strip()[:160]

    with open(URL_LOG_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "url": url,
                "prediction": result.prediction,
                "phishing_probability": round(result.probability, 6),
                "url_probability": None if result.url_probability is None else round(result.url_probability, 6),
                "image_probability": None if result.image_probability is None else round(result.image_probability, 6),
                "ocr_probability": None if result.ocr_probability is None else round(result.ocr_probability, 6),
                "ocr_text_excerpt": excerpt,
                "image_name": image_name or "",
            }
        )

    return URL_LOG_PATH


def delete_url_log_entry(row_id: int) -> bool:
    if not URL_LOG_PATH.exists():
        return False

    frame = pd.read_csv(URL_LOG_PATH)
    if row_id < 0 or row_id >= len(frame):
        return False

    frame = frame.drop(index=row_id).reset_index(drop=True)
    if frame.empty:
        URL_LOG_PATH.unlink(missing_ok=True)
        return True

    frame.to_csv(URL_LOG_PATH, index=False)
    return True


def read_url_logs(limit: int = 50) -> pd.DataFrame:
    if not URL_LOG_PATH.exists():
        return pd.DataFrame(columns=["row_id", *LOG_COLUMNS])
    frame = pd.read_csv(URL_LOG_PATH)
    frame = frame.reset_index(names="row_id")
    return frame.tail(limit).iloc[::-1].reset_index(drop=True)
