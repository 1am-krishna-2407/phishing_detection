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
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")

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
IMAGE_MODEL_PATH = MODELS_DIR / "image_model_phase2.pt"
THRESHOLD_CONFIG_PATH = MODELS_DIR / "branch_thresholds.json"

HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "Krishna787/phishing_detection").strip()
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "").strip() or None
HF_MODEL_TOKEN = os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip() or None
HF_AUTO_DOWNLOAD = os.getenv("HF_AUTO_DOWNLOAD", "1").strip().lower() in {"1", "true", "yes"}
HF_CACHE_DIR = os.getenv("HF_HOME", "").strip() or os.getenv("HUGGINGFACE_HUB_CACHE", "").strip() or None
HF_TOKENIZER_REPO_ID = os.getenv("HF_TOKENIZER_REPO_ID", "distilbert-base-uncased").strip()
HF_TOKENIZER_REVISION = os.getenv("HF_TOKENIZER_REVISION", "").strip() or None
HF_TOKENIZER_DIR = MODELS_DIR / "tokenizers" / HF_TOKENIZER_REPO_ID.replace("/", "__")
PRELOAD_URL_MODEL = os.getenv("PRELOAD_URL_MODEL", "1").strip().lower() in {"1", "true", "yes"}
HF_TOKENIZER_PATTERNS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
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
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _download_from_hugging_face(target_path: Path) -> Path:
    if target_path.exists():
        return target_path

    if not HF_MODEL_REPO_ID:
        raise ServiceConfigurationError(
            "Missing model checkpoint and no Hugging Face repo is configured. "
            f"Either add `{target_path.name}` to `models/` or set `HF_MODEL_REPO_ID`."
        )

    try:
        _emit_runtime_log(
            f"Starting Hugging Face model download for `{target_path.name}` from `{HF_MODEL_REPO_ID}`."
        )
        downloaded_path = hf_hub_download(
            repo_id=HF_MODEL_REPO_ID,
            filename=target_path.name,
            revision=HF_MODEL_REVISION,
            token=HF_MODEL_TOKEN,
            cache_dir=HF_CACHE_DIR,
            tqdm_class=_LoggingTqdm,
        )
    except Exception as exc:
        raise ServiceConfigurationError(
            f"Failed to download `{target_path.name}` from Hugging Face repo "
            f"`{HF_MODEL_REPO_ID}`. Check the repo contents and any required token."
        ) from exc

    _emit_runtime_log(f"Finished Hugging Face model download for `{target_path.name}`.")
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


def _load_text_checkpoint(
    model: "TextPhishingModel",
    checkpoint_path: Path,
    device: "torch.device",
) -> int:
    import torch

    checkpoint_path = _ensure_model_file(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    max_len = 128

    if isinstance(checkpoint, dict) and "max_len" in checkpoint:
        max_len = int(checkpoint["max_len"])

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.bert.load_state_dict(checkpoint)

    model.eval()
    return max_len


def get_distilbert_tokenizer() -> "AutoTokenizer":
    global _tokenizer_cache

    if _tokenizer_cache is not None:
        return _tokenizer_cache

    with _tokenizer_lock:
        if _tokenizer_cache is not None:
            return _tokenizer_cache

        from transformers import AutoTokenizer

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

    if _url_bundle_cache is not None:
        return _url_bundle_cache

    with _url_bundle_lock:
        if _url_bundle_cache is not None:
            return _url_bundle_cache

        tokenizer = get_distilbert_tokenizer()

        from src.text_model import TextPhishingModel

        device = get_device()
        model = TextPhishingModel().to(device)
        max_len = _load_text_checkpoint(model, TEXT_MODEL_PATH, device)
        _url_bundle_cache = (model, tokenizer, max_len, device)
        return _url_bundle_cache


def get_ocr_bundle() -> tuple["TextPhishingModel", "AutoTokenizer", int, "torch.device"]:
    global _ocr_bundle_cache

    if _ocr_bundle_cache is not None:
        return _ocr_bundle_cache

    with _ocr_bundle_lock:
        if _ocr_bundle_cache is not None:
            return _ocr_bundle_cache

        tokenizer = get_distilbert_tokenizer()

        from src.text_model import TextPhishingModel

        device = get_device()
        model = TextPhishingModel().to(device)
        max_len = _load_text_checkpoint(model, OCR_MODEL_PATH, device)
        _ocr_bundle_cache = (model, tokenizer, max_len, device)
        return _ocr_bundle_cache


def get_image_transform() -> Any:
    global _image_transform_cache

    if _image_transform_cache is not None:
        return _image_transform_cache

    with _image_transform_lock:
        if _image_transform_cache is not None:
            return _image_transform_cache

        from torchvision import transforms

        _image_transform_cache = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        return _image_transform_cache


def get_image_bundle() -> tuple["ImagePhishingModel", "torch.device"]:
    global _image_bundle_cache

    if _image_bundle_cache is not None:
        return _image_bundle_cache

    with _image_bundle_lock:
        if _image_bundle_cache is not None:
            return _image_bundle_cache

        import torch

        from src.image_model import ImagePhishingModel

        device = get_device()
        model = ImagePhishingModel().to(device)
        image_checkpoint_path = _ensure_model_file(IMAGE_MODEL_PATH)
        model.load_state_dict(torch.load(image_checkpoint_path, map_location=device))
        model.eval()
        _image_bundle_cache = (model, device)
        return _image_bundle_cache


def get_runtime_diagnostics() -> list[str]:
    issues: list[str] = []

    for required_path in (TEXT_MODEL_PATH, OCR_MODEL_PATH, IMAGE_MODEL_PATH):
        if not required_path.exists() and (not HF_MODEL_REPO_ID or not HF_AUTO_DOWNLOAD):
            issues.append(f"Missing model checkpoint: `{required_path.relative_to(PROJECT_ROOT)}`")
    if (
        HF_AUTO_DOWNLOAD
        and HF_MODEL_REPO_ID
        and not all(path.exists() for path in (TEXT_MODEL_PATH, OCR_MODEL_PATH, IMAGE_MODEL_PATH))
    ):
        issues.append(
            "Model checkpoints are not stored locally yet. They will be downloaded from "
            f"Hugging Face repo `{HF_MODEL_REPO_ID}` on first use."
        )

    missing_tokenizer_files = [
        filename for filename in HF_TOKENIZER_PATTERNS if not (HF_TOKENIZER_DIR / filename).exists()
    ]
    if missing_tokenizer_files and not HF_AUTO_DOWNLOAD:
        issues.append(
            "DistilBERT tokenizer files are missing locally. URL predictions will use the "
            "fast heuristic fallback until the tokenizer is added."
        )
    elif missing_tokenizer_files and HF_AUTO_DOWNLOAD:
        issues.append(
            "DistilBERT tokenizer files are missing locally. They will be downloaded from "
            f"Hugging Face repo `{HF_TOKENIZER_REPO_ID}` on first text prediction."
        )

    if shutil.which("tesseract") is None:
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

    if not PRELOAD_URL_MODEL:
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
    import torch

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


def predict_url_probability(url: str) -> float:
    normalized_url = normalize_url(url)
    prepared_url = prepare_text_for_model(normalized_url, "urls_phase1_csv")
    heuristic_probability = heuristic_url_risk(normalized_url)

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

    from src.ocr_extractor import preprocess_image

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_array = preprocess_image(np.array(image)[:, :, ::-1])
    text = pytesseract.image_to_string(image_array, config="--psm 6")
    return text.strip()


def predict_ocr_probability(text: str) -> float:
    return _predict_text_probability(text, get_ocr_bundle)


def try_predict_ocr_probability(text: str) -> float | None:
    try:
        return predict_ocr_probability(text)
    except ServiceConfigurationError as exc:
        _emit_runtime_log(f"Skipping OCR text model: {exc}")
        return None


def predict_image_probability(image_bytes: bytes) -> float:
    import torch
    from PIL import Image

    model, device = get_image_bundle()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = get_image_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor).view(-1)
        return float(torch.sigmoid(logits)[0].cpu())


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
    image_probability = predict_image_probability(image_bytes) if image_bytes else None
    ocr_text = extract_ocr_text_from_bytes(image_bytes) if image_bytes else None
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
        if len(available_scores) == 1:
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
