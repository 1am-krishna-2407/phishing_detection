from __future__ import annotations

import html
import json
import os
import re
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import pandas as pd


SOURCE_PREFIX = {
    "urls_phase1_csv": "[URL]",
    "urls_json": "[URL]",
    "texts_json": "[TEXT]",
    "webs_json": "[WEB]",
}


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_html_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(
        r"(?is)<(script|style).*?>.*?</\1>",
        " ",
        text,
    )
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return _collapse_whitespace(text)


def prepare_text_for_model(text: str, source: str) -> str:
    raw = "" if text is None else str(text)

    if source == "webs_json":
        cleaned = _clean_html_text(raw)
    else:
        cleaned = _collapse_whitespace(raw)

    if not cleaned:
        cleaned = "[NO_TEXT]"

    prefix = SOURCE_PREFIX.get(source, "[TEXT]")
    return f"{prefix} {cleaned}"


def _load_json_records(path: str, source: str) -> pd.DataFrame:
    import pandas as pd

    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    frame = pd.DataFrame(records)
    if "text" not in frame.columns or "label" not in frame.columns:
        raise ValueError(f"{path} must contain 'text' and 'label' columns")

    frame = frame[["text", "label"]].copy()
    frame["source"] = source
    return frame


def _load_csv_records(path: str, source: str) -> pd.DataFrame:
    import pandas as pd

    frame = pd.read_csv(path)
    if "text" not in frame.columns or "label" not in frame.columns:
        raise ValueError(f"{path} must contain 'text' and 'label' columns")

    frame = frame[["text", "label"]].copy()
    frame["source"] = source
    return frame


def load_phase1_sources(
    processed_dir: str,
    allowed_sources: list[str] | None = None,
) -> pd.DataFrame:
    import pandas as pd

    sources = [
        ("urls_phase1.csv", "urls_phase1_csv", _load_csv_records),
        ("urls.json", "urls_json", _load_json_records),
        ("texts.json", "texts_json", _load_json_records),
        ("webs.json", "webs_json", _load_json_records),
    ]

    frames = []
    for filename, source, loader in sources:
        path = os.path.join(processed_dir, filename)
        if not os.path.exists(path):
            continue
        frames.append(loader(path, source))

    if not frames:
        raise FileNotFoundError(f"No phase-1 text sources found in {processed_dir}")

    frame = pd.concat(frames, ignore_index=True)
    if allowed_sources:
        frame = frame[frame["source"].isin(allowed_sources)].copy()
        if frame.empty:
            raise FileNotFoundError(
                f"No phase-1 text rows left after filtering for sources: {allowed_sources}"
            )
    frame["label"] = frame["label"].astype(int)
    frame["raw_text"] = frame["text"].fillna("").astype(str)
    frame["text"] = [
        prepare_text_for_model(text, source)
        for text, source in zip(frame["raw_text"], frame["source"])
    ]
    frame = frame.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return frame


def _sample_frame(frame: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    if n >= len(frame):
        return frame.copy()
    return frame.sample(n=n, random_state=random_state)


def build_integrated_phase1_dataset(
    processed_dir: str,
    output_csv: str,
    target_per_class: int = 40000,
    random_state: int = 42,
    allowed_sources: list[str] | None = None,
) -> pd.DataFrame:
    import pandas as pd

    frame = load_phase1_sources(processed_dir, allowed_sources=allowed_sources)
    sources = list(frame["source"].drop_duplicates())

    selected_parts = []
    for label in sorted(frame["label"].unique()):
        label_frame = frame[frame["label"] == label].copy()
        per_source_target = max(target_per_class // max(len(sources), 1), 1)

        chosen_indices = []
        leftovers = []

        for source_idx, source in enumerate(sources):
            source_frame = label_frame[label_frame["source"] == source]
            take = min(len(source_frame), per_source_target)
            if take > 0:
                sample = _sample_frame(
                    source_frame,
                    take,
                    random_state + (label * 100) + source_idx,
                )
                chosen_indices.extend(sample.index.tolist())

            if len(source_frame) > take:
                leftovers.append(source_frame.drop(index=chosen_indices, errors="ignore"))

        selected = label_frame.loc[sorted(set(chosen_indices))]
        remaining = target_per_class - len(selected)

        if remaining > 0:
            remainder_frame = label_frame.drop(index=selected.index)
            fill = _sample_frame(
                remainder_frame,
                remaining,
                random_state + 1000 + label,
            )
            selected = pd.concat([selected, fill], ignore_index=False)

        selected_parts.append(selected.copy())

    integrated = pd.concat(selected_parts, ignore_index=True)
    integrated = integrated.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    integrated.insert(0, "id", [f"text_phase1_{idx:06d}" for idx in range(len(integrated))])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    integrated[["id", "text", "label", "source"]].to_csv(output_csv, index=False)
    return integrated


def summarize_by_source(frame: pd.DataFrame) -> Iterable[str]:
    summary = (
        frame.groupby(["source", "label"])
        .size()
        .sort_index()
    )
    for (source, label), count in summary.items():
        yield f"{source} label={label}: {count}"
