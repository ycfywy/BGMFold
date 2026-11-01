"""Resolve local GB.RNA encoder weight shards."""

import json
import re
from pathlib import Path


def directory_plan(source, fallback_config):
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError(f"Expected model DIRECTORY containing weight shards: {source}")
    config = source / "config.json"
    fallback = not config.is_file()
    if fallback:
        config = Path(fallback_config).resolve() / "config.json"
    data = json.loads(config.read_text())
    if data.get("model_type") != "rnabert":
        raise ValueError(f"Expected GB.RNA rnabert configuration: {config}")
    # Prefer the supplied .bin index. Do not load duplicate .safetensors copies.
    index = None
    for name in (
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
        "pytorch_model.safetensors.index.json",
    ):
        if (source / name).is_file():
            index = source / name
            mapping = json.loads(index.read_text())["weight_map"]
            if not mapping:
                raise ValueError(f"Empty weight index: {index}")
            files = [source / name for name in sorted(set(mapping.values()))]
            break
    else:
        mapping = None
        if (source / "pytorch_model.bin").is_file():
            files = [source / "pytorch_model.bin"]
        elif list(source.glob("pytorch_model-*-of-*.bin")):
            files = sorted(source.glob("pytorch_model-*-of-*.bin"))
        elif (source / "model.safetensors").is_file():
            files = [source / "model.safetensors"]
        else:
            files = sorted(source.glob("*-*-of-*.safetensors"))
        if not files:
            raise FileNotFoundError(
                f"No supported local .bin/.safetensors weights: {source}"
            )
        if len(files) > 1 or "-of-" in files[0].name:
            parts = [
                re.fullmatch(r".*-(\d+)-of-(\d+)\.(?:bin|safetensors)", p.name)
                for p in files
            ]
            if not all(parts):
                raise ValueError(f"Cannot verify shard numbering in {source}")
            totals = {int(m.group(2)) for m in parts}
            if len(totals) != 1 or {int(m.group(1)) for m in parts} != set(
                range(1, next(iter(totals)) + 1)
            ):
                raise ValueError(f"Incomplete or ambiguous numbered shards: {source}")
    for file in files:
        if (
            not file.resolve().is_relative_to(source)
            or not file.is_file()
            or file.stat().st_size == 0
        ):
            raise ValueError(f"Missing/empty/unsafe shard path: {file}")
        if file.suffix not in (".bin", ".safetensors"):
            raise ValueError(f"Unsupported indexed weight file: {file}")
    return dict(
        source=str(source),
        config=str(config),
        config_fallback=fallback,
        index=str(index) if index else None,
        files=list(map(str, files)),
        weight_map=mapping,
    )
