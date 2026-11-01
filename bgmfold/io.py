"""Small local result I/O helpers. No platform service or Git dependency."""

import csv
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bundle_reports(output):
    output = Path(output)
    path = output / "analysis_bundle.zip"
    temporary = path.with_suffix(".partial")
    with ZipFile(temporary, "w", ZIP_DEFLATED, allowZip64=True) as archive:
        for folder in ("reports", "logs", "configs"):
            if (output / folder).is_dir():
                for file in sorted((output / folder).rglob("*")):
                    if file.is_file() and ".partial" not in file.name:
                        archive.write(file, file.relative_to(output))
    temporary.replace(path)
    return path
