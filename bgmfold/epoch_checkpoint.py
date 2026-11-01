"""Retain closed checkpoint files without a second model serialization."""

import logging
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4


def link_or_copy(source, destination):
    """A later atomic replacement of last.pt cannot change this snapshot."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial." + uuid4().hex)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def publish_file(source, destination):
    """cp+mv with I/O retries; no remote size/hash checks."""
    source, destination = Path(source), Path(destination)
    for attempt in range(1, 4):
        temporary = destination.with_name(destination.name + ".partial." + uuid4().hex)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
            return
        except OSError:
            if attempt == 3:
                raise
            time.sleep(attempt)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logging.warning("Could not remove temporary checkpoint file: %s", temporary)


def retain_epoch_checkpoint(last_path, completed_epoch, publish_dir=None, logger=None):
    logger = logger or logging.getLogger(__name__)
    path = Path(last_path).with_name(f"epoch_{completed_epoch:04d}.pt")
    link_or_copy(last_path, path)
    event = dict(
        epoch=completed_epoch,
        filename=path.name,
        local_path=str(path),
        published_during_training=False,
    )
    logger.info("Saved epoch checkpoint: %s", path)
    if publish_dir:
        destination = Path(publish_dir) / path.name
        event["destination"] = str(destination)
        try:
            publish_file(path, destination)
            event["published_during_training"] = True
            logger.info("Published epoch checkpoint: %s", destination)
        except OSError as exc:
            event["error"] = str(exc)
            logger.exception(
                "Failed to publish epoch checkpoint; keeping local copy and retrying after training: %s", path
            )
    return event
