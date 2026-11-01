"""CPU-only validation selection/reporting for checkpoint continuation runs."""

from __future__ import annotations

import math


def require_complete_initialization(report: dict) -> None:
    if report["random_target"] or report["skipped_source"]:
        raise ValueError(
            "Continuation requires the complete matching task model: "
            f"random target tensors={len(report['random_target'])}, "
            f"unused/incompatible source tensors={len(report['skipped_source'])}"
        )


class ContinuationProgress:
    """Epoch 0 is eligible; only full validation F1 drives decisions."""

    def __init__(
        self, source: str, max_epochs: int, patience_epochs: int, min_delta: float = 0.0
    ):
        if max_epochs < 1 or patience_epochs < 1 or min_delta < 0:
            raise ValueError("Invalid continuation stopping policy")
        self.payload = {
            "source_checkpoint": source,
            "max_epochs": max_epochs,
            "patience_epochs": patience_epochs,
            "min_delta": min_delta,
            "selection": "full validation per-RNA positive-pair F1; test never selects",
            "best_epoch": None,
            "best_val_f1": None,
            "last_improvement_epoch": None,
            "best_threshold": None,
            "evaluations": [],
            "training_epochs": [],
            "status": "running",
        }

    def evaluate(
        self,
        epoch: int,
        step: int,
        threshold: float,
        val: dict,
        test: dict | None = None,
    ) -> dict:
        if not val:
            raise ValueError("Full validation results required")
        score = sum(float(m["f1"]) for m in val.values()) / len(val)
        if not math.isfinite(score):
            raise ValueError("Non-finite validation F1")
        previous = self.payload["evaluations"]
        if (not previous and epoch != 0) or (
            previous and epoch <= previous[-1]["epoch"]
        ):
            raise ValueError(
                "Continuation evaluations must start at epoch 0 and increase"
            )
        best = self.payload["best_val_f1"]
        save_best = best is None or score > best
        if save_best:
            self.payload.update(
                best_epoch=epoch, best_val_f1=score, best_threshold=threshold
            )
        # Patience counts completed epochs, not optimizer steps/evaluation calls.
        reference = self.payload.get("patience_reference_f1")
        if reference is None or score > reference + self.payload["min_delta"]:
            self.payload.update(
                patience_reference_f1=score, last_improvement_epoch=epoch
            )
        idle = epoch - self.payload["last_improvement_epoch"]
        should_stop = epoch > 0 and idle >= self.payload["patience_epochs"]
        row = {
            "epoch": epoch,
            "global_step": step,
            "threshold": threshold,
            "val": val,
            "test": test,
            "val_per_rna_f1": score,
            "new_best": save_best,
            "epochs_without_improvement": idle,
            "should_stop": should_stop,
        }
        previous.append(row)
        return row

    def add_test(self, epoch: int, results: dict) -> None:
        if self.payload["evaluations"][-1]["epoch"] != epoch:
            raise ValueError("Test diagnostics must match the current validation epoch")
        self.payload["evaluations"][-1]["test"] = results

    def finish(self, epochs_completed: int, early_stopped: bool) -> None:
        self.payload.update(
            status="completed",
            epochs_completed=epochs_completed,
            stop_reason="validation_patience" if early_stopped else "max_epochs",
        )
