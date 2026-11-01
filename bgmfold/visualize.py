"""Visualization helpers for contact-map predictions and training curves."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless server-safe backend
import matplotlib.pyplot as plt
import numpy as np


def _imshow(ax, mat: np.ndarray, title: str, cmap: str = "Greys"):
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def save_sample_visualization(
    save_dir: str,
    dataset_name: str,
    epoch: int,
    sample_id: str,
    length: int,
    gt_map: np.ndarray,
    prob_map: np.ndarray,
    pred_map: np.ndarray,
) -> str:
    """Save a multi-panel comparison figure for a single sample.

    Panels: ground truth, predicted probability map, predicted binary map,
    concatenated GT|prediction, and an error map (FP=red, FN=blue).

    `dataset_name` should already encode which split/dataset the sample came
    from (e.g. "val_bprna-spot-val", "test_bprna-new", "test_bprna-spot0",
    "test_archiveii") so that visualizations from different datasets are kept
    visually and by-filename separate.

    Filename encodes dataset name, epoch, sample id, and sequence length, e.g.
    `test_bprna-new_epoch10_idbpRNA_CRW_1_len120.png`.
    """
    os.makedirs(save_dir, exist_ok=True)

    gt_map = gt_map[:length, :length]
    prob_map = prob_map[:length, :length]
    pred_map = pred_map[:length, :length]

    concat_map = np.concatenate([gt_map, pred_map], axis=1)

    fp = (pred_map == 1) & (gt_map == 0)
    fn = (pred_map == 0) & (gt_map == 1)
    error_map = np.ones((length, length, 3))
    error_map[fp] = [1.0, 0.0, 0.0]  # false positive -> red
    error_map[fn] = [0.0, 0.0, 1.0]  # false negative -> blue

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    try:
        _imshow(axes[0], gt_map, "Ground Truth")
        _imshow(axes[1], prob_map, "Predicted Prob.")
        _imshow(axes[2], pred_map, "Predicted Binary")
        axes[3].imshow(
            concat_map, cmap="Greys", vmin=0, vmax=1, interpolation="nearest"
        )
        axes[3].set_title("GT | Pred", fontsize=10)
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        axes[4].imshow(error_map, interpolation="nearest")
        axes[4].set_title("Error (FP=red, FN=blue)", fontsize=10)
        axes[4].set_xticks([])
        axes[4].set_yticks([])

        fig.suptitle(
            f"{dataset_name} | epoch {epoch} | id {sample_id} | len {length}",
            fontsize=11,
        )
        fig.tight_layout()

        safe_id = str(sample_id).replace("/", "_")
        filename = f"{dataset_name}_epoch{epoch}_id{safe_id}_len{length}.png"
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)
    return path


class TrainingCurveLogger:
    """Accumulates scalar training/eval history in memory + on disk (JSON),
    and periodically re-renders PNG curve plots so progress can be inspected
    just by looking at image files (no TensorBoard needed, though TensorBoard
    logging can still be used alongside this).

    Given a single training run directory (`run_dir`, e.g.
    `outputs/<task_name>_<CST timestamp>/`), this class owns:

        <run_dir>/training_dashboard.png   # single combined overview image:
                                            # loss, lr, train/val
                                            # {f1,precision,recall}, and test
                                            # {f1,precision,recall}.
        <run_dir>/logs/history.json        # full structured scalar history
        <run_dir>/dashboards/<stage>.png   # one dashboard per sequential config
    """

    _METRICS = ("f1", "precision", "recall")

    def __init__(
        self,
        run_dir: str,
        stage_name: str | None = None,
        dashboard_path: str | None = None,
    ):
        self.run_dir = run_dir
        self.log_dir = os.path.join(run_dir, "logs")
        self.stage_name = stage_name
        self.history_path = os.path.join(self.log_dir, "history.json")
        self.dashboard_path = dashboard_path or os.path.join(
            run_dir, "training_dashboard.png"
        )
        os.makedirs(os.path.dirname(self.dashboard_path), exist_ok=True)

        self.history = {"train": [], "eval": {}}
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path) as f:
                    self.history = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

    def log_train_step(self, global_step: int, epoch: int, loss: float, lr: float):
        entry = {"step": global_step, "epoch": epoch, "loss": loss, "lr": lr}
        if self.stage_name is not None:
            entry["stage"] = self.stage_name
        self.history["train"].append(entry)

    def log_eval(
        self, split: str, dataset_name: str, epoch: int, global_step: int, metrics: dict
    ):
        """`split` is e.g. "train_eval", "val", or "test" -- keeps periodic
        evaluation results separate from optimization-time training loss.
        `dataset_name` keeps different datasets within the same split (e.g.
        bpRNA-new / ArchiveII / bpRNA-spot0 test sets) separate in all plots."""
        key = f"{split}/{dataset_name}"
        self.history["eval"].setdefault(key, [])
        entry = {"epoch": epoch, "step": global_step}
        if self.stage_name is not None:
            entry["stage"] = self.stage_name
        entry.update(metrics)
        self.history["eval"][key].append(entry)

    def save(self):
        temporary = f"{self.history_path}.partial.{os.getpid()}"
        try:
            with open(temporary, "w") as f:
                json.dump(self.history, f, indent=2)
            os.replace(temporary, self.history_path)
        finally:
            try:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            except OSError:
                pass

    def render(self):
        """Render only the stage dashboard; detail curves are intentionally disabled."""
        self._render_dashboard()

    def _plot_train_scalar(
        self,
        key: str,
        title: str,
        path: str | None,
        log_scale: bool = False,
        ax=None,
        stage_name: str | None = None,
    ):
        train = self.history["train"]
        if stage_name is not None:
            train = [entry for entry in train if entry.get("stage") == stage_name]
        if not train:
            if ax is not None:
                ax.set_title(f"{title} (no data yet)")
            return
        steps = [e["step"] for e in train]
        values = [e[key] for e in train]
        owns_fig = ax is None
        if owns_fig:
            fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, values, color="tab:blue", linewidth=1.2)
        ax.set_xlabel("global step")
        ax.set_ylabel(key)
        ax.set_title(title)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        if owns_fig:
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)

    def _plot_metric_on_ax(self, ax, metric: str, splits=("val", "test")) -> bool:
        """Plot `metric` (e.g. "f1") vs. epoch on `ax`, one line per dataset,
        restricted to the given splits. Returns True if anything was plotted."""
        eval_hist = self.history["eval"]
        has_data = False
        for key, entries in sorted(eval_hist.items()):
            split = key.split("/", 1)[0]
            if split not in splits:
                continue
            xs = [e["epoch"] for e in entries if metric in e]
            ys = [e[metric] for e in entries if metric in e]
            if not xs:
                continue
            styles = {
                "train_eval": (":", "^"),
                "val": ("-", "o"),
                "test": ("--", "s"),
            }
            style, marker = styles.get(split, ("-", "o"))
            ax.plot(
                xs, ys, style, marker=marker, markersize=4, label=key, linewidth=1.4
            )
            has_data = True
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{metric} (per-RNA macro)")
        if has_data:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        return has_data

    def _render_dashboard(self):
        """Render ONE combined overview image with 8 panels:
            row 1: train loss | train/val f1 | train/val precision | train/val recall
            row 2: learning rate | test f1 | test precision | test recall
        One line per dataset within each val/test panel (e.g. bpRNA-new,
        ArchiveII, bpRNA-spot0 test sets stay visually separate).
        """
        fig, axes = plt.subplots(2, 4, figsize=(26, 10))
        try:
            self._plot_train_scalar(
                "loss",
                "Training Loss (current config)",
                path=None,
                ax=axes[0, 0],
                stage_name=self.stage_name,
            )
            self._plot_train_scalar(
                "lr", "Learning Rate", path=None, log_scale=True, ax=axes[1, 0]
            )

            for col, metric in enumerate(self._METRICS, start=1):
                self._plot_metric_on_ax(
                    axes[0, col], metric, splits=("train_eval", "val")
                )
                axes[0, col].set_title(f"Train/Val macro {metric}")

            for col, metric in enumerate(self._METRICS, start=1):
                self._plot_metric_on_ax(axes[1, col], metric, splits=("test",))
                axes[1, col].set_title(f"Test macro {metric}")

            title = "BGMFold Training Dashboard"
            if self.stage_name is not None:
                title += f" — {self.stage_name}"
            fig.suptitle(title, fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(self.dashboard_path, dpi=120)
        finally:
            plt.close(fig)
