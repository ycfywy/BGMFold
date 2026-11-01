"""Train the discriminative RNA secondary-structure contact-map predictor."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from bgmfold.batch_accumulation import backward_original_batch
from bgmfold.continuation import ContinuationProgress, require_complete_initialization
from bgmfold.data.datasets import (
    build_length_bucketed_eval_dataloaders,
    build_length_bucketed_train_dataloader,
)
from bgmfold.losses import (
    compute_multitask_losses,
    masked_contact_loss,
)
from bgmfold.metrics import (
    confusion_counts_per_sequence,
    greedy_project_contacts,
    logits_to_probs_and_preds,
    metrics_from_confusion_counts,
    metrics_per_sequence_from_confusion_counts,
    probability_threshold_logit,
)
from bgmfold.models.discriminative_contact_map_model import (
    build_discriminative_contact_map_model,
)
from bgmfold.threshold_grid import default_threshold_candidates
from bgmfold.utils import (
    AverageMeter,
    build_optimizer,
    build_pair_mask,
    build_scheduler,
    get_logger,
    load_checkpoint,
    load_compatible_initialization,
    now_cst_str,
    save_checkpoint,
    set_seed,
)
from bgmfold.visualize import TrainingCurveLogger, save_sample_visualization

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument("--resume-run-dir", type=str, default=None)
    parser.add_argument("--dashboard-path", type=str, default=None)
    parser.set_defaults(
        stage_name=None,
        stage_num_epochs=None,
        stage_resume_from=None,
        stage_reset_optimizer=False,
        stage_scheduler_num_epochs=None,
    )
    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    timeout_minutes = int(os.environ.get("DDP_TIMEOUT_MINUTES", "180"))
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout_minutes))
    return True, rank, local_rank, world_size


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            # Never mask the original training exception with teardown noise.
            pass


def broadcast_object(value, is_distributed: bool, rank: int):
    if not is_distributed:
        return value
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def broadcast_model_state(model: torch.nn.Module, is_distributed: bool) -> None:
    """Synchronize model parameters/buffers without making every rank read a checkpoint."""
    if not is_distributed:
        return
    for parameter in model.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)


def epoch_seed(base_seed: int, epoch: int, rank: int, world_size: int) -> int:
    """Stable per-epoch/per-rank RNG seed, including after epoch-boundary resume."""
    return int(base_seed) + int(epoch) * int(world_size) + int(rank)


def publish_checkpoint(local_path: str, publish_dir: str | None, logger=None) -> None:
    """Atomically copy a completed checkpoint to persistent storage."""
    if not publish_dir:
        return
    os.makedirs(publish_dir, exist_ok=True)
    destination = os.path.join(publish_dir, os.path.basename(local_path))
    temporary = f"{destination}.partial.{os.getpid()}"
    try:
        shutil.copy2(local_path, temporary)
        os.replace(temporary, destination)
        if logger:
            logger.info(f"Published checkpoint: {destination}")
    except OSError as exc:
        if logger:
            logger.warning(
                f"Failed to publish checkpoint {local_path} -> {destination}: {exc}"
            )
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass


def save_json_atomic(path: str, payload: dict) -> str:
    """Atomically persist small checkpoint metadata sidecars."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.partial.{os.getpid()}"
    try:
        with open(temporary, "w") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, path)
    finally:
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass
    return path


def run_optional_artifact(callback, description: str, logger=None):
    """Run non-essential reporting without sacrificing a training job."""
    try:
        return callback()
    except Exception as exc:  # plotting backends and FUSE writes vary by host
        if logger:
            logger.warning(f"Skipped {description} after a non-fatal error: {exc}")
        return None


def _count_sample_accumulation_windows(
    global_batch_sizes: list[int], target_global_samples: int
) -> int:
    """Count optimizer updates produced by sample-count accumulation."""
    if target_global_samples <= 0:
        return len(global_batch_sizes)
    updates = 0
    accumulated = 0
    for batch_size in global_batch_sizes:
        accumulated += int(batch_size)
        if accumulated >= target_global_samples:
            updates += 1
            accumulated = 0
    return updates + int(accumulated > 0)


def _scheduler_steps_for_epochs(
    batch_sampler,
    world_size: int,
    target_global_samples: int,
    num_epochs: int,
) -> int:
    """Replay deterministic sampler schedules to size the LR scheduler exactly."""
    original_epoch = int(getattr(batch_sampler, "epoch", 0))
    total = 0
    try:
        for epoch in range(int(num_epochs)):
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)
            total += _count_sample_accumulation_windows(
                [len(batch) * world_size for batch in batch_sampler],
                target_global_samples,
            )
    finally:
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(original_epoch)
    return total


def _normalize_accumulated_gradients(
    model: torch.nn.Module, accumulated_global_samples: int
) -> None:
    """Convert accumulated global per-RNA gradient sums back to their mean."""
    denominator = max(float(accumulated_global_samples), 1.0)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(denominator)


def build_direct_model(model_cfg: dict) -> tuple[torch.nn.Module, str]:
    model_type = model_cfg.get("type", "discriminative_contact_map")
    if model_type == "discriminative_contact_map":
        return build_discriminative_contact_map_model(model_cfg), model_type
    raise ValueError(
        f"Unknown model.type={model_type!r}; only 'discriminative_contact_map' is supported."
    )


def compute_multitask_eval_counts(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    mask: torch.Tensor,
    min_sequence_separation: int,
) -> dict[str, torch.Tensor]:
    """Return per-RNA auxiliary counts for macro and micro evaluation."""
    counts = {}
    if "pairedness_logits" in outputs:
        valid = mask
        predicted = outputs["pairedness_logits"] > 0
        target = batch["paired_tokens"] > 0.5
        tp = (predicted & target & valid).sum(dim=1)
        fp = (predicted & ~target & valid).sum(dim=1)
        fn = (~predicted & target & valid).sum(dim=1)
        counts["pairedness"] = torch.stack((tp, fp, fn), dim=1).to(torch.float64)
    if "pair_type_logits" in outputs:
        pair_mask = build_pair_mask(
            mask, min_sequence_separation=min_sequence_separation
        )
        target = batch["pair_type_map"]
        valid = pair_mask & (target >= 0)
        predicted = outputs["pair_type_logits"].argmax(dim=-1)
        correct = ((predicted == target) & valid).sum(dim=(1, 2))
        total = valid.sum(dim=(1, 2))
        counts["pair_type"] = torch.stack((correct, total), dim=1).to(torch.float64)
    return counts


def _truncate_batch(batch: dict, size: int) -> dict:
    """Slice samples and repad labels to the retained sequences' maximum length.

    The encoder pads its output from ``sequences`` again. Dropping the longest
    sample must therefore also shrink every length-dependent target/mask.
    """
    result = {}
    batch_size = len(batch["sequences"])
    for key, value in batch.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == batch_size
        ):
            result[key] = value[:size]
        elif isinstance(value, (list, tuple)) and len(value) == batch_size:
            result[key] = value[:size]
        else:
            result[key] = value
    max_length = max((len(sequence) for sequence in result["sequences"]), default=0)
    for key in ("contact_map", "pair_type_map"):
        if key in result:
            result[key] = result[key][:, :max_length, :max_length]
    for key in ("paired_tokens", "mask"):
        if key in result:
            result[key] = result[key][:, :max_length]
    return result


@torch.no_grad()
def evaluate_supervised(
    model,
    dataloaders,
    device,
    threshold,
    min_sequence_separation=1,
    greedy_projection=False,
    logger=None,
    visualize_cfg=None,
    vis_dir=None,
    epoch=0,
    max_samples_per_dataset=None,
    split="eval",
    multitask_enabled=False,
    amp_enabled=False,
    amp_dtype=torch.bfloat16,
    distributed=False,
    rank=0,
    world_size=1,
    greedy_logit_ranking=False,
    prediction_cache_dir=None,
):
    model.eval()
    all_results = {}
    for name, loader in dataloaders.items():
        contact_counts = torch.zeros(4, dtype=torch.float64, device=device)
        contact_metric_sums = torch.zeros(5, dtype=torch.float64, device=device)
        pairedness_counts = torch.zeros(3, dtype=torch.float64, device=device)
        pairedness_f1_sum = torch.zeros((), dtype=torch.float64, device=device)
        pair_type_counts = torch.zeros(2, dtype=torch.float64, device=device)
        pair_type_accuracy_sum = torch.zeros((), dtype=torch.float64, device=device)
        pairedness_samples = 0
        pair_type_samples = 0
        sample_offset = 0
        vis_indices = set()
        if (
            visualize_cfg
            and visualize_cfg.get("enabled", False)
            and vis_dir is not None
        ):
            num_samples = min(
                visualize_cfg.get("num_samples_per_dataset", 3), len(loader.dataset)
            )
            if num_samples > 0:
                g = torch.Generator().manual_seed(epoch + 1234)
                vis_indices = set(
                    torch.randperm(len(loader.dataset), generator=g)[
                        :num_samples
                    ].tolist()
                )

        samples_seen = 0
        local_sample_limit = max_samples_per_dataset
        if distributed and max_samples_per_dataset is not None:
            global_limit = max(0, int(max_samples_per_dataset))
            local_sample_limit = (
                0
                if global_limit <= rank
                else (global_limit - 1 - rank) // world_size + 1
            )
        for batch in loader:
            if local_sample_limit is not None and samples_seen >= local_sample_limit:
                break
            if local_sample_limit is not None:
                remaining = int(local_sample_limit) - samples_seen
                if remaining < len(batch["sequences"]):
                    batch = _truncate_batch(batch, remaining)
            sequences = batch["sequences"]
            target = batch["contact_map"].to(device)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                if multitask_enabled:
                    model_output, mask = model(sequences, return_auxiliary=True)
                else:
                    model_output, mask = model(sequences)
            if multitask_enabled:
                logits = model_output["contact_logits"]
                if "pairedness_logits" in model_output:
                    batch["paired_tokens"] = batch["paired_tokens"].to(device)
                if "pair_type_logits" in model_output:
                    batch["pair_type_map"] = batch["pair_type_map"].to(device)
                auxiliary_counts = compute_multitask_eval_counts(
                    model_output, batch, mask, min_sequence_separation
                )
            else:
                logits = model_output
                auxiliary_counts = {}
            if prediction_cache_dir:
                from bgmfold.evaluation_snapshot import save_batch

                run_optional_artifact(
                    lambda: save_batch(
                        prediction_cache_dir, epoch + 1, split, batch, logits, target
                    ),
                    description=f"{split}/{name} prediction snapshot",
                    logger=logger,
                )
            probs, preds = logits_to_probs_and_preds(logits, threshold=threshold)
            if greedy_projection:
                preds = greedy_project_contacts(
                    probs,
                    mask,
                    threshold=threshold,
                    min_sequence_separation=min_sequence_separation,
                    ranking_logits=logits if greedy_logit_ranking else None,
                )
            batch_contact_counts = confusion_counts_per_sequence(
                preds, target, mask, min_sequence_separation=min_sequence_separation
            )
            contact_counts += batch_contact_counts.sum(dim=0)
            contact_metric_sums += metrics_per_sequence_from_confusion_counts(
                batch_contact_counts
            ).sum(dim=0)
            if "pairedness" in auxiliary_counts:
                batch_pairedness_counts = auxiliary_counts["pairedness"]
                pairedness_counts += batch_pairedness_counts.sum(dim=0)
                tp, fp, fn = batch_pairedness_counts.unbind(dim=1)
                pairedness_f1_sum += (
                    2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1e-8)
                ).sum()
                pairedness_samples += int(batch_pairedness_counts.shape[0])
            if "pair_type" in auxiliary_counts:
                batch_pair_type_counts = auxiliary_counts["pair_type"]
                pair_type_counts += batch_pair_type_counts.sum(dim=0)
                correct, total = batch_pair_type_counts.unbind(dim=1)
                valid_sequences = total > 0
                pair_type_accuracy_sum += (
                    correct[valid_sequences] / total[valid_sequences]
                ).sum()
                pair_type_samples += int(valid_sequences.sum().item())

            if vis_indices and vis_dir is not None:
                for local_i in range(len(sequences)):
                    global_i = sample_offset + local_i
                    if global_i in vis_indices:
                        seq_len = batch["lengths"][local_i].item()
                        run_optional_artifact(
                            lambda local_i=local_i, seq_len=seq_len: (
                                save_sample_visualization(
                                    save_dir=vis_dir,
                                    dataset_name=f"{split}_{name}",
                                    epoch=epoch,
                                    sample_id=batch["ids"][local_i],
                                    length=seq_len,
                                    gt_map=target[local_i].float().cpu().numpy(),
                                    prob_map=probs[local_i].float().cpu().numpy(),
                                    pred_map=preds[local_i].float().cpu().numpy(),
                                )
                            ),
                            description=f"{split}/{name} visualization",
                            logger=logger,
                        )
            sample_offset += len(sequences)
            samples_seen += len(sequences)

        if distributed:
            aggregate = torch.cat(
                (
                    contact_counts,
                    contact_metric_sums,
                    pairedness_counts,
                    pairedness_f1_sum.unsqueeze(0),
                    pair_type_counts,
                    pair_type_accuracy_sum.unsqueeze(0),
                    torch.tensor(
                        [samples_seen, pairedness_samples, pair_type_samples],
                        dtype=torch.float64,
                        device=device,
                    ),
                )
            )
            dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
            contact_counts = aggregate[:4]
            contact_metric_sums = aggregate[4:9]
            pairedness_counts = aggregate[9:12]
            pairedness_f1_sum = aggregate[12]
            pair_type_counts = aggregate[13:15]
            pair_type_accuracy_sum = aggregate[15]
            samples_seen = int(aggregate[16].item())
            pairedness_samples = int(aggregate[17].item())
            pair_type_samples = int(aggregate[18].item())

        metric_names = ("precision", "recall", "f1", "mcc", "accuracy")
        results = {
            metric: float((contact_metric_sums[index] / max(samples_seen, 1)).item())
            for index, metric in enumerate(metric_names)
        }
        micro_results = metrics_from_confusion_counts(contact_counts)
        results.update(
            {f"{metric}_micro": value for metric, value in micro_results.items()}
        )
        if pairedness_samples:
            tp, fp, fn = pairedness_counts.unbind()
            results["pairedness_f1"] = float(
                (pairedness_f1_sum / pairedness_samples).item()
            )
            results["pairedness_f1_micro"] = float(
                (2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1e-8)).item()
            )
        if pair_type_samples:
            correct, total = pair_type_counts.unbind()
            results["pair_type_accuracy"] = float(
                (pair_type_accuracy_sum / pair_type_samples).item()
            )
            results["pair_type_accuracy_micro"] = float(
                (correct / total.clamp_min(1.0)).item()
            )
        all_results[name] = results
        if logger:
            metrics_str = " ".join(f"{k}={v:.4f}" for k, v in results.items())
            logger.info(
                f"[Eval:{split}] dataset={name} n={samples_seen} "
                f"aggregation=per_sequence_macro {metrics_str}"
            )
    return all_results


@torch.no_grad()
def select_validation_threshold(
    model,
    dataloaders,
    device,
    thresholds,
    min_sequence_separation,
    greedy_projection,
    multitask_enabled,
    amp_enabled,
    amp_dtype,
    distributed=False,
    prefer_higher_on_tie=False,
    greedy_logit_ranking=False,
):
    """Select one decision threshold using validation data only."""
    candidates = sorted({float(value) for value in thresholds})
    if not candidates or candidates[0] < 0.0 or candidates[-1] > 1.0:
        raise ValueError("eval.threshold_candidates must contain values within [0, 1]")
    model.eval()
    by_dataset = {}
    for dataset_name, loader in dataloaders.items():
        f1_sums = {
            threshold: torch.zeros((), dtype=torch.float64, device=device)
            for threshold in candidates
        }
        sample_count = torch.zeros((), dtype=torch.float64, device=device)
        for batch in loader:
            sequences = batch["sequences"]
            target = batch["contact_map"].to(device)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                if multitask_enabled:
                    outputs, mask = model(sequences, return_auxiliary=True)
                    logits = outputs["contact_logits"]
                else:
                    logits, mask = model(sequences)
            probs = torch.sigmoid(logits.float())
            greedy_selected = (
                greedy_project_contacts(
                    probs,
                    mask,
                    # Edges below the smallest candidate can never appear in
                    # any scanned result, so do not sort/match the full L^2 map.
                    threshold=candidates[0],
                    min_sequence_separation=min_sequence_separation,
                    ranking_logits=logits if greedy_logit_ranking else None,
                )
                if greedy_projection
                else None
            )
            for threshold in candidates:
                preds = (
                    (logits.double() >= probability_threshold_logit(threshold)).float()
                    if greedy_logit_ranking
                    else (probs >= threshold).float()
                )
                if greedy_selected is not None:
                    preds = preds * greedy_selected
                per_sequence_counts = confusion_counts_per_sequence(
                    preds,
                    target,
                    mask,
                    min_sequence_separation=min_sequence_separation,
                )
                f1_sums[threshold] += metrics_per_sequence_from_confusion_counts(
                    per_sequence_counts
                )[:, 2].sum()
            sample_count += len(sequences)
        if distributed:
            aggregate = torch.cat(
                (
                    torch.stack([f1_sums[threshold] for threshold in candidates]),
                    sample_count.unsqueeze(0),
                )
            )
            dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
            f1_sums = {
                threshold: aggregate[index]
                for index, threshold in enumerate(candidates)
            }
            sample_count = aggregate[-1]
        by_dataset[dataset_name] = {
            threshold: float((value / sample_count.clamp_min(1.0)).item())
            for threshold, value in f1_sums.items()
        }

    mean_f1 = {
        threshold: sum(results[threshold] for results in by_dataset.values())
        / max(len(by_dataset), 1)
        for threshold in candidates
    }
    selected = max(
        candidates,
        key=lambda threshold: (
            mean_f1[threshold],
            threshold if prefer_higher_on_tie else -threshold,
        ),
    )
    return selected, {
        "aggregation": "per_sequence_macro",
        "mean_f1": mean_f1,
        "by_dataset": by_dataset,
    }


def report_single_gpu_threshold_scan(
    model,
    val_loaders,
    test_loaders,
    device,
    eval_cfg,
    epoch,
    global_step,
    log_dir,
    logger,
    curve_logger,
    writer,
    min_sequence_separation,
    greedy_projection,
    multitask_enabled,
    amp_enabled,
    amp_dtype,
):
    """Extra diagnostic only: fixed-threshold training selection stays untouched."""

    def limited_batches(loader, limit):
        seen = 0
        for batch in loader:
            if limit is not None and seen >= limit:
                break
            if limit is not None and len(batch["sequences"]) > limit - seen:
                batch = _truncate_batch(batch, limit - seen)
            seen += len(batch["sequences"])
            yield batch

    val_limit = eval_cfg.get("max_val_samples_per_dataset")
    scan_loaders = {
        name: limited_batches(loader, val_limit) for name, loader in val_loaders.items()
    }
    # Extra diagnostic passes must not advance the training Torch RNG stream.
    was_training = model.training
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        try:
            selected, scan = select_validation_threshold(
                model,
                scan_loaders,
                device,
                default_threshold_candidates(),
                min_sequence_separation,
                greedy_projection,
                multitask_enabled,
                amp_enabled,
                amp_dtype,
                prefer_higher_on_tie=bool(
                    eval_cfg.get("threshold_tie_prefer_higher", False)
                ),
                greedy_logit_ranking=bool(eval_cfg.get("greedy_logit_ranking", False)),
            )
            payload = {
                "epoch": epoch + 1,
                "selected_threshold": selected,
                "threshold_selected_on": "val",
                "selection_affects_checkpoint": False,
                "validation_scan": scan,
                "metrics": {},
            }
            for split, loaders, limit in (
                ("val_scanned", val_loaders, val_limit),
                (
                    "test_scanned",
                    test_loaders,
                    eval_cfg.get("max_test_samples_per_dataset"),
                ),
            ):
                if not loaders:
                    continue
                logger.info(
                    "[Threshold scan] epoch=%d split=%s threshold=%.4f selected_on=val; "
                    "fixed_threshold=%.4f remains checkpoint criterion",
                    epoch + 1,
                    split,
                    selected,
                    float(eval_cfg.get("threshold", 0.5)),
                )
                results = evaluate_supervised(
                    model,
                    loaders,
                    device,
                    selected,
                    min_sequence_separation=min_sequence_separation,
                    greedy_projection=greedy_projection,
                    logger=logger,
                    epoch=epoch,
                    max_samples_per_dataset=limit,
                    split=split,
                    multitask_enabled=multitask_enabled,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    greedy_logit_ranking=bool(
                        eval_cfg.get("greedy_logit_ranking", False)
                    ),
                )
                payload["metrics"][split] = results
                for name, metrics in results.items():
                    curve_logger.log_eval(split, name, epoch, global_step, metrics)
                    if writer:
                        for metric, value in metrics.items():
                            writer.add_scalar(
                                f"{split}/{name}/{metric}", value, global_step
                            )
                if writer:
                    writer.add_scalar(
                        f"{split}/selected_threshold", selected, global_step
                    )
            save_json_atomic(
                os.path.join(
                    log_dir, f"diagnostic_threshold_scan_epoch_{epoch + 1:04d}.json"
                ),
                payload,
            )
            curve_logger.save()
        finally:
            model.train(was_training)


def main():
    args = parse_args()
    if len(args.config) != 1:
        raise ValueError("Use one complete training configuration")

    config_path = args.config[0]
    is_distributed, rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    trainer_type = cfg.get("trainer", {}).get("type")
    if trainer_type != "direct_contact_map":
        cleanup_distributed(is_distributed)
        raise ValueError(
            f"Unsupported trainer.type={trainer_type!r}; only 'direct_contact_map' is supported."
        )
    expected_world_size = cfg.get("distributed", {}).get("expected_world_size")
    if expected_world_size is not None and int(expected_world_size) != world_size:
        cleanup_distributed(is_distributed)
        raise RuntimeError(
            f"Expected {int(expected_world_size)} distributed processes, got {world_size}. "
            f"Launch this config with torchrun --nproc_per_node={int(expected_world_size)}."
        )

    if args.stage_num_epochs is not None:
        cfg["train"]["num_epochs"] = args.stage_num_epochs
    if args.stage_resume_from is not None:
        cfg["train"]["resume_from"] = args.stage_resume_from

    exp_cfg = cfg["experiment"]
    run_dir = None
    if is_main:
        run_dir = args.resume_run_dir or os.path.join(
            exp_cfg["output_dir"], f"{exp_cfg['name']}_{now_cst_str('%Y%m%d_%H%M%S')}"
        )
    run_dir = broadcast_object(run_dir, is_distributed, rank)
    log_dir = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    vis_dir = os.path.join(run_dir, "visualizations")
    checkpoint_publish_root = cfg.get("train", {}).get("checkpoint_publish_root")
    checkpoint_publish_dir = (
        os.path.join(
            checkpoint_publish_root, os.path.basename(os.path.normpath(run_dir))
        )
        if checkpoint_publish_root
        else None
    )
    if is_main:
        for d in (log_dir, ckpt_dir, vis_dir):
            os.makedirs(d, exist_ok=True)
        if checkpoint_publish_dir:
            os.makedirs(checkpoint_publish_dir, exist_ok=True)
        shutil.copy2(config_path, os.path.join(run_dir, "resolved_config.yaml"))
        resume_best_from = cfg.get("train", {}).get("resume_best_from")
        if resume_best_from:
            shutil.copy2(resume_best_from, os.path.join(ckpt_dir, "best.pt"))
    if is_distributed:
        dist.barrier()

    set_seed(int(exp_cfg["seed"]) + rank)
    reseed_each_epoch = bool(cfg["train"].get("reseed_each_epoch", True))
    logger = get_logger(
        f"bgmfold.train_supervised_pair.rank{rank}",
        log_file=os.path.join(log_dir, "train.log") if is_main else None,
    )
    if is_main:
        logger.info(f"Run directory: {run_dir}")

    device = torch.device(
        f"cuda:{local_rank}"
        if is_distributed
        else (exp_cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        allow_tf32 = bool(cfg.get("train", {}).get("allow_tf32", False))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    logger.info(f"Using device: {device}, rank={rank}/{world_size}")

    data_cfg = cfg["data"]
    sampler_seed = int(data_cfg.get("sampler_seed", exp_cfg["seed"]))
    if cfg.get("eval", {}).get("report_threshold_scan") and (
        is_distributed or cfg["eval"].get("threshold_candidates")
    ):
        raise ValueError(
            "report_threshold_scan is single-GPU diagnostic mode; "
            "do not combine it with checkpoint-selecting threshold_candidates"
        )
    train_loader = build_length_bucketed_train_dataloader(
        data_cfg["train_datasets"],
        data_cfg.get("max_length"),
        data_cfg["num_workers"],
        data_cfg["bucket_width"],
        data_cfg["budget"],
        data_cfg["length_power"],
        data_cfg["min_batch_size"],
        data_cfg["max_batch_size"],
        augmentation_cfg=data_cfg.get("covariation_augmentation"),
        distributed=is_distributed,
        world_size=world_size,
        rank=rank,
        distributed_batch_mode=data_cfg.get("distributed_batch_mode", "same_bucket"),
        seed=sampler_seed,
        batch_size_by_bucket=data_cfg.get("batch_size_by_bucket"),
    )
    if data_cfg.get("batch_size_by_bucket") is not None:
        logger.info(
            "Measured training batch table=%s; batches/epoch=%d; sampler_seed=%d; initialization_seed=%d",
            data_cfg["batch_size_by_bucket"],
            len(train_loader),
            sampler_seed,
            int(exp_cfg["seed"]),
        )

    def build_eval_loaders(dataset_cfgs):
        return build_length_bucketed_eval_dataloaders(
            dataset_cfgs,
            data_cfg.get("max_length"),
            data_cfg.get("eval_num_workers", data_cfg["num_workers"]),
            data_cfg["bucket_width"],
            data_cfg.get("eval_budget", data_cfg["budget"]),
            data_cfg["length_power"],
            data_cfg.get("eval_min_batch_size", data_cfg["min_batch_size"]),
            data_cfg.get("eval_max_batch_size", data_cfg["max_batch_size"]),
            distributed=is_distributed,
            world_size=world_size,
            rank=rank,
        )

    val_loaders = build_eval_loaders(data_cfg["val_datasets"])
    train_eval_every = int(cfg["train"].get("train_eval_every_n_epochs", 0))
    train_eval_loaders = (
        build_eval_loaders(data_cfg["train_datasets"]) if train_eval_every > 0 else None
    )
    test_loaders = (
        build_eval_loaders(data_cfg["test_datasets"])
        if int(cfg["train"].get("test_every_n_epochs", 0)) > 0
        else None
    )
    if is_main:
        logger.info(
            f"Train dataset size: {len(train_loader.dataset)}, local batches/epoch: {len(train_loader)}, "
            f"world_size={world_size}"
        )

    model, model_type = build_direct_model(cfg["model"])
    resume_path = cfg["train"].get("resume_from")
    initialization_cfg = cfg.get("initialization", {}) or {}
    initialization_path = initialization_cfg.get("checkpoint")
    if (
        cfg["model"].get("rna_encoder_initialize_from_config_only", False)
        and not initialization_path
    ):
        raise ValueError(
            "model.rna_encoder_initialize_from_config_only requires "
            "initialization.checkpoint; otherwise the encoder would remain random."
        )
    if resume_path and initialization_path:
        raise ValueError(
            "train.resume_from and initialization.checkpoint are mutually exclusive: "
            "resume restores training state, while initialization starts a new run."
        )
    initialization_report = None
    initialization_summary = None
    if initialization_path:
        initialization_result = None
        if is_main:
            try:
                initialization_report = load_compatible_initialization(
                    initialization_path,
                    model,
                    map_location="cpu",
                    require_encoder=bool(
                        initialization_cfg.get("require_encoder", True)
                    ),
                    min_encoder_coverage=float(
                        initialization_cfg.get("min_encoder_coverage", 0.99)
                    ),
                )
                if initialization_cfg.get("require_complete_model", False):
                    require_complete_initialization(initialization_report)
                initialization_result = {
                    "report": initialization_report,
                    "error": None,
                }
            except Exception as exc:
                initialization_result = {
                    "report": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        initialization_result = broadcast_object(
            initialization_result, is_distributed, rank
        )
        if initialization_result["error"]:
            raise RuntimeError(
                "Failed to initialize from task checkpoint: "
                f"{initialization_result['error']}"
            )
        initialization_report = initialization_result["report"]
        initialization_report["source_checkpoint"] = initialization_cfg.get(
            "source_checkpoint", initialization_path
        )
        initialization_summary = {
            "source_checkpoint": initialization_report["source_checkpoint"],
            "encoder_coverage": initialization_report["encoder_coverage"],
            "loaded_tensor_count": initialization_report["loaded_tensor_count"],
            "target_tensor_count": initialization_report["target_tensor_count"],
            "components": initialization_report["components"],
        }
        if is_main:
            report_path = save_json_atomic(
                os.path.join(log_dir, "initialization_report.json"),
                initialization_report,
            )
            logger.info(
                "Initialized model from task checkpoint: "
                f"{initialization_path}; report={report_path}; "
                f"loaded={initialization_report['loaded_tensor_count']}/"
                f"{initialization_report['target_tensor_count']} target tensors; "
                f"encoder_coverage={initialization_report['encoder_coverage']:.2%}."
            )
            if is_distributed:
                logger.info(
                    "Only rank 0 read the initialization checkpoint; DDP will "
                    "broadcast the initialized parameters to all other ranks."
                )
            for component, summary in initialization_report["components"].items():
                logger.info(
                    f"[Init:{component}] loaded={summary['loaded_tensors']}/"
                    f"{summary['target_tensors']} tensors "
                    f"coverage={summary['coverage']:.2%}"
                )
            for skipped in initialization_report["skipped_source"]:
                logger.warning(
                    f"[Init:skipped-source] key={skipped['source_key']} "
                    f"reason={skipped['reason']}"
                )
            for random_target in initialization_report["random_target"]:
                logger.warning(
                    f"[Init:random-target] key={random_target['target_key']} "
                    f"reason={random_target['reason']}"
                )
    encoder_freeze_epochs = max(0, int(cfg["train"].get("encoder_freeze_epochs", 0)))
    if encoder_freeze_epochs > 0:
        if bool(
            cfg["model"].get(
                "rna_encoder_freeze", cfg["model"].get("rnafm_freeze", True)
            )
        ):
            raise ValueError(
                "train.encoder_freeze_epochs requires model.rna_encoder_freeze=false "
                "so the encoder can be unfrozen later."
            )
        if is_distributed and (
            bool(cfg.get("distributed", {}).get("static_graph", True))
            or not bool(cfg.get("distributed", {}).get("find_unused_parameters", False))
        ):
            raise ValueError(
                "Phased encoder freezing under DDP requires static_graph=false and "
                "find_unused_parameters=true."
            )
        model.rnafm.set_runtime_freeze(True)
    model = model.to(device)
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(
                cfg.get("distributed", {}).get("find_unused_parameters", False)
            ),
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            bucket_cap_mb=int(cfg.get("distributed", {}).get("bucket_cap_mb", 100)),
            static_graph=bool(cfg.get("distributed", {}).get("static_graph", True)),
        )
    raw_model = model.module if hasattr(model, "module") else model
    optimizer = build_optimizer(model, cfg["optim"])
    scheduler_epochs = args.stage_scheduler_num_epochs or cfg["train"]["num_epochs"]
    accumulation_target = max(
        0,
        int(cfg["train"].get("gradient_accumulation_target_global_samples", 0)),
    )
    scheduler_steps = _scheduler_steps_for_epochs(
        train_loader.batch_sampler,
        world_size,
        accumulation_target,
        scheduler_epochs,
    )
    scheduler_cfg = dict(cfg["optim"])
    if scheduler_cfg.get("scheduler") == "cosine_with_encoder_warmup":
        # Resolve epochs to exact optimizer-update counts. Length-bucket counts
        # can vary by epoch; do not approximate with len(loader) * epochs.
        def steps_through(epochs):
            return _scheduler_steps_for_epochs(
                train_loader.batch_sampler, world_size, accumulation_target, epochs
            )

        task_warmup_epochs = int(scheduler_cfg["warmup_epochs"])
        encoder_warmup_epochs = int(scheduler_cfg["rna_encoder_warmup_epochs"])
        if not (
            0
            < task_warmup_epochs
            <= encoder_freeze_epochs
            < encoder_freeze_epochs + encoder_warmup_epochs
            < scheduler_epochs
        ):
            raise ValueError(
                "Phased warmup must fit within encoder freeze and full training epochs"
            )
        scheduler_cfg["warmup_steps"] = steps_through(task_warmup_epochs)
        scheduler_cfg["rna_encoder_warmup_start_steps"] = steps_through(
            encoder_freeze_epochs
        )
        scheduler_cfg["rna_encoder_warmup_steps"] = (
            steps_through(encoder_freeze_epochs + encoder_warmup_epochs)
            - scheduler_cfg["rna_encoder_warmup_start_steps"]
        )
        if is_main:
            logger.info(
                "Fixed phased cosine: task_warmup_steps=%d encoder_start_step=%d "
                "encoder_warmup_steps=%d total_steps=%d min_lr_ratio=%s",
                scheduler_cfg["warmup_steps"],
                scheduler_cfg["rna_encoder_warmup_start_steps"],
                scheduler_cfg["rna_encoder_warmup_steps"],
                scheduler_steps,
                scheduler_cfg.get("min_lr_ratio", 0.1),
            )
    scheduler = build_scheduler(optimizer, scheduler_cfg, scheduler_steps)
    writer = (
        SummaryWriter(log_dir=log_dir)
        if (is_main and SummaryWriter is not None)
        else None
    )
    curve_logger = (
        TrainingCurveLogger(
            run_dir=run_dir,
            stage_name=args.stage_name,
            dashboard_path=args.dashboard_path,
        )
        if is_main
        else None
    )

    amp_enabled = cfg["train"].get("amp", True) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if cfg["train"].get("amp_dtype", "bfloat16") == "bfloat16"
        else torch.float16
    )
    loss_cfg = cfg["loss"]
    eval_cfg = cfg["eval"]
    multitask_model_cfg = cfg.get("model", {}).get("multitask", {})
    multitask_enabled = bool(multitask_model_cfg.get("enabled", False))
    pairedness_enabled = multitask_enabled and bool(
        multitask_model_cfg.get("pairedness", {}).get("enabled", True)
    )
    pair_type_enabled = multitask_enabled and bool(
        multitask_model_cfg.get("pair_type", {}).get("enabled", True)
    )
    if is_main and multitask_enabled:
        logger.info("Multi-task learning enabled: contact + configured auxiliary heads")
    min_sequence_separation = 1
    microbatch_splits = max(1, int(cfg["train"].get("microbatch_splits", 1)))
    if is_distributed and microbatch_splits != 1:
        raise ValueError(
            "DDP training uses per-rank batches and requires microbatch_splits=1"
        )
    distributed_batch_reduction = is_distributed and bool(
        cfg["train"].get("global_weighted_loss", True)
    )
    if (
        distributed_batch_reduction
        and loss_cfg.get("normalization", "pair") == "pair"
        and (
            float(loss_cfg.get("dice_weight", 0.0)) != 0.0
            or float(loss_cfg.get("degree_weight", 0.0)) != 0.0
        )
    ):
        raise ValueError(
            "global_weighted_loss with pair normalization requires "
            "dice_weight=degree_weight=0; per-sequence modes support them"
        )
    if microbatch_splits > 1 and (
        loss_cfg.get("dice_weight", 0) or loss_cfg.get("degree_weight", 0)
    ):
        raise ValueError("microbatch_splits requires dice_weight=degree_weight=0")
    if microbatch_splits > 1 and (
        loss_cfg.get("normalization") == "per_sequence_class_balanced"
        or loss_cfg.get("multitask", {}).get("normalization")
        == "per_sequence_class_balanced"
    ):
        raise ValueError(
            "microbatch_splits does not yet support per_sequence_class_balanced; "
            "use an unsplit batch or implement exact per-class full-batch denominators"
        )
    if microbatch_splits > 1 and accumulation_target > 0:
        raise ValueError(
            "gradient_accumulation_target_global_samples cannot be combined "
            "with microbatch_splits"
        )
    greedy_projection = bool(eval_cfg.get("greedy_projection", False))
    if is_main:
        logger.info(
            f"Direct model type: {model_type}; distributed={is_distributed}; "
            f"global_weighted_loss={distributed_batch_reduction}; "
            f"gradient_accumulation_target_global_samples={accumulation_target}; "
            f"scheduler_steps={scheduler_steps}; "
            f"encoder_freeze_epochs={encoder_freeze_epochs}; "
            f"torch={torch.__version__}; cuda={torch.version.cuda}; "
            f"gpu={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}; "
            f"tf32={bool(cfg['train'].get('allow_tf32', False))}"
        )
    save_optimizer_state = bool(cfg["train"].get("save_optimizer_state", True))
    early_cfg = cfg["train"].get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = max(1, int(early_cfg.get("patience", 3)))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    stage_best_metric = -float("inf")
    bad_eval_count = 0
    if is_main and early_enabled:
        logger.info(
            f"Early stopping enabled: monitor=val_f1 patience={early_patience} "
            f"evals min_delta={early_min_delta:.4f}"
        )
    best_metric = -float("inf")
    current_eval_threshold = float(eval_cfg["threshold"])
    global_step = 0
    start_epoch = 0
    if resume_path:
        resume_optimizer = None if args.stage_reset_optimizer else optimizer
        resume_scheduler = None if args.stage_reset_optimizer else scheduler
        state = load_checkpoint(
            resume_path,
            model,
            resume_optimizer,
            resume_scheduler,
            map_location=str(device),
        )
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_metric = state.get("best_metric", -float("inf"))
        checkpoint_freeze_epochs = state.get("encoder_freeze_epochs")
        if (
            checkpoint_freeze_epochs is not None
            and int(checkpoint_freeze_epochs) != encoder_freeze_epochs
        ):
            raise ValueError(
                "Resume checkpoint encoder_freeze_epochs does not match the current config: "
                f"{checkpoint_freeze_epochs} != {encoder_freeze_epochs}"
            )
        if state.get("stage_name") == args.stage_name:
            stage_best_metric = state.get("stage_best_metric", -float("inf"))
            bad_eval_count = int(state.get("bad_eval_count", 0))
        resume_state_path = os.path.join(
            os.path.dirname(resume_path), "training_state.json"
        )
        if os.path.isfile(resume_state_path):
            with open(resume_state_path) as handle:
                resume_state = json.load(handle)
            if int(resume_state.get("checkpoint_epoch", -1)) == int(state["epoch"]):
                best_metric = float(resume_state.get("best_metric", best_metric))
                if resume_state.get("stage_name") == args.stage_name:
                    stage_best_metric = float(
                        resume_state.get("stage_best_metric", stage_best_metric)
                    )
                    bad_eval_count = int(
                        resume_state.get("bad_eval_count", bad_eval_count)
                    )
        logger.info(
            f"Resumed from {resume_path} at epoch {start_epoch}, "
            f"global_step={global_step}; "
            f"optimizer/scheduler {'reset for changed trainable parameters' if args.stage_reset_optimizer else 'state restored'}."
        )

    def save_training_state(checkpoint_epoch: int) -> None:
        state_path = os.path.join(ckpt_dir, "training_state.json")
        save_json_atomic(
            state_path,
            {
                "checkpoint_epoch": checkpoint_epoch,
                "best_metric": best_metric,
                "stage_name": args.stage_name,
                "stage_best_metric": stage_best_metric,
                "bad_eval_count": bad_eval_count,
                "encoder_freeze_epochs": encoder_freeze_epochs,
            },
        )
        publish_checkpoint(state_path, checkpoint_publish_dir, logger)

    epoch_checkpoint_records = []

    def save_last_training_checkpoint(checkpoint_epoch: int) -> None:
        if not cfg["train"].get("save_last_checkpoint", True):
            save_training_state(checkpoint_epoch)
            return
        last_path = os.path.join(ckpt_dir, "last.pt")
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            checkpoint_epoch,
            global_step,
            best_metric,
            extra={
                "stage_name": args.stage_name,
                "stage_best_metric": stage_best_metric,
                "bad_eval_count": bad_eval_count,
                "epoch_seed_scheme": (
                    "base_seed + epoch * world_size + rank"
                    if reseed_each_epoch
                    else "startup_seed_only; continuous_rng"
                ),
                "encoder_freeze_epochs": encoder_freeze_epochs,
                "initialization": initialization_summary,
            },
            save_optimizer_state=save_optimizer_state,
        )
        if cfg["train"].get("keep_epoch_checkpoints", False):
            from bgmfold.epoch_checkpoint import retain_epoch_checkpoint

            epoch_checkpoint_records.append(
                retain_epoch_checkpoint(
                    last_path,
                    checkpoint_epoch + 1,
                    cfg["train"].get("epoch_checkpoint_publish_dir"),
                    logger=logger,
                )
            )
            save_json_atomic(
                os.path.join(log_dir, "epoch_checkpoints.json"),
                {"checkpoints": epoch_checkpoint_records},
            )
        publish_checkpoint(last_path, checkpoint_publish_dir, logger)
        save_training_state(checkpoint_epoch)

    continuation = None
    continuation_path = os.path.join(log_dir, "continuation_summary.json")
    if cfg["train"].get("evaluate_initial_checkpoint", False):
        if resume_path or not initialization_path or encoder_freeze_epochs:
            raise ValueError(
                "Continuation baseline requires fresh task initialization without freeze/resume"
            )
        if not early_enabled or not initialization_cfg.get(
            "require_complete_model", False
        ):
            raise ValueError(
                "Continuation requires complete task initialization and epoch-based early stopping"
            )
        if eval_cfg.get("max_val_samples_per_dataset") is not None:
            raise ValueError(
                "Continuation checkpoint selection must use the full validation set"
            )
        if cfg["train"].get("test_every_n_epochs") != cfg["train"].get(
            "eval_every_n_epochs"
        ):
            raise ValueError(
                "Continuation test diagnostics must align with validation evaluation epochs"
            )
        continuation = ContinuationProgress(
            initialization_cfg.get("source_checkpoint", initialization_path),
            int(cfg["train"]["num_epochs"]),
            int(early_cfg["patience_epochs"]),
            early_min_delta,
        )
        candidates = eval_cfg["threshold_candidates"]
        current_eval_threshold, scan = select_validation_threshold(
            raw_model,
            val_loaders,
            device,
            candidates,
            min_sequence_separation,
            greedy_projection,
            multitask_enabled,
            amp_enabled,
            amp_dtype,
            distributed=is_distributed,
            prefer_higher_on_tie=bool(
                eval_cfg.get("threshold_tie_prefer_higher", False)
            ),
            greedy_logit_ranking=bool(eval_cfg.get("greedy_logit_ranking", False)),
        )
        initial_results = {}
        for split, loaders in (("val", val_loaders), ("test", test_loaders)):
            if loaders is None:
                raise ValueError("Continuation baseline requires val and test loaders")
            initial_results[split] = evaluate_supervised(
                raw_model,
                loaders,
                device,
                current_eval_threshold,
                min_sequence_separation=min_sequence_separation,
                greedy_projection=greedy_projection,
                logger=logger if is_main else None,
                epoch=-1,
                split=split + "_source",
                multitask_enabled=multitask_enabled,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                distributed=is_distributed,
                rank=rank,
                world_size=world_size,
                greedy_logit_ranking=bool(eval_cfg.get("greedy_logit_ranking", False)),
            )
        continuation.evaluate(
            0,
            0,
            current_eval_threshold,
            initial_results["val"],
            initial_results["test"],
        )
        best_metric = stage_best_metric = continuation.payload["best_val_f1"]
        if is_main:
            save_checkpoint(
                os.path.join(ckpt_dir, "best.pt"),
                model,
                optimizer,
                scheduler,
                -1,
                0,
                best_metric,
                extra={
                    "selected_threshold": current_eval_threshold,
                    "initialization": initialization_summary,
                    "continuation_best_epoch": 0,
                },
                save_optimizer_state=False,
            )
            for split, results in initial_results.items():
                for name, metrics in results.items():
                    curve_logger.log_eval(split, name, -1, 0, metrics)
            curve_logger.save()
            save_json_atomic(
                os.path.join(log_dir, "threshold_scan_epoch_0000.json"),
                {
                    "epoch": 0,
                    "selected_threshold": current_eval_threshold,
                    "results": scan,
                },
            )
            save_json_atomic(continuation_path, continuation.payload)
            logger.info(
                f"Epoch 0 source checkpoint eligible: val per-RNA F1={best_metric:.6f}"
            )
        if is_distributed:
            dist.barrier()

    final_epoch = start_epoch - 1
    stopped_early = False
    for epoch in range(start_epoch, cfg["train"]["num_epochs"]):
        final_epoch = epoch
        should_stop = False
        if reseed_each_epoch:
            set_seed(epoch_seed(int(exp_cfg["seed"]), epoch, rank, world_size))
        encoder_frozen_for_epoch = epoch < encoder_freeze_epochs
        if encoder_freeze_epochs > 0:
            raw_model.rnafm.set_runtime_freeze(encoder_frozen_for_epoch)
            if is_main and (epoch == start_epoch or epoch == encoder_freeze_epochs):
                phase = (
                    "head-only encoder-frozen"
                    if encoder_frozen_for_epoch
                    else "joint encoder-unfrozen"
                )
                logger.info(
                    f"Training phase transition: epoch={epoch} phase={phase} "
                    f"freeze_epochs={encoder_freeze_epochs}"
                )
        model.train()
        train_loader.dataset.set_epoch(epoch)
        train_loader.batch_sampler.set_epoch(epoch)
        loss_meter = AverageMeter()
        loss_meters = {
            "contact": AverageMeter(),
            "pairedness": AverageMeter(),
            "pair_type": AverageMeter(),
        }
        local_samples_seen = 0
        local_tokens_seen = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_train_start = time.perf_counter()
        accumulated_global_samples = 0
        accumulation_window_counts = {}
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            sequences = batch["sequences"]
            local_samples_seen += len(sequences)
            local_tokens_seen += sum(map(len, sequences))
            target = batch["contact_map"].to(device, non_blocking=True)
            batch["contact_map"] = target
            if pairedness_enabled:
                batch["paired_tokens"] = batch["paired_tokens"].to(
                    device, non_blocking=True
                )
            if pair_type_enabled:
                batch["pair_type_map"] = batch["pair_type_map"].to(
                    device, non_blocking=True
                )
            batch["mask"] = batch["mask"].to(device, non_blocking=True)

            def loss_fn(model_output, micro_batch, mask):
                if multitask_enabled:
                    return compute_multitask_losses(
                        model_output,
                        micro_batch,
                        mask,
                        loss_cfg,
                        min_sequence_separation=min_sequence_separation,
                    )
                value = masked_contact_loss(
                    model_output,
                    micro_batch["contact_map"],
                    mask,
                    loss_cfg,
                    min_sequence_separation=min_sequence_separation,
                )
                return {"contact": value, "total": value}

            if microbatch_splits > 1:
                losses, mask = backward_original_batch(
                    model,
                    batch,
                    loss_fn,
                    loss_cfg,
                    min_sequence_separation,
                    device,
                    splits=microbatch_splits,
                )
            else:
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
                ):
                    if multitask_enabled:
                        model_output, mask = model(sequences, return_auxiliary=True)
                        losses = compute_multitask_losses(
                            model_output,
                            batch,
                            mask,
                            loss_cfg,
                            min_sequence_separation=min_sequence_separation,
                            distributed_batch_reduction=distributed_batch_reduction,
                        )
                    else:
                        model_output, mask = model(sequences)
                        contact_loss = masked_contact_loss(
                            model_output,
                            target,
                            mask,
                            loss_cfg,
                            min_sequence_separation=min_sequence_separation,
                            distributed_batch_reduction=distributed_batch_reduction,
                        )
                        losses = {"contact": contact_loss, "total": contact_loss}
                global_batch_samples = len(sequences) * world_size
                if accumulation_target > 0:
                    # distributed_weighted_mean returns a global per-RNA mean.
                    # Convert it to a sum before accumulating variable-size
                    # batches, then divide the completed window exactly once.
                    (losses["total"] * global_batch_samples).backward()
                    accumulated_global_samples += global_batch_samples
                else:
                    losses["total"].backward()
            loss = losses["total"]
            loss_meter.update(loss.item(), n=len(sequences))
            for name in loss_meters:
                if name in losses:
                    loss_meters[name].update(losses[name].item(), n=len(sequences))
            should_update = (
                accumulation_target <= 0
                or accumulated_global_samples >= accumulation_target
                or step + 1 == len(train_loader)
            )
            if not should_update:
                continue
            if accumulation_target > 0:
                _normalize_accumulated_gradients(model, accumulated_global_samples)
            if cfg["train"].get("grad_clip_norm"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["train"]["grad_clip_norm"]
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_window_samples = accumulated_global_samples
            window_size = completed_window_samples or len(sequences) * world_size
            accumulation_window_counts[window_size] = (
                accumulation_window_counts.get(window_size, 0) + 1
            )
            accumulated_global_samples = 0
            global_step += 1
            if is_main and global_step % cfg["train"]["log_every_n_steps"] == 0:
                lr = optimizer.param_groups[0]["lr"]
                mem_gb = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0
                )
                batch_size, length = mask.shape
                loss_details = " ".join(
                    f"{name}_loss={meter.avg:.4f}"
                    for name, meter in loss_meters.items()
                    if meter.count
                )
                logger.info(
                    f"[Train] epoch={epoch} step={step}/{len(train_loader)} global_step={global_step} "
                    f"loss={loss_meter.avg:.4f} {loss_details} lr={lr:.2e} "
                    f"B={batch_size} L={length} "
                    f"accum_global_samples={completed_window_samples or len(sequences) * world_size} "
                    f"peak_mem={mem_gb:.1f}GB"
                )
                if writer:
                    writer.add_scalar("train/loss", loss_meter.avg, global_step)
                    for name, meter in loss_meters.items():
                        if meter.count:
                            writer.add_scalar(
                                f"train/loss_{name}", meter.avg, global_step
                            )
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("train/peak_mem_gb", mem_gb, global_step)
                curve_logger.log_train_step(global_step, epoch, loss_meter.avg, lr)
                run_optional_artifact(
                    curve_logger.save, "training history write", logger
                )
                # Matplotlib rendering stalls rank 0 while all other DDP ranks
                # wait at their next collective. Render at evaluation points instead.

        # Gradients from the final optimizer step are otherwise kept alive
        # throughout validation, reducing the headroom available to eval batches.
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_train_seconds = time.perf_counter() - epoch_train_start
        if is_distributed:
            epoch_time = torch.tensor(
                epoch_train_seconds, dtype=torch.float64, device=device
            )
            dist.all_reduce(epoch_time, op=dist.ReduceOp.MAX)
            epoch_train_seconds = float(epoch_time.item())
            volume = torch.tensor(
                [local_samples_seen, local_tokens_seen],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(volume, op=dist.ReduceOp.SUM)
            global_samples_seen, global_tokens_seen = map(float, volume.tolist())
        else:
            global_samples_seen = float(local_samples_seen)
            global_tokens_seen = float(local_tokens_seen)
        if is_main:
            samples_per_second = global_samples_seen / max(epoch_train_seconds, 1e-9)
            tokens_per_second = global_tokens_seen / max(epoch_train_seconds, 1e-9)
            logger.info(
                f"[Train] epoch={epoch} finished. avg_loss={loss_meter.avg:.4f} "
                f"train_time={epoch_train_seconds:.1f}s samples/s={samples_per_second:.1f} "
                f"nt/s={tokens_per_second:.1f}"
            )
            if continuation is not None:
                continuation.payload["training_epochs"].append(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "loss": loss_meter.avg,
                        "loss_components": {
                            name: meter.avg
                            for name, meter in loss_meters.items()
                            if meter.count
                        },
                        "learning_rates": [
                            group["lr"] for group in optimizer.param_groups
                        ],
                        "train_seconds": epoch_train_seconds,
                        "samples": global_samples_seen,
                        "samples_per_second": samples_per_second,
                        "actual_global_batch_histogram": accumulation_window_counts,
                    }
                )
                save_json_atomic(continuation_path, continuation.payload)
            if writer:
                writer.add_scalar("train/epoch_seconds", epoch_train_seconds, epoch)
                writer.add_scalar("train/samples_per_second", samples_per_second, epoch)
                writer.add_scalar(
                    "train/nucleotides_per_second", tokens_per_second, epoch
                )
        stop_after_epoch = cfg["train"].get("stop_after_epoch")
        reached_stop_after = stop_after_epoch is not None and (epoch + 1) >= int(
            stop_after_epoch
        )
        checkpoint_every = max(1, int(cfg["train"].get("checkpoint_every_n_epochs", 1)))
        is_final_epoch = (epoch + 1) >= int(cfg["train"]["num_epochs"])
        checkpoint_due = (
            (epoch + 1) % checkpoint_every == 0 or is_final_epoch or reached_stop_after
        )
        # Save the expensive, resumable state before evaluation. Evaluation and
        # visualization are non-essential reporting work and must not be able to
        # discard a full checkpoint interval if they fail.
        if is_main and checkpoint_due:
            save_last_training_checkpoint(epoch)
        if is_distributed:
            dist.barrier()
        do_vis = (
            cfg["visualization"].get("enabled", False)
            and (epoch + 1) % cfg["visualization"].get("every_n_epochs", 5) == 0
        )
        if (
            val_loaders
            and (epoch + 1) % cfg["train"].get("eval_every_n_epochs", 1) == 0
        ):
            threshold_candidates = eval_cfg.get("threshold_candidates")
            threshold_results = None
            if threshold_candidates:
                current_eval_threshold, threshold_results = select_validation_threshold(
                    raw_model,
                    val_loaders,
                    device,
                    threshold_candidates,
                    min_sequence_separation,
                    greedy_projection,
                    multitask_enabled,
                    amp_enabled,
                    amp_dtype,
                    distributed=is_distributed,
                    prefer_higher_on_tie=bool(
                        eval_cfg.get("threshold_tie_prefer_higher", False)
                    ),
                    greedy_logit_ranking=bool(
                        eval_cfg.get("greedy_logit_ranking", False)
                    ),
                )
            if is_main and eval_cfg.get("report_threshold_scan"):
                logger.info(
                    "[Fixed threshold] epoch=%d split=val threshold=%.4f",
                    epoch + 1,
                    current_eval_threshold,
                )
            val_results = evaluate_supervised(
                raw_model,
                val_loaders,
                device,
                current_eval_threshold,
                min_sequence_separation=min_sequence_separation,
                greedy_projection=greedy_projection,
                logger=logger if is_main else None,
                visualize_cfg=cfg["visualization"] if is_main else None,
                vis_dir=vis_dir if (is_main and do_vis) else None,
                epoch=epoch,
                max_samples_per_dataset=eval_cfg.get("max_val_samples_per_dataset"),
                split="val",
                prediction_cache_dir=eval_cfg.get("prediction_cache_dir"),
                multitask_enabled=multitask_enabled,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                distributed=is_distributed,
                rank=rank,
                world_size=world_size,
                greedy_logit_ranking=bool(eval_cfg.get("greedy_logit_ranking", False)),
            )
            if is_main:
                if threshold_results is not None:
                    threshold_path = os.path.join(
                        log_dir, f"threshold_scan_epoch_{epoch + 1:04d}.json"
                    )
                    run_optional_artifact(
                        lambda: save_json_atomic(
                            threshold_path,
                            {
                                "epoch": epoch + 1,
                                "selected_threshold": current_eval_threshold,
                                "results": threshold_results,
                            },
                        ),
                        "validation threshold report write",
                        logger,
                    )
                    logger.info(
                        f"Validation-selected threshold={current_eval_threshold:.4f} "
                        f"for epoch={epoch + 1}."
                    )
                for name, metrics in val_results.items():
                    for k, v in metrics.items():
                        if writer:
                            writer.add_scalar(f"val/{name}/{k}", v, global_step)
                    curve_logger.log_eval("val", name, epoch, global_step, metrics)
                run_optional_artifact(
                    curve_logger.save, "validation history write", logger
                )
                run_optional_artifact(
                    curve_logger.render, "validation dashboard render", logger
                )
                mean_f1 = sum(m["f1"] for m in val_results.values()) / max(
                    len(val_results), 1
                )
                continuation_decision = None
                if continuation is not None:
                    continuation_decision = continuation.evaluate(
                        epoch + 1, global_step, current_eval_threshold, val_results
                    )
                if mean_f1 > best_metric:
                    best_metric = mean_f1
                    save_checkpoint(
                        os.path.join(ckpt_dir, "best.pt"),
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        best_metric,
                        extra={
                            "selected_threshold": current_eval_threshold,
                            "initialization": initialization_summary,
                        },
                        save_optimizer_state=False,
                    )
                    publish_checkpoint(
                        os.path.join(ckpt_dir, "best.pt"),
                        checkpoint_publish_dir,
                        logger,
                    )
                    logger.info(
                        f"New best model (mean val macro-F1={best_metric:.4f}) saved."
                    )
                if continuation_decision is not None:
                    should_stop = continuation_decision["should_stop"]
                    stage_best_metric = continuation.payload["patience_reference_f1"]
                    bad_eval_count = continuation_decision["epochs_without_improvement"]
                    save_json_atomic(continuation_path, continuation.payload)
                    logger.info(
                        f"Continuation selection: epoch={epoch + 1}, "
                        f"best_epoch={continuation.payload['best_epoch']}, "
                        f"no_improvement_epochs={bad_eval_count}/{early_cfg['patience_epochs']}"
                    )
                elif mean_f1 > stage_best_metric + early_min_delta:
                    stage_best_metric = mean_f1
                    bad_eval_count = 0
                elif early_enabled:
                    bad_eval_count += 1
                    logger.info(
                        f"Early stopping: no val F1 improvement for {bad_eval_count}/"
                        f"{early_patience} evaluations (stage best={stage_best_metric:.4f})."
                    )
                    should_stop = bad_eval_count >= early_patience

        if train_eval_every > 0 and (epoch + 1) % train_eval_every == 0:
            train_eval_results = evaluate_supervised(
                raw_model,
                train_eval_loaders,
                device,
                current_eval_threshold,
                min_sequence_separation=min_sequence_separation,
                greedy_projection=greedy_projection,
                logger=logger if is_main else None,
                # Train-set evaluation is diagnostic only. Do not generate a
                # second set of sample images from the much larger train split.
                visualize_cfg=None,
                vis_dir=None,
                epoch=epoch,
                max_samples_per_dataset=eval_cfg.get("max_train_samples_per_dataset"),
                split="train_eval",
                multitask_enabled=multitask_enabled,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                distributed=is_distributed,
                rank=rank,
                world_size=world_size,
                greedy_logit_ranking=bool(eval_cfg.get("greedy_logit_ranking", False)),
            )
            if is_main:
                for name, metrics in train_eval_results.items():
                    for k, v in metrics.items():
                        if writer:
                            writer.add_scalar(f"train_eval/{name}/{k}", v, global_step)
                    curve_logger.log_eval(
                        "train_eval", name, epoch, global_step, metrics
                    )
                run_optional_artifact(
                    curve_logger.save, "train-set evaluation history write", logger
                )
                run_optional_artifact(
                    curve_logger.render, "train/validation dashboard render", logger
                )

        test_every = int(cfg["train"].get("test_every_n_epochs", 0))
        if test_every > 0 and (epoch + 1) % test_every == 0:
            if is_main and eval_cfg.get("report_threshold_scan"):
                logger.info(
                    "[Fixed threshold] epoch=%d split=test threshold=%.4f",
                    epoch + 1,
                    current_eval_threshold,
                )
            test_results = evaluate_supervised(
                raw_model,
                test_loaders,
                device,
                current_eval_threshold,
                min_sequence_separation=min_sequence_separation,
                greedy_projection=greedy_projection,
                logger=logger if is_main else None,
                visualize_cfg=cfg["visualization"] if is_main else None,
                vis_dir=vis_dir if (is_main and do_vis) else None,
                epoch=epoch,
                max_samples_per_dataset=eval_cfg.get("max_test_samples_per_dataset"),
                split="test",
                prediction_cache_dir=eval_cfg.get("prediction_cache_dir"),
                multitask_enabled=multitask_enabled,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                distributed=is_distributed,
                rank=rank,
                world_size=world_size,
                greedy_logit_ranking=bool(eval_cfg.get("greedy_logit_ranking", False)),
            )
            if is_main:
                if continuation is not None:
                    continuation.add_test(epoch + 1, test_results)
                    save_json_atomic(continuation_path, continuation.payload)
                for name, metrics in test_results.items():
                    for k, v in metrics.items():
                        if writer:
                            writer.add_scalar(f"test/{name}/{k}", v, global_step)
                    curve_logger.log_eval("test", name, epoch, global_step, metrics)
                run_optional_artifact(curve_logger.save, "test history write", logger)
                run_optional_artifact(
                    curve_logger.render, "test dashboard render", logger
                )

        if (
            eval_cfg.get("report_threshold_scan")
            and (epoch + 1) % cfg["train"].get("eval_every_n_epochs", 1) == 0
        ):
            # This separate path neither changes current_eval_threshold nor
            # best_metric/early stopping, and never uses Test to choose a threshold.
            report_single_gpu_threshold_scan(
                raw_model,
                val_loaders,
                test_loaders
                if test_every > 0 and (epoch + 1) % test_every == 0
                else {},
                device,
                eval_cfg,
                epoch,
                global_step,
                log_dir,
                logger,
                curve_logger,
                writer,
                min_sequence_separation,
                greedy_projection,
                multitask_enabled,
                amp_enabled,
                amp_dtype,
            )

        if reached_stop_after:
            should_stop = True
            if is_main:
                logger.info(
                    f"Stop-after-epoch reached at epoch={epoch + 1}; "
                    "evaluation and checkpointing are complete."
                )
        if is_main:
            if checkpoint_due:
                # The large checkpoint already exists. Refresh only the small
                # post-evaluation state (best metric / early-stopping counters).
                save_training_state(epoch)
            elif should_stop:
                save_last_training_checkpoint(epoch)
        if is_distributed:
            stop_tensor = torch.tensor(int(should_stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
            dist.barrier()
        if should_stop:
            stopped_early = True
            if is_main and not reached_stop_after:
                logger.info(
                    f"Early stopping triggered at epoch={epoch}; "
                    f"best stage val F1={stage_best_metric:.4f}."
                )
            break

    if is_main and continuation is not None:
        continuation.finish(final_epoch + 1, stopped_early)
        save_json_atomic(continuation_path, continuation.payload)
    if is_distributed:
        dist.barrier()
    if bool(cfg["train"].get("test_best_at_end", False)):
        best_path = os.path.join(ckpt_dir, "best.pt")
        best_available = broadcast_object(
            os.path.isfile(best_path) if is_main else None, is_distributed, rank
        )
        if best_available:
            best_state = None
            if is_main or not is_distributed:
                best_state = load_checkpoint(
                    best_path, raw_model, map_location=str(device)
                )
            broadcast_model_state(raw_model, is_distributed)
            if is_main:
                logger.info(
                    "Loaded best validation checkpoint for one final test evaluation."
                )
            selected_threshold = float(
                best_state.get("selected_threshold", eval_cfg["threshold"])
                if best_state is not None
                else eval_cfg["threshold"]
            )
            selected_threshold = broadcast_object(
                selected_threshold if is_main else None, is_distributed, rank
            )
            threshold_candidates = eval_cfg.get("threshold_candidates")
            if threshold_candidates:
                selected_threshold, threshold_results = select_validation_threshold(
                    raw_model,
                    val_loaders,
                    device,
                    threshold_candidates,
                    min_sequence_separation,
                    greedy_projection,
                    multitask_enabled,
                    amp_enabled,
                    amp_dtype,
                    distributed=is_distributed,
                    prefer_higher_on_tie=bool(
                        eval_cfg.get("threshold_tie_prefer_higher", False)
                    ),
                    greedy_logit_ranking=bool(
                        eval_cfg.get("greedy_logit_ranking", False)
                    ),
                )
                if is_main:
                    threshold_path = os.path.join(log_dir, "threshold_scan.json")
                    threshold_saved = run_optional_artifact(
                        lambda: save_json_atomic(
                            threshold_path,
                            {
                                "selected_threshold": selected_threshold,
                                "results": threshold_results,
                            },
                        ),
                        "validation threshold report write",
                        logger,
                    )
                    if threshold_saved:
                        logger.info(
                            f"Validation-selected final threshold={selected_threshold:.4f}; "
                            f"details saved to {threshold_path}."
                        )
                    if writer:
                        writer.add_scalar(
                            "eval/selected_threshold", selected_threshold, global_step
                        )
            if test_loaders is None:
                test_loaders = build_eval_loaders(data_cfg["test_datasets"])
            test_results = evaluate_supervised(
                raw_model,
                test_loaders,
                device,
                selected_threshold,
                min_sequence_separation=min_sequence_separation,
                greedy_projection=greedy_projection,
                logger=logger if is_main else None,
                visualize_cfg=cfg["visualization"] if is_main else None,
                vis_dir=vis_dir if is_main else None,
                epoch=final_epoch,
                max_samples_per_dataset=eval_cfg.get("max_test_samples_per_dataset"),
                split="test_best",
                multitask_enabled=multitask_enabled,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                distributed=is_distributed,
                rank=rank,
                world_size=world_size,
                greedy_logit_ranking=bool(eval_cfg.get("greedy_logit_ranking", False)),
            )
            if is_main:
                for name, metrics in test_results.items():
                    for key, value in metrics.items():
                        if writer:
                            writer.add_scalar(
                                f"test_best/{name}/{key}", value, global_step
                            )
                    curve_logger.log_eval(
                        "test", name, final_epoch, global_step, metrics
                    )
                run_optional_artifact(
                    curve_logger.save, "final-test history write", logger
                )
                run_optional_artifact(
                    curve_logger.render, "final-test dashboard render", logger
                )
        elif is_main:
            logger.warning(
                f"Best checkpoint not found; skipped final test: {best_path}"
            )
    if is_distributed:
        dist.barrier()

    if writer:
        run_optional_artifact(writer.close, "TensorBoard writer close", logger)
    if is_main:
        logger.info("Training complete.")
    cleanup_distributed(is_distributed)


def run_training_entrypoint():
    """Report failures before NCCL teardown can hide the original traceback."""
    try:
        main()
    except BaseException:
        sys.stderr.write(
            f"\n[Training failed: rank={os.environ.get('RANK', '0')}] "
            "Original exception follows; skipping graceful NCCL teardown on failure.\n"
        )
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        # Graceful ncclCommDestroy may wait forever when another rank is still
        # inside a collective. Exit nonzero so torchrun can stop sibling ranks;
        # bypass interpreter/CUDA destructors on this failed distributed rank.
        if dist.is_available() and dist.is_initialized():
            os._exit(1)
        raise
    else:
        cleanup_distributed(True)


if __name__ == "__main__":
    run_training_entrypoint()
