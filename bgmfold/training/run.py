"""Portable two-GPU multitask / eight-GPU CPTM training orchestration."""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from bgmfold.io import write_json
from bgmfold.threshold_grid import default_threshold_candidates

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("contact", "pairedness", "pairtype", "multitask")


def configure(base, role, encoder, output, data_root, resume=False):
    cfg = copy.deepcopy(base)
    cfg["experiment"].update(
        name="bgmfold_" + role, output_dir=str(output.parent), device="cuda:0"
    )
    cfg["model"].update(
        rna_encoder_path=str(Path(encoder).expanduser().resolve()),
        rna_encoder_initialize_from_config_only=False,
    )
    tasks = cfg["model"]["multitask"]
    tasks["enabled"] = role != "contact"
    tasks["pairedness"]["enabled"] = role in ("pairedness", "multitask")
    tasks["pair_type"]["enabled"] = role in ("pairtype", "multitask")
    for head in ("pairedness", "pair_type"):
        if not tasks[head]["enabled"]:
            cfg["loss"]["multitask"][head + "_weight"] = 0.0
    for split in ("train", "val", "test"):
        for ds in cfg["data"][split + "_datasets"]:
            relative = Path(ds["path"]).relative_to("data")
            ds["path"] = str((Path(data_root) / relative).resolve())
    cfg["eval"]["threshold_candidates"] = default_threshold_candidates()
    early = cfg["train"]["early_stopping"]
    if early.get("enabled"):
        interval = int(cfg["train"]["eval_every_n_epochs"])
        patience = int(early["patience_epochs"])
        if patience <= 0 or patience % interval:
            raise ValueError(
                "patience_epochs must be a positive multiple of validation interval"
            )
        early["patience"] = patience // interval
    if resume:
        cfg["train"]["resume_from"] = str(output / "checkpoints/last.pt")
    return cfg


def plan(args):
    cfg = yaml.safe_load(args.config.read_text())
    roles = ("multitask",) if args.experiment == "rnastralign" else ROLES
    visible = [
        s.strip()
        for s in os.environ.get(
            "CUDA_VISIBLE_DEVICES", ",".join(map(str, range(2 * len(roles))))
        ).split(",")
    ]
    if (
        len(visible) < 2 * len(roles)
        or len(set(visible)) != len(visible)
        or any(not d for d in visible)
    ):
        raise ValueError(
            f"{args.experiment} needs {2 * len(roles)} distinct visible GPUs"
        )
    jobs = []
    for i, role in enumerate(roles):
        directory = args.output / role
        resolved = configure(
            cfg, role, args.encoder_path, directory, args.data_root, args.resume
        )
        file = directory / "configs/training.yaml"
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            "-m",
            "bgmfold.training.engine",
            "--config",
            str(file),
            "--resume-run-dir",
            str(directory),
        ]
        jobs.append(
            dict(
                role=role,
                gpus=visible[2 * i : 2 * i + 2],
                directory=directory,
                config=file,
                settings=resolved,
                command=cmd,
            )
        )
    return jobs


def completed_role(job):
    """Keep completed/early-stopped roles stopped when resuming a suite."""
    sidecar = job["directory"] / "checkpoints/training_state.json"
    if not sidecar.is_file():
        return None
    state = json.loads(sidecar.read_text())
    settings = job["settings"]["train"]
    if int(state.get("checkpoint_epoch", -1)) + 1 >= int(settings["num_epochs"]):
        return "max_epochs"
    early = settings.get("early_stopping", {})
    if early.get("enabled") and int(state.get("bad_eval_count", 0)) >= int(
        early["patience"]
    ):
        return "validation_patience"
    return None


def run(args):
    args.output = args.output.expanduser().resolve()
    jobs = plan(args)
    if not args.dry_run:
        if not args.encoder_path.is_dir():
            raise FileNotFoundError(
                "Pass a complete local pretrained encoder directory"
            )
        if args.output.exists() and not args.resume:
            raise FileExistsError("Use a fresh output directory, or --resume")
        if args.resume:
            for job in jobs:
                if not (job["directory"] / "checkpoints/last.pt").is_file():
                    raise FileNotFoundError(
                        f"Missing resume checkpoint for {job['role']}"
                    )
    for job in jobs:
        job["config"].parent.mkdir(parents=True, exist_ok=True)
        job["config"].write_text(
            yaml.safe_dump(job["settings"], sort_keys=False), encoding="utf-8"
        )
    if args.dry_run:
        for job in jobs:
            print(
                json.dumps(
                    dict(role=job["role"], gpus=job["gpus"], command=job["command"]),
                    ensure_ascii=False,
                )
            )
        return 0
    manifest = dict(experiment=args.experiment, status="running", models={})
    if args.resume:
        pending = []
        for job in jobs:
            reason = completed_role(job)
            if reason:
                manifest["models"][job["role"]] = dict(
                    status="completed",
                    exit_code=0,
                    resumed_without_restart=True,
                    stop_reason=reason,
                )
            else:
                pending.append(job)
        jobs = pending
    write_json(args.output / "training_status.json", manifest)
    processes = []
    lock = threading.Lock()
    stop = threading.Event()

    def terminate(*_):
        stop.set()
        with lock:
            for proc in processes:
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)

    def worker(job):
        env = dict(
            os.environ, CUDA_VISIBLE_DEVICES=",".join(job["gpus"]), PYTHONUNBUFFERED="1"
        )
        for key in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "GROUP_RANK",
            "ROLE_RANK",
            "TORCHELASTIC_RUN_ID",
        ):
            env.pop(key, None)
        env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        env.setdefault("OMP_NUM_THREADS", "2")
        folder = job["directory"] / "logs"
        folder.mkdir(exist_ok=True)
        with (folder / "launcher.log").open("a", encoding="utf-8") as log:
            with lock:
                if stop.is_set():
                    return 130
                proc = subprocess.Popen(
                    job["command"],
                    env=env,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                processes.append(proc)
            for line in proc.stdout:
                print(f"[{job['role']}] {line}", end="", flush=True)
                log.write(line)
                log.flush()
            return proc.wait()

    code = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
            futures = {pool.submit(worker, job): job["role"] for job in jobs}
            for future in as_completed(futures):
                role = futures[future]
                try:
                    result = future.result()
                    info = {
                        "exit_code": result,
                        "status": "completed" if result == 0 else "failed",
                    }
                except Exception as exc:
                    result = 1
                    info = {"status": "failed", "error": str(exc)}
                manifest["models"][role] = info
                if result:
                    code = 1
                write_json(args.output / "training_status.json", manifest)
    except KeyboardInterrupt:
        code = 130
    manifest.update(
        status="completed" if code == 0 else "partial_or_failed", exit_code=code
    )
    write_json(args.output / "training_status.json", manifest)
    print(f"Training finished: {manifest['status']}; output={args.output}", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("rnastralign", "bprna"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write resolved configs and commands without importing torch or loading weights",
    )
    args = parser.parse_args()
    if args.config is None:
        args.config = (
            ROOT
            / "configs"
            / (
                "train_rnastralign_multitask.yaml"
                if args.experiment == "rnastralign"
                else "train_bprna_cptm.yaml"
            )
        )
    if args.output is None:
        args.output = (
            ROOT
            / "outputs"
            / ("rnastralign" if args.experiment == "rnastralign" else "bprna_cptm")
        )
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
