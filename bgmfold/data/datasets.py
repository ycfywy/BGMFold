"""RNA contact-map dataset built on top of the provided parquet files.

Each parquet file is expected to contain at least the columns:
    - `id`: unique sample identifier
    - `sequence`: RNA sequence string (A/C/G/U/N ...)
    - `secondary_structure`: dot-bracket notation (supports pseudoknot tiers)
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from bgmfold.data.covariation import PriFoldCovariationAugmentation
from bgmfold.data.dotbracket import (
    dotbracket_to_contact_map,
    dotbracket_to_pair_type_map,
)
from bgmfold.data.length_bucket_sampler import (
    DistributedLengthBucketBatchSampler,
    DistributedRandomFixedBatchSampler,
    LengthBucketBatchSampler,
)


def distributed_eval_indices(
    dataset_size: int, world_size: int, rank: int
) -> list[int]:
    """Return a disjoint, exhaustive strided shard without padding duplicates."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(
            f"Invalid distributed eval shard rank={rank}, world_size={world_size}"
        )
    return list(range(rank, int(dataset_size), world_size))


class RNADataset(Dataset):
    """Loads parquet RNA samples and optionally augments training sequences."""

    def __init__(
        self,
        paths: List[str],
        max_length: Optional[int] = None,
        name: str = "dataset",
        covariation_augmentation: Optional[Dict] = None,
    ):
        self.name = name
        self.epoch = 0
        frames = [pd.read_parquet(p) for p in paths]
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

        if max_length is not None:
            df = df[df["sequence"].str.len() <= max_length].reset_index(drop=True)

        self.df = df
        self.covariation_augmentation = None
        if covariation_augmentation and covariation_augmentation.get("enabled", False):
            self.covariation_augmentation = PriFoldCovariationAugmentation(
                select=covariation_augmentation.get("select", 0.1),
                replace=covariation_augmentation.get("replace", 0.3),
                seed=covariation_augmentation.get("seed", 42),
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        sequence = row["sequence"]
        structure = row["secondary_structure"]
        contact_map = dotbracket_to_contact_map(structure)
        if self.covariation_augmentation is not None:
            sequence = self.covariation_augmentation(
                sequence,
                contact_map,
                sample_id=f"{self.name}:{row['id']}",
                epoch=self.epoch,
            )
        pair_type_map = dotbracket_to_pair_type_map(sequence, structure)
        paired_tokens = (contact_map.max(axis=1) > 0.5).astype(np.float32)
        return {
            "id": row["id"],
            "sequence": sequence,
            "contact_map": contact_map,
            "pair_type_map": pair_type_map,
            "paired_tokens": paired_tokens,
            "length": len(sequence),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    ids = [b["id"] for b in batch]
    sequences = [b["sequence"] for b in batch]
    lengths = [b["length"] for b in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    contact_maps = torch.zeros(batch_size, max_len, max_len, dtype=torch.float32)
    pair_type_maps = torch.full((batch_size, max_len, max_len), -1, dtype=torch.long)
    paired_tokens = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, sample in enumerate(batch):
        length = sample["length"]
        contact_maps[i, :length, :length] = torch.from_numpy(sample["contact_map"])
        pair_type_maps[i, :length, :length] = torch.from_numpy(sample["pair_type_map"])
        paired_tokens[i, :length] = torch.from_numpy(sample["paired_tokens"])
        mask[i, :length] = True

    return {
        "ids": ids,
        "sequences": sequences,
        "contact_map": contact_maps,
        "pair_type_map": pair_type_maps,
        "paired_tokens": paired_tokens,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


def build_train_dataloader(
    dataset_cfgs: List[Dict],
    max_length: Optional[int],
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Joins all configured training datasets into a single RNADataset for joint training."""

    paths = [cfg["path"] for cfg in dataset_cfgs]
    dataset = RNADataset(
        paths, max_length=max_length, name="+".join(cfg["name"] for cfg in dataset_cfgs)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )


def build_eval_dataloaders(
    dataset_cfgs: List[Dict],
    max_length: Optional[int],
    batch_size: int,
    num_workers: int = 2,
) -> Dict[str, DataLoader]:
    """Builds one DataLoader per evaluation dataset, keyed by dataset name (kept separate on purpose)."""

    loaders = {}
    for cfg in dataset_cfgs:
        dataset = RNADataset([cfg["path"]], max_length=max_length, name=cfg["name"])
        loaders[cfg["name"]] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )
    return loaders


class DistributedBatchSampler(Sampler[List[int]]):
    """Shard an existing batch sampler across DDP ranks while keeping the same
    number of batches on every rank (padding by repeating early batches if needed).
    """

    def __init__(self, batch_sampler: Sampler[List[int]], num_replicas: int, rank: int):
        self.batch_sampler = batch_sampler
        self.num_replicas = num_replicas
        self.rank = rank

    def set_epoch(self, epoch: int):
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        batches = list(self.batch_sampler)
        if not batches:
            return iter(())
        total_size = int(
            math.ceil(len(batches) / self.num_replicas) * self.num_replicas
        )
        if total_size > len(batches):
            batches = batches + batches[: total_size - len(batches)]
        return iter(batches[self.rank : total_size : self.num_replicas])

    def __len__(self) -> int:
        return int(math.ceil(len(self.batch_sampler) / self.num_replicas))


def build_length_bucketed_train_dataloader(
    dataset_cfgs: List[Dict],
    max_length: Optional[int],
    num_workers: int = 4,
    bucket_width: int = 32,
    budget: int = 64 * 128**3,
    length_power: float = 3.0,
    min_batch_size: int = 1,
    max_batch_size: int = 64,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    augmentation_cfg: Optional[Dict] = None,
    distributed_batch_mode: str = "shard_batches",
    seed: int = 0,
    batch_size_by_bucket: Optional[Dict[int, int]] = None,
) -> DataLoader:
    """Training dataloader that batches samples of similar length together, and
    dynamically shrinks batch size for longer sequences to keep peak memory
    roughly constant. See `LengthBucketBatchSampler` for the memory model.
    """
    if (
        distributed
        and distributed_batch_mode == "random_fixed_batch"
        and batch_size_by_bucket is not None
    ):
        raise ValueError("random_fixed_batch cannot use batch_size_by_bucket")
    paths = [cfg["path"] for cfg in dataset_cfgs]
    dataset = RNADataset(
        paths,
        max_length=max_length,
        name="+".join(cfg["name"] for cfg in dataset_cfgs),
        covariation_augmentation=augmentation_cfg,
    )
    lengths = dataset.df["sequence"].str.len().tolist()

    if distributed and distributed_batch_mode in {"same_bucket", "random_fixed_batch"}:
        if distributed_batch_mode == "random_fixed_batch":
            if min_batch_size != max_batch_size:
                raise ValueError(
                    "random_fixed_batch requires min_batch_size == max_batch_size"
                )
            batch_sampler = DistributedRandomFixedBatchSampler(
                len(dataset),
                num_replicas=world_size,
                rank=rank,
                local_batch_size=min_batch_size,
                shuffle=True,
                seed=seed,
            )
        else:
            batch_sampler = DistributedLengthBucketBatchSampler(
                lengths,
                num_replicas=world_size,
                rank=rank,
                bucket_width=bucket_width,
                budget=budget,
                length_power=length_power,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                shuffle=True,
                seed=seed,
                batch_size_by_bucket=batch_size_by_bucket,
            )
    else:
        batch_sampler = LengthBucketBatchSampler(
            lengths,
            bucket_width=bucket_width,
            budget=budget,
            length_power=length_power,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            shuffle=True,
            drop_last=False,
            seed=seed,
            batch_size_by_bucket=batch_size_by_bucket,
        )
        if distributed:
            batch_sampler = DistributedBatchSampler(
                batch_sampler, num_replicas=world_size, rank=rank
            )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        # Workers are created on iteration, after NCCL/CUDA initialization.
        # Forking that parent is unsafe; spawn also gives each epoch's dataset
        # a fresh copy after set_epoch (do not enable persistent_workers here).
        **(
            {"multiprocessing_context": "spawn"}
            if distributed and num_workers > 0
            else {}
        ),
    )


def build_length_bucketed_eval_dataloaders(
    dataset_cfgs: List[Dict],
    max_length: Optional[int],
    num_workers: int = 2,
    bucket_width: int = 32,
    budget: int = 64 * 128**3,
    length_power: float = 3.0,
    min_batch_size: int = 1,
    max_batch_size: int = 64,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> Dict[str, DataLoader]:
    """Same length-bucketed strategy as the training loader, but one DataLoader per
    eval dataset (kept separate, no shuffling needed but bucketing still helps memory)."""
    loaders = {}
    for cfg in dataset_cfgs:
        full_dataset = RNADataset(
            [cfg["path"]], max_length=max_length, name=cfg["name"]
        )
        full_lengths = full_dataset.df["sequence"].str.len().tolist()
        if distributed:
            indices = distributed_eval_indices(len(full_dataset), world_size, rank)
            lengths = [full_lengths[index] for index in indices]
            # Retain only this rank's rows instead of keeping eight full copies
            # of every evaluation dataframe alive for the entire training run.
            full_dataset.df = full_dataset.df.iloc[indices].reset_index(drop=True)
            dataset = full_dataset
        else:
            dataset = full_dataset
            lengths = full_lengths
        batch_sampler = LengthBucketBatchSampler(
            lengths,
            bucket_width=bucket_width,
            budget=budget,
            length_power=length_power,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            shuffle=False,
            drop_last=False,
        )
        loaders[cfg["name"]] = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            **(
                {"multiprocessing_context": "spawn"}
                if distributed and num_workers > 0
                else {}
            ),
        )
    return loaders
