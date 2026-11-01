"""Length-aware batch sampler for RNA contact-map training.

Two problems addressed here:

1. Padding waste: naive random batching mixes very short and very long
   sequences into the same batch, forcing all of them to be padded up to
   the max length in the batch. We instead sort samples into length
   buckets and only batch together samples from (approximately) the same
   bucket.

2. Super-linear memory growth with sequence length: the contact map and
   attention features are O(L^2).  Measurements from the discriminative
   GB.RNA pipeline are conservatively fitted with an L^2.6 budget so long
   sequences do not OOM while short-sequence batches remain large, e.g.:

       L=64  -> ~0.21 GB/sample
       L=128 -> ~0.94 GB/sample
       L=256 -> ~4.78 GB/sample
       L=384 -> ~13.5 GB/sample
       L=512 -> ~28.8 GB/sample

   We therefore pick batch size dynamically per-bucket as:

       batch_size(bucket) = max(1, floor(budget / L_max_in_bucket^length_power))

   with `length_power=2.6` by default (empirically calibrated as above), so
   peak activation memory stays much more stable across all batches
   regardless of sequence length, instead of OOMing on the longest
   sequences while under-utilizing the GPU on short ones. `length_power` is
   configurable in case a different backbone has different scaling behavior.
"""

from __future__ import annotations

import random
from typing import Iterator, List, Mapping, Optional, Sequence

from torch.utils.data import Sampler


class DistributedRandomFixedBatchSampler(Sampler[List[int]]):
    """Shuffle the full dataset and give every DDP rank a fixed local batch.

    Unlike :class:`DistributedLengthBucketBatchSampler`, this sampler does not
    inspect sequence lengths.  It pads the final global batch by repeating the
    first shuffled indices so that every rank always receives exactly
    ``local_batch_size`` samples and executes the same number of steps.
    """

    def __init__(
        self,
        dataset_size: int,
        num_replicas: int,
        rank: int,
        local_batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if dataset_size < 1:
            raise ValueError("dataset_size must be positive")
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if local_batch_size < 1:
            raise ValueError("local_batch_size must be positive")
        self.dataset_size = int(dataset_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.local_batch_size = int(local_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    @property
    def global_batch_size(self) -> int:
        return self.local_batch_size * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(self.dataset_size))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)

        remainder = len(indices) % self.global_batch_size
        if remainder:
            repeats = self.global_batch_size - remainder
            indices.extend(
                indices[index % self.dataset_size] for index in range(repeats)
            )

        for start in range(0, len(indices), self.global_batch_size):
            global_batch = indices[start : start + self.global_batch_size]
            rank_start = self.rank * self.local_batch_size
            yield global_batch[rank_start : rank_start + self.local_batch_size]

    def __len__(self) -> int:
        return (
            self.dataset_size + self.global_batch_size - 1
        ) // self.global_batch_size


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Groups dataset indices into fixed-width length buckets, then yields batches
    whose size is inversely proportional to (bucket max length) ** length_power.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        bucket_width: int = 32,
        budget: int = 64 * 128**3,
        length_power: float = 3.0,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        batch_size_by_bucket: Optional[Mapping[int, int]] = None,
    ):
        """
        Args:
            lengths: per-sample sequence length, len(lengths) == len(dataset).
            bucket_width: sequences are grouped into buckets of this many
                nucleotides (e.g. width=32 -> buckets [0,32), [32,64), ...).
            budget: target B * L_max^length_power per batch; batch size is
                derived from this so longer buckets automatically get smaller
                batches. Tune this against measured peak GPU memory (see
                EXPERIMENT_SETUP.md for a worked calibration example).
            length_power: empirical memory-scaling exponent w.r.t. sequence
                length for the backbone in use. 2.0 is the pair-tensor lower
                bound; the default remains conservative for encoder activations.
            min_batch_size / max_batch_size: hard clamps on the derived batch size.
            shuffle: shuffle sample order within each bucket, and shuffle
                batch order across buckets, every epoch.
            drop_last: drop the final (possibly smaller) batch of each bucket.
            batch_size_by_bucket: optional measured batch sizes keyed by
                floor(length / bucket_width); replaces the budget formula.
        """
        self.lengths = list(lengths)
        self.bucket_width = bucket_width
        self.budget = budget
        self.length_power = length_power
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Opt-in measured capacity table; absent preserves the original budget rule.
        self.batch_size_by_bucket = (
            {int(k): int(v) for k, v in batch_size_by_bucket.items()}
            if batch_size_by_bucket is not None
            else None
        )

        self._buckets = self._build_buckets()
        if self.batch_size_by_bucket is not None:
            missing = set(self._buckets) - self.batch_size_by_bucket.keys()
            if missing:
                raise ValueError(
                    f"Missing measured batch sizes for buckets: {sorted(missing)}"
                )
            if any(
                not min_batch_size <= v <= max_batch_size
                for v in self.batch_size_by_bucket.values()
            ):
                raise ValueError("Measured batch sizes must respect min/max_batch_size")

    def _build_buckets(self):
        buckets = {}
        for idx, length in enumerate(self.lengths):
            bucket_id = length // self.bucket_width
            buckets.setdefault(bucket_id, []).append(idx)
        return buckets

    def _batch_size_for_bucket(self, bucket_id: int) -> int:
        if self.batch_size_by_bucket is not None:
            return self.batch_size_by_bucket[bucket_id]
        bucket_max_len = (bucket_id + 1) * self.bucket_width
        size = self.budget / (bucket_max_len**self.length_power)
        return int(max(self.min_batch_size, min(self.max_batch_size, size)))

    def set_epoch(self, epoch: int):
        """Call once per epoch (mirrors DistributedSampler API) for reproducible shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        all_batches: List[List[int]] = []

        for bucket_id, indices in self._buckets.items():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            batch_size = self._batch_size_for_bucket(bucket_id)
            for i in range(0, len(indices), batch_size):
                batch = indices[i : i + batch_size]
                if len(batch) < batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        total = 0
        for bucket_id, indices in self._buckets.items():
            batch_size = self._batch_size_for_bucket(bucket_id)
            n_batches = len(indices) // batch_size
            if not self.drop_last and len(indices) % batch_size != 0:
                n_batches += 1
            total += n_batches
        return total


class DistributedLengthBucketBatchSampler(Sampler[List[int]]):
    """Build synchronized, same-length local batches for data-parallel training.

    Every rank follows the same shuffled bucket schedule.  A global batch is
    formed inside one length bucket and then split evenly across ranks.  The
    final global batch in a bucket shrinks its local batch size and is padded
    only to a multiple of the world size. Thus at most ``world_size - 1``
    samples are repeated per bucket while every rank still executes the same
    number of forwards with the same tensor shapes.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        num_replicas: int,
        rank: int,
        bucket_width: int = 32,
        budget: int = 64 * 128**3,
        length_power: float = 3.0,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 0,
        batch_size_by_bucket: Optional[Mapping[int, int]] = None,
    ):
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        self.lengths = list(lengths)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.bucket_width = int(bucket_width)
        self.budget = int(budget)
        self.length_power = float(length_power)
        self.min_batch_size = int(min_batch_size)
        self.max_batch_size = int(max_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self._buckets = self._build_buckets()
        self.batch_size_by_bucket = (
            {int(k): int(v) for k, v in batch_size_by_bucket.items()}
            if batch_size_by_bucket is not None
            else None
        )
        if self.batch_size_by_bucket is not None:
            missing = set(self._buckets) - self.batch_size_by_bucket.keys()
            if missing:
                raise ValueError(
                    f"Missing measured batch sizes for buckets: {sorted(missing)}"
                )
            if any(
                not self.min_batch_size <= v <= self.max_batch_size
                for v in self.batch_size_by_bucket.values()
            ):
                raise ValueError("Measured batch sizes must respect min/max_batch_size")

    def _build_buckets(self) -> dict[int, list[int]]:
        buckets: dict[int, list[int]] = {}
        for index, length in enumerate(self.lengths):
            buckets.setdefault(length // self.bucket_width, []).append(index)
        return buckets

    def _local_batch_size(self, bucket_id: int) -> int:
        if self.batch_size_by_bucket is not None:
            return self.batch_size_by_bucket[bucket_id]
        bucket_max_len = (bucket_id + 1) * self.bucket_width
        size = self.budget / (bucket_max_len**self.length_power)
        return int(max(self.min_batch_size, min(self.max_batch_size, size)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        global_batches: list[tuple[int, list[int]]] = []
        for bucket_id, source_indices in self._buckets.items():
            indices = list(source_indices)
            if self.shuffle:
                rng.shuffle(indices)
            local_batch_size = self._local_batch_size(bucket_id)
            global_batch_size = local_batch_size * self.num_replicas
            for start in range(0, len(indices), global_batch_size):
                batch = indices[start : start + global_batch_size]
                if len(batch) < global_batch_size:
                    local_batch_size = max(
                        1, (len(batch) + self.num_replicas - 1) // self.num_replicas
                    )
                    repeats = local_batch_size * self.num_replicas - len(batch)
                    batch.extend(indices[i % len(indices)] for i in range(repeats))
                global_batches.append((local_batch_size, batch))
        if self.shuffle:
            rng.shuffle(global_batches)
        for local_batch_size, batch in global_batches:
            start = self.rank * local_batch_size
            yield batch[start : start + local_batch_size]

    def __len__(self) -> int:
        total = 0
        for bucket_id, indices in self._buckets.items():
            global_batch_size = self._local_batch_size(bucket_id) * self.num_replicas
            total += (len(indices) + global_batch_size - 1) // global_batch_size
        return total
