"""Length-sorted padded-attention budget batching, independent of torch."""


def attention_batches(items, length, max_batch_size=8, max_pair_tokens=1_000_000):
    if max_batch_size < 1 or max_pair_tokens < 1:
        raise ValueError("Batch size and pair-token budget must be positive")
    batch = []
    longest = 0
    for item in sorted(items, key=length):
        padded = length(item) + 2  # CLS and SEP also consume attention memory.
        if batch and (
            len(batch) >= max_batch_size
            or (len(batch) + 1) * max(longest, padded) ** 2 > max_pair_tokens
        ):
            yield batch
            batch = []
            longest = 0
        batch.append(item)
        longest = max(longest, padded)
    if batch:
        yield batch
