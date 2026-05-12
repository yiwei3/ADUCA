def construct_block_range(begin: int, end: int, block_size: int):
    """
    Build contiguous blocks covering [begin, end) using slice objects.

    Using slices keeps downstream NumPy/SciPy indexing fast (views instead of
    fancy indexing) and avoids converting ranges to lists of indices.
    """
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if begin >= end:
        return []

    # Normalize types to plain Python ints to keep range fast.
    begin = int(begin)
    end = int(end)
    block_size = int(block_size)

    block_count = (end - begin + block_size - 1) // block_size
    last_start = begin + (block_count - 1) * block_size

    blocks = [slice(start, start + block_size) for start in range(begin, last_start, block_size)]
    blocks.append(slice(last_start, end))
    return blocks
