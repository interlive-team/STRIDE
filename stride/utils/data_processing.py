#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
import math

import torch
from torch.utils.data import Sampler


class LengthGroupedSampler(Sampler):
    def __init__(self, global_batch_size, lengths, generator=None):
        self.global_batch_size = global_batch_size
        self.lengths = lengths
        self.generator = generator

    def __iter__(self):
        indices = torch.randperm(len(self.lengths), generator=self.generator).tolist()
        chunk_size = [len(indices) // self.global_batch_size] * self.global_batch_size
        for i in range(len(indices) % self.global_batch_size):
            chunk_size[i] += 1
        groups = [[] for _ in range(chunk_size[0])]
        start = 0
        for size in chunk_size:
            chunk = indices[start : start + size]
            chunk.sort(key=lambda x: self.lengths[x], reverse=True)
            for g, idx in zip(groups, chunk):
                g.append(idx)
            start += size
        if len(groups[-1]) < self.global_batch_size:
            groups[-1] = sorted(
                (groups[-1] * self.global_batch_size)[: self.global_batch_size]
            )
        group_orders = torch.randperm(len(groups) - 1, generator=self.generator)
        group_orders = [0] + (group_orders + 1).tolist()
        for i in group_orders:
            yield from groups[i]

    def __len__(self):
        return (
            math.ceil(len(self.lengths) / self.global_batch_size)
            * self.global_batch_size
        )
