"""Uncompiled candidate retained solely for reproducing comparison benchmarks."""

import torch


def _path_signature(path: torch.Tensor, depth: int) -> torch.Tensor:
    """Compute a truncated tensor signature with the origin prepended to the path.

    The returned levels are concatenated in degree order, with each level flattened
    in lexicographic tensor-product order. This matches signatory's basepoint=True
    convention: prepend the increment from the origin to the first path point.
    """
    if path.ndim != 3:
        raise ValueError("path must have shape [batch, length, channels]")
    if path.shape[1] == 0:
        raise ValueError("path must contain at least one point")
    if depth < 1:
        raise ValueError("depth must be a positive integer")

    increments = torch.cat((path[:, :1], path[:, 1:] - path[:, :-1]), dim=1)
    batch_size, _, channels = path.shape
    levels = [path.new_ones(batch_size, 1)] + [
        path.new_zeros(batch_size, channels**degree) for degree in range(1, depth + 1)
    ]
    for increment in increments.unbind(dim=1):
        segment = [levels[0]]
        for degree in range(1, depth + 1):
            previous = segment[-1]
            segment.append(
                torch.bmm(previous.unsqueeze(-1), increment.unsqueeze(1)).flatten(1)
                / degree
            )
        updated = []
        for degree in range(depth + 1):
            terms = [
                torch.bmm(
                    levels[left].unsqueeze(-1), segment[degree - left].unsqueeze(1)
                ).flatten(1)
                for left in range(degree + 1)
            ]
            updated.append(torch.stack(terms).sum(0))
        levels = updated
    return torch.cat(levels[1:], dim=1)
