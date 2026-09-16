"""Minimal data utilities required by the curated ReDial code flow."""

from typing import List, Optional, Union

import torch


def padded_tensor(
    items: List[Union[List[int], torch.LongTensor]],
    pad_idx: int = 0,
    pad_tail: bool = True,
    max_len: Optional[int] = None,
    debug: bool = False,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = False,
) -> torch.LongTensor:
    """Pad an uneven list of integer sequences into a 2-D tensor."""
    lens = [len(item) for item in items]
    target_len = max(max(lens), 1)
    if debug and max_len is not None:
        target_len = max(target_len, max_len)
    if use_amp:
        target_len = max((target_len + 7) // 8 * 8, 8)

    output = torch.full(
        (len(items), target_len),
        fill_value=pad_idx,
        dtype=torch.long,
        device=device,
    )
    for index, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            continue
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item, dtype=torch.long, device=device)
        if pad_tail:
            output[index, :length] = item
        else:
            output[index, target_len - length :] = item
    return output
