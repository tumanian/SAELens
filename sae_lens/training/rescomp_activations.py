"""SAE training activations read from a rescomp capture instead of a live model."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from sae_lens.training.mixing_buffer import mixing_buffer

try:
    import rescomp  # pyright: ignore[reportMissingImports]
except ImportError:  # rescomp is an optional dependency
    rescomp = None

RESCOMP_INSTALL_HINT = (
    "this needs the rescomp package (https://github.com/tumanian/rescomp): "
    "pip install -e path/to/rescomp"
)


def rescomp_chunks(
    path: str,
    layer: int,
    device: str | torch.device = "cpu",
    exclude_first_position: bool = True,
) -> Iterator[torch.Tensor]:
    """
    Endless iterator over a rescomp capture (https://github.com/tumanian/rescomp):
    one (n_tokens, d_in) float32 tensor per stored chunk, each chunk decoded once
    per pass, in storage order. A chunk holds the sequences of one forward call of
    the captured model. Padding positions are left out using the stored attention
    mask, and with ``exclude_first_position`` so is the first real position of every
    sequence (normally the BOS token).
    """
    if rescomp is None:
        raise ImportError(RESCOMP_INSTALL_HINT)
    reader = rescomp.open(path)
    groups = [
        [sid for sid, start in zip(chunk["seq_ids"], chunk["starts"]) if start == 0]
        for chunk in reader.chunks
    ]
    first = 1 if exclude_first_position else 0
    while True:
        for seq_ids in groups:
            acts = reader.read(layer, seqs=seq_ids)
            masks = reader.attention_mask(seq_ids)
            rows = [
                (x if mask is None else x[mask.bool()])[first:]
                for x, mask in zip(acts, masks)
            ]
            yield torch.cat(rows).to(device)


def rescomp_capture_info(path: str, layer: int) -> dict[str, int | str | None]:
    """
    What an SAE config needs from a capture: ``d_in`` (the model width) and
    ``hook_name`` (the captured model's hook for ``layer``, when recorded).
    """
    if rescomp is None:
        raise ImportError(RESCOMP_INSTALL_HINT)
    reader = rescomp.open(path)
    hooks = reader.header["layers"][reader.layers.index(layer)]["hook"]
    hook_name = None if hooks is None else next(iter(hooks.values()))
    return {"d_in": int(reader.d), "hook_name": hook_name}


def rescomp_activations(
    path: str,
    layer: int,
    train_batch_size_tokens: int,
    buffer_size_tokens: int,
    device: str | torch.device = "cpu",
    exclude_first_position: bool = True,
    mix_fraction: float = 0.5,
) -> Iterator[torch.Tensor]:
    """
    A DataProvider for ``SAETrainer``: shuffled (train_batch_size_tokens, d_in)
    batches of one stored layer of a rescomp capture, through the same
    ``mixing_buffer`` that ``ActivationsStore`` uses. The captured model is not
    needed; training runs from the files alone.
    """
    return mixing_buffer(
        buffer_size=buffer_size_tokens,
        batch_size=train_batch_size_tokens,
        activations_loader=rescomp_chunks(path, layer, device, exclude_first_position),
        mix_fraction=mix_fraction,
    )
