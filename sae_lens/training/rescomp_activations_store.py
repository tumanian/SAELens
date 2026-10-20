"""SAE training activations read from a rescomp capture instead of a live model."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch

from sae_lens import logger
from sae_lens.training.mixing_buffer import mixing_buffer
from sae_lens.training.prefetch import PrefetchingIterator

try:
    import rescomp  # pyright: ignore[reportMissingImports]
except ImportError:  # rescomp is an optional dependency
    rescomp = None

RESCOMP_INSTALL_HINT = (
    "RescompActivationsStore needs the rescomp package "
    "(https://github.com/tumanian/rescomp): pip install -e path/to/rescomp"
)


class RescompActivationsStore(Iterator[torch.Tensor]):
    """
    A DataProvider that reads one layer's residual-stream activations from a
    rescomp capture (https://github.com/tumanian/rescomp) and serves shuffled
    training batches of shape (train_batch_size_tokens, d_in). The captured
    model is not needed: SAE training runs from the files alone.

    A capture holds the activations of every forward call the model made under
    a ``rescomp.Sink``, compressed, together with the token ids. The store reads
    it back in those chunks, so each chunk is decoded once per epoch. Each chunk
    becomes an (n_tokens, d_in) float tensor: padding positions are dropped using
    the stored attention mask, and the first real position of every sequence
    (normally the BOS token) is dropped when ``exclude_first_position`` is True.
    Chunks then go through the same ``mixing_buffer`` shuffle that
    ``ActivationsStore`` uses. The store loops over the capture for ever and
    warns at each new epoch, like ``ActivationsStore``.

    Args:
        path: the capture stem (``<stem>.meta.parquet`` plus ``<stem>.NNNNN.rsc``
            files), a local path or an s3:// URI
        layer: the stored layer to read
        train_batch_size_tokens: rows per training batch
        buffer_size_tokens: rows held in the mixing buffer; at least
            ``train_batch_size_tokens``
        device: device of the yielded batches
        dtype: dtype of the yielded batches
        exclude_first_position: drop position 0 of every sequence (the BOS token)
        mix_fraction: fraction of the buffer kept back for mixing with the next
            chunks; 0 serves the capture in order (see ``mixing_buffer``)
        prefetch: when > 0, decode chunks in a background thread that keeps up
            to this many chunks ready, so decoding overlaps the SAE training step
        verify: check every blob's CRC32 while reading
    """

    def __init__(
        self,
        path: str,
        layer: int,
        train_batch_size_tokens: int,
        buffer_size_tokens: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        exclude_first_position: bool = True,
        mix_fraction: float = 0.5,
        prefetch: int = 0,
        verify: bool = True,
    ):
        if rescomp is None:
            raise ImportError(RESCOMP_INSTALL_HINT)
        if buffer_size_tokens < train_batch_size_tokens:
            raise ValueError(
                "buffer_size_tokens must be at least train_batch_size_tokens"
            )
        self.path = path
        self.layer = layer
        self.train_batch_size_tokens = train_batch_size_tokens
        self.buffer_size_tokens = buffer_size_tokens
        self.device = torch.device(device)
        self.dtype = dtype
        self.exclude_first_position = exclude_first_position
        self.mix_fraction = mix_fraction
        self.prefetch = prefetch
        self.reader = rescomp.open(path, verify=verify)
        if layer not in self.reader.layers:
            raise ValueError(
                f"layer {layer} is not stored in {path}; stored layers: {self.reader.layers}"
            )
        # Sequences grouped by the chunk they start in, in chunk order. Reading a
        # chunk's sequences together decodes that chunk once.
        self._sequence_groups: list[list[int]] = [
            [sid for sid, start in zip(chunk["seq_ids"], chunk["starts"]) if start == 0]
            for chunk in self.reader.chunks
        ]
        self._sequence_groups = [g for g in self._sequence_groups if g]
        self.n_epochs_started = 0
        self._prefetcher: PrefetchingIterator[torch.Tensor] | None = None
        self._dataloader: Iterator[torch.Tensor] | None = None

    @property
    def header(self) -> dict[str, Any]:
        """The capture header: model info, spec, layers, fidelity, byte counts."""
        return self.reader.header

    @property
    def d_in(self) -> int:
        return int(self.reader.d)

    @property
    def hook_name(self) -> str | None:
        """The captured model's hook name for ``layer``, when the capture recorded it."""
        hooks = self.header["layers"][self.reader.layers.index(self.layer)]["hook"]
        if hooks is None:
            return None
        return next(iter(hooks.values()))

    @property
    def n_sequences(self) -> int:
        return int(self.reader.n_seq)

    @property
    def n_training_tokens(self) -> int:
        """Rows served per epoch: stored tokens minus padding and dropped first positions."""
        table = self.reader.sequences()
        n_tokens = table.column("n_tokens").to_pylist()
        masks = table.column("mask").to_pylist()
        total = 0
        for n, mask in zip(n_tokens, masks):
            real = int(sum(mask)) if mask is not None else int(n)
            total += max(0, real - 1) if self.exclude_first_position else real
        return total

    def _chunk_rows(self, seq_ids: list[int]) -> torch.Tensor:
        """The training rows of one chunk's sequences: (n_tokens, d_in) on ``device``."""
        acts = self.reader.read(self.layer, seqs=seq_ids, dtype=self.dtype)
        masks = self.reader.attention_mask(seq_ids)
        rows: list[torch.Tensor] = []
        for x, mask in zip(acts, masks):
            if mask is not None:
                x = x[mask.bool()]
            if self.exclude_first_position:
                x = x[1:]
            rows.append(x)
        return torch.cat(rows, dim=0).to(self.device)

    def iterate_epoch(self) -> Iterator[torch.Tensor]:
        """One pass over the capture, one (n_tokens, d_in) tensor per chunk, in storage order."""
        for seq_ids in self._sequence_groups:
            yield self._chunk_rows(seq_ids)

    def _iterate_chunks(self) -> Iterator[torch.Tensor]:
        while True:
            if self.n_epochs_started > 0:
                logger.warning(
                    f"All {self.n_sequences} sequences of {self.path} have been served, beginning new epoch."
                )
            self.n_epochs_started += 1
            yield from self.iterate_epoch()

    def get_data_loader(self) -> Iterator[torch.Tensor]:
        """An endless stream of shuffled (train_batch_size_tokens, d_in) batches."""
        chunks: Iterator[torch.Tensor] = self._iterate_chunks()
        if self.prefetch > 0:
            self._prefetcher = PrefetchingIterator(chunks, prefetch=self.prefetch)
            chunks = self._prefetcher
        return mixing_buffer(
            buffer_size=self.buffer_size_tokens,
            batch_size=self.train_batch_size_tokens,
            activations_loader=chunks,
            mix_fraction=self.mix_fraction,
        )

    def __next__(self) -> torch.Tensor:
        if self._dataloader is None:
            self._dataloader = self.get_data_loader()
        return next(self._dataloader)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def close(self) -> None:
        """Stop the decode thread, if any, and close the capture's files."""
        if self._prefetcher is not None:
            self._prefetcher.close()
            self._prefetcher = None
        self._dataloader = None
        self.reader.close()
