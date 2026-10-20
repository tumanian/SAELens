from pathlib import Path

import pytest
import rescomp  # pyright: ignore[reportMissingImports]
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from sae_lens.config import SAETrainerConfig
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
from sae_lens.training import rescomp_activations as module
from sae_lens.training.rescomp_activations import (
    rescomp_activations,
    rescomp_chunks,
)
from sae_lens.training.sae_trainer import SAETrainer

D_MODEL = 16
LAYER = 1
HOOK = f"blocks.{LAYER}.hook_resid_post"


def build_model() -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=D_MODEL,
        d_head=4,
        n_heads=4,
        d_mlp=32,
        d_vocab=50,
        n_ctx=8,
        act_fn="gelu",
        normalization_type="LN",
        device="cpu",
    )
    return HookedTransformer(cfg).eval()


def write_capture(
    tmp_path: Path, model: HookedTransformer, n_sequences: int, chunk_size: int
) -> tuple[str, torch.Tensor]:
    """A capture of fp32 activations of random tokens (BOS at position 0), written in chunks; returns
    the stem and the live activations (n_sequences, 8, d_model)."""
    tokens = torch.randint(1, 50, (n_sequences, 8))
    tokens[:, 0] = 0
    stem = str(tmp_path / "capture" / "capture")
    sink = rescomp.Sink(stem, rescomp.bind(model, [LAYER]), rescomp.Codec("fp32"))
    with sink, torch.no_grad():
        for i in range(0, n_sequences, chunk_size):
            model(tokens[i : i + chunk_size])
    _, cache = model.run_with_cache(tokens, names_filter=HOOK)
    return stem, cache[HOOK]


def test_rescomp_chunks_yields_the_non_bos_activations_chunk_by_chunk_and_loops(
    tmp_path: Path,
):
    stem, live = write_capture(tmp_path, build_model(), n_sequences=6, chunk_size=4)
    expected = live[:, 1:].reshape(-1, D_MODEL)

    chunks = rescomp_chunks(stem, LAYER)
    first, second, third = next(chunks), next(chunks), next(chunks)

    assert torch.equal(first, expected[: 4 * 7])
    assert torch.equal(second, expected[4 * 7 :])
    assert torch.equal(third, first)


def test_rescomp_chunks_keeps_position_0_when_asked(tmp_path: Path):
    stem, live = write_capture(tmp_path, build_model(), n_sequences=3, chunk_size=3)

    chunk = next(rescomp_chunks(stem, LAYER, exclude_first_position=False))

    assert torch.equal(chunk, live.reshape(-1, D_MODEL))


def test_rescomp_activations_batches_are_shuffled_rows_of_the_capture(tmp_path: Path):
    stem, live = write_capture(tmp_path, build_model(), n_sequences=64, chunk_size=16)
    live = live[:, 1:].reshape(-1, D_MODEL)
    provider = rescomp_activations(
        stem, LAYER, train_batch_size_tokens=32, buffer_size_tokens=224
    )

    served = torch.cat([next(provider) for _ in range(8)])

    assert served.shape == (256, D_MODEL)
    is_a_live_row = (served[:, None, :] == live[None, :, :]).all(-1).any(-1)
    assert is_a_live_row.all()
    assert not torch.equal(served, live[:256])


def test_sae_trainer_fits_an_sae_from_rescomp_activations(tmp_path: Path):
    stem, live = write_capture(tmp_path, build_model(), n_sequences=64, chunk_size=16)
    live = live[:, 1:].reshape(-1, D_MODEL)
    provider = rescomp_activations(
        stem, LAYER, train_batch_size_tokens=64, buffer_size_tokens=256
    )
    sae = TopKTrainingSAE(
        TopKTrainingSAEConfig(d_in=D_MODEL, d_sae=4 * D_MODEL, k=4, device="cpu")
    )
    mse_before = (sae.decode(sae.encode(live)) - live).pow(2).mean().item()
    cfg = SAETrainerConfig(
        total_training_samples=64 * 100,
        train_batch_size_samples=64,
        lr=3e-3,
        lr_end=3e-4,
        device="cpu",
    )

    trained = SAETrainer(cfg, sae, provider).fit()

    mse_after = (trained.decode(trained.encode(live)) - live).pow(2).mean().item()
    assert mse_after < 0.5 * mse_before


def test_rescomp_chunks_raises_a_clear_error_when_rescomp_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(module, "rescomp", None)
    with pytest.raises(ImportError, match="pip install -e path/to/rescomp"):
        next(rescomp_chunks(str(tmp_path / "none"), LAYER))
