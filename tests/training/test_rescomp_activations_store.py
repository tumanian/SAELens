from pathlib import Path

import pytest
import rescomp  # pyright: ignore[reportMissingImports]
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from sae_lens.config import SAETrainerConfig
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
from sae_lens.training import rescomp_activations_store
from sae_lens.training.rescomp_activations_store import RescompActivationsStore
from sae_lens.training.sae_trainer import SAETrainer
from tests.helpers import assert_close

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
    tmp_path: Path,
    model: HookedTransformer,
    tokens: torch.Tensor,
    chunk_size: int,
    codec: str = "fp32",
    mask: torch.Tensor | None = None,
) -> str:
    stem = str(tmp_path / "capture" / "capture")
    sink = rescomp.Sink(stem, rescomp.bind(model, [LAYER]), rescomp.Codec(codec))
    with sink, torch.no_grad():
        for i in range(0, tokens.shape[0], chunk_size):
            if mask is not None:
                sink.tokens(tokens[i : i + chunk_size], mask[i : i + chunk_size])
            model(tokens[i : i + chunk_size])
    return stem


def live_activations(model: HookedTransformer, tokens: torch.Tensor) -> torch.Tensor:
    _, cache = model.run_with_cache(tokens, names_filter=HOOK)
    return cache[HOOK]


def random_tokens(n_sequences: int, context: int = 8) -> torch.Tensor:
    tokens = torch.randint(1, 50, (n_sequences, context))
    tokens[:, 0] = 0  # the BOS position
    return tokens


def test_iterate_epoch_yields_each_non_bos_activation_once_in_storage_order(
    tmp_path: Path,
):
    model = build_model()
    tokens = random_tokens(6)
    stem = write_capture(tmp_path, model, tokens, chunk_size=4)
    store = RescompActivationsStore(
        stem, LAYER, train_batch_size_tokens=7, buffer_size_tokens=7
    )

    chunks = list(store.iterate_epoch())
    expected = live_activations(model, tokens)[:, 1:].reshape(-1, D_MODEL)

    assert [c.shape for c in chunks] == [(4 * 7, D_MODEL), (2 * 7, D_MODEL)]
    assert torch.equal(torch.cat(chunks), expected)
    assert store.n_training_tokens == 6 * 7
    assert store.d_in == D_MODEL
    assert store.n_sequences == 6
    assert store.hook_name == HOOK
    store.close()


def test_exclude_first_position_false_keeps_the_bos_position(tmp_path: Path):
    model = build_model()
    tokens = random_tokens(3)
    stem = write_capture(tmp_path, model, tokens, chunk_size=3)
    store = RescompActivationsStore(
        stem,
        LAYER,
        train_batch_size_tokens=8,
        buffer_size_tokens=8,
        exclude_first_position=False,
    )

    chunk = next(store.iterate_epoch())

    assert torch.equal(chunk, live_activations(model, tokens).reshape(-1, D_MODEL))
    assert store.n_training_tokens == 3 * 8
    store.close()


def test_padding_positions_are_dropped_using_the_stored_mask(tmp_path: Path):
    model = build_model()
    tokens = random_tokens(4)
    mask = torch.ones_like(tokens)
    mask[0, 6:] = 0
    mask[2, 3:] = 0
    stem = write_capture(tmp_path, model, tokens, chunk_size=4, mask=mask)
    store = RescompActivationsStore(
        stem, LAYER, train_batch_size_tokens=4, buffer_size_tokens=4
    )

    chunk = next(store.iterate_epoch())
    acts = live_activations(model, tokens)
    expected = torch.cat([acts[i, 1 : int(mask[i].sum())] for i in range(4)])

    assert torch.equal(chunk, expected)
    assert store.n_training_tokens == expected.shape[0]
    store.close()


def test_batches_are_shuffled_rows_of_the_capture(tmp_path: Path):
    model = build_model()
    tokens = random_tokens(64)
    stem = write_capture(tmp_path, model, tokens, chunk_size=16)
    store = RescompActivationsStore(
        stem, LAYER, train_batch_size_tokens=32, buffer_size_tokens=224
    )
    live = live_activations(model, tokens)[:, 1:].reshape(-1, D_MODEL)

    served = torch.cat([next(store) for _ in range(8)])

    assert served.shape == (256, D_MODEL)
    is_a_live_row = (served[:, None, :] == live[None, :, :]).all(-1).any(-1)
    assert is_a_live_row.all()
    assert not torch.equal(served, live[:256])
    store.close()


def test_mix_fraction_zero_serves_the_capture_in_order_and_loops_over_epochs(
    tmp_path: Path,
):
    model = build_model()
    tokens = random_tokens(6)
    stem = write_capture(tmp_path, model, tokens, chunk_size=3)
    store = RescompActivationsStore(
        stem, LAYER, train_batch_size_tokens=21, buffer_size_tokens=21, mix_fraction=0
    )
    live = live_activations(model, tokens)[:, 1:].reshape(-1, D_MODEL)

    batches = [next(store) for _ in range(3)]

    assert torch.equal(batches[0], live[:21])
    assert torch.equal(batches[1], live[21:])
    assert torch.equal(batches[2], live[:21])
    assert store.n_epochs_started == 2
    store.close()


def test_prefetch_thread_serves_the_same_batches(tmp_path: Path):
    model = build_model()
    tokens = random_tokens(12)
    stem = write_capture(tmp_path, model, tokens, chunk_size=4)
    plain = RescompActivationsStore(
        stem, LAYER, train_batch_size_tokens=14, buffer_size_tokens=14, mix_fraction=0
    )
    prefetched = RescompActivationsStore(
        stem,
        LAYER,
        train_batch_size_tokens=14,
        buffer_size_tokens=14,
        mix_fraction=0,
        prefetch=2,
    )

    for _ in range(8):
        assert torch.equal(next(plain), next(prefetched))
    plain.close()
    prefetched.close()


def test_lossy_capture_rows_are_close_to_the_live_activations(tmp_path: Path):
    model = build_model()
    tokens = random_tokens(8)
    stem = write_capture(tmp_path, model, tokens, chunk_size=8, codec="faithful")
    store = RescompActivationsStore(
        stem, LAYER, train_batch_size_tokens=8, buffer_size_tokens=8
    )
    live = live_activations(model, tokens)[:, 1:].reshape(-1, D_MODEL)

    chunk = next(store.iterate_epoch())

    assert chunk.shape == live.shape
    assert_close(chunk, live, rtol=0, atol=0.05 * live.abs().max().item())
    assert not torch.equal(chunk, live)
    store.close()


def test_sae_trainer_fits_an_sae_from_the_store(tmp_path: Path):
    model = build_model()
    tokens = random_tokens(64)
    stem = write_capture(tmp_path, model, tokens, chunk_size=16)
    store = RescompActivationsStore(
        stem, LAYER, train_batch_size_tokens=64, buffer_size_tokens=256
    )
    live = live_activations(model, tokens)[:, 1:].reshape(-1, D_MODEL)
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

    trained = SAETrainer(cfg, sae, store).fit()
    store.close()

    mse_after = (trained.decode(trained.encode(live)) - live).pow(2).mean().item()
    assert mse_after < 0.5 * mse_before


def test_raises_a_clear_error_when_rescomp_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(rescomp_activations_store, "rescomp", None)
    with pytest.raises(ImportError, match="pip install -e path/to/rescomp"):
        RescompActivationsStore(
            str(tmp_path / "none"),
            LAYER,
            train_batch_size_tokens=1,
            buffer_size_tokens=1,
        )


def test_raises_for_a_layer_that_is_not_stored(tmp_path: Path):
    model = build_model()
    stem = write_capture(tmp_path, model, random_tokens(2), chunk_size=2)
    with pytest.raises(ValueError, match="layer 0 is not stored"):
        RescompActivationsStore(
            stem, 0, train_batch_size_tokens=1, buffer_size_tokens=1
        )
