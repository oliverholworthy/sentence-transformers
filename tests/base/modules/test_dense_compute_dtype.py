from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizer

from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.base.modules import Dense, Router, Transformer
from sentence_transformers.sentence_transformer.modules import Pooling

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("autocast", [False, True])
def test_fp32_dense_keeps_nearby_scores_distinct(device: str, autocast: bool) -> None:
    head = Dense(2, 1, bias=False, activation_function=None, compute_dtype="float32").to(device)
    with torch.no_grad():
        head.linear.weight.fill_(1)
    inputs = torch.tensor([[1, 0.001], [1, 0.002]], device=device, dtype=torch.bfloat16)
    # Both inputs are representable in BF16, but their projected scores round to the same BF16 value.
    rounded = F.linear(inputs, head.linear.weight.bfloat16())
    assert rounded[0] == rounded[1]

    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        scores = head({"sentence_embedding": inputs})["sentence_embedding"]
    assert scores.dtype == torch.float32
    assert scores[0] != scores[1]
    torch.testing.assert_close(scores, inputs.double().sum(dim=-1, keepdim=True).float(), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("out_features, use_residual", [(3, False), (3, True), (4, True)])
@pytest.mark.parametrize("bias", [False, True])
def test_compute_dtype_matches_fp32_forward_and_gradients(
    device: str, out_features: int, use_residual: bool, bias: bool
) -> None:
    torch.manual_seed(42)
    head = Dense(4, out_features, bias=bias, use_residual=use_residual, compute_dtype=torch.float32).to(device)
    reference = deepcopy(head)
    reference.compute_dtype = None
    inputs = torch.randn(2, 4, device=device, dtype=torch.bfloat16, requires_grad=True)
    reference_inputs = inputs.detach().float().requires_grad_()

    with torch.autocast(device, dtype=torch.bfloat16):
        actual = head({"sentence_embedding": inputs})["sentence_embedding"]
    expected = reference({"sentence_embedding": reference_inputs})["sentence_embedding"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad.bfloat16(), rtol=0, atol=0)
    for parameter, reference_parameter in zip(head.parameters(), reference.parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize("compute_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("safe_serialization", [False, True])
def test_compute_dtype_round_trip(tmp_path: Path, compute_dtype: torch.dtype, safe_serialization: bool) -> None:
    head = Dense(4, 1, use_residual=True, compute_dtype=compute_dtype)
    head.save(str(tmp_path), safe_serialization=safe_serialization)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["compute_dtype"] == str(compute_dtype).removeprefix("torch.")
    loaded = Dense.load(str(tmp_path), local_files_only=True)
    assert loaded.compute_dtype == compute_dtype
    for name, parameter in loaded.named_parameters():
        assert parameter.dtype == compute_dtype
        torch.testing.assert_close(parameter, head.state_dict()[name], rtol=0, atol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        scores = loaded({"sentence_embedding": torch.ones(2, 4)})["sentence_embedding"]
    assert scores.dtype == compute_dtype


@pytest.mark.parametrize("compute_dtype", ["not_a_dtype", "int8", torch.int64, "auto", 32])
def test_invalid_compute_dtype(compute_dtype) -> None:
    with pytest.raises(ValueError, match="compute_dtype"):
        Dense(2, 1, compute_dtype=compute_dtype)


def test_default_dense_keeps_legacy_config_and_autocast_behavior(tmp_path: Path) -> None:
    head = Dense(4, 1, activation_function=None)
    head.save(str(tmp_path))
    assert "compute_dtype" not in json.loads((tmp_path / "config.json").read_text())
    loaded = Dense.load(str(tmp_path), local_files_only=True)
    assert loaded.compute_dtype is None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        scores = loaded({"sentence_embedding": torch.ones(2, 4)})["sentence_embedding"]
    assert scores.dtype == torch.bfloat16


def test_compute_dtype_preserves_module_forward_hooks_and_activation_parameters() -> None:
    class OffsetLinear(nn.Linear):
        def forward(self, inputs):
            return super().forward(inputs) + 0.1

    head = Dense(4, 2, activation_function=nn.PReLU(), compute_dtype="float32")
    head.linear = OffsetLinear(4, 2)
    reference = deepcopy(head).float()
    reference.compute_dtype = None
    inputs = torch.tensor([[-1, 1, -2, 2]], dtype=torch.bfloat16)
    observed = []
    hook = head.linear.register_forward_hook(lambda module, args, output: observed.append(output.dtype))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = head({"sentence_embedding": inputs})["sentence_embedding"]
    hook.remove()
    expected = reference({"sentence_embedding": inputs.float()})["sentence_embedding"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert observed == [torch.float32]
    actual.sum().backward()
    assert head.activation_function.weight.grad is not None
    assert torch.isfinite(head.activation_function.weight.grad).all()
    assert head.linear.weight.dtype == torch.float32


@pytest.mark.parametrize("part", ["head", "activation", "residual"])
def test_compute_dtype_rejects_incompatible_explicit_casts(part: str) -> None:
    head = Dense(4, 2, activation_function=nn.PReLU(), use_residual=True, compute_dtype="float32")
    module = {"head": head, "activation": head.activation_function, "residual": head.residual}[part]
    module.bfloat16()
    with pytest.raises(RuntimeError, match="Set the backbone dtype through model_kwargs"):
        head({"sentence_embedding": torch.ones(2, 4, dtype=torch.bfloat16)})


def test_compute_dtype_preserves_activation_running_statistics() -> None:
    head = Dense(2, 2, bias=False, activation_function=nn.BatchNorm1d(2), compute_dtype="float32")
    with torch.no_grad():
        head.linear.weight.copy_(torch.eye(2))
    reference = deepcopy(head)
    reference.compute_dtype = None
    inputs = torch.tensor([[2, 4], [4, 8]], dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = head({"sentence_embedding": inputs})["sentence_embedding"]
    expected = reference({"sentence_embedding": inputs.float()})["sentence_embedding"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, buffer in head.named_buffers():
        torch.testing.assert_close(buffer, reference.state_dict()[name], rtol=0, atol=0)
    assert head.activation_function.running_mean.abs().sum() > 0


def test_compute_dtype_without_device_autocast_support(monkeypatch: pytest.MonkeyPatch) -> None:
    def unsupported_autocast(*args, **kwargs):
        raise RuntimeError("User specified an unsupported autocast device_type")

    # Older PyTorch versions reject MPS even with enabled=False. No context is needed there.
    monkeypatch.setattr(torch, "autocast", unsupported_autocast)
    head = Dense(2, 1, bias=False, activation_function=None, compute_dtype="float32")
    inputs = torch.ones(2, 2, dtype=torch.bfloat16)
    actual = head({"sentence_embedding": inputs})["sentence_embedding"]
    torch.testing.assert_close(actual, F.linear(inputs.float(), head.linear.weight), rtol=0, atol=0)


def test_compute_dtype_concurrent_inference_preserves_parameters() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    head = Dense(2, 1, activation_function=None, compute_dtype="float32")
    inputs = torch.ones(2, 2, dtype=torch.bfloat16)
    expected = head({"sentence_embedding": inputs})["sentence_embedding"].detach()
    weight = head.linear.weight
    barrier = Barrier(2, timeout=10)

    def hook(module, args):
        barrier.wait()
        assert module.weight is weight

    handle = head.linear.register_forward_pre_hook(hook)

    def predict():
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            return head({"sentence_embedding": inputs})["sentence_embedding"]

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(predict) for _ in range(2)]
            for future in futures:
                torch.testing.assert_close(future.result(), expected, rtol=0, atol=0)
    finally:
        handle.remove()
    assert head.linear.weight is weight


@pytest.mark.parametrize("offload", ["cpu", "disk"])
def test_compute_dtype_with_offloaded_linear(tmp_path: Path, offload: str) -> None:
    from accelerate import cpu_offload, disk_offload

    head = Dense(4, 1, activation_function=None, compute_dtype="float32")
    inputs = torch.ones(2, 4, dtype=torch.bfloat16)
    expected = head({"sentence_embedding": inputs})["sentence_embedding"].detach()
    if offload == "cpu":
        cpu_offload(head.linear, execution_device=torch.device("cpu"))
    else:
        disk_offload(head.linear, str(tmp_path), execution_device=torch.device("cpu"))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = head({"sentence_embedding": inputs})["sentence_embedding"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert head.linear.weight.device.type == "meta"


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("delegated", [False, True])
def test_model_alignment_preserves_dense_weight_precision(nested: bool, delegated: bool) -> None:
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    weights = torch.tensor([[0.1001, 0.2002, 0.3003, 0.4004]])
    assert not torch.equal(weights, weights.bfloat16().float())
    head = Dense(4, 1, init_weight=weights.clone(), compute_dtype="float32", use_residual=True)
    expected = deepcopy(head.state_dict())
    module = nn.Sequential(nn.Linear(4, 4), head) if nested else head
    backbone = nn.Linear(4, 4, dtype=torch.bfloat16)
    if delegated:
        add_hook_to_module(backbone, AlignDevicesHook(execution_device="cpu"))
    model = SentenceTransformer(
        modules=[backbone, module],
        device="cpu",
    )
    assert model._placement_is_delegated == delegated
    if delegated:
        model._resolve_inference_device(None)  # First inference aligns auxiliary modules a second time.
    assert backbone.weight.dtype == torch.bfloat16
    if nested:
        assert module[0].weight.dtype == torch.bfloat16
    for name, parameter in head.named_parameters():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("delegated", [False, True])
@pytest.mark.parametrize("share_module", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_model_alignment_preserves_shared_dense_precision(delegated: bool, share_module: bool, reverse: bool) -> None:
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    weights = torch.tensor([[0.1001, 0.2002, 0.3003, 0.4004]])
    head = Dense(4, 1, bias=False, init_weight=weights.clone(), compute_dtype="float32")
    head.linear.weight.grad = torch.full_like(weights, 0.1001)
    ordinary = Dense(4, 1, bias=False)
    if share_module:
        ordinary.linear = head.linear
    else:
        ordinary.linear.weight = head.linear.weight
    routes = {"query": [head], "document": [ordinary]}
    if reverse:
        routes = dict(reversed(list(routes.items())))
    backbone = nn.Linear(4, 4, dtype=torch.bfloat16)
    if delegated:
        add_hook_to_module(backbone, AlignDevicesHook(execution_device="cpu"))
    SentenceTransformer(modules=[backbone, Router(routes)], device="cpu")
    assert head.linear.weight is ordinary.linear.weight
    torch.testing.assert_close(head.linear.weight, weights, rtol=0, atol=0)
    torch.testing.assert_close(head.linear.weight.grad, torch.full_like(weights, 0.1001), rtol=0, atol=0)


@pytest.mark.parametrize("delegated", [False, True])
def test_model_alignment_rejects_conflicting_shared_dtypes(delegated: bool) -> None:
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    first = Dense(4, 1, compute_dtype="float32")
    second = Dense(4, 1, compute_dtype="float64")
    second.linear.weight = first.linear.weight
    expected = first.linear.weight.detach().clone()
    backbone = nn.Linear(4, 4, dtype=torch.bfloat16)
    if delegated:
        add_hook_to_module(backbone, AlignDevicesHook(execution_device="cpu"))
    with pytest.raises(ValueError, match="sharing tensors must use the same compute_dtype"):
        SentenceTransformer(modules=[backbone, first, second], device="cpu")
    torch.testing.assert_close(first.linear.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device_map", [None, {"": "cpu"}])
def test_cross_encoder_load_predict_and_reload(tmp_path: Path, device_map: dict | None) -> None:
    backbone_path = tmp_path / "backbone"
    config = BertConfig(vocab_size=5, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16)
    BertModel(config).bfloat16().save_pretrained(backbone_path)
    BertTokenizer(
        vocab={token: i for i, token in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])}
    ).save_pretrained(backbone_path)
    transformer = Transformer(str(backbone_path), model_kwargs={"dtype": torch.bfloat16})
    weights = torch.linspace(0.1001, 0.8008, 8).reshape(1, 8)
    model = CrossEncoder(
        modules=[
            transformer,
            Pooling(8, pooling_mode="mean"),
            Dense(
                8,
                1,
                bias=False,
                activation_function=None,
                init_weight=weights.clone(),
                module_output_name="scores",
                compute_dtype="float32",
            ),
        ],
        device="cpu",
        activation_fn=nn.Identity(),
    )
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(str(checkpoint), create_model_card=False)
    pairs = [("hello", "world"), ("longer query", "another document")]
    expected = model.predict(pairs, convert_to_tensor=True, show_progress_bar=False)

    for _ in range(2):
        loaded = CrossEncoder(
            str(checkpoint),
            device=None if device_map else "cpu",
            model_kwargs={"device_map": device_map} if device_map else None,
            local_files_only=True,
        )
        assert next(loaded[0].parameters()).dtype == torch.bfloat16
        assert loaded[2].compute_dtype == torch.float32
        torch.testing.assert_close(loaded[2].linear.weight, weights, rtol=0, atol=0)
        observed = []
        hook = loaded[2].register_forward_hook(lambda module, args, output: observed.append(output["scores"].dtype))
        actual = loaded.predict(pairs, convert_to_tensor=True, show_progress_bar=False)
        hook.remove()
        assert observed and all(dtype == torch.float32 for dtype in observed)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(loaded[2].linear.weight, weights, rtol=0, atol=0)
        loaded.save_pretrained(str(checkpoint), create_model_card=False)
