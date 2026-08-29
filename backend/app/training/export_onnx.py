"""Export the multi-task model to ONNX (+ dynamic INT8 quantization) and
benchmark CPU inference: PyTorch eager vs ONNX FP32 vs ONNX INT8.

Outputs land next to the checkpoint (faceinsight.onnx, faceinsight-int8.onnx)
and the numbers feed the README's serving section. The production API keeps
the PyTorch path by default; ONNX is the documented optimization route for
CPU-bound deployments.

Usage:
    python -m app.training.export_onnx --checkpoint app/models/checkpoints/best.pth
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.config import CHECKPOINTS_DIR
from app.models.multitask_model import TASK_DIMS, MultiTaskFaceModel

TASK_ORDER = list(TASK_DIMS)  # fixed head order for the ONNX signature


class _TupleModel(nn.Module):
    """ONNX needs tensor outputs, not a dict: expose heads as a tuple."""

    def __init__(self, model: MultiTaskFaceModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        return tuple(outputs[task] for task in TASK_ORDER)


def benchmark(fn, warmup: int = 5, iters: int = 50) -> tuple[float, float]:
    """Return (mean_ms, p95_ms) of fn() over iters runs."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "best.pth")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = MultiTaskFaceModel(pretrained=False)
    model.load_state_dict(ckpt["model"])
    wrapped = _TupleModel(model)
    wrapped.eval()  # on the wrapper: a fresh nn.Module defaults to train mode

    dummy = torch.randn(1, 3, 224, 224)
    fp32_path = args.checkpoint.parent / "faceinsight.onnx"
    int8_path = args.checkpoint.parent / "faceinsight-int8.onnx"

    print("exporting ONNX (opset 17)…")
    # dynamo=False: the classic exporter honors opset 17 directly; the dynamo
    # path exports opset 18 and its downconversion breaks ORT quantization
    torch.onnx.export(
        wrapped, dummy, str(fp32_path),
        input_names=["face"], output_names=TASK_ORDER,
        dynamic_axes={"face": {0: "batch"}, **{t: {0: "batch"} for t in TASK_ORDER}},
        opset_version=17,
        dynamo=False,
    )
    print(f"quantizing (dynamic INT8)…")
    # QUInt8: onnxruntime's CPU ConvInteger kernel only implements uint8
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)

    # --- numerical parity (FP32 export vs eager) ---
    session_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    session_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        eager_out = wrapped(dummy)
    onnx_out = session_fp32.run(None, {"face": dummy.numpy()})
    max_diff = max(
        float((eager.numpy() - onnx).max()) for eager, onnx in zip(eager_out, onnx_out)
    )
    print(f"FP32 parity: max |eager - onnx| = {max_diff:.2e}")

    # --- INT8 sanity: emotion argmax must agree with eager on the dummy ---
    int8_out = session_int8.run(None, {"face": dummy.numpy()})
    agree = int(eager_out[0].argmax()) == int(np.argmax(int8_out[0]))
    print(f"INT8 emotion argmax agrees with eager on probe input: {agree}")

    # --- benchmarks ---
    x = dummy.numpy()
    results = {
        "pytorch eager": benchmark(lambda: wrapped(dummy), iters=args.iters),
        "onnx fp32": benchmark(lambda: session_fp32.run(None, {"face": x}), iters=args.iters),
        "onnx int8": benchmark(lambda: session_int8.run(None, {"face": x}), iters=args.iters),
    }
    size_mb = {
        "onnx fp32": fp32_path.stat().st_size / 1e6,
        "onnx int8": int8_path.stat().st_size / 1e6,
    }
    print(f"\n{'engine':<15} {'mean':>9} {'p95':>9} {'size':>9}")
    for name, (mean, p95) in results.items():
        size = f"{size_mb[name]:.0f} MB" if name in size_mb else "—"
        print(f"{name:<15} {mean:7.1f}ms {p95:7.1f}ms {size:>9}")


if __name__ == "__main__":
    main()
