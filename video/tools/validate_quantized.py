# Local numerical validation of the int8-quantized MLVC-S 1280x720 bundle
# against the original fp16 bundle, on a held-out real clip (not the one
# used for calibration), using the project's own FrameLoop harness so the
# comparison exercises the real recurrent state machinery (I-frame every 96
# frames, feature reset every 32 - see PLAN.md's reference-drift lessons)
# rather than a hand-rolled loop.
#
# Both bundles are loaded through ONNX Runtime's CPUExecutionProvider with
# graph optimizations disabled. This works around a real onnxruntime CPU-EP
# bug (not a bug in our quantized model): with ORT_ENABLE_ALL, onnxruntime's
# own QDQ->QOperator fusion pass produces an internal QLinearConcat node
# with a float16 scale input, which its schema rejects
# (INVALID_GRAPH: "Type 'tensor(float16)' ... of operator (QLinearConcat)
# ... is invalid"). This is CPU-EP-only: conversion/_model_wrapper.py's
# _load_onnx_model() already uses ORT_DISABLE_ALL for every non-CPU
# provider (DirectML, OpenVINO, QNN, CoreML) - QNN is the real deployment
# target and never exercises this fusion path. We monkeypatch just our own
# process's loader (no project files touched) to match that same
# ORT_DISABLE_ALL behavior for this local sanity check.
#
# Usage (from mlvc/video, using the project venv):
#   PYTHONPATH=/Users/sajil/genzee/mlvc/video \
#       /Users/sajil/genzee/mlvc/.venv/bin/python3 tools/validate_quantized.py [frame_count]
import sys
from pathlib import Path

import onnxruntime as ort

import conversion._model_wrapper as _mw
from conversion import load_split_model, FrameLoop
from conversion.types import RuntimeParams, OnnxExecutionProvider


def _load_onnx_model_no_opt(model_path, runtime_params):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"])


_mw._load_onnx_model = _load_onnx_model_no_opt

FP16_BUNDLE = "output/models/mlvc_s-mlvc-s-psnr-v1/onnx-qualcomm/1280x720"
INT8_BUNDLE = "output/models/mlvc_s-mlvc-s-psnr-v1/onnx-qualcomm-int8/1280x720"
# capture_natural.yuv, NOT capture_motion.yuv - the latter was used for
# calibration, this is an independent held-out clip, same 1280x720 real
# camera capture.
VAL_CLIP = "/Users/sajil/genzee/mlvc/demo_asset/capture_natural.yuv"


def run(bundle_dir: str, frame_count: int, q_index: int):
    params = RuntimeParams(onnx_execution_provider=OnnxExecutionProvider.CPU)
    split_model = load_split_model(bundle_dir, params)
    loop = FrameLoop(
        split_model=split_model,
        video_path=VAL_CLIP,
        image_width=1280,
        image_height=720,
        frame_count=frame_count,
    )
    return loop.run(q_index=q_index, reset_period=32, progress_bar=False)


def main() -> None:
    frame_count = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    q_index = 63

    print(f"Validating on {VAL_CLIP} ({frame_count} frames, q_index={q_index})\n")

    print("=== fp16 (original) ===")
    fp16_results = run(FP16_BUNDLE, frame_count, q_index)
    fp16_psnr = [f.metrics.psnr for f in fp16_results.frames]
    fp16_bpp = [f.metrics.bpp for f in fp16_results.frames]

    print("=== int8 (quantized) ===")
    int8_results = run(INT8_BUNDLE, frame_count, q_index)
    int8_psnr = [f.metrics.psnr for f in int8_results.frames]
    int8_bpp = [f.metrics.bpp for f in int8_results.frames]

    print(f"\n{'frame':>5} {'fp16 psnr':>10} {'int8 psnr':>10} {'delta':>8} {'fp16 bpp':>10} {'int8 bpp':>10}")
    for i in range(frame_count):
        print(
            f"{i:>5} {fp16_psnr[i]:>10.3f} {int8_psnr[i]:>10.3f} "
            f"{int8_psnr[i]-fp16_psnr[i]:>8.3f} {fp16_bpp[i]:>10.5f} {int8_bpp[i]:>10.5f}"
        )

    def median(xs):
        s = sorted(xs)
        return s[len(s) // 2]

    print(f"\nmedian PSNR: fp16={median(fp16_psnr):.3f} dB  int8={median(int8_psnr):.3f} dB  "
          f"delta={median(int8_psnr)-median(fp16_psnr):+.3f} dB")
    print(f"median bpp:  fp16={median(fp16_bpp):.5f}  int8={median(int8_bpp):.5f}")
    print(f"min PSNR:    fp16={min(fp16_psnr):.3f} dB  int8={min(int8_psnr):.3f} dB")


if __name__ == "__main__":
    main()
