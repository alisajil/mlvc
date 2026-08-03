# Local PTQ of the MLVC-S 1280x720 ONNX graphs via onnxruntime.quantization,
# entirely bypassing AI Hub's own PTQ pipeline (see PLAN.md's "W8A16 PTQ
# ATTEMPT - BLOCKED ON AI HUB QUANTIZER" section, 2026-08-02: AI Hub's
# quantizer failed identically 3 times during generateActivations on
# q_index_shifted's batch/file-count bookkeeping - a structural issue in
# their tooling, not something calibration-data tweaks could route around).
#
# This script quantizes each of MLVCEncoderPart1 / MLVCEncoderPart2 /
# MLVCDecoder independently (they are separate ONNX graphs on disk) using
# real captured recurrent-state calibration data (see
# capture_calib_data_720p.py in the session scratchpad), then copies the
# bundle's sidecar metadata so the quantized bundle can be loaded via
# conversion.load_split_model exactly like the original fp16 bundle.
#
# Usage (from mlvc/video, using the project venv):
#   PYTHONPATH=/Users/sajil/genzee/mlvc/video \
#       /Users/sajil/genzee/mlvc/.venv/bin/python3 tools/quantize_ptq.py
import shutil
import sys
import tempfile
from pathlib import Path

import onnx
from onnx import version_converter
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from tools.quant_calibration import ListCalibrationDataReader, load_calibration_feeds

SRC_BUNDLE = Path("output/models/mlvc_s-mlvc-s-psnr-v1/onnx-qualcomm/1280x720")
DST_BUNDLE = Path("output/models/mlvc_s-mlvc-s-psnr-v1/onnx-qualcomm-int8/1280x720")

MODEL_NAMES = ["MLVCEncoderPart1", "MLVCEncoderPart2", "MLVCDecoder"]

SIDECAR_FILES = ["metadata.json", "gaussian_pmf.json", "bit_estimator_pmf.json"]

# Tensors we ATTEMPTED to protect from quantization, per model part: these
# are either the codec's own entropy-coding symbols (z_raw/y_raw_0/y_raw_1 -
# produced by an explicit Floor/round op in MLVCEncoderPart1, i.e. these ARE
# the actual transmitted integer-valued codes, losslessly entropy-coded in
# the real pipeline) or the recurrent memory state ("feature", carried
# across frames as ref_feature).
#
# IMPORTANT - this exclusion turned out NOT to fix the accuracy problem
# (kept only because it's still the conceptually-correct thing to do, and
# harmless): diag_per_part.py (scratchpad) showed IDENTICAL corruption of
# z_raw/feature with and without these exclusions. Root cause is NOT
# re-quantization at this exact boundary - it's ordinary int8 activation
# noise accumulated through the *preceding* conv stack, which is already
# corrupted by the time it reaches the Floor op this exclusion protects.
# Excluding the last node doesn't help if its input was already wrong.
# See the accuracy caveat below quantize_static() for the real status.
BOUNDARY_TENSORS = {
    "MLVCEncoderPart1": ["feature", "z_raw", "y_raw_0", "y_raw_1"],  # all graph outputs
    "MLVCEncoderPart2": ["feature"],  # graph input
    "MLVCDecoder": ["feature", "z_raw", "y_raw_0", "y_raw_1"],  # feature=output, rest=inputs
}


def _name_boundary_nodes(model: onnx.ModelProto, tensor_names: list[str]) -> list[str]:
    """Give stable, unique names to whichever nodes produce (if the tensor
    is a graph output) or consume (if it's a graph input) each of the given
    tensors - many nodes in this export have name=="", so we cannot pass
    that to nodes_to_exclude without matching every other unnamed node too.
    Returns the list of names to exclude from quantization."""
    graph_inputs = {i.name for i in model.graph.input}
    graph_outputs = {o.name for o in model.graph.output}
    excluded = []
    for tensor_name in tensor_names:
        if tensor_name in graph_outputs:
            for n in model.graph.node:
                if tensor_name in n.output:
                    n.name = f"keep_fp16_producer_{tensor_name}"
                    excluded.append(n.name)
        elif tensor_name in graph_inputs:
            count = 0
            for n in model.graph.node:
                if tensor_name in n.input:
                    n.name = f"keep_fp16_consumer_{tensor_name}_{count}"
                    excluded.append(n.name)
                    count += 1
        else:
            raise ValueError(f"{tensor_name} is neither a graph input nor output")
    return excluded


# The deployed graphs are opset 18. DequantizeLinear/QuantizeLinear's
# x_scale input is hardcoded to tensor(float) on the pre-19 schema - only
# opset 19+ widens it to {float, float16, bfloat16} (verified via
# onnx.defs.get_schema; bit-exact-identical outputs on random input before
# vs after conversion for this graph, so this is a pure metadata bump here,
# not an operator behavior change). Since these graphs declare fp16 I/O
# natively, onnxruntime.quantization emits fp16-scale Q/DQ nodes to match -
# which is invalid at opset 18 (ORT CPU EP raises INVALID_GRAPH loading it).
# Upgrade to opset 19 first so the emitted QDQ nodes are schema-valid.
TARGET_OPSET = 19


def _prepare_source_model(model_name: str, src_path: Path, dst_path: Path) -> list[str]:
    """Upgrade to opset 19 (for fp16-scale QDQ support) and stamp stable
    names onto this model's boundary nodes (see BOUNDARY_TENSORS) so they
    can be passed to nodes_to_exclude. Returns that exclude list."""
    model = onnx.load(str(src_path))
    current = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), None)
    if current is not None and current < TARGET_OPSET:
        model = version_converter.convert_version(model, TARGET_OPSET)

    excluded = _name_boundary_nodes(model, BOUNDARY_TENSORS[model_name])
    onnx.save(model, str(dst_path))
    return excluded


def quantize_one(model_name: str, calib_dir: str, tmp_dir: Path) -> None:
    src_path = SRC_BUNDLE / f"{model_name}.onnx"
    opset19_path = tmp_dir / f"{model_name}.opset19.onnx"
    dst_path = DST_BUNDLE / f"{model_name}.onnx"

    print(f"\n=== {model_name} ===")
    nodes_to_exclude = _prepare_source_model(model_name, src_path, opset19_path)
    print(f"excluding from quantization (boundary/symbol tensors): {nodes_to_exclude}")

    feeds = load_calibration_feeds(calib_dir, model_name, str(opset19_path))
    print(f"{len(feeds)} calibration samples loaded from {calib_dir}")
    reader = ListCalibrationDataReader(feeds)

    quantize_static(
        model_input=str(opset19_path),
        model_output=str(dst_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        # Plain int8: QUInt8 activations (Qualcomm HTP's own default
        # asymmetric-activation convention), QInt8 per-channel weights.
        #
        # ACCURACY CAVEAT (unresolved - see final report): this configuration
        # produces a valid, loadable graph, but validate_quantized.py measured
        # ~35 dB PSNR loss vs the fp16 original on a held-out clip, WORSENING
        # frame over frame (recurrent drift on top of an already-bad
        # per-frame baseline). Root cause (diag_per_part.py, scratchpad):
        # int8's 256 levels are too coarse once noise compounds through the
        # conv stack and reaches the codec's own Floor/round step (a
        # discontinuous op - noise that flips which side of a rounding
        # boundary a value lands on changes the transmitted integer code
        # outright).
        #
        # w8a16 (QUInt16/QInt16 activations) was tried as the fix - matches
        # AI Hub's own --quantize_full_type w8a16 flag and should have ~256x
        # less quantization noise - but hit a reproducible onnxruntime.
        # quantization bug in this environment (onnxruntime 1.27.0): output
        # collapses to a literal constant 0 across 4 variants (QUInt16 and
        # QInt16 activation types, with and without nodes_to_exclude,
        # per_channel True and False - see test_w8a16_*.py in the session
        # scratchpad). Per this project's 3-strikes debugging rule, stopped
        # pursuing w8a16 rather than keep varying parameters blindly. Left
        # at int8 here as the best of the two: it at least produces a real
        # (if inaccurate) numeric result, not a hard zero.
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=nodes_to_exclude,
    )
    before = src_path.stat().st_size / 1e6
    after = dst_path.stat().st_size / 1e6
    print(f"{model_name}: {before:.1f} MB -> {after:.1f} MB")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: quantize_ptq.py <calib_dir containing model_data_N.npz>")
        sys.exit(1)
    calib_dir = sys.argv[1]

    DST_BUNDLE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlvc_opset19_") as tmp:
        tmp_dir = Path(tmp)
        for model_name in MODEL_NAMES:
            quantize_one(model_name, calib_dir, tmp_dir)

    for name in SIDECAR_FILES:
        shutil.copy2(SRC_BUNDLE / name, DST_BUNDLE / name)

    print(f"\nDone. Quantized bundle at {DST_BUNDLE}")


if __name__ == "__main__":
    main()
