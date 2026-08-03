# Capture real-frame calibration data for LOCAL ONNX quantization, at the
# 1280x720 resolution actually deployed (onnx-qualcomm/1280x720). This is
# the 720p sibling of the calibration capture used in the earlier AI-Hub-PTQ
# attempt (PLAN.md's "W8A16 PTQ ATTEMPT" section, which captured at
# 1920x1088 to match a different bundle).
#
# Reuses the existing, already-validated FrameLoop machinery (recurrent
# state handling: ref_frame/ref_feature/ref_exists, I-frame schedule every
# 96 frames, feature reset every 32) via the CoreML split model (fast local
# execution on this Mac, NPU/ANE backed). save_debug_data=True dumps exact
# ONNX-input-named tensors per frame, so captured field names line up 1:1
# with the onnx-qualcomm/1280x720 graphs' input names.
#
# Source footage: demo_asset/capture_motion.yuv - real camera capture
# (not synthetic), confirmed 1280x720 4:2:0, 300 frames
# (414720000 bytes = 1280*720*1.5*300 exactly). demo_asset/capture_natural.yuv
# and demo_asset/capture_from_live2.yuv are the same resolution and make
# good independent held-out clips for validate_quantized.py.
#
# Usage (from mlvc/video, using the project venv):
#   PYTHONPATH=/Users/sajil/genzee/mlvc/video \
#       /Users/sajil/genzee/mlvc/.venv/bin/python3 tools/capture_calib_data_720p.py <output_dir> [frame_count]
import sys
from pathlib import Path

from conversion import load_split_model
from conversion._frame_loop import FrameLoop
from conversion.types import RuntimeParams

MODEL_DIR = "output/models/mlvc_s-mlvc-s-psnr-v1/coreml-apple/1280x720"
CLIP = "/Users/sajil/genzee/mlvc/demo_asset/capture_motion.yuv"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: capture_calib_data_720p.py <output_dir> [frame_count=100]")
        sys.exit(1)
    out_dir = sys.argv[1]
    frame_count = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    params = RuntimeParams.from_dict({"coreml_compute_units": "npu", "coreml_fast_prediction": True})
    split_model = load_split_model(MODEL_DIR, params)

    loop = FrameLoop(
        split_model=split_model,
        video_path=CLIP,
        image_width=1280,
        image_height=720,
        frame_count=frame_count,
        output_data_dir=out_dir,
    )
    results = loop.run(q_index=63, progress_bar=True, save_debug_data=True)

    out_path = Path(out_dir) / "q_index=63"
    files = sorted(out_path.glob("model_data_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
    total_bytes = sum(f.stat().st_size for f in files)
    print(f"\n{len(files)} files, {total_bytes / 1e9:.2f} GB total")

    psnrs = sorted(f.metrics.psnr for f in results.frames if f.metrics is not None)
    if psnrs:
        print(f"PSNR median~ {psnrs[len(psnrs) // 2]:.2f} dB (n={len(psnrs)})")
    print("done, debug data in", out_path)


if __name__ == "__main__":
    main()
