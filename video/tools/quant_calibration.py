# Shared calibration-data plumbing for local ONNX PTQ of the MLVC-S codec.
#
# Reuses real per-frame ONNX-level tensors captured via
# conversion.FrameLoop(save_debug_data=True) - see capture_calib_data_720p.py
# (scratchpad driver used to produce the 1280x720 calibration set). Each
# captured model_data_N.npz holds one dict entry per input/output of every
# model part, named f"{ModelPartId.value}_input_{name}" /
# f"{ModelPartId.value}_output_{name}" (see conversion/_frame_loop.py's
# _save_model_data). This module extracts just the inputs for a single
# requested model part (e.g. "MLVCEncoderPart1") and feeds them to
# onnxruntime.quantization by way of a CalibrationDataReader.
#
# q_index_shifted (the scalar int32 conditioning input that broke AI Hub's
# own quantizer - see PLAN.md's W8A16 PTQ ATTEMPT section) needs no special
# exclusion here: it is consumed only as a Gather *index* in every model
# part (indexing per-q-index PMF/scale tables), never as a float activation,
# so onnxruntime's quantizer naturally leaves it untouched - it is already
# declared int32 in the ONNX graph and stays that way through calibration.
import glob
import os
import re
from typing import Iterator, Optional

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader

_ONNX_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
}

_FRAME_INDEX_RE = re.compile(r"model_data_(\d+)\.npz$")


def sorted_npz_paths(calib_dir: str) -> list[str]:
    """List calibration npz files in frame order (not glob/lexicographic
    order, which would sort model_data_10 before model_data_2)."""
    paths = glob.glob(os.path.join(calib_dir, "model_data_*.npz"))
    return sorted(paths, key=lambda p: int(_FRAME_INDEX_RE.search(p).group(1)))


def onnx_input_dtypes(onnx_path: str) -> dict[str, np.dtype]:
    """Query the actual declared numpy dtype of every graph input - the
    calibration feed must match this exactly (the deployed graphs declare
    fp16 I/O directly, not fp32, which is why AI Hub's HTP backend rejected
    them on non-fp16-capable devices in the first place)."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return {i.name: _ONNX_TO_NUMPY[i.type] for i in session.get_inputs()}


def load_calibration_feeds(
    calib_dir: str,
    model_part: str,
    onnx_path: str,
    max_samples: Optional[int] = None,
) -> list[dict[str, np.ndarray]]:
    """Load every captured frame's inputs for one model part, cast to the
    dtypes the given ONNX graph actually declares for each input."""
    input_dtypes = onnx_input_dtypes(onnx_path)
    input_names = set(input_dtypes)
    prefix = f"{model_part}_input_"

    paths = sorted_npz_paths(calib_dir)
    if max_samples is not None:
        paths = paths[:max_samples]

    feeds = []
    for path in paths:
        data = np.load(path)
        feed = {}
        for key in data.files:
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if name not in input_names:
                continue
            feed[name] = np.ascontiguousarray(data[key].astype(input_dtypes[name], copy=False))
        missing = input_names - feed.keys()
        if missing:
            raise ValueError(f"{path}: missing inputs for {model_part}: {sorted(missing)}")
        feeds.append(feed)
    return feeds


class ListCalibrationDataReader(CalibrationDataReader):
    """Replays a pre-built list of feed dicts, one per get_next() call."""

    def __init__(self, feeds: list[dict[str, np.ndarray]]):
        self._feeds = feeds
        self._iter: Iterator[dict[str, np.ndarray]] = iter(feeds)

    def get_next(self) -> Optional[dict[str, np.ndarray]]:
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = iter(self._feeds)

    def __len__(self) -> int:
        return len(self._feeds)
