# Converts the exported PMF JSON tables into one flat binary blob the C++
# encoder mmaps. Layout (all little-endian):
#   magic 'MPMF' u32, version u32
#   gaussian:  numRows u32, tableLen u32, scaleLevels u32,
#              lengths i32[numRows], offsets i32[numRows], table i32[tableLen]
#   bitest:    qpNum u32, channels u32, numRows u32, tableLen u32,
#              lengths i32[numRows], offsets i32[numRows], table i32[tableLen]
import json
import struct
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    src_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    g = json.loads((src_dir / "gaussian_pmf.json").read_text())
    b = json.loads((src_dir / "bit_estimator_pmf.json").read_text())

    def arrays(d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lengths = np.array(d["pmf_lengths"], dtype=np.int32)
        offsets = np.array(d["pmf_offsets"], dtype=np.int32)
        table = np.array(d["pmf_table"], dtype=np.int32)
        assert lengths.sum() == len(table), "table not flat-concatenated by lengths"
        return lengths, offsets, table

    gl, go, gt = arrays(g)
    bl, bo, bt = arrays(b)

    with out_path.open("wb") as f:
        f.write(struct.pack("<II", 0x4D504D46, 1))
        f.write(struct.pack("<III", len(gl), len(gt), g["scale_levels"]))
        f.write(gl.tobytes())
        f.write(go.tobytes())
        f.write(gt.tobytes())
        f.write(struct.pack("<IIII", b["qp_num"], b["channels"], len(bl), len(bt)))
        f.write(bl.tobytes())
        f.write(bo.tobytes())
        f.write(bt.tobytes())

    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
