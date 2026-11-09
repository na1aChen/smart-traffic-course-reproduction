from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report true aria2 progress from the TT100K control file."
    )
    parser.add_argument("--archive", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    control_path = args.archive.with_name(args.archive.name + ".aria2")
    if not control_path.exists():
        raise FileNotFoundError(f"aria2 control file does not exist: {control_path}")

    data = control_path.read_bytes()
    if data[:2] != b"\x00\x01":
        raise RuntimeError("Only aria2 control-file version 0001 is supported")

    info_hash_length = struct.unpack_from(">I", data, 6)[0]
    cursor = 10
    if info_hash_length:
        cursor += info_hash_length
    piece_length = struct.unpack_from(">I", data, cursor)[0]
    total_length = struct.unpack_from(">Q", data, cursor + 4)[0]
    bitfield_length = struct.unpack_from(">I", data, cursor + 20)[0]
    bitfield = data[cursor + 24 : cursor + 24 + bitfield_length]
    if len(bitfield) != bitfield_length:
        raise RuntimeError("Truncated aria2 bitfield")

    total_pieces = math.ceil(total_length / piece_length)
    completed_pieces = sum(byte.bit_count() for byte in bitfield)
    valid_bits_in_last_byte = total_pieces % 8
    if valid_bits_in_last_byte:
        padding_bits = 8 - valid_bits_in_last_byte
        completed_pieces -= (bitfield[-1] & ((1 << padding_bits) - 1)).bit_count()
    completed_pieces = min(completed_pieces, total_pieces)

    completed_bytes = completed_pieces * piece_length
    if completed_pieces == total_pieces:
        completed_bytes = total_length
    remaining_bytes = max(total_length - completed_bytes, 0)
    report = {
        "archive": str(args.archive.resolve()),
        "control_file": str(control_path.resolve()),
        "total_bytes": total_length,
        "piece_bytes": piece_length,
        "total_pieces": total_pieces,
        "completed_pieces": completed_pieces,
        "completed_gib": round(completed_bytes / 1024**3, 3),
        "remaining_gib": round(remaining_bytes / 1024**3, 3),
        "progress_percent": round(100 * completed_pieces / total_pieces, 3),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
