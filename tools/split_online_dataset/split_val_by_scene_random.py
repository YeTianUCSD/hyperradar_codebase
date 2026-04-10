import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


TARGET_CLASSES = ("Car", "Pedestrian", "Cyclist")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split val by contiguous scene chunks, with random prefix/suffix direction per chunk."
    )
    parser.add_argument(
        "--val_txt",
        type=str,
        default="/home/code/hyperradar/dataset/view_of_delft_PUBLIC/radar_5frames/ImageSets/val.txt",
        help="Path to val split txt",
    )
    parser.add_argument(
        "--val_info",
        type=str,
        default="/home/code/hyperradar/dataset/view_of_delft_PUBLIC/radar_5frames/kitti_infos_val.pkl",
        help="Path to kitti_infos_val.pkl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/code/hyperradar/dataset/view_of_delft_PUBLIC/radar_5frames",
        help="Dataset root directory where output txt/pkl/report will be written",
    )
    parser.add_argument(
        "--stream_name",
        type=str,
        default="val_stream_by_scene_random",
        help="Name of stream split to create",
    )
    parser.add_argument(
        "--eval_name",
        type=str,
        default="val_eval_by_scene_random",
        help="Name of eval split to create",
    )
    parser.add_argument(
        "--stream_ratio",
        type=float,
        default=0.5,
        help="Within each contiguous scene chunk, split by this ratio before random direction assignment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed controlling per-chunk direction assignment",
    )
    return parser.parse_args()


def _load_lines(path: Path) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _load_infos(path: Path):
    with open(path, "rb") as f:
        infos = pickle.load(f)
    if not isinstance(infos, list):
        raise RuntimeError(f"Expected list in info pkl, got {type(infos)}")
    return infos


def _sample_id_from_info(info: dict) -> str:
    point_cloud = info.get("point_cloud", {})
    sample_id = point_cloud.get("lidar_idx", None)
    if sample_id is None:
        raise RuntimeError("Info entry missing point_cloud.lidar_idx")
    return str(sample_id)


def _class_count_from_info(info: dict) -> Counter:
    annos = info.get("annos", None)
    counts = Counter({k: 0 for k in TARGET_CLASSES})
    if not annos or "name" not in annos:
        return counts
    for name in annos["name"]:
        if name in TARGET_CLASSES:
            counts[str(name)] += 1
    return counts


def _prepare_samples(val_ids: List[str], infos: List[dict]) -> List[dict]:
    if len(val_ids) != len(set(val_ids)):
        dup = [sid for sid, c in Counter(val_ids).items() if c > 1]
        raise RuntimeError(f"val.txt contains duplicated sample ids, e.g. {dup[:5]}")

    info_ids = [_sample_id_from_info(info) for info in infos]
    if len(info_ids) != len(set(info_ids)):
        dup = [sid for sid, c in Counter(info_ids).items() if c > 1]
        raise RuntimeError(f"info pkl contains duplicated sample ids, e.g. {dup[:5]}")

    info_by_id = {sid: info for sid, info in zip(info_ids, infos)}
    order_by_id = {sid: idx for idx, sid in enumerate(val_ids)}
    missing = [sid for sid in val_ids if sid not in info_by_id]
    extra = [sid for sid in info_by_id.keys() if sid not in set(val_ids)]
    if missing:
        raise RuntimeError(f"{len(missing)} sample ids from val.txt missing in info pkl, e.g. {missing[:5]}")

    samples = []
    for sid in val_ids:
        info = info_by_id[sid]
        counts = _class_count_from_info(info)
        samples.append(
            {
                "sample_id": sid,
                "sample_num": int(sid),
                "info": info,
                "counts": counts,
                "order": int(order_by_id[sid]),
            }
        )
    if extra:
        print(f"[WARN] info pkl contains {len(extra)} extra samples not present in val.txt; they will be ignored")
    return samples


def _vec_add(a: Counter, b: Counter) -> Counter:
    out = Counter(a)
    for k, v in b.items():
        out[k] += int(v)
    return out


def _write_txt(path: Path, sample_ids: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sid in sample_ids:
            f.write(f"{sid}\n")


def _write_pkl(path: Path, infos: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(infos, f)


def _summary(samples: List[dict]) -> Dict:
    counts = Counter({k: 0 for k in TARGET_CLASSES})
    for sample in samples:
        counts = _vec_add(counts, sample["counts"])
    return {
        "num_samples": len(samples),
        "class_object_counts": {k: int(counts[k]) for k in TARGET_CLASSES},
        "avg_objects_per_sample": {
            k: (float(counts[k]) / max(1, len(samples))) for k in TARGET_CLASSES
        },
    }


def _compute_overlap(a: List[dict], b: List[dict]) -> int:
    set_a = {x["sample_id"] for x in a}
    set_b = {x["sample_id"] for x in b}
    return len(set_a & set_b)


def _find_contiguous_chunks(samples: List[dict]) -> List[List[dict]]:
    if not samples:
        return []
    chunks: List[List[dict]] = []
    cur = [samples[0]]
    for sample in samples[1:]:
        if sample["sample_num"] == cur[-1]["sample_num"] + 1:
            cur.append(sample)
        else:
            chunks.append(cur)
            cur = [sample]
    chunks.append(cur)
    return chunks


def _split_chunk(chunk: List[dict], stream_ratio: float) -> Tuple[List[dict], List[dict]]:
    n = len(chunk)
    if n <= 1:
        raise RuntimeError(
            f"Contiguous chunk {chunk[0]['sample_id']}..{chunk[-1]['sample_id']} has length {n}; "
            "cannot split a scene chunk of length <= 1."
        )
    stream_n = int(round(n * float(stream_ratio)))
    stream_n = max(1, min(n - 1, stream_n))
    return chunk[:stream_n], chunk[stream_n:]


def _chunk_report_entry(
    idx: int,
    chunk: List[dict],
    stream_chunk: List[dict],
    eval_chunk: List[dict],
    direction: str,
) -> Dict:
    return {
        "chunk_id": int(idx),
        "full_range": [chunk[0]["sample_id"], chunk[-1]["sample_id"]],
        "num_samples": int(len(chunk)),
        "direction": direction,
        "stream_range": [stream_chunk[0]["sample_id"], stream_chunk[-1]["sample_id"]],
        "stream_samples": int(len(stream_chunk)),
        "eval_range": [eval_chunk[0]["sample_id"], eval_chunk[-1]["sample_id"]],
        "eval_samples": int(len(eval_chunk)),
    }


def main():
    args = parse_args()

    val_txt = Path(args.val_txt)
    val_info = Path(args.val_info)
    output_dir = Path(args.output_dir)
    imagesets_dir = output_dir / "ImageSets"

    if not val_txt.is_file():
        raise FileNotFoundError(f"val_txt not found: {val_txt}")
    if not val_info.is_file():
        raise FileNotFoundError(f"val_info not found: {val_info}")

    stream_ratio = float(args.stream_ratio)
    if not (0.0 < stream_ratio < 1.0):
        raise RuntimeError(f"stream_ratio must be in (0, 1), got {stream_ratio}")

    val_ids = _load_lines(val_txt)
    infos = _load_infos(val_info)
    samples = _prepare_samples(val_ids, infos)
    if len(samples) != len(val_ids):
        raise RuntimeError("Prepared sample count does not match val.txt length")

    chunks = _find_contiguous_chunks(samples)
    if not chunks:
        raise RuntimeError("No contiguous chunks found in val split")

    rng = random.Random(int(args.seed))
    stream_samples: List[dict] = []
    eval_samples: List[dict] = []
    chunk_reports: List[Dict] = []

    for idx, chunk in enumerate(chunks):
        prefix, suffix = _split_chunk(chunk, stream_ratio=stream_ratio)
        if rng.random() < 0.5:
            stream_chunk, eval_chunk = prefix, suffix
            direction = "prefix_to_stream"
        else:
            stream_chunk, eval_chunk = suffix, prefix
            direction = "suffix_to_stream"

        stream_samples.extend(stream_chunk)
        eval_samples.extend(eval_chunk)
        chunk_reports.append(_chunk_report_entry(idx, chunk, stream_chunk, eval_chunk, direction))

    stream_samples = sorted(stream_samples, key=lambda x: x["order"])
    eval_samples = sorted(eval_samples, key=lambda x: x["order"])

    overlap = _compute_overlap(stream_samples, eval_samples)
    if overlap != 0:
        raise RuntimeError(f"Split overlap must be 0, got {overlap}")

    stream_ids = [x["sample_id"] for x in stream_samples]
    eval_ids = [x["sample_id"] for x in eval_samples]
    stream_infos = [x["info"] for x in stream_samples]
    eval_infos = [x["info"] for x in eval_samples]

    stream_txt = imagesets_dir / f"{args.stream_name}.txt"
    eval_txt = imagesets_dir / f"{args.eval_name}.txt"
    stream_pkl = output_dir / f"kitti_infos_{args.stream_name}.pkl"
    eval_pkl = output_dir / f"kitti_infos_{args.eval_name}.pkl"
    report_json = output_dir / f"split_report_{args.stream_name}_{args.eval_name}.json"

    _write_txt(stream_txt, stream_ids)
    _write_txt(eval_txt, eval_ids)
    _write_pkl(stream_pkl, stream_infos)
    _write_pkl(eval_pkl, eval_infos)

    report = {
        "mode": "scene_random_direction_split",
        "seed": int(args.seed),
        "stream_ratio": float(stream_ratio),
        "source_val_txt": str(val_txt),
        "source_val_info": str(val_info),
        "stream_name": str(args.stream_name),
        "eval_name": str(args.eval_name),
        "stream_txt": str(stream_txt),
        "eval_txt": str(eval_txt),
        "stream_pkl": str(stream_pkl),
        "eval_pkl": str(eval_pkl),
        "num_scene_chunks": int(len(chunks)),
        "chunks": chunk_reports,
        "stream": _summary(stream_samples),
        "eval": _summary(eval_samples),
        "overlap_samples": int(overlap),
        "total_samples": int(len(samples)),
    }
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)

    print("[DONE] by-scene-random split created")
    print(f"[OUT] stream txt : {stream_txt}")
    print(f"[OUT] eval txt   : {eval_txt}")
    print(f"[OUT] stream pkl : {stream_pkl}")
    print(f"[OUT] eval pkl   : {eval_pkl}")
    print(f"[OUT] report     : {report_json}")
    print(f"[STAT] scene chunks={len(chunks)}")
    print(f"[STAT] stream samples={len(stream_samples)}, eval samples={len(eval_samples)}, overlap={overlap}")
    print(f"[STAT] stream class counts={report['stream']['class_object_counts']}")
    print(f"[STAT] eval class counts={report['eval']['class_object_counts']}")


if __name__ == "__main__":
    main()
