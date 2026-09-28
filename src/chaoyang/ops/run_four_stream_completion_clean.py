"""Registered C01 replay: frozen propagation, one output-domain repair, CPU only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from chaoyang.ops.run_human_to_robot_shared_delivery_clean import (
    _apply_lama_residual, _lama_shape_matches,
)
from chaoyang.pipeline.v5_scene import composite_clean

TASK = "four_stream_completion_20260928_scene"
PREFIX = "_run/current/four_stream_completion_20260928/attempts"
OLD = "_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001"
BASE = "_run/current/human_to_robot_result_breakthrough_20260924/attempts/attempt_0001/lanes/clean/POKER_076_091_INPUT_REPAIR"
V3 = OLD + "/lanes/clean/POKER_076_091_LAMA_RESIDUAL_V3"
WEIGHT = "assets/models/lama-onnx/lama_fp32.onnx"
WEIGHT_SHA = "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def ref(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("UNSTABLE_REFERENCE")
    return {"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest()}

def verify(item):
    if ref(item["path"]) != item:
        raise RuntimeError("REFERENCE_DRIFT:" + item["path"])
    return Path(item["path"])

def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

def safe_context(repo, attempt):
    repo = Path(repo).resolve(strict=True)
    if "/" in attempt or attempt in ("", ".", ".."):
        raise ValueError("INVALID_ATTEMPT")
    packet_path = repo / "tasks/current" / TASK / "TASK_PACKET.json"
    packet = read(packet_path)
    rows = read(repo / "tasks/current/INDEX.json").get("task_packets", [])
    parent_rows = [row for row in rows if row.get("task_id") == "four_stream_completion_20260928" and row.get("execution_allowed") is True]
    child_rows = [row for row in rows if row.get("task_id") == TASK]
    if len(parent_rows) != 1 or len(child_rows) != 1:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    for row in (parent_rows[0], child_rows[0]):
        expected = repo / "tasks/current" / row["task_id"] / "TASK_PACKET.json"
        if (repo / row["packet_path"]).resolve() != expected.resolve() or ref(expected)["sha256"] != row["packet_sha256"]:
            raise RuntimeError("TASK_PACKET_SHA_DRIFT")
    if packet.get("task_id") != TASK:
        raise RuntimeError("TASK_ID_DRIFT")
    parent = read(repo / parent_rows[0]["packet_path"])
    if parent["writer"] != packet["writer"]:
        raise RuntimeError("PARENT_WRITER_DRIFT")
    writer = packet["writer"]
    stat = Path(f"/proc/{int(writer['pid'])}/stat").read_text()
    ticks = int(stat.rsplit(")", 1)[1].split()[19])
    if ticks != int(writer["proc_start_ticks"]):
        raise RuntimeError("WRITER_IDENTITY_CHANGED")
    lane = repo / PREFIX / attempt / "lanes/scene"
    if lane.resolve() != Path(packet["output_root"]).resolve() or not lane.resolve().is_relative_to(repo):
        raise ValueError("OUTPUT_NOT_IN_TASK_WRITE_SET")
    if str(lane.relative_to(repo)) not in packet.get("write_set", []):
        raise ValueError("OUTPUT_NOT_IN_WRITE_SET")
    for part in [lane, *lane.parents]:
        if part == repo:
            break
        if part.is_symlink():
            raise ValueError("OUTPUT_SYMLINK")
    return repo, lane

def image(path, gray=False):
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if value is None:
        raise RuntimeError("IMAGE_UNREADABLE:" + str(path))
    return value

def save(path, array):
    if path.exists():
        raise FileExistsError(path)
    if not cv2.imwrite(str(path), array):
        raise RuntimeError("IMAGE_WRITE_FAILED")

def freeze(repo, lane):
    lane.mkdir(parents=True, exist_ok=True)
    target = lane / "C01_FROZEN_INPUT.json"
    if target.exists():
        raise FileExistsError(target)
    base = read(repo / BASE / "INPUT.json")
    trace = read(repo / V3 / "propagation_trace/TRACE_MANIFEST.json")
    if base.get("session_id") != "play_cards_0902_042" or base.get("frame_range") != [76, 91]:
        raise RuntimeError("SESSION_OR_WINDOW_DRIFT")
    if len(base["rows"]) != 16 or len(trace["propagation_rows"]) != 16:
        raise RuntimeError("FRAME_COUNT_DRIFT")
    rows = []
    for i, (source, prop) in enumerate(zip(base["rows"], trace["propagation_rows"], strict=True)):
        if source["frame_id"] != 76+i or source["local_index"] != i or prop["source_frame"] != 76+i or prop["local_index"] != i:
            raise RuntimeError("FRAME_MAPPING_DRIFT")
        row = {"frame_id": 76+i}
        for key in ("raw", "write", "protect", "frames", "model_masks"):
            row[key] = ref(verify(source[key]))
        for key in ("propagated_frame", "residual_propagation_mask"):
            path = (repo / V3 / "propagation_trace" / prop[key]).resolve()
            if not path.is_relative_to((repo / V3 / "propagation_trace").resolve()):
                raise RuntimeError("TRACE_ESCAPE")
            row[key] = ref(path)
        row["previous_clean"] = ref(repo / V3 / "clean" / f"{76+i:06d}.png")
        rows.append(row)
    weight = ref(repo / WEIGHT)
    if weight["sha256"] != WEIGHT_SHA:
        raise RuntimeError("MODEL_SHA_DRIFT")
    payload = {
        "schema_version": "COMPLETION_C01_FROZEN_V1", "task_id": TASK,
        "session_id": base["session_id"], "frame_range": [76, 91],
        "weight": weight, "runtime": ref(repo / OLD / "environments/onnxruntime-1.22.1/INSTALL_RECEIPT.json"),
        "source_manifest": ref(repo / BASE / "INPUT.json"),
        "propagation_manifest": ref(repo / V3 / "propagation_trace/TRACE_MANIFEST.json"),
        "predecessor": ref(repo / V3 / "RESULT.json"),
        "code": [ref(Path(__file__)), ref(Path(sys.modules[_apply_lama_residual.__module__].__file__))],
        "rows": rows,
        "change": "REMOVE_OUTPUT_MULTIPLY_255_ONLY; REUSE_IDENTICAL_PROPAGATION",
        "input_resize": [512, 512], "output_resize": [960, 720],
        "full_session_allowed": False,
    }
    write_json(target, payload)
    return {"frozen": str(target), "frames": 16}

class AuditedSession:
    def __init__(self, session):
        self.session, self.rows = session, []
    def run(self, outputs, inputs):
        prediction = self.session.run(outputs, inputs)
        arr = prediction[0]
        self.rows.append({"input_range": [float(inputs["image"].min()), float(inputs["image"].max())],
                          "output_range": [float(arr.min()), float(arr.max())],
                          "output_p50": float(np.median(arr)), "output_dtype": str(arr.dtype),
                          "output_shape": list(arr.shape)})
        return prediction

def apply_with_probe(audited, output_name, prop, mask):
    before = len(audited.rows)
    generated = _apply_lama_residual(audited, output_name, prop, mask)
    called = len(audited.rows) > before
    return generated, {"model_called": called, "probe": audited.rows[-1] if called else None}

def run(repo, lane):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU_ONLY_REQUIRES_EMPTY_CUDA_VISIBLE_DEVICES")
    cv2.setNumThreads(2)
    frozen_path = lane / "C01_FROZEN_INPUT.json"
    data = read(frozen_path)
    for key in ("weight", "runtime", "source_manifest", "propagation_manifest", "predecessor"):
        verify(data[key])
    for item in data["code"]:
        verify(item)
    runtime = repo / OLD / "environments/onnxruntime-1.22.1"
    sys.path.insert(0, str(runtime))
    import onnxruntime as ort
    if ort.__version__ != "1.22.1":
        raise RuntimeError("ORT_VERSION_DRIFT")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    core = ort.InferenceSession(str(repo / WEIGHT), sess_options=options, providers=["CPUExecutionProvider"])
    ins = {x.name: x for x in core.get_inputs()}
    outs = core.get_outputs()
    if set(ins) != {"image", "mask"} or len(outs) != 1:
        raise RuntimeError("ONNX_NAMES")
    for node, channels in ((ins["image"], 3), (ins["mask"], 1), (outs[0], 3)):
        if node.type != "tensor(float)" or not _lama_shape_matches(node.shape, channels):
            raise RuntimeError("ONNX_CONTRACT")
    destination = lane / "C01_ADAPTER_FIX1"
    destination.mkdir()
    for folder in ("clean", "internal", "panels"):
        (destination / folder).mkdir()
    audited = AuditedSession(core)
    started = time.monotonic()
    results = []
    for row in data["rows"]:
        for key, value in row.items():
            if isinstance(value, dict):
                verify(value)
        fid = row["frame_id"]
        raw = image(row["raw"]["path"])
        write = image(row["write"]["path"], True) > 0
        protect = image(row["protect"]["path"], True) > 0
        prop = image(row["propagated_frame"]["path"])
        mask = image(row["residual_propagation_mask"]["path"], True)
        generated, frame_probe = apply_with_probe(audited, outs[0].name, prop, mask)
        cleaned = composite_clean(raw, cv2.resize(generated, (1280, 960)), write, protect)
        changed = np.any(cleaned != raw, axis=2)
        if np.any(changed & ~write) or np.any(changed & protect):
            raise RuntimeError(f"WRITE_BOUNDARY:{fid}")
        before = image(row["previous_clean"]["path"])
        save(destination / "clean" / f"{fid:06d}.png", cleaned)
        save(destination / "internal" / f"{fid:06d}.png", generated)
        panel = np.concatenate([cv2.resize(x, (640, 480)) for x in (raw, before, cleaned)], axis=1)
        for x, label in ((0, "RAW"), (640, "PREVIOUS INVALID *255"), (1280, "ADAPTER FIX CANDIDATE")):
            cv2.putText(panel, f"{fid} {label}", (x+10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
        save(destination / "panels" / f"{fid:06d}.png", panel)
        results.append({"frame_id": fid, **frame_probe,
                        "clean": ref(destination / "clean" / f"{fid:06d}.png"),
                        "outside_write_changed": 0, "protected_changed": 0,
                        "old_white_write_px": int((np.all(before >= 250, axis=2) & write).sum()),
                        "new_white_write_px": int((np.all(cleaned >= 250, axis=2) & write).sum()),
                        "changed_px": int(changed.sum())})
    result = {"schema_version": "COMPLETION_C01_RESULT_V1", "task_id": TASK,
              "execution": "REAL_CPU_LAMA_16_FRAME_REPLAY", "structure": "PASS",
              "quality": "PENDING_FIXED_WINDOW_REVIEW", "adoption": "NOT_ADOPTED",
              "human_review": "NOT_PERFORMED", "full_171": "NOT_RUN",
              "frozen": ref(frozen_path), "rows": results, "elapsed_seconds": time.monotonic()-started,
              "providers": core.get_providers(), "gpu_used": False,
              "propainter_executed_this_attempt": False,
              "claim_limit": "Only numerical adapter repaired; semantic quality is not certified."}
    write_json(destination / "RESULT.json", result)
    return {"result": str(destination / "RESULT.json"), "frames": len(results)}

def review(repo, lane):
    source = lane / "C01_ADAPTER_FIX1/RESULT.json"
    value = read(source)
    for row in value["rows"]:
        verify(row["clean"])
    target = lane / "C01_ADAPTER_FIX1/REVIEW_MANIFEST.json"
    write_json(target, {"result": ref(source), "frames": [r["frame_id"] for r in value["rows"]],
                        "fixed_frames": [80, 84, 91], "human_review": "NOT_PERFORMED",
                        "quality": "PENDING_FIXED_WINDOW_REVIEW",
                        "required": ["human/device residue", "protected objects", "temporal flicker"],
                        "panels": [ref(lane / "C01_ADAPTER_FIX1/panels" / f"{r['frame_id']:06d}.png") for r in value["rows"]]})
    return {"review_manifest": str(target)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--attempt", default="attempt_0001")
    args = parser.parse_args()
    repo, lane = safe_context(args.repo_root, args.attempt)
    print(json.dumps({"freeze": freeze, "run": run, "review": review}[args.stage](repo, lane), ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
