from pathlib import Path

import pytest

from chaoyang.inpainting_candidate.publication import HELPER, WRAPPERS, runtime_publication
from chaoyang.governance.common import artifact_ref, atomic_json


def fixture(root):
    refs = []
    for name in sorted(WRAPPERS):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen wrapper", encoding="utf-8")
        refs.append(artifact_ref(path))
    helper = root / HELPER
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text("frozen helper", encoding="utf-8")
    manifest = {"baseline_id": "INPAINTING_RAW_TO_ROBOT_V1", "runtime_closure": refs,
                "dependencies": refs[:2], "algorithm_core": []}
    manifest_path = root / "inpainting_v1/MANIFEST.json"
    manifest_path.parent.mkdir(parents=True)
    atomic_json(manifest_path, manifest)
    selection = next(ref for ref in refs if ref["path"].endswith("inpainting_v1_selection.py"))
    parent = {"baseline_pointer": artifact_ref(manifest_path), "original_wrapper": selection, "current_wrapper": selection}
    parent_path = root / "inpainting_v1/GOVERNANCE_PUBLICATION_R02.json"
    atomic_json(parent_path, parent)
    packet = {"task_id": "inpainting_generalization_g0_publication_test", "deletion_allowed": False,
              "algorithm_execution_allowed": False, "write_set": [*WRAPPERS, "inpainting_v1/GOVERNANCE_PUBLICATION_R03.json"]}
    task = root / "tasks/TASK_PACKET.json"
    task.parent.mkdir(parents=True)
    atomic_json(task, packet)
    value = {"schema": "inpainting-v1-runtime-publication-r03", "baseline_id": manifest["baseline_id"],
             "baseline_pointer": artifact_ref(manifest_path), "parent_supplement": artifact_ref(parent_path),
             "algorithm_changed": False, "files_deleted": 0, "model_calls": 0, "training_steps": 0, "external_writes": 0,
             "wrapper_overrides": [{"relative_path": str(Path(ref["path"]).relative_to(root)), "original": ref,
                                     "parent": ref, "current": ref, "role": "GOVERNANCE_RUNTIME_ONLY"} for ref in refs],
             "additional_runtime": [artifact_ref(helper)], "task_packet": artifact_ref(task)}
    path = root / "inpainting_v1/GOVERNANCE_PUBLICATION_R03.json"
    atomic_json(path, value)
    return manifest, value, path


def test_only_signed_wrappers_and_helper_are_rebound(tmp_path):
    manifest, _, _ = fixture(tmp_path)
    runtime, dependencies, supplement = runtime_publication(tmp_path, manifest)
    assert len(runtime) == 4 and len(dependencies) == 2
    assert supplement["path"].endswith("R03.json")


@pytest.mark.parametrize("mutation", ["extra_override", "changed_wrapper", "algorithm", "training", "helper", "task_scope"])
def test_publication_drift_rejected(tmp_path, mutation):
    manifest, value, path = fixture(tmp_path)
    if mutation == "extra_override":
        value["wrapper_overrides"].append({"relative_path": "assets/model.pt"})
    elif mutation == "changed_wrapper":
        Path(value["wrapper_overrides"][0]["current"]["path"]).write_text("drift")
    elif mutation == "algorithm":
        value["algorithm_changed"] = True
    elif mutation == "training":
        value["training_steps"] = 1
    elif mutation == "helper":
        value["additional_runtime"] = []
    elif mutation == "task_scope":
        task_path = Path(value["task_packet"]["path"])
        atomic_json(task_path, {"task_id": "inpainting_generalization_g0_publication_test",
                               "deletion_allowed": False, "algorithm_execution_allowed": False, "write_set": []})
        value["task_packet"] = artifact_ref(task_path)
    atomic_json(path, value)
    with pytest.raises(RuntimeError):
        runtime_publication(tmp_path, manifest)
