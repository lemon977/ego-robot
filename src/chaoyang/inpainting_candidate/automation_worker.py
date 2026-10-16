"""Thin six-stage V1 adapters, invoked only by the existing fenced MCO worker.

No model, geometry or original pipeline is imported before authorization. The
original V1 implementations remain at their frozen paths; only stage separation
and immutable receipts live here. Render buffers can survive a Mask/Clean edit.
"""
from __future__ import annotations

from pathlib import Path

from .automation import DEPENDENCIES, read_reference, require, verify_reference, write_once_json
from .contracts import SCHEMA, STAGES, file_ref


def _arrays(reference):
    import numpy as np
    verify_reference(reference)
    with np.load(reference["path"], allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _save_arrays(path, **arrays):
    import numpy as np
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return file_ref(path)


def _outputs(reference):
    receipt = read_reference(reference)
    root = Path(reference["path"]).resolve(strict=True).parent
    return {Path(r["path"]).name: r for r in receipt["outputs"] if Path(r["path"]).parent == root}


class V1Builders:
    def __init__(self, packet, authorization):
        self.packet, self.authority = packet, authorization
        from chaoyang.paths import get_paths
        self.repo = Path(packet.get("project_root", get_paths().repo_root)).resolve(strict=True)
        self.models = {}
        self.sam_budget = None
        self.clean_calls, self.render_calls = 0, 0

    def builders(self):
        return {stage: getattr(self, stage.lower()) for stage in STAGES}

    def _start(self, output_dir):
        self.authority()
        root = Path(output_dir).resolve()
        require(root != self.repo and root.is_relative_to(self.repo) and not root.exists(),
                "AUTOMATION_FRESH_PROJECT_STAGE_REQUIRED")
        root.mkdir(parents=True, exist_ok=False)
        return root

    def _seal(self, stage, row, root, signature, upstream, extra=None):
        self.authority()
        outputs = [file_ref(p) for p in sorted(root.rglob("*")) if p.is_file()]
        receipt = {"schema": SCHEMA, "stage": stage, "status": "SEALED_COMPLETE",
            "component_signature": signature, "session_id": row["session_id"], "native_frames": row["native_frames"],
            "upstream": {s: upstream[s] for s in DEPENDENCIES[stage]},
            "outputs": outputs, "mode": "v1-compatible", "visual_quality": "NOT_REVIEWED",
            "human_review": "PENDING", "training_eligible": False, "control_ground_truth": False,
            "candidate_adopted": False, **(extra or {})}
        write_once_json(root / "RECEIPT.json", receipt)
        return receipt

    def _session(self, upstream):
        from chaoyang.pick_place_1008.source import load_prepared
        from .automation_paths import legacy_source_evidence_paths
        with legacy_source_evidence_paths(self.repo):
            return load_prepared(Path(upstream["Source"]["path"]).parent / "Prepared")

    def _rgb(self, upstream):
        return _arrays(_outputs(upstream["Source"])["RAW_RGB.npz"])["rgb"]

    def _mapping(self, upstream):
        return read_reference(_outputs(upstream["Source"])["FRAME_MAP.json"])

    def _models(self, stage):
        if stage not in self.models:
            from .model_stages import load_frozen_models, _AuthorizedSam, _gate
            models = load_frozen_models(self.repo, self.packet["output_dir"], authorization=self.authority,
                frozen_inputs=[*self.packet["runtime_code"], *self.packet["model_weights"]],
                need_sam=stage == "Mask", need_clean=stage == "Clean")
            if stage == "Mask":
                from .model_worker import CountedSam, ModelCallBudget, MODEL_CALLS_UNIT
                maximum = self.packet["budgets"]["sam_api_calls_max"]
                self.sam_budget = ModelCallBudget(maximum, MODEL_CALLS_UNIT)
                models["sam"] = CountedSam(_AuthorizedSam(models["sam"], self.authority, _gate(self.authority)), self.sam_budget)
            self.models[stage] = models
        return self.models[stage]

    def source(self, *, row, output_dir, signature, upstream, request):
        import hashlib
        import numpy as np
        from chaoyang.pick_place_1008 import source
        from .automation_paths import legacy_source_evidence_paths
        root = self._start(output_dir)
        config = read_reference(request["parent_inputs"]["camera"])
        with legacy_source_evidence_paths(self.repo):
            session = source.prepare_session(Path(row["raw_session_dir"]), root / "Prepared", config, row["native_frames"])
        require(session.session_id == "pick_place_1008_" + row["session_id"]
            and session.frame_count == row["native_frames"], "AUTOMATION_PREPARED_SOURCE_IDENTITY_DRIFT")
        rgb = source.load_rgb_frames(session, 640, 480)
        require(rgb.dtype == np.uint8 and rgb.shape == (row["native_frames"], 480, 640, 3),
                "AUTOMATION_RAW_RGB_NATIVE_DOMAIN_DRIFT")
        mapping = [{"session_id": row["session_id"], "native_frame_index": i, "decoded_display_index": i,
            "model_input_frame": i, "output_frame": i, "qpc_declared_ns": int(t),
            "pixel_sha256": hashlib.sha256(rgb[i].tobytes()).hexdigest(),
            "physical_eye": session.calibration.get("physical_eye"), "sbs_index": session.calibration["sbs_index"],
            "acquisition_frame_id": "UNKNOWN", "frame_mapping_qualified": False,
            "timebase_status": "TIMEBASE_UNVERIFIED_HOST_ARRIVAL_QPC_DECLARED_NS"}
            for i, t in enumerate(session.qpc_ns)]
        _save_arrays(root / "RAW_RGB.npz", rgb=rgb)
        write_once_json(root / "FRAME_MAP.json", mapping)
        write_once_json(root / "SOURCE_REPORT.json", {"session_id": row["session_id"],
            "native_source_session_id": session.session_id, "native_sidecar_frames": len(rgb),
            "raw_session_dir": row["raw_session_dir"], "actual_inputs": row["raw_inputs"],
            "camera": session.calibration, "frame_mapping": mapping, "historical_pixels_consumed": False,
            "hdf5_datasets_read": False, "training_eligible": False, "control_ground_truth": False})
        return self._seal("Source", row, root, signature, upstream, {"actual_inputs": row["raw_inputs"],
            "camera_config": request["parent_inputs"]["camera"], "models_called": 0})

    def motion(self, *, row, output_dir, signature, upstream, request):
        import numpy as np
        from chaoyang.pick_place_1008.pipeline import motion_worker
        from .automation_paths import legacy_source_evidence_paths
        root = self._start(output_dir)
        with legacy_source_evidence_paths(self.repo):
            motion_worker(str(self.repo), str(root), str(Path(upstream["Source"]["path"]).parent / "Prepared"))
        motion = _arrays(file_ref(root / "actions.npz"))
        expected = (row["native_frames"],)
        require(motion["q_arm"].shape == (*expected, 2, 7) and motion["q_hand"].shape == (*expected, 2, 22)
            and np.array_equal(motion["qpc_ns"], [m["qpc_declared_ns"] for m in self._mapping(upstream)]),
            "AUTOMATION_CURRENT_RAW_ACTION_NATIVE_IDENTITY_DRIFT")
        return self._seal("Motion", row, root, signature, upstream, {"models_called": 0,
            "motion_source": "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS", "historical_actions_consumed": False})

    def mask(self, *, row, output_dir, signature, upstream, request):
        import numpy as np
        from chaoyang.pick_place_1008 import visual
        from chaoyang.pick_place_1008 import mask_seeds
        from .automation_paths import seed_evidence_alias_reader
        root = self._start(output_dir)
        session, rgb = self._session(upstream), self._rgb(upstream)
        with seed_evidence_alias_reader(mask_seeds, self.repo):
            visible = mask_seeds.load_visible_seed(request["parent_inputs"]["mask_seeds"]["path"], row["session_id"], session, rgb)
        require(visible["seed_reference"] == row["seed"], "AUTOMATION_ONLY_EXPLICIT_PARENT_SEED_ALLOWED")
        models = self._models("Mask")
        before = self.sam_budget.total
        masks = visual.build_visual_masks(models["sam"], rgb, session, visible_seeds=visible)
        require(masks["metadata"].get("point_tracking", {}).get("status") == "COMPLETE",
                "AUTOMATION_MASK_POINT_TRACKING_INCOMPLETE")
        original_arrays = {k: v for k, v in masks.items() if isinstance(v, np.ndarray) and v.dtype == np.bool_ and v.shape == rgb.shape[:3]}
        masks["effective_removal"], _ = visual.effective_clean_masks(masks["removal"], masks["protected_visible"])
        original_arrays["effective_removal"] = masks["effective_removal"]
        _save_arrays(root / "V1_BASE_MASK_ARRAYS.npz", **original_arrays)
        write_once_json(root / "V1_BASE_MASK_REPORT.json", {"metadata": masks["metadata"], "seed": row["seed"],
            "quality_failures_retained": masks["metadata"].get("quality_problems", [])})
        from .automation_mask import repair_current_masks
        from .contracts import canonical_sha
        sam_refs = [r for r in self.packet["component_weights"]["Mask"] if Path(r["path"]).name == "sam3.1_multiplex.pt"]
        require(len(sam_refs) == 1, "AUTOMATION_EXACT_SAM_WEIGHT_REQUIRED")
        mask_config = self.packet.get("component_config", {}).get("Mask", {})
        repair_rounds = mask_config.get("repair_rounds", 1)
        require(type(repair_rounds) is int and repair_rounds == request.get("repair_rounds", 1),
                "AUTOMATION_REPAIR_ROUNDS_CONFIG_DRIFT")
        snapshots = {}
        snapshot_options = {"round_snapshots": snapshots} if repair_rounds == 3 else {}
        masks, repair, proposals, hints, round_1_masks = repair_current_masks(sam=models["sam"], rgb=rgb, session=session,
            original_masks=masks, original_objects=visible["objects"], raw_source_sha256=row["raw_inputs"]["raw_vst"]["sha256"],
            frame_map_sha256=canonical_sha(self._mapping(upstream)), model_sha256=sam_refs[0]["sha256"],
            glove_text_prompt=mask_config.get("glove_text_prompt", "a gloved hand"),
            repair_rounds=repair_rounds, return_round_masks=True, **snapshot_options)
        _save_arrays(root / "IDENTITY_HINTS.npz", xy=hints)
        _save_arrays(root / "RGB_PROPOSALS.npz", **{f"track_{i}": np.packbits(p.masks.reshape(len(rgb), -1), axis=1)
            for i, p in enumerate(proposals)})
        round_1_arrays = {k: v for k, v in round_1_masks.items()
            if isinstance(v, np.ndarray) and v.dtype == np.bool_ and v.shape == rgb.shape[:3]}
        round_1_ref = _save_arrays(root / "ROUND_1_MASK_ARRAYS.npz", **round_1_arrays)
        snapshot_refs = {}
        if repair_rounds == 3:
            require(2 in snapshots, "AUTOMATION_ROUND_TWO_SNAPSHOT_REQUIRED")
            round_2_arrays = {k:v for k,v in snapshots[2].items()
                if isinstance(v,np.ndarray) and v.dtype == np.bool_ and v.shape == rgb.shape[:3]}
            require(set(round_2_arrays) == set(round_1_arrays), "AUTOMATION_ROUND_TWO_MASK_ROLES_DRIFT")
            snapshot_refs["round_2_masks"] = _save_arrays(root / "ROUND_2_MASK_ARRAYS.npz", **round_2_arrays)
        write_once_json(root / "RGB_PROPOSALS.json", {"input_bindings": repair["input_bindings"],
            "arrays": file_ref(root / "RGB_PROPOSALS.npz"), "proposals": [{"array_key": f"track_{i}",
            "role": p.role, "track_id": p.track_id, "native_shape": list(p.masks.shape), "source": p.source}
            for i, p in enumerate(proposals)]})
        arrays = {k: v for k, v in masks.items() if isinstance(v, np.ndarray) and v.dtype == np.bool_ and v.shape == rgb.shape[:3]}
        require({"core", "removal", "protected_visible", "unknown", "sam_core", "equipment"} <= arrays.keys(),
                "AUTOMATION_V1_MASK_ROLES_REQUIRED")
        require(set(original_arrays) == set(round_1_arrays) == set(arrays),
                "AUTOMATION_ROUND_MASK_ROLES_DRIFT")
        for previous, current in ((original_arrays, round_1_arrays), (round_1_arrays, arrays)):
            require(all(np.array_equal(previous[role], current[role]) for role in
                ("equipment", "protected_visible", "unknown", "cup_observed", "semantic_conflict") if role in previous),
                "AUTOMATION_ROUND_FIXED_MASK_SUPPORT_DRIFT")
            require(all(not np.any(previous[role] & ~current[role]) for role in ("core", "sam_core", "removal")),
                    "AUTOMATION_ROUND_PREVIOUS_SUPPORT_REMOVED")
            effective, _ = visual.effective_clean_masks(current["removal"], current["protected_visible"])
            require(np.array_equal(effective, current["effective_removal"]), "AUTOMATION_ROUND_EFFECTIVE_MASK_DRIFT")
        final_ref = _save_arrays(root / "MASK_ARRAYS.npz", **arrays)
        repair.update(round_1_masks=round_1_ref, final_masks=final_ref, **snapshot_refs)
        write_once_json(root / "MASK_REPAIR.json", repair)
        write_once_json(root / "MASK_REPORT.json", {"metadata": masks["metadata"], "seed": row["seed"],
            "sam_api_attempted_calls": self.sam_budget.total - before, "historical_masks_consumed": False,
            "repair_audit": file_ref(root / "MASK_REPAIR.json"), "manual_points_added": 0})
        return self._seal("Mask", row, root, signature, upstream, {"sam_budget": self.sam_budget.receipt(),
            "sam_api_calls": self.sam_budget.total - before, "repair_rounds": repair_rounds,
            "round_1_masks": round_1_ref, "final_masks": final_ref, **snapshot_refs})

    def clean(self, *, row, output_dir, signature, upstream, request):
        import numpy as np
        from chaoyang.pick_place_1008.visual import ProtectedProPainter
        root = self._start(output_dir)
        rgb = self._rgb(upstream)
        masks = _arrays(_outputs(upstream["Mask"])["MASK_ARRAYS.npz"])
        paired = self.packet.get("component_config", {}).get("Clean", {}).get("paired_clean") is True
        before = self.clean_calls
        if paired:
            from .v1_clean_pair import run_fresh_clean_pair, runtime_backend
            import torch
            self.authority()
            backend = runtime_backend(torch, torch.device("cuda"))
            def on_call(variant):
                self.authority()
                require(self.clean_calls < self.packet["budgets"]["propainter_calls_max"], "AUTOMATION_PROPAINTER_CALL_BUDGET_EXHAUSTED")
                self.clean_calls += 1
            pair = run_fresh_clean_pair(self._models("Clean")["propainter"], rgb, masks["removal"], masks["protected_visible"],
                authority=self.authority, on_call=on_call, backend_before_load=backend)
            clean, effective, validation = pair["A"], pair["effective_removal"], pair["validations"]
            _save_arrays(root / "CLEAN_A_RGB.npz", rgb=pair["A"])
            _save_arrays(root / "CLEAN_B_RGB.npz", rgb=pair["B"])
            _save_arrays(root / "CLEAN_PAIR_RNG.npz", **pair["rng_arrays"])
            write_once_json(root / "CLEAN_PAIR.json", {**pair["receipt"], "source_receipt": upstream["Source"],
                "mask_receipt": upstream["Mask"], "same_sealed_mask": True})
            from .source_frames import write_review_video
            write_review_video(root / "clean_A.mp4", pair["A"], self._mapping(upstream))
            write_review_video(root / "clean_B.mp4", pair["B"], self._mapping(upstream))
        else:
            require(self.clean_calls < self.packet["budgets"]["propainter_calls_max"], "AUTOMATION_PROPAINTER_CALL_BUDGET_EXHAUSTED")
            painter = ProtectedProPainter(self._models("Clean")["propainter"])
            self.authority()
            self.clean_calls += 1
            clean = painter.inpaint(rgb, masks["removal"], masks["protected_visible"])
            self.authority()
            effective, validation = painter.effective_masks, painter.last_validation
        require(clean.shape == rgb.shape and clean.dtype == np.uint8 and not np.any(clean[masks["protected_visible"]] != rgb[masks["protected_visible"]])
            and not np.any(clean[~effective] != rgb[~effective]), "AUTOMATION_CLEAN_PIXEL_PROTECTION_FAILED")
        _save_arrays(root / "CLEAN_RGB.npz", rgb=clean, effective_removal=effective)
        write_once_json(root / "CLEAN_REPORT.json", {"validation": validation,
            "settings": self.models["Clean"]["propainter"].settings, "propainter_calls": self.clean_calls - before,
            "paired_clean": paired, "primary_clean_variant": "A", "historical_clean_consumed": False})
        return self._seal("Clean", row, root, signature, upstream, {"propainter_calls": self.clean_calls - before,
            "paired_clean": paired})

    def render(self, *, row, output_dir, signature, upstream, request):
        import numpy as np
        from chaoyang.exact78_baseline import render as rd
        from chaoyang.exact78_baseline.pipeline import _robot
        root = self._start(output_dir)
        session = self._session(upstream)
        motion = _arrays(_outputs(upstream["Motion"])["actions.npz"])
        renderer = rd.RobotRenderer(_robot(self.repo))
        K = rd.scaled_intrinsics(session.K, 640, 480, (session.width, session.height))
        external = rd.external_camera(np.linalg.inv(session.c2w[0]) @ motion["T_world_base"])
        ego_rgb, ego_alpha, external_rgb, external_alpha = [], [], [], []
        for i in range(row["native_frames"]):
            self.authority()
            require(self.render_calls + 2 <= self.packet["budgets"]["render_calls_max"], "AUTOMATION_RENDER_CALL_BUDGET_EXHAUSTED")
            camera = np.linalg.inv(session.c2w[i]) @ motion["T_world_base"]
            self.render_calls += 1
            bgr, alpha, _ = renderer.render(motion["q_arm"][i], motion["q_hand"][i], camera, K, complete_robot=False)
            self.render_calls += 1
            ebgr, ealpha, _ = renderer.render(motion["q_arm"][i], motion["q_hand"][i], external, rd.EXTERNAL_K, complete_robot=True)
            require(bgr.shape == (480, 640, 3) and bgr.dtype == np.uint8 and alpha.shape == (480, 640)
                and ebgr.shape == bgr.shape and ebgr.dtype == np.uint8 and ealpha.shape == alpha.shape,
                "AUTOMATION_PARENT_RENDER_NATIVE_DOMAIN_DRIFT")
            ego_rgb.append(bgr[..., ::-1]); ego_alpha.append(alpha)
            external_rgb.append(ebgr[..., ::-1]); external_alpha.append(ealpha)
        _save_arrays(root / "ROBOT_BUFFERS.npz", ego_rgb=np.stack(ego_rgb), ego_alpha=np.stack(ego_alpha),
            external_rgb=np.stack(external_rgb), external_alpha=np.stack(external_alpha))
        write_once_json(root / "FRAME_MAP.json", self._mapping(upstream))
        return self._seal("Render", row, root, signature, upstream, {"render_calls": 2 * row["native_frames"],
            "ego_complete_robot": False, "external_complete_robot": True, "recipe": "EXACT_FROZEN_V1_RENDER_POLICY",
            "scene_depth": "ABSENT", "geometry": "UNKNOWN", "Mask_consumed": False, "Clean_consumed": False})

    def composite(self, *, row, output_dir, signature, upstream, request):
        import numpy as np
        from chaoyang.pick_place_1008 import visual
        from chaoyang.exact78_baseline import render as rd
        from .source_frames import write_review_video
        root = self._start(output_dir)
        rgb, mapping = self._rgb(upstream), self._mapping(upstream)
        masks = _arrays(_outputs(upstream["Mask"])["MASK_ARRAYS.npz"])
        clean_arrays = _arrays(_outputs(upstream["Clean"])["CLEAN_RGB.npz"])
        clean, effective = clean_arrays["rgb"], clean_arrays["effective_removal"]
        robots = _arrays(_outputs(upstream["Render"])["ROBOT_BUFFERS.npz"])
        motion = _arrays(_outputs(upstream["Motion"])["actions.npz"])
        require(read_reference(_outputs(upstream["Render"])["FRAME_MAP.json"]) == mapping,
                "AUTOMATION_RENDER_SOURCE_MAP_DRIFT")
        require(clean.shape == rgb.shape and clean.dtype == np.uint8 and effective.dtype == np.bool_
            and effective.shape == rgb.shape[:3] and all(v.shape == rgb.shape[:3] and v.dtype == np.bool_ for v in masks.values())
            and robots["ego_rgb"].shape == rgb.shape and robots["external_rgb"].shape == rgb.shape
            and robots["ego_alpha"].shape == rgb.shape[:3] and robots["external_alpha"].shape == rgb.shape[:3],
            "AUTOMATION_COMPOSITE_FULL_NATIVE_DOMAIN_REQUIRED")
        require(not np.any(clean[masks["protected_visible"]] != rgb[masks["protected_visible"]])
            and not np.any(clean[~effective] != rgb[~effective]), "AUTOMATION_COMPOSITE_CLEAN_PROTECTION_DRIFT")
        final, review, states = [], [], []
        for i in range(row["native_frames"]):
            self.authority()
            raw_bgr, clean_bgr = rgb[i][..., ::-1], clean[i][..., ::-1]
            robot_bgr, alpha = robots["ego_rgb"][i][..., ::-1], robots["ego_alpha"][i]
            composed = visual.compose_frame(raw_bgr, clean_bgr, robot_bgr, alpha,
                {k: masks[k][i] for k in ("core", "protected_visible", "unknown")}, robot_observed=True, masks_available=True)
            pixels = rd.over(clean_bgr, robot_bgr, composed["draw"])
            pixels[composed["object_front"]] = clean_bgr[composed["object_front"]]
            pixels[masks["protected_visible"][i]] = clean_bgr[masks["protected_visible"][i]]
            final.append(pixels[..., ::-1])
            review.append(np.concatenate((rgb[i], clean[i], pixels[..., ::-1]), axis=1))
            states.append({**mapping[i], "visible_protected_changed_pixels": 0, "hidden_regions": "UNKNOWN",
                "qpc_velocity_violation": motion["qpc_velocity_violation"][i].tolist(),
                "qpc_acceleration_violation": motion["qpc_acceleration_violation"][i].tolist(),
                "motion_diagnostics_reference": _outputs(upstream["Motion"])["motion.json"],
                "visual_quality": "NOT_REVIEWED", "human_review": "PENDING", "training_eligible": False,
                "control_ground_truth": False})
        final = np.stack(final)
        _save_arrays(root / "FINAL_RGB.npz", rgb=final)
        final_video = write_review_video(root / "final.mp4", final, mapping)
        review_video = write_review_video(root / "review.mp4", np.stack(review), mapping)
        pair_videos = {}
        clean_outputs = _outputs(upstream["Clean"])
        if "CLEAN_B_RGB.npz" in clean_outputs:
            candidate_clean = _arrays(clean_outputs["CLEAN_B_RGB.npz"])["rgb"]
            require(candidate_clean.shape == rgb.shape and candidate_clean.dtype == np.uint8
                and not np.any(candidate_clean[masks["protected_visible"]] != rgb[masks["protected_visible"]])
                and not np.any(candidate_clean[~effective] != rgb[~effective]), "AUTOMATION_COMPOSITE_PAIRED_CLEAN_PROTECTION_DRIFT")
            candidate_final = []
            for i in range(row["native_frames"]):
                self.authority()
                cbgr = candidate_clean[i][..., ::-1]
                rbgr = robots["ego_rgb"][i][..., ::-1]
                composed = visual.compose_frame(rgb[i][..., ::-1], cbgr, rbgr, robots["ego_alpha"][i],
                    {k: masks[k][i] for k in ("core", "protected_visible", "unknown")}, robot_observed=True, masks_available=True)
                pixels = rd.over(cbgr, rbgr, composed["draw"])
                pixels[composed["object_front"]] = cbgr[composed["object_front"]]
                pixels[masks["protected_visible"][i]] = cbgr[masks["protected_visible"][i]]
                candidate_final.append(pixels[..., ::-1])
            _save_arrays(root / "FINAL_A_RGB.npz", rgb=final)
            _save_arrays(root / "FINAL_B_RGB.npz", rgb=np.stack(candidate_final))
            pair_videos["B"] = write_review_video(root / "final_B.mp4", np.stack(candidate_final), mapping)
            pair_videos["A"] = final_video
        write_once_json(root / "FRAME_STATUS.json", states)
        write_once_json(root / "SESSION.json", {"session_id": row["session_id"], "native_frames": row["native_frames"],
            "execution_complete": True, "final_encoding": final_video, "review_encoding": review_video,
            "paired_final_encoding": pair_videos, "primary_clean_variant": "A",
            "stage_receipts": upstream, "visual_quality": "NOT_REVIEWED", "human_review": "PENDING",
            "training_eligible": False, "control_ground_truth": False})
        return self._seal("Composite", row, root, signature, upstream, {"model_calls": 0, "render_calls": 0,
            "robot_video_produced": True, "compositor": "FROZEN_V1_COMPOSE_FRAME_AND_SOFT_ALPHA_PROTECTION"})
