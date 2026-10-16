"""Bounded CPU-only Raw donor registration tests, no Torch or neural calls."""
from dataclasses import replace
import json
import numpy as np
import pytest
from chaoyang.inpainting_candidate import clean_flow as f
from chaoyang.inpainting_candidate.clean_references import select_reference_indices
from chaoyang.inpainting_candidate.contracts import SCHEMA, canonical_sha, ContractError

@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "PROJECT_ROOT", tmp_path)
    counter = [0]
    def create(kind=""):
        counter[0] += 1
        root = tmp_path / str(counter[0]); root.mkdir()
        raw = np.arange(5*4*5*3).reshape(5,4,5,3).astype(np.uint8)
        mapping = [{"native_frame_index": i} for i in range(5)]
        masks = {k: np.zeros((5,4,5), bool) for k in ("effective_removal","protected_visible","unknown","equipment","held")}
        masks["effective_removal"][2,1,1] = True
        ff = np.zeros((4,2,4,5),np.float32); bb = ff.copy()
        cf = np.ones((4,4,5),bool); cb = cf.copy()
        if kind == "shift":
            ff[:,0] = .5; bb[:,0] = -.5
        if kind in masks:
            masks[kind][4,1,1] = True
        if kind == "fb_forward": cf[2,1,1] = False
        if kind == "fb_backward": cb[2,1,1] = False
        if kind == "nan": ff[2,:,1,1] = np.nan
        if kind == "bounds": ff[2,0,1,1] = 10
        raw_ref = f.write_npz(root/"RAW_RGB.npz",rgb=raw)
        map_ref = f.write_json(root/"FRAME_MAP.json",mapping)
        source = {"schema":SCHEMA,"stage":"Source","status":"SEALED_COMPLETE","component_signature":"a"*64,"native_frames":5,"session_id":"cpu_only","outputs":[raw_ref,map_ref]}
        sr = f.write_json(root/"source.json",source)
        mask_ref = f.write_npz(root/"MASK_ARRAYS.npz",**masks)
        actual = {"frames_sha256":f.frames_sha256(raw),"frame_map_sha256":canonical_sha(mapping),"raw_source":raw_ref}
        mr = f.write_json(root/"mask.json",{"schema":SCHEMA,"stage":"Mask","status":"SEALED_COMPLETE","component_signature":"b"*64,"actual_inputs":actual,"outputs":[mask_ref]})
        arrays = f.write_npz(root/"flows.npz",forward=ff[None],backward=bb[None])
        index = f.write_json(root/"CHUNK_INDEX.json",[{"first_edge":0,"end_edge_exclusive":4,"arrays":arrays,"forward_array_sha256":f.array_sha(ff[None]),"backward_array_sha256":f.array_sha(bb[None])}])
        flow_actual = {**actual,"source_receipt":sr,"raw_rgb":raw_ref,"frame_map":map_ref}
        fr = f.write_json(root/"flow.json",{"schema":SCHEMA,"stage":"RawFlow","status":"SEALED_COMPLETE","component_signature":"c"*64,"session_id":"cpu_only","actual_native_shape":[5,4,5],"flow_source":"RAW_RGB_FORWARD_BACKWARD_FLOW","completed_inpainted_flow_consumed":False,"actual_inputs":flow_actual,"outputs":[arrays,index]})
        binding = {"source_receipt":sr,"mask_receipt":mr,"raw_flow_receipt":fr,"fb_consistent_forward_sha256":f.array_sha(cf),"fb_consistent_backward_sha256":f.array_sha(cb),"vendor_fb_diagnostic_policy":"EXISTING_VENDOR_DEFAULTS_NOT_BACKGROUND_CONFIDENCE"}
        return dict(session_id="cpu_only",raw_rgb=raw,frame_map=mapping,masks=masks,forward=ff,backward=bb,fb_consistent_forward=cf,fb_consistent_backward=cb,binding=binding)
    return create

def test_composes_forward_and_backward_target_native_support(make):
    b=f.RawVisibleReferenceBuilder(**make())
    rows=b.build(2,[0,4])
    assert [r.reference_frame for r in rows]==[0,4]
    assert all(r.visible_background[1,1] and r.flow_error[1,1]==0 for r in rows)
    assert all(r.support_source=="RAW_UNREMOVED_VISIBLE_SUPPORT" for r in rows)

def test_fractional_multiedge_translation_uses_all_raw_donor_samples(make):
    data=make("shift"); b=f.RawVisibleReferenceBuilder(**data)
    assert b.build(2,[4])[0].visible_background[1,1]
    # Target-native support stays at the target pixel, not the source-frame x=2.
    assert not b.build(2,[4])[0].visible_background[1,2]

def test_e_p_u_equipment_and_held_are_excluded_without_background_claim(make):
    for key in ("effective_removal","protected_visible","unknown","equipment","held"):
        assert not f.RawVisibleReferenceBuilder(**make(key)).build(2,[4])[0].visible_background.any()

def test_both_vendor_fb_directions_are_required(make):
    for kind in ("fb_forward","fb_backward"):
        row=f.RawVisibleReferenceBuilder(**make(kind)).build(2,[4])[0]
        assert not row.visible_background.any() and np.isinf(row.flow_error).all()

def test_nonfinite_and_out_of_bounds_paths_stay_unknown(make):
    for kind in ("nan","bounds"):
        assert not f.RawVisibleReferenceBuilder(**make(kind)).build(2,[4])[0].visible_background.any()

def test_actual_flow_and_fb_hash_drift_reject_before_composition(make):
    data=make();data["forward"][0,0,1,1]=1
    with pytest.raises(ContractError,match="ACTUAL_FLOW_DRIFT"):f.RawVisibleReferenceBuilder(**data)
    data=make();data["fb_consistent_forward"][0,1,1]=False
    with pytest.raises(ContractError,match="VENDOR_FB_BINDING"):f.RawVisibleReferenceBuilder(**data)

def test_cross_session_and_native_domain_reject(make):
    data=make();data["session_id"]="other"
    with pytest.raises(ContractError,match="SOURCE_DOMAIN"):f.RawVisibleReferenceBuilder(**data)
    b=f.RawVisibleReferenceBuilder(**make())
    with pytest.raises(ContractError,match="NATIVE_DOMAIN"):b.build(2,[2])

def test_rank_keeps_frozen_count_and_fallback_is_not_applied(make):
    b=f.RawVisibleReferenceBuilder(**make())
    e=b.build(2,[0,4]); hole=np.zeros((4,5),bool);hole[1,1]=True
    e[0].visible_background[:]=False
    r=select_reference_indices(2,[2],5,session_id="cpu_only",effective_delete_mask=hole,evidence=e)
    assert r["references"]==[4] and r["reference_count"]==1
    assert r["status"]=="RANKED_RAW_VISIBLE_SUPPORT_AND_FLOW" and not r["background_geometry_qualified"]
    e[1].visible_background[:]=False
    r=select_reference_indices(2,[2],5,session_id="cpu_only",effective_delete_mask=hole,evidence=e)
    assert r["references"]==r["original_references"]==[0]
    assert r["status"]=="BASELINE_INSUFFICIENT_RAW_SUPPORTED_REFERENCES"

def test_prepare_once_two_calls_no_io_and_input_arrays_are_frozen(make,monkeypatch):
    data=make();b=f.RawVisibleReferenceBuilder(**data)
    def forbidden(*a,**kw):raise AssertionError("setup repeated during build")
    monkeypatch.setattr(f,"sealed_receipt",forbidden);monkeypatch.setattr(f,"verify_ref",forbidden)
    monkeypatch.setattr(np,"load",forbidden);monkeypatch.setattr(f,"frames_sha256",forbidden)
    data["forward"][:]=99;data["masks"]["effective_removal"][:]=False
    assert b.build(2,[4])[0].visible_background[1,1]
    assert b.build(2,[0])[0].visible_background[1,1]

