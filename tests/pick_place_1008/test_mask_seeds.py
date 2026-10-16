import copy,json
from types import SimpleNamespace
import cv2,numpy as np,pytest
from chaoyang.pick_place_1008 import mask_seeds as m,source

def setup(tmp_path):
    rgb=np.zeros((2,480,640,3),np.uint8);rgb[:]=[31,72,119]
    image=tmp_path/"raw.png";cv2.imwrite(str(image),rgb[0,:,:,::-1])
    objects=[{"obj_id":i+1,"role":role,"side":side,"points_px":[[100+i*30,200],[120+i*30,220]],"point_labels":[1,1],"visible_confirmed":True}
             for i,(side,role) in enumerate([("left","glove"),("left","controller"),("right","glove"),("right","controller")])]
    record={"schema":"pick_place_visible_mask_seeds_v1","session_id":"001","device_serial":"deviceA","raw_frame_index":0,"qpc_ns":1000,"image":source.ref(image),"visibility_review_status":"AI_CONFIRMED_VISIBLE_PIXELS","training_eligible":False,"physical_ground_truth":False,"objects":objects}
    leaf=tmp_path/"seed.json";leaf.write_text(json.dumps(record))
    index=tmp_path/"index.json"
    def save():
        leaf.write_text(json.dumps(record));index.write_text(json.dumps({"schema":"pick_place_mask_seed_index_v1","records":[{"session_id":"001","seed":source.ref(leaf)}]}))
    save()
    session=SimpleNamespace(calibration={"device_serial":"deviceA"},qpc_ns=np.array([1000,2000]))
    return index,record,session,rgb,save,leaf

def test_exact_visible_seed_record(tmp_path):
    index,record,session,rgb,_,leaf=setup(tmp_path)
    loaded=m.load_visible_seed(index,"001",session,rgb)
    assert loaded["objects"]==record["objects"] and loaded["seed_reference"]==source.ref(leaf)
    assert m.seed_reference(index,"001")==source.ref(leaf)

@pytest.mark.parametrize("key,value,error",[("device_serial","other","IDENTITY"),("qpc_ns",1001,"QPC"),("raw_frame_index",2,"QPC"),("physical_ground_truth",True,"IDENTITY"),("visibility_review_status","UNREVIEWED","IDENTITY")])
def test_seed_admission_rejects_wrong_binding(tmp_path,key,value,error):
    index,record,session,rgb,save,_=setup(tmp_path);record[key]=value;save()
    with pytest.raises(ValueError,match=error):m.load_visible_seed(index,"001",session,rgb)

def test_seed_rejects_original_pixel_difference(tmp_path):
    index,_,session,rgb,_,_=setup(tmp_path);rgb[0,0,0,0]+=1
    with pytest.raises(ValueError,match="RGB_PIXEL"):m.load_visible_seed(index,"001",session,rgb)

def test_seed_rejects_leaf_drift(tmp_path):
    index,_,session,rgb,_,leaf=setup(tmp_path);leaf.write_text(leaf.read_text()+" ")
    with pytest.raises(Exception,match="REFERENCE_DRIFT"):m.load_visible_seed(index,"001",session,rgb)

@pytest.mark.parametrize("edit,error",[("duplicate","ROLE_SIDE"),("outside","POINT_DOMAIN"),("missing_controller","REQUIRED_GLOVE_CONTROLLER"),("no_foreground","POINT_DOMAIN")])
def test_seed_object_safety(tmp_path,edit,error):
    index,record,session,rgb,save,_=setup(tmp_path)
    if edit=="duplicate":record["objects"][1]["obj_id"]=1
    elif edit=="outside":record["objects"][0]["points_px"][0][0]=640
    elif edit=="missing_controller":record["objects"].pop()
    else:record["objects"][0]["point_labels"]=[0,0]
    save()
    with pytest.raises(ValueError,match=error):m.load_visible_seed(index,"001",session,rgb)

def setup_v2(tmp_path):
    index, record, session, _, save, leaf = setup(tmp_path)
    rgb = np.zeros((3, 480, 640, 3), np.uint8)
    images = []
    for frame in range(3):
        rgb[frame] = [31 + frame, 72 + 2 * frame, 119 + 3 * frame]
        path = tmp_path / f"raw_{frame}.png"
        cv2.imwrite(str(path), rgb[frame, :, :, ::-1])
        images.append(source.ref(path))
    session.qpc_ns = np.array([1000, 2000, 3000])
    record.update(schema="pick_place_visible_mask_seeds_v2", image=images[0])
    return index, record, session, rgb, save, leaf, images


def test_v2_each_object_binds_its_actual_native_frame_and_min_anchor(tmp_path):
    index, record, session, rgb, save, _, images = setup_v2(tmp_path)
    for obj, frame in zip(record["objects"], [2, 1, 2, 1]):
        obj.update(raw_frame_index=frame, qpc_ns=int(session.qpc_ns[frame]), image=images[frame])
    save()
    loaded = m.load_visible_seed(index, "001", session, rgb)
    assert loaded["anchor_frame"] == 1
    assert [obj["frame_idx"] for obj in loaded["objects"]] == [2, 1, 2, 1]
    assert [obj["image"] for obj in loaded["objects"]] == [images[2], images[1], images[2], images[1]]
    assert loaded["unseeded_roles"] == []


@pytest.mark.parametrize("mutation,error", [
    ("qpc", "QPC"), ("missing_qpc", "QPC"), ("other_native_image", "RGB_PIXEL"),
    ("own_pixel_changed", "RGB_PIXEL"), ("image_hash_drift", "REFERENCE_DRIFT"),
    ("aliases", "FRAME_ALIASES"), ("device", "DEVICE"),
])
def test_v2_own_anchor_evidence_is_verified_not_top_anchor_only(tmp_path, mutation, error):
    index, record, session, rgb, save, _, images = setup_v2(tmp_path)
    obj = record["objects"][0]
    obj.update(frame_idx=2, qpc_ns=3000, image=images[2])
    if mutation == "qpc":
        obj["qpc_ns"] = 3001
    elif mutation == "missing_qpc":
        obj.pop("qpc_ns")
    elif mutation == "other_native_image":
        obj["image"] = images[0]
    elif mutation == "own_pixel_changed":
        rgb[2, 2, 3, 0] += 1
    elif mutation == "image_hash_drift":
        from pathlib import Path
        path = Path(images[2]["path"])
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "aliases":
        obj["raw_frame_index"] = 1
    else:
        obj["device_serial"] = "other-device"
    save()
    with pytest.raises(Exception, match=error):
        m.load_visible_seed(index, "001", session, rgb)


def invisible_proof(record, session, images):
    record["objects"] = [obj for obj in record["objects"] if (obj["side"], obj["role"]) != ("left", "controller")]
    record["unseeded_roles"] = [{
        "side": "left", "role": "controller",
        "visibility": "NOT_VISIBLE_IN_INSPECTED_NATIVE_FRAMES",
        "review": "AI_CONFIRMED_NOT_VISIBLE",
        "inspected_frames": [{"raw_frame_index": frame, "qpc_ns": int(session.qpc_ns[frame]), "image": images[frame]}
                             for frame in [0, 1, 2]],
    }]


def test_v2_missing_role_needs_actual_zero_mid_last_native_evidence(tmp_path):
    index, record, session, rgb, save, _, images = setup_v2(tmp_path)
    invisible_proof(record, session, images)
    save()
    loaded = m.load_visible_seed(index, "001", session, rgb)
    item = loaded["unseeded_roles"][0]
    assert item["validation"] == m.UNSEEN_VALIDATION
    assert item["unreviewed_visibility_status"] == "UNKNOWN"
    assert item["later_visibility_review_needed"] is True
    assert item["global_all_frames_invisible"] is False
    assert [row["image"] for row in item["inspected_frames"]] == images


@pytest.mark.parametrize("mutation,error", [
    ("no_proof", "REQUIRED_GLOVE_CONTROLLER"), ("only_anchor", "ZERO_MID_LAST"),
    ("duplicate_frames", "ZERO_MID_LAST"), ("wrong_image", "RGB_PIXEL"),
    ("wrong_qpc", "QPC"), ("not_reviewed", "VISIBILITY_PROOF"),
    ("global_claim", "VISIBILITY_PROOF"), ("physical_truth", "VISIBILITY_PROOF"),
    ("visible_claim", "VISIBILITY_PROOF"), ("not_unknown", "VISIBILITY_PROOF"),
    ("seed_and_unseeded", "VISIBILITY_PROOF"), ("v1", "UNSEEDED_V2"),
])
def test_v2_unseen_role_does_not_allow_false_or_missing_proof(tmp_path, mutation, error):
    index, record, session, rgb, save, _, images = setup_v2(tmp_path)
    original = copy.deepcopy(record["objects"])
    invisible_proof(record, session, images)
    item = record["unseeded_roles"][0]
    if mutation == "no_proof":
        record.pop("unseeded_roles")
    elif mutation == "only_anchor":
        item["inspected_frames"] = item["inspected_frames"][:1]
    elif mutation == "duplicate_frames":
        item["inspected_frames"][1] = copy.deepcopy(item["inspected_frames"][0])
    elif mutation == "wrong_image":
        item["inspected_frames"][1]["image"] = images[0]
    elif mutation == "wrong_qpc":
        item["inspected_frames"][2]["qpc_ns"] = 3001
    elif mutation == "not_reviewed":
        item["review"] = "PENDING"
    elif mutation == "global_claim":
        item["global_all_frames_invisible"] = True
    elif mutation == "physical_truth":
        item["physical_ground_truth"] = True
    elif mutation == "visible_claim":
        item["visibility"] = "NOT_VISIBLE_ANYWHERE"
    elif mutation == "not_unknown":
        item["unreviewed_visibility_status"] = "NOT_VISIBLE"
    elif mutation == "seed_and_unseeded":
        record["objects"] = original
    else:
        record["schema"] = "pick_place_visible_mask_seeds_v1"
    save()
    with pytest.raises(ValueError, match=error):
        m.load_visible_seed(index, "001", session, rgb)
