"""Private pure-array visual candidate. No file/model/q/FK/render entry."""
import numpy as np
FIELDS=("core","sam_core","equipment","removal","cup_observed","semantic_conflict","protected_visible","held","unknown","effective_removal")
OLD_SCHEMA="pick-place-visual-masks-observed-point-v2"
NEW_SCHEMA="pick-place-q-frozen-cup-conflict-raw-restore-masks-candidate-v1"

def require(condition, message):
    if not condition:
        raise ValueError(message)

def unpack_big(raw, clean, packed, bitorder, mask_schema):
    require(bitorder=="big", "EXPLICIT_BIG_BITORDER_REQUIRED")
    require(mask_schema==OLD_SCHEMA, "EXACT_OLD_MASK_SCHEMA_REQUIRED")
    require(isinstance(raw,np.ndarray) and isinstance(clean,np.ndarray), "RGB_ARRAY_REQUIRED")
    require(raw.dtype==clean.dtype==np.uint8, "RGB_UINT8_REQUIRED")
    require(raw.ndim==4 and raw.shape==clean.shape and raw.shape[-1]==3 and all(type(x)is int and x>0 for x in raw.shape), "RGB_NHWC3_NONEMPTY")
    n,h,w,_=raw.shape
    require(h*w%8==0, "NO_PACKED_PADDING")
    require(isinstance(packed,dict) and set(packed)==set(FIELDS), "EXACT_TEN_FIELDS_REQUIRED")
    masks={}
    for name in FIELDS:
        value=packed[name]
        require(isinstance(value,np.ndarray) and value.dtype==np.uint8 and value.shape==(n,h*w//8), "PACKED_BIG_UINT8_SHAPE:"+name)
        masks[name]=np.unpackbits(value,axis=1,count=h*w,bitorder="big").reshape(n,h,w).astype(np.bool_)
    return masks

def derive_visual_candidate(raw, old_clean, packed, *, bitorder, baseline_mask_schema):
    m=unpack_big(raw,old_clean,packed,bitorder,baseline_mask_schema)
    core=m["core"];cup=m["cup_observed"];P=m["protected_visible"];E=m["effective_removal"];semantic=m["semantic_conflict"]
    require(np.array_equal(core,m["sam_core"]|m["equipment"]), "OLD_CORE_ROLE_UNION")
    require(np.array_equal(P,cup&~core), "OLD_P_IDENTITY")
    require(np.array_equal(semantic,cup&core), "OLD_SEMANTIC_IDENTITY")
    require(np.array_equal(m["unknown"],P), "OLD_UNKNOWN_IDENTITY")
    require(not np.any(m["held"]&~P), "OLD_HELD_SUBSET_P")
    require(not np.any(E&P), "OLD_E_P_DISJOINT")
    changed=np.any(raw!=old_clean,axis=-1)
    require(not np.any(changed&~E), "OLD_CLEAN_OUTSIDE_E")
    require(not np.any(changed&P), "OLD_CLEAN_PROTECTED_DRIFT")
    restore=semantic&E
    clean=old_clean.copy()
    clean[restore]=raw[restore]
    new={name:value.copy() for name,value in m.items()}
    new["protected_visible"]=cup.copy()
    new["effective_removal"]=E&~cup
    require(not np.any(new["effective_removal"]&new["protected_visible"]), "NEW_E_P_DISJOINT")
    require(not np.any(np.any(raw!=clean,axis=-1)&~new["effective_removal"]), "NEW_CLEAN_OUTSIDE_E")
    require(np.array_equal(clean[cup],raw[cup]), "NEW_CUP_PIXELS_BYTE_EXACT")
    require(np.array_equal(clean[~restore],old_clean[~restore]), "RESTORE_ONLY_SEMANTIC_AND_OLD_E")
    require(np.array_equal(new["semantic_conflict"],cup&core), "NEW_SEMANTIC_DIAGNOSTIC")
    require(np.array_equal(new["unknown"],cup&~core), "NEW_UNKNOWN_OLD_VISIBLE_AMBIGUITY")
    require(not np.any(new["held"]&~new["unknown"]), "NEW_HELD_OLD_UNKNOWN_SUBSET")
    packed_new={name:np.packbits(value.reshape(len(value),-1),axis=1,bitorder="big") for name,value in new.items()}
    for name in FIELDS:
        if name not in ("protected_visible","effective_removal"):
            require(np.array_equal(packed_new[name],packed[name]), "OLD_FIELD_BITEXACT:"+name)
    per_frame=lambda x:np.count_nonzero(x,axis=(1,2)).tolist()
    metadata={
        "schema_version":NEW_SCHEMA,
        "baseline_mask_schema":OLD_SCHEMA,
        "bitorder":"big",
        "single_factor":"RAW_RESTORE_SEMANTIC_CONFLICT_AND_OLD_E_WITH_CUP_PROTECTION",
        "raw_restore_region":"old_semantic_conflict & old_effective_removal",
        "new_protected_visible":"old_cup_observed",
        "new_effective_removal":"old_effective_removal & ~old_cup_observed",
        "semantic_conflict":"unchanged old cup & core, diagnostic only; now subset of protected_visible",
        "unknown":"unchanged old unknown == cup & ~core; not expanded to newly protected conflict",
        "held":"unchanged old held, subset of old unknown and new protection; no new attachment truth",
        "removal":"unchanged historical ProPainter request; current edit authority is new effective_removal",
        "restored_mask_pixels_per_frame":per_frame(restore),
        "actually_restored_changed_pixels_per_frame":per_frame(restore&changed),
        "protected_core_overlap_pixels_per_frame":per_frame(cup&core),
        "cup_foreground_conflict_not_actual_visible_cup_truth":True,
        "boundary_repair_or_false_positive_removal_claimed":False,
        "q_source_camera_seed_models_modified":False,
        "new_ProPainter_or_SAM_or_FK_or_Render_calls":0,
        "Contact_requires_fresh_derived_recompute":True,
        "physical_contact_ground_truth":False,
        "control_ground_truth":False,
        "training_eligible":False,
        "human_review":"PENDING",
        "quality_accepted":False,
    }
    return {"clean_rgb":clean,"packed_masks":packed_new,"metadata":metadata}
