"""Opt-in stage-local Clean adapter; shared V1 stage implementations unchanged."""
from __future__ import annotations
from pathlib import Path
from types import FunctionType
import numpy as np
from .automation_worker import V1Builders,_arrays,_outputs,_save_arrays
from .automation import read_reference,verify_reference,write_once_json,require

POLICY='EXCLUDE_UNCERTAIN_EQUIPMENT_RGB'

class ContextExclusionBuilders(V1Builders):
    def clean(self,*,row,output_dir,signature,upstream,request):
        require(self.packet['component_config']['Clean'].get('reference_policy')==POLICY,
                'CONTEXT_BUILDER_EXPLICIT_POLICY_REQUIRED')
        root=self._start(output_dir);raw=self._rgb(upstream)
        mask_outputs=_outputs(upstream['Mask']);masks=_arrays(mask_outputs['MASK_ARRAYS.npz'])
        report=read_reference(mask_outputs['MASK_REPORT.json']);seed=read_reference(row['seed'])
        require(report['seed']==row['seed'],'CONTEXT_BUILDER_SEED_NOT_PARENT_MASK')
        verify_reference(seed['image'])
        from PIL import Image
        require(seed['session_id']==row['session_id']and
            np.array_equal(np.asarray(Image.open(seed['image']['path']).convert('RGB')),raw[seed['raw_frame_index']]),
            'CONTEXT_BUILDER_SEED_SOURCE_RGB_NOT_EXACT')
        from .v1_equipment_rgb import extend
        from .v1_context_exclusion import restore_fixed_write
        masks['metadata']=report['metadata']
        working,support,audit=extend(raw,masks,seed,authority=self.authority,motion_veto=False)
        _save_arrays(root/'CONTEXT_WORKING_MASKS.npz',removal=working['removal'],effective_removal=working['effective_removal'],
            write_effective_removal=masks['effective_removal'],rgb_uncertain_support=support)
        write_once_json(root/'CONTEXT_REFERENCE_AUDIT.json',{'source_receipt':upstream['Source'],'mask_receipt':upstream['Mask'],
            'seed':row['seed'],'reference_policy':POLICY,'motion_veto':False,'working_support_role':'UNCERTAIN_CONTEXT_ONLY_NOT_DELETION',
            'rgb_recipe':audit,'new_manual_points':0,'hidden':'UNKNOWN','training_eligible':False})
        require(self.clean_calls<self.packet['budgets']['propainter_calls_max'],'AUTOMATION_PROPAINTER_CALL_BUDGET_EXHAUSTED')
        from chaoyang.pick_place_1008.visual import ProtectedProPainter
        painter=ProtectedProPainter(self._models('Clean')['propainter']);self.authority();self.clean_calls+=1
        work=painter.inpaint(raw,working['removal'],masks['protected_visible']);self.authority()
        clean,validation=restore_fixed_write(raw,work,masks['effective_removal'],masks['protected_visible'])
        _save_arrays(root/'CLEAN_RGB.npz',rgb=clean,effective_removal=masks['effective_removal']);_save_arrays(root/'WORKING_CLEAN_RGB.npz',rgb=work)
        _save_arrays(root/'EFFECTIVE_MASKS.npz',effective_removal=masks['effective_removal'])
        write_once_json(root/'CLEAN_VALIDATION.json',{'final_write_validation':validation,'working_validation':painter.last_validation,
            'protected_changed_pixels_per_frame':validation['protected_changed_pixels_per_frame'],
            'outside_delete_changed_pixels_per_frame':validation['outside_delete_changed_pixels_per_frame'],
            'reference_policy':POLICY,'primary_clean_variant':'CONTEXT_EXCLUSION','status':'PASS'})
        write_once_json(root/'CLEAN_REPORT.json',{'validation':validation,'settings':painter.settings,
            'propainter_calls':1,'paired_clean':False,'primary_clean_variant':'CONTEXT_EXCLUSION',
            'reference_policy':POLICY,'historical_clean_consumed':False})
        from .source_frames import write_review_video
        write_review_video(root/'clean.mp4',clean,self._mapping(upstream))
        return self._seal('Clean',row,root,signature,upstream,{'propainter_calls':1,'sam_api_calls':0,
            'reference_policy':POLICY,'final_write_mask':'UNCHANGED_SEALED_MASK_STAGE','primary_clean_variant':'CONTEXT_EXCLUSION',
            'new_manual_points':0,'candidate_adopted':False})

    def composite(self,**kwargs):
        # Use the original algorithm bytecode, changing only the metadata writer
        # before immutable SESSION.json is first written. No module mutation.
        original=V1Builders.composite;namespace=dict(original.__globals__)
        writer=namespace['write_once_json']
        def scoped_writer(path,value):
            if Path(path).name=='SESSION.json':
                value={**value,'primary_clean_variant':'CONTEXT_EXCLUSION','clean_reference_policy':POLICY}
            return writer(path,value)
        namespace['write_once_json']=scoped_writer
        clone=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
        return clone(self,**kwargs)
