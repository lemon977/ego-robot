"""Bounded read-only PP stage observer. Never changes tensors or return values."""
from __future__ import annotations
from copy import copy
import numpy as np


class OriginalPathObserver:
    def __init__(self,model,effective,settings,anchors=(0,13,28,64,88,96,119,127),*,pixel_context_length=None):
        self.model=model;self.effective=effective;self.settings=dict(settings)
        self.anchors=set(f for f in anchors if 0<=f<len(effective))
        self.arrays={};self.records=[];self.pixel_calls=0;self.forward_calls=0;self.raw_bytes=0
        n=len(effective);sub=settings['subvideo_length'];prop=min(100,sub) if pixel_context_length is None else pixel_context_length
        self.pixel_contexts=[(max(0,f-10),min(n,f+prop+10)) for f in range(0,n,prop)] if n>prop else [(0,n)]
        self.neighbors=[];self.ids=[]
        from chaoyang.exact78_baseline.clean import _get_ref_index
        stride=settings['neighbor_length']//2;ref_num=sub//settings['ref_stride'] if n>sub else -1
        for f in range(0,n,stride):
            near=list(range(max(0,f-stride),min(n,f+stride+1)))
            self.neighbors.append(near);self.ids.append(near+_get_ref_index(f,near,n,settings['ref_stride'],ref_num))

    def __getattr__(self,name):return getattr(self.model,name)

    def save(self,key,array):
        value=np.array(array,copy=True)
        if key in self.arrays:
            if not np.array_equal(self.arrays[key],value):raise ValueError('PP_TRACE_REPEATED_NATIVE_INPUT_CHANGED')
            return key
        self.raw_bytes+=value.nbytes
        if self.raw_bytes>128*1024*1024:raise ValueError('PP_TRACE_DIAGNOSTIC_CAPACITY')
        self.arrays[key]=value;return key

    @staticmethod
    def numpy(tensor):return tensor.detach().float().cpu().numpy()

    def rgb(self,tensor):
        value=self.numpy(tensor)
        return np.clip(np.rint((value.transpose(1,2,0)+1)*127.5),0,255).astype(np.uint8)

    def mask_record(self,prefix,mask,native):
        actual=self.numpy(mask).reshape(self.effective.shape[1:])
        key=self.save(prefix,actual.astype(np.float16))
        return {'array':key,'minimum':float(actual.min()),'maximum':float(actual.max()),
                'mismatch_vs_frozen_effective_pixels':int(np.count_nonzero(actual!=self.effective[native]))}

    def img_propagation(self,masked,flows,masks,interpolation='nearest'):
        call=self.pixel_calls;self.pixel_calls+=1
        if call>=len(self.pixel_contexts):raise ValueError('PP_TRACE_PIXEL_CONTEXT_COUNT')
        start,end=self.pixel_contexts[call]
        if tuple(masked.shape)!=(1,end-start,3,*self.effective.shape[1:]):raise ValueError('PP_TRACE_PIXEL_NATIVE_SHAPE')
        record={'stage':'PIXEL_PROPAGATION','call':call,'native_start':start,'native_end_exclusive':end,'samples':[]}
        for native in sorted(self.anchors.intersection(range(start,end))):
            local=native-start;prefix=f'PIXEL_C{call:02d}_F{native:03d}'
            m=self.mask_record(prefix+'_INPUT_MASK',masks[0,local,0],native)
            before=self.numpy(masked[0,local])
            row={'native_frame_index':native,'mask':m,
                'masked_input_rgb':self.save(prefix+'_MASKED_RGB',self.rgb(masked[0,local])),
                'nonzero_masked_normalized_values_in_effective':int(np.count_nonzero(before[:,self.effective[native]]))}
            record['samples'].append(row)
        result=self.model.img_propagation(masked,flows,masks,interpolation)
        images,unknown=result
        images=images.reshape(1,end-start,3,*self.effective.shape[1:])
        unknown=unknown.reshape(1,end-start,1,*self.effective.shape[1:])
        for row in record['samples']:
            native=row['native_frame_index'];local=native-start;prefix=f'PIXEL_C{call:02d}_F{native:03d}'
            row.update(propagated_rgb=self.save(prefix+'_PROPAGATED_RGB',self.rgb(images[0,local])),
                       unfilled_mask=self.save(prefix+'_UNFILLED_MASK',self.numpy(unknown[0,local,0]).astype(np.float16)))
        self.records.append(record)
        return result

    def __call__(self,frames,flows,masks,updated_masks,neighbor_count):
        call=self.forward_calls;self.forward_calls+=1
        if call>=len(self.ids):raise ValueError('PP_TRACE_FORWARD_CALL_COUNT')
        ids=self.ids[call];near=self.neighbors[call]
        if tuple(frames.shape)!=(1,len(ids),3,*self.effective.shape[1:]) or neighbor_count!=len(near):
            raise ValueError('PP_TRACE_FORWARD_NATIVE_MAPPING')
        record={'stage':'PREDICTION','call':call,'actual_input_native_ids':ids,'actual_neighbor_native_ids':near,'samples':[]}
        for native in sorted(self.anchors.intersection(ids)):
            local=ids.index(native);prefix=f'MODEL_INPUT_F{native:03d}'
            row={'native_frame_index':native,'role':'NEIGHBOR' if native in near else 'REFERENCE',
                'updated_rgb':self.save(prefix+'_RGB',self.rgb(frames[0,local])),
                'mask':self.mask_record(prefix+'_MASK',masks[0,local,0],native),
                'unfilled_mask':self.save(prefix+'_UNFILLED_MASK',self.numpy(updated_masks[0,local,0]).astype(np.float16))}
            record['samples'].append(row)
        result=self.model(frames,flows,masks,updated_masks,neighbor_count)
        values=result.reshape(len(near),3,*self.effective.shape[1:])
        for row in record['samples']:
            native=row['native_frame_index']
            if native not in near:continue
            tensor=values[near.index(native)];value=self.numpy(tensor);prefix=f'PRED_C{call:02d}_F{native:03d}'
            row.update(predicted_rgb=self.save(prefix+'_RGB',self.rgb(tensor)),
                predicted_normalized=self.save(prefix+'_NORMALIZED',value.astype(np.float16)),
                prediction_minimum=float(value.min()),prediction_maximum=float(value.max()))
        self.records.append(record)
        return result

    def metadata(self):
        if self.pixel_calls!=len(self.pixel_contexts) or self.forward_calls!=len(self.ids):
            raise ValueError('PP_TRACE_INCOMPLETE_CONTEXT_OBSERVATION')
        return {'schema':'inpainting-original-pp-readonly-trace-v1','native_frames':len(self.effective),
            'anchors':sorted(self.anchors),'pixel_calls':self.pixel_calls,'forward_calls':self.forward_calls,
            'uncompressed_array_bytes':self.raw_bytes,'records':self.records,'tensors_modified':False,
            'return_values_modified':False,'rgb_display':'RINT_FROM_NORMALIZED_TENSOR_NOT_RAW_BYTE_TRUTH',
            'quality_pass':False,'human_review':'PENDING','hidden':'UNKNOWN'}


def observed_parent(parent,effective,*,pixel_context_length=None):
    wrapped=copy(parent);wrapped.settings=dict(parent.settings)
    observer=OriginalPathObserver(parent.model,effective,parent.settings,pixel_context_length=pixel_context_length)
    wrapped.model=observer
    return wrapped,observer
