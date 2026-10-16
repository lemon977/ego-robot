"""Direct local affine donor proposals; no geometric qualification is implied."""
from __future__ import annotations
import cv2
import numpy as np
from .v1_scene_donor import _project


def split_affine(source,target):
    matrices=np.full((2,3,3),np.nan);evidence=[]
    if len(source)<6:return matrices,evidence
    for split in range(2):
        cv2.setRNGSeed(0)
        A,inliers=cv2.estimateAffine2D(np.ascontiguousarray(source[split::2]),np.ascontiguousarray(target[split::2]),method=cv2.RANSAC,
            ransacReprojThreshold=3.,maxIters=2000,confidence=.995,refineIters=10)
        if A is None or not np.isfinite(A).all() or np.linalg.matrix_rank(A[:,:2])!=2:continue
        matrices[split]=np.vstack([A,[0.,0.,1.]])
        prediction,valid=_project(source[1-split::2],matrices[split])
        errors=np.linalg.norm(prediction-target[1-split::2],axis=1)
        evidence.append({'split':split,'fit_points':source[split::2].tolist(),
            'fit_destination_points':target[split::2].tolist(),'fit_inliers':inliers.reshape(-1).tolist(),
            'validation_points':source[1-split::2].tolist(),'validation_errors_px':np.where(valid,errors,-1).tolist(),
            'validation_median_px':float(np.median(errors[valid])) if valid.any() else None})
    return matrices,evidence


class LocalAffineMappings:
    def __init__(self,raw,excluded,effective,authority=lambda:None):
        cv2.setNumThreads(2);sift=cv2.SIFT_create(nfeatures=2000)
        self.features=[];self.effective=effective;self.diagnostics=[];self.authority=authority
        self.current=None;self.cache={};self.labels=None
        for f,image in enumerate(raw):
            authority()
            kp,desc=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),(~excluded[f]).astype(np.uint8)*255)
            self.features.append((np.asarray([k.pt for k in kp],np.float64).reshape(-1,2),desc))

    def __call__(self,target,donor,points,split):
        if self.current!=target:
            self.current=target;self.cache={}
            _,self.labels,_,self.centers=cv2.connectedComponentsWithStats(self.effective[target].astype(np.uint8),8)
        if donor not in self.cache:
            self.authority()
            src,desc=self.features[target];dst,dst_desc=self.features[donor]
            coords=np.full((2,len(points),2),np.nan);valid=np.zeros((2,len(points)),bool)
            row={'target':target,'donor':donor,'qualification':'UNVERIFIED_LOCAL_AFFINE','components':[]}
            matches=[] if desc is None or dst_desc is None else cv2.BFMatcher(cv2.NORM_L2,crossCheck=True).match(desc,dst_desc)
            row['mutual_matches']=len(matches)
            if matches:
                a=np.asarray([src[m.queryIdx] for m in matches]);b=np.asarray([dst[m.trainIdx] for m in matches])
                ids=self.labels[points[:,1],points[:,0]]
                for label in np.unique(ids):
                    if label==0:continue
                    distance=np.sum((a-self.centers[label])**2,axis=1)
                    order=np.argsort(distance,kind='stable')[:64]
                    H,diag=split_affine(a[order],b[order]);selection=ids==label
                    for k in range(2):
                        if not np.isfinite(H[k]).all():continue
                        coords[k,selection],valid[k,selection]=_project(points[selection],H[k])
                    row['components'].append({'component':int(label),'center':self.centers[label].tolist(),
                        'matrices':np.where(np.isfinite(H),H,0).tolist(),
                        'fit_available':[bool(np.isfinite(H[k]).all()) for k in range(2)],'evidence':diag})
            self.diagnostics.append(row);self.cache[donor]=(coords,valid)
        return self.cache[donor][0][split],self.cache[donor][1][split]
