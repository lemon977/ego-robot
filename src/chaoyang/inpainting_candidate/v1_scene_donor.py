"""Offline Raw-visible donor proposal; planar registration is NOT geometry truth.

No model, file IO, manual seeds, task identity or hidden object inference. Every
changed pixel is an actual same-session source sample, with its integer address.
All unsupported pixels fall back to the caller's frozen Clean. Source exclusion
is limited by the caller's masks; it cannot prove that those masks are complete.
"""
from __future__ import annotations
import cv2
import numpy as np


def _project(points, matrix):
    p=np.concatenate([points,np.ones((len(points),1))],axis=1) @ matrix.T
    valid=np.isfinite(p).all(axis=1) & (np.abs(p[:,2])>np.finfo(float).eps)
    result=np.full((len(points),2),np.nan)
    result[valid]=p[valid,:2]/p[valid,2,None]
    return result,valid


def _fit(source,target):
    if len(source)<4:return None,None
    cv2.setRNGSeed(0)
    H,inliers=cv2.findHomography(source,target,cv2.RANSAC,3.0,maxIters=2000,confidence=0.995)
    if H is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H)!=3:return None,None
    return H,inliers.reshape(-1).astype(bool)


def registrations(raw,excluded,authority=lambda:None):
    """Two disjoint match splits; errors saved, never called calibrated confidence."""
    cv2.setNumThreads(2)
    sift=cv2.SIFT_create(nfeatures=2000)
    feature=[]
    for frame,image in enumerate(raw):
        authority()
        key,desc=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),
                                      (~excluded[frame]).astype(np.uint8)*255)
        feature.append((np.asarray([k.pt for k in key],np.float64).reshape(-1,2),desc))
    anchor=max(range(len(raw)),key=lambda i:(len(feature[i][0]),-i))
    anchor_points,anchor_desc=feature[anchor]
    matrices=np.full((len(raw),2,3,3),np.nan)
    diagnostics=[]
    for i,(points,desc) in enumerate(feature):
        authority()
        row={'native_frame_index':i,'keypoints':len(points),'anchor':anchor,
             'qualification':'UNVERIFIED_PLANAR_REGISTRATION','heldout_errors_px':[]}
        if i==anchor:
            matrices[i]=np.eye(3);row.update(status='ANCHOR_IDENTITY',mutual_matches=len(points))
        elif desc is None or anchor_desc is None:
            row.update(status='NO_DESCRIPTORS',mutual_matches=0)
        else:
            matches=cv2.BFMatcher(cv2.NORM_L2,crossCheck=True).match(desc,anchor_desc)
            matches=sorted(matches,key=lambda m:(m.queryIdx,m.trainIdx))
            row['mutual_matches']=len(matches)
            if len(matches)<8:row['status']='INSUFFICIENT_DISJOINT_FIT_SUPPORT'
            else:
                src=np.asarray([points[m.queryIdx] for m in matches])
                dst=np.asarray([anchor_points[m.trainIdx] for m in matches])
                for split in range(2):
                    H,inliers=_fit(src[split::2],dst[split::2])
                    if H is None:continue
                    matrices[i,split]=H
                    prediction,valid=_project(src[1-split::2],H)
                    errors=np.linalg.norm(prediction-dst[1-split::2],axis=1)
                    row['heldout_errors_px'].append({'fit_split':split,'fit_inliers':int(inliers.sum()),
                        'validation_points':len(errors),'median':float(np.median(errors[valid])) if valid.any() else None,
                        'p95':float(np.percentile(errors[valid],95)) if valid.any() else None,
                        'errors':np.where(valid,errors,-1).tolist()})
                row['status']='TWO_FITS_AVAILABLE' if np.isfinite(matrices[i]).all() else 'FIT_UNAVAILABLE'
        diagnostics.append(row)
    return matrices,diagnostics,anchor


def donor_fill(raw,clean,effective,protected,excluded,matrices,*,donor_count=12,authority=lambda:None,mapping_provider=None):
    if (raw.dtype!=np.uint8 or raw.ndim!=4 or raw.shape[-1]!=3 or clean.shape!=raw.shape
        or clean.dtype!=np.uint8 or effective.shape!=raw.shape[:3] or protected.shape!=effective.shape
        or excluded.shape!=effective.shape or any(v.dtype!=bool for v in (effective,protected,excluded))
        or np.any(effective & protected)):
        raise ValueError('DONOR_FROZEN_NATIVE_DOMAIN_REQUIRED')
    n,h,w=raw.shape[:3]
    if ((mapping_provider is None and matrices.shape!=(n,2,3,3)) or n>32767 or max(h,w)>32767):
        raise ValueError('DONOR_NATIVE_MATRIX_DOMAIN_REQUIRED')
    donors=sorted(set(min(n-1,((2*i+1)*n)//(2*min(n,donor_count))) for i in range(min(n,donor_count))))
    result=clean.copy()
    provenance=np.full((n,h,w,3),-1,np.int16)
    counts=np.zeros((n,h,w),np.uint8)
    changes=[]
    for target in range(n):
        authority()
        yy,xx=np.nonzero(effective[target] & ~protected[target])
        if not len(xx) or (mapping_provider is None and not np.isfinite(matrices[target]).all()):
            changes.append({'frame':target,'supported_pixels':0,'changed_pixels':0});continue
        points=np.stack([xx,yy],axis=1)
        colors=[];addresses=[];validity=[]
        for donor in donors:
            if donor==target or (mapping_provider is None and not np.isfinite(matrices[donor]).all()):continue
            projected=[];ok=np.ones(len(points),bool)
            for split in range(2):
                if mapping_provider is None:
                    M=np.linalg.inv(matrices[donor,split]) @ matrices[target,split]
                    coords,valid=_project(points,M)
                else:
                    coords,valid=mapping_provider(target,donor,points,split)
                finite=np.where(np.isfinite(coords),coords,-1)
                indices=np.rint(np.clip(finite,-32768,32767)).astype(np.int32)
                projected.append(indices);ok &= valid
            a,b=projected
            ok &= np.all(a==b,axis=1) & (a[:,0]>=0) & (a[:,0]<w) & (a[:,1]>=0) & (a[:,1]<h)
            sx=np.clip(a[:,0],0,w-1);sy=np.clip(a[:,1],0,h-1)
            ok &= ~excluded[donor,sy,sx]
            colors.append(raw[donor,sy,sx]);validity.append(ok)
            addresses.append(np.stack([np.full(len(xx),donor),sx,sy],axis=1))
        if not colors:
            changes.append({'frame':target,'supported_pixels':0,'changed_pixels':0});continue
        valid=np.stack(validity);values=np.stack(colors);address=np.stack(addresses)
        support=valid.sum(axis=0);counts[target,yy,xx]=support
        admitted=support>=3
        # Median only selects a medoid: output itself is an actual source byte.
        masked=np.where(valid[:,:,None],values.astype(np.float32),np.nan)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            center=np.nanmedian(masked,axis=0)
        cost=np.sum(np.abs(values.astype(np.float32)-center[None]),axis=2)
        cost[~valid]=np.inf;cost[~np.isfinite(cost)]=np.inf
        chosen=np.argmin(cost,axis=0);ids=np.arange(len(xx))
        result[target,yy[admitted],xx[admitted]]=values[chosen[admitted],ids[admitted]]
        provenance[target,yy[admitted],xx[admitted]]=address[chosen[admitted],ids[admitted]]
        changes.append({'frame':target,'supported_pixels':int(admitted.sum()),
                        'changed_pixels':int(np.any(result[target]!=clean[target],axis=-1).sum())})
    if np.any(np.any(result!=raw,axis=-1) & (~effective | protected)):
        raise ValueError('DONOR_PIXEL_PRESERVATION_FAILED')
    return result,provenance,counts,{'donor_frames':donors,'per_frame':changes,
       'actual_source_sample':True,'registration_qualified':False,'hidden':'UNKNOWN',
       'time_mode':'OFFLINE_NONCAUSAL','training_eligible':False}
