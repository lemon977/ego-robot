import numpy as np
from chaoyang.inpainting_candidate.v1_scene_donor_local import split_affine
from chaoyang.inpainting_candidate.v1_scene_donor import donor_fill
from tests.unit.test_inpainting_scene_donor import synthetic


def test_disjoint_local_affine_translation():
    rng=np.random.default_rng(9);src=rng.uniform(0,200,(64,2));dst=src+[13,-8]
    H,diag=split_affine(src,dst)
    assert len(diag)==2 and np.allclose(H[:,:2,2],[13,-8],atol=1e-5)
    assert all(r['validation_median_px']<1e-4 for r in diag)
    a={tuple(v) for v in diag[0]['fit_points']};b={tuple(v) for v in diag[1]['fit_points']}
    assert not a.intersection(b)


def test_local_provider_preserves_source_contract():
    raw,clean,e,p,_=synthetic()
    def provider(t,d,points,split):return points.astype(float),np.ones(len(points),bool)
    out,prov,_,_=donor_fill(raw,clean,e,p,e|p,np.empty((0,2,3,3)),mapping_provider=provider)
    assert np.array_equal(out,raw) and np.all(prov[~e]==-1)


def test_underconstrained_local_fit_unavailable():
    H,d=split_affine(np.zeros((5,2)),np.zeros((5,2)))
    assert np.isnan(H).all() and not d
