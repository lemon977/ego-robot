"""Fixed cross-body triangle/convex query canary, never a production collision gate."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def rigid(transform):
    t=np.asarray(transform,dtype=float)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]):
        raise ValueError('finite homogeneous transform required')
    r=t[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(r),1.):
        raise ValueError('rigid proper rotation required')
    return t[:3,3].tolist(),Rotation.from_matrix(r).as_quat().tolist()


def _summary(points):
    distances=[float(x[8]) for x in points]
    return dict(query_returned_points=len(distances),minimum_signed_distance_m=min(distances) if distances else None,
                status='DISTANCE_RETURNED' if distances else 'NO_CLOSE_POINT_NOT_PROOF_OF_SEPARATION')


def query_mesh_pair(mesh_a, transform_a, mesh_b, transform_b, *, concave_a, margin_m=None):
    """Static concave A vs dynamic convex B only, in separate bodies."""
    import pybullet as p
    client=p.connect(p.DIRECT)
    try:
        shape_a=p.createCollisionShape(p.GEOM_MESH,fileName=str(mesh_a),
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH if concave_a else 0,physicsClientId=client)
        shape_b=p.createCollisionShape(p.GEOM_MESH,fileName=str(mesh_b),physicsClientId=client)
        pa,qa=rigid(transform_a);pb,qb=rigid(transform_b)
        a=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape_a,basePosition=pa,baseOrientation=qa,physicsClientId=client)
        b=p.createMultiBody(baseMass=1,baseCollisionShapeIndex=shape_b,basePosition=pb,baseOrientation=qb,physicsClientId=client)
        if margin_m is not None:
            if not np.isfinite(margin_m) or margin_m<0:raise ValueError('invalid query margin')
            for body in (a,b):p.changeDynamics(body,-1,collisionMargin=float(margin_m),physicsClientId=client)
        # Record defaults or explicitly requested diagnostic-only common margin.
        margins=[float(p.getDynamicsInfo(body,-1,physicsClientId=client)[11]) for body in (a,b)]
        if margin_m is not None and not np.allclose(margins,[margin_m,margin_m],atol=1e-12,rtol=0):
            raise ValueError('engine did not apply requested common margin')
        points=p.getClosestPoints(a,b,distance=.1,physicsClientId=client)
        result=_summary(points)
        if points:
            deepest=min(points,key=lambda x:x[8])
            result['closest_witness_world']=dict(point_on_a=list(deepest[5]),point_on_b=list(deepest[6]),normal_on_b=list(deepest[7]))
        result.update(concave_static_a=bool(concave_a),convex_dynamic_b=True,cross_body=True,
                      collision_margins_m=margins,query_radius_m=.1)
        return result
    finally:p.disconnect(client)


def triangle_backend_controls(margin_m=None):
    import pybullet as p
    rows=[]
    for concave in (False,True):
        for center,expected in ((.012,-.003),(.030,.015)):
            client=p.connect(p.DIRECT)
            try:
                mesh=trimesh.creation.box(extents=[.02]*3)
                a_shape=p.createCollisionShape(p.GEOM_MESH,vertices=mesh.vertices.tolist(),
                    indices=mesh.faces.reshape(-1).tolist(),flags=p.GEOM_FORCE_CONCAVE_TRIMESH if concave else 0,
                    physicsClientId=client)
                b_shape=p.createCollisionShape(p.GEOM_SPHERE,radius=.005,physicsClientId=client)
                a=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=a_shape,physicsClientId=client)
                b=p.createMultiBody(baseMass=1,baseCollisionShapeIndex=b_shape,basePosition=[center,0,0],physicsClientId=client)
                if margin_m is not None:
                    for body in (a,b):p.changeDynamics(body,-1,collisionMargin=float(margin_m),physicsClientId=client)
                q=_summary(p.getClosestPoints(a,b,distance=.1,physicsClientId=client))
                actual=q['minimum_signed_distance_m']
                # Bullet mesh margin may contribute up to 1 mm, explicitly bounded.
                passed=actual is not None and np.sign(actual)==np.sign(expected) and abs(actual-expected)<=.00101
                rows.append(dict(concave_static_a=concave,expected_surface_distance_m=expected,
                    **q,pass_check=bool(passed),margin_allowance_m=.00101))
            finally:p.disconnect(client)
    return rows


def file_mesh_controls(folder, margin):
    folder.mkdir(exist_ok=False)
    path=folder/'box.stl'
    trimesh.creation.box(extents=[.02]*3).export(path)
    rows=[]
    for concave in (False,True):
        for distance in (.015,.04):
            a=np.eye(4);b=np.eye(4);b[0,3]=distance
            row=query_mesh_pair(path,a,path,b,concave_a=concave,margin_m=margin)
            expected=distance-.02-2*margin
            actual=row['minimum_signed_distance_m']
            row.update(expected_box_backend_distance_m=expected,
                       pass_check=actual is not None and abs(actual-expected)<1e-7,
                       oracle_scope='AXIS_ALIGNED_BOX_FIXTURE_ONLY_NOT_GENERAL_MESH_DEPTH')
            rows.append(row)
    if not all(x['pass_check'] for x in rows):
        raise ValueError('file-backed common-margin canary failed; do not interpret asset queries')
    data=path.read_bytes()
    return dict(rows=rows,fixture=dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))


def fixed_fk_pair_probe(models, root, pin, *, same_margin=False, control_root=None):
    from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
    controls=triangle_backend_controls(margin_m=.001 if same_margin else None)
    mesh_controls=file_mesh_controls(control_root,.001) if same_margin else None
    if not all(x['pass_check'] for x in controls):
        raise ValueError('cross-body triangle query failed penetration/separation controls')
    pinned={x['path']:x for x in pin['files']};verified={};rows=[]
    for side,model in zip(('left','right'),models,strict=True):
        tree=ET.parse(model.path).getroot();links={x.get('name'):x for x in tree.findall('link')}
        moving=[j for j in model.joints if j.joint_type!='fixed']
        transforms=forward_kinematics(model,{j.name:(j.lower+j.upper)*.5 for j in moving})
        prefix='hand_l' if side=='left' else 'hand_r'
        pairs=[(f'{prefix}_base_link',f'{prefix}_{finger}_link2') for finger in ('thumb','index','middle','ring','pinky')]
        pairs.append((f'{prefix}_thumb_link3',f'{prefix}_thumb_link5'))
        for a,b in pairs:
            mesh_paths=[];refs=[]
            for name in (a,b):
                node=links[name].find('collision');shape=node.find('geometry/mesh');origin=node.find('origin')
                xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
                rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
                scale=np.fromstring(shape.get('scale','1 1 1'),sep=' ')
                if not (np.array_equal(xyz,np.zeros(3)) and np.array_equal(rpy,np.zeros(3)) and np.array_equal(scale,np.ones(3))):
                    raise ValueError('this bounded probe requires pinned zero-origin unscaled mesh')
                path=(model.path.parent/shape.get('filename')).resolve(strict=True)
                relative=str(path.relative_to(root));ref=pinned[relative]
                if relative not in verified:
                    data=path.read_bytes()
                    if len(data)!=ref['bytes'] or hashlib.sha256(data).hexdigest()!=ref['sha256']:
                        raise ValueError('mesh pin drift')
                    verified[relative]=True
                mesh_paths.append(path);refs.append(ref)
            margin=.001 if same_margin else None
            baseline=query_mesh_pair(mesh_paths[0],transforms[a],mesh_paths[1],transforms[b],concave_a=False,margin_m=margin)
            triangle=query_mesh_pair(mesh_paths[0],transforms[a],mesh_paths[1],transforms[b],concave_a=True,margin_m=margin)
            surface={}
            if same_margin and 'thumb' in b:
                reverse=query_mesh_pair(mesh_paths[1],transforms[b],mesh_paths[0],transforms[a],concave_a=True,margin_m=.001)
                surface['reverse_triangle_b_convex_a']=reverse
                probes=[]
                for direction,result,order in [('forward',triangle,(0,1)),('reverse',reverse,(1,0))]:
                    witness=result.get('closest_witness_world')
                    if witness is None:continue
                    for role,index in zip(('a','b'),order,strict=True):
                        name=(a,b)[index];transform=transforms[name]
                        local=transform[:3,:3].T@(np.asarray(witness['point_on_'+role])-transform[:3,3])
                        mesh=trimesh.load(mesh_paths[index],force='mesh',process=False)
                        closest,distance,face=trimesh.proximity.closest_point_naive(mesh,local[None])
                        probes.append(dict(direction=direction,query_role=role,link=name,witness_local_m=local.tolist(),
                            nearest_original_triangle_local_m=closest[0].tolist(),surface_distance_m=float(distance[0]),face_index=int(face[0])))
                surface['original_surface_witness_checks']=probes
            rows.append(dict(side=side,pair=[a,b],mesh_refs=refs,configuration='MID_LIMITS_FIXED_FK',
                transform_a=transforms[a].tolist(),transform_b=transforms[b].tolist(),
                convex_a_convex_b=baseline,concave_static_a_convex_b=triangle,thumb_surface_check=surface))
    return dict(backend_controls=controls,pairs=rows,production_geometry_modified=False,
                common_margin_m=.001 if same_margin else None,
                file_backed_controls=mesh_controls,
                quality_adoption=False,limitations=[
                    'No concave-concave or same-body concave query is used.',
                    'Base STL is not watertight; surface queries cannot certify containment or physical clearance.',
                    'B remains a convex approximation. An empty query is not accepted as proof of no collision.',
                    'This canary does not modify exclusion rules, solver targets or quality thresholds.'])
