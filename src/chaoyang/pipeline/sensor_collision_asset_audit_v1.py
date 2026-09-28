"""Small CPU asset audit. Convex/concave probes are not physical certification."""
from collections import deque
import hashlib
import io
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def joint_path(tree, start, finish):
    graph = {}
    for joint in tree.findall('joint'):
        a, b = joint.find('parent').get('link'), joint.find('child').get('link')
        item = dict(joint=joint.get('name'), type=joint.get('type'), parent=a, child=b)
        graph.setdefault(a, []).append((b,item))
        graph.setdefault(b, []).append((a,item))
    queue = deque([(start, [])]); seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == finish:
            return path
        for neighbor, item in graph.get(node,[]):
            if neighbor not in seen:
                seen.add(neighbor); queue.append((neighbor,path+[item]))
    raise ValueError('disconnected link pair')


def mesh_summary(mesh):
    hull = mesh.convex_hull
    volume = float(abs(mesh.volume))
    return dict(vertices=len(mesh.vertices), faces=len(mesh.faces),
                bounds_m=mesh.bounds.tolist(), watertight=bool(mesh.is_watertight),
                signed_volume_m3=float(mesh.volume), convex_hull_volume_m3=float(hull.volume),
                volume_ratio_mesh_to_hull=volume/float(hull.volume) if hull.volume else None,
                caveat='Volume ratio alone does not prove an illegal collision or a cavity at the contact.')


def primitive_self_collision_fixture():
    """Two non-adjacent box links with known penetration/separation; no masks."""
    import pybullet as p
    results = []
    for distance, expected in ((.01,True),(.05,False)):
        client = p.connect(p.DIRECT)
        try:
            box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.01]*3,physicsClientId=client)
            body = p.createMultiBody(baseMass=0,baseCollisionShapeIndex=box,
                linkMasses=[.1,.1],linkCollisionShapeIndices=[-1,box],linkVisualShapeIndices=[-1,-1],
                linkPositions=[[0,0,0],[distance,0,0]],linkOrientations=[[0,0,0,1]]*2,
                linkInertialFramePositions=[[0,0,0]]*2,linkInertialFrameOrientations=[[0,0,0,1]]*2,
                linkParentIndices=[0,1],linkJointTypes=[p.JOINT_REVOLUTE,p.JOINT_REVOLUTE],
                linkJointAxis=[[0,0,1]]*2,
                flags=p.URDF_USE_SELF_COLLISION|p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                physicsClientId=client)
            p.performCollisionDetection(physicsClientId=client)
            distances = [float(c[8]) for c in p.getContactPoints(body,body,physicsClientId=client)
                         if set((c[3],c[4])) == {-1,1}]
            actual = any(d < 0 for d in distances)
            results.append(dict(distance_m=distance, expected_penetration=expected,
                                actual_penetration=actual,pass_check=actual==expected,
                                contact_distances_m=distances))
        finally:
            p.disconnect(client)
    return results


def small_joint_perturbations(models):
    import pybullet as p
    output=[]
    for side,model in zip(('left','right'),models,strict=True):
        client=p.connect(p.DIRECT)
        try:
            body=p.loadURDF(str(model.path),useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION|p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                physicsClientId=client)
            info=[p.getJointInfo(body,i,physicsClientId=client) for i in range(p.getNumJoints(body,physicsClientId=client))]
            ids={j[1].decode():i for i,j in enumerate(info)}
            links={-1:p.getBodyInfo(body,physicsClientId=client)[0].decode()}
            links.update({i:j[12].decode() for i,j in enumerate(info)})
            moving=[j for j in model.joints if j.joint_type!='fixed']
            neutral={j.name:(j.lower+j.upper)*.5 for j in moving}
            prefix='hand_l' if side=='left' else 'hand_r'
            names=[f'{prefix}_{finger}_joint2' for finger in ('thumb','index','middle','ring','pinky')]+[f'{prefix}_thumb_joint5']
            cases=[('mid_limits',None,0.)]+[(name,name,delta) for name in names for delta in (-.05,.05)]
            for label,joint,delta in cases:
                q=neutral.copy()
                if joint is not None:q[joint]+=delta
                if any(not j.lower<=q[j.name]<=j.upper for j in moving):
                    raise ValueError('planned synthetic perturbation outside limits')
                for name,value in q.items():p.resetJointState(body,ids[name],value,physicsClientId=client)
                p.performCollisionDetection(physicsClientId=client)
                pairs={}
                for c in p.getContactPoints(body,body,physicsClientId=client):
                    if c[8]<0:
                        key=tuple(sorted((links[c[3]],links[c[4]])))
                        pairs[key]=max(pairs.get(key,0.),-float(c[8]))
                output.append(dict(side=side,case=label,delta_rad=delta,
                    pairs=[dict(links=list(k),penetration_m=v) for k,v in sorted(pairs.items())]))
        finally:p.disconnect(client)
    return output


def audit_assets(models, root, pin):
    """Read pinned STL geometry, audit adjacency and fixed-axis tip proxies."""
    table = {row['path']:row for row in pin['files']}
    reports=[]
    for side, model in zip(('left','right'),models,strict=True):
        tree=ET.parse(model.path).getroot()
        links={link.get('name'):link for link in tree.findall('link')}
        prefix='hand_l' if side=='left' else 'hand_r'
        pairs=[(f'{prefix}_base_link',f'{prefix}_{finger}_link2')
               for finger in ('thumb','index','middle','ring','pinky')]
        pairs.append((f'{prefix}_thumb_link3',f'{prefix}_thumb_link5'))
        cached={}
        def geometry(link_name, kind):
            node=links[link_name].find(kind)
            if node is None: return None
            shape=node.find('geometry/mesh')
            if shape is None: raise ValueError('registered asset must use mesh')
            path=(model.path.parent/shape.get('filename')).resolve(strict=True)
            relative=str(path.relative_to(root))
            if relative not in table: raise ValueError('mesh outside pinned asset closure')
            data=path.read_bytes(); ref=table[relative]
            if len(data)!=ref['bytes'] or hashlib.sha256(data).hexdigest()!=ref['sha256']:
                raise ValueError('asset mesh bytes/SHA drift')
            if relative not in cached:
                cached[relative]=trimesh.load(io.BytesIO(data),file_type='stl',force='mesh',process=True)
            origin=node.find('origin')
            return dict(path=relative,bytes=len(data),sha256=ref['sha256'],
                        xyz=origin.get('xyz','0 0 0') if origin is not None else '0 0 0',
                        rpy=origin.get('rpy','0 0 0') if origin is not None else '0 0 0',
                        scale=shape.get('scale','1 1 1'))
        rows=[]
        for a,b in pairs:
            path=joint_path(tree,a,b)
            node_names=list(dict.fromkeys([a]+[x['parent'] for x in path]+[x['child'] for x in path]))
            rows.append(dict(pair=[a,b],joint_distance=len(path),joint_chain=path,
                             links={name:dict(collision=geometry(name,'collision'),visual=geometry(name,'visual'))
                                    for name in node_names}))
        base=geometry(f'{prefix}_base_link','collision')
        base_mesh=mesh_summary(cached[base['path']])
        tips=[]
        for finger in ('thumb','index','middle','ring','pinky'):
            number=6 if finger=='thumb' else 4
            link=f'{prefix}_{finger}_link{number}'
            ref=geometry(link,'visual');mesh=cached[ref['path']]
            v=np.asarray(mesh.vertices);axis=0 if finger=='thumb' else 2;sign=1 if finger=='thumb' else -1
            projection=sign*v[:,axis]; chosen=v[projection>=projection.max()-.0005]
            opposite=v[projection<=projection.min()+.0005]
            tip=chosen.mean(axis=0);other=opposite.mean(axis=0)
            singular=np.linalg.svd(v-v.mean(axis=0),full_matrices=False)[2][0]
            incoming=next(j for j in tree.findall('joint') if j.find('child').get('link')==link)
            tips.append(dict(finger=finger,terminal_link=link,mesh=ref,
                             fixed_axis=['x','y','z'][axis],sign=sign,tip_local_m=tip.tolist(),
                             opposite_extreme_local_m=other.tolist(),proxy_separation_m=float(np.linalg.norm(tip-other)),
                             pca_absolute_axis_alignment=float(abs(singular[axis])),
                             terminal_joint_axis=incoming.find('axis').get('xyz'),
                             authority='MESH_EXTREME_PROXY_NOT_INDEPENDENT_FINGERTIP_OR_PAD_MEASUREMENT'))
        reports.append(dict(side=side,pairs=rows,base_mesh=base_mesh,tip_proxies=tips))
    fixtures=primitive_self_collision_fixture()
    if not all(row['pass_check'] for row in fixtures):
        raise ValueError('non-adjacent collision checker positive/negative fixture failed')
    return dict(sides=reports,primitive_fixtures=fixtures,
                single_joint_perturbations=small_joint_perturbations(models),
                changes_to_geometry_or_exclusions=False,
                limitations=['No exact triangle-triangle collision oracle is installed.',
                             'Convex volume discrepancy is diagnostic, not proof of which pairs are false positives.',
                             'No learned optimizer or real-data q was changed.'])
