"""Static-only signed robot asset closure. No NumPy, Robot, FK, mesh geometry or renderer."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
ROBOT_PIN='assets/robot/ROBOT_ASSET_PIN.json'
FLANGE='assets/robot/V2相机固定法兰PRO版本.STL'
PROXY='assets/robot/hardware_handoff/kaihand_flange_adapter_v1/excluded_reference_only/MOUNT_VISUAL_PROXY_CONTRACT.json'
STRICT_IO='src/chaoyang/ops/immutable_artifact_io.py'
STRICT_IO_SHA='c0cdcea8b541083de65b2ccd7d1011305e2000bbf49694eef601599d7fbcb5de'
def need(x,m):
 if not x:raise ValueError(m)
def ref(path):
 p=Path(path);h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8388608):h.update(b)
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def project_file(root,path):
 root=Path(root);p=Path(path)
 need(p.is_absolute() and '..' not in p.parts and p.is_relative_to(root),'FK2_ASSET_PROJECT_TYPED_PATH')
 for a in (p,*p.parents):
  need(not a.is_symlink(),'FK2_ASSET_NO_SYMLINK')
  if a==root:break
 need(p.is_file(),'FK2_ASSET_REGULAR_REQUIRED')
 return p
def exact(root,pin):
 need(isinstance(pin,dict) and set(pin)=={'path','bytes','sha256'} and type(pin['bytes']) is int and pin['bytes']>=0 and isinstance(pin['sha256'],str) and len(pin['sha256'])==64 and all(c in '0123456789abcdef' for c in pin['sha256']),'FK2_ASSET_TYPED_REF')
 p=project_file(root,pin['path']);need(ref(p)==pin,'FK2_ASSET_SHA_OR_SIZE_DRIFT:'+str(p));return p
def resolve(root,urdf,filename):
 if filename.startswith('package://marvin_description/'):candidate=root/'assets/robot/tianji/marvin_description'/filename.removeprefix('package://marvin_description/')
 elif filename.startswith('package://'):raise ValueError('FK2_UNSUPPORTED_MESH_PACKAGE_URI')
 else:candidate=urdf.parent/filename
 # Same frozen resolver's normalization plus stricter symlink rejection.
 need(not candidate.is_symlink(),'FK2_MESH_SYMLINK')
 p=candidate.resolve(strict=True);need(p.is_relative_to(root/'assets/robot'),'FK2_MESH_OUTSIDE_ROBOT')
 return project_file(root,p)
def static_closure(root):
 root=Path(root);pinpath=project_file(root,root/ROBOT_PIN);pin=json.loads(pinpath.read_text())
 need(pin.get('status')=='T0_IDENTITY_PIN_NOT_EXECUTION_AUTHORIZATION','FK2_IDENTITY_PIN_STATUS')
 files=pin.get('files');need(isinstance(files,list) and len({x['path'] for x in files})==len(files),'FK2_PIN_FILE_UNIQUE')
 table={x['path']:x for x in files};urdfs=pin.get('urdfs');need(isinstance(urdfs,list) and len(urdfs)==3,'FK2_EXACT_3_URDF')
 assets={};components=[];meshes=set()
 for row in urdfs:
  path=project_file(root,root/row['path']);pr=ref(path);entry=table.get(row['path'])
  need(entry is not None and pr['bytes']==entry['bytes'] and pr['sha256']==entry['sha256'],'FK2_PIN_URDF_DRIFT')
  xml=ET.fromstring(path.read_bytes());links=[v.get('name') for v in xml.findall('link')]
  need(links and all(links) and len(set(links))==len(links),'FK2_URDF_LINK_IDENTITIES')
  children=[j.find('child').get('link') for j in xml.findall('joint')];roots=set(links)-set(children)
  need(len(roots)==1,'FK2_URDF_UNIQUE_ROOT')
  assets[str(path)]=pr;count=0
  for link in xml.findall('link'):
   for v in link.findall('visual'):
    mesh=v.find('geometry/mesh');need(mesh is not None and bool(mesh.get('filename')),'FK2_VISUAL_MESH_REQUIRED')
    p=resolve(root,path,mesh.get('filename'));rel=p.relative_to(root).as_posix();value=ref(p);entry=table.get(rel)
    need(entry is not None and value['bytes']==entry['bytes'] and value['sha256']==entry['sha256'],'FK2_PIN_MESH_DRIFT')
    assets[str(p)]=value;meshes.add(str(p));count+=1
  components.append({'path':str(path),'root_link':next(iter(roots)),'visual_count':count})
 need({x['root_link'] for x in components}=={'base_link','hand_l_base_link','hand_r_base_link'} and sum(x['visual_count'] for x in components)==len(meshes)==63 and len(assets)==66,'FK2_ACTUAL_3_MODEL_63_MESH_CLOSURE')
 for raw in [FLANGE,PROXY]:assets[str(root/raw)]=ref(project_file(root,root/raw))
 need(len(assets)==68,'FK2_HELPER_ASSETS68')
 data=sorted([*assets.values(),ref(pinpath)],key=lambda x:x['path'])
 io=ref(project_file(root,root/STRICT_IO));need(io['sha256']==STRICT_IO_SHA,'FK2_FIXED_STRICT_IO_CODE_DRIFT')
 return {'asset_input_refs':data,'renderer_helper_asset_refs':sorted(assets.values(),key=lambda x:x['path']),'fixed_existing_runtime_code':[io],'components':components}
def validate_contract_assets(contract,root,verify=True):
 data=contract.get('robot_asset_input_refs');runtime=contract.get('fixed_existing_runtime_code')
 need(isinstance(data,list) and len(data)==69 and len({x['path'] for x in data})==69,'FK2_EXPLICIT_69_ASSET_INPUT_REFS')
 need(isinstance(runtime,list) and len(runtime)==1 and runtime[0]['path']==str(Path(root)/STRICT_IO) and runtime[0]['sha256']==STRICT_IO_SHA,'FK2_FIXED_EXISTING_IO_RUNTIME1')
 if verify:
  observed=static_closure(root)
  need(observed['asset_input_refs']==data and observed['fixed_existing_runtime_code']==runtime,'FK2_STATIC_PRE_REGISTER_ASSET_CLOSURE_DRIFT')
 return data,runtime
def validate_packet_asset_inputs(packet,contract,root,verify=True):
 data,code=validate_contract_assets(contract,root,verify)
 need(all(x in packet['inputs']['files'] for x in data),'FK2_PACKET_SIGNED_ASSETS_MISSING')
 need(all(x in packet['runtime_code'] for x in code),'FK2_PACKET_STRICT_IO_CODE_UNBOUND')
 return True
def validate_live_renderer_assets(assets,packet,contract,root):
 expected=contract['robot_asset_input_refs'];need(len(assets)==68 and len({x['path'] for x in assets})==68,'FK2_ACTUAL_HELPER_ASSETS68')
 need(sorted(assets,key=lambda x:x['path'])==[x for x in expected if x['path']!=str(Path(root)/ROBOT_PIN)],'FK2_ACTUAL_MODEL_ASSET_IDENTITY_DIFF')
 need(all(x in packet['inputs']['files'] for x in assets),'FK2_ASSETS_NOT_SIGNED_PACKET_INPUTS')
 return True
