"""Single-FD SHA verified uint8 NPY streams; project-local streaming outputs."""
from __future__ import annotations
from pathlib import Path
import hashlib,os,zipfile
import numpy as np
from .helper import FIELDS
def need(v,s):
 if not v:raise ValueError(s)
def identity(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def exactread(s,n):
 b=s.read(n);need(len(b)==n,'CUP_STREAM_SHORT');return b
class Frames:
 def __init__(self,pin,shapes,tick=lambda:None):
  self.pin=pin;self.tick=tick;self.fd=None;self.z=None;self.members={};self.shapes=shapes;self.count=0;self.finished=False
  try:
   need(set(pin)=={'path','bytes','sha256'} and type(pin['bytes']) is int,'CUP_STREAM_PIN')
   pp=Path(pin['path']);root=Path('/path/to/chaoyang')
   need(pp.is_absolute() and '..' not in pp.parts and pp.is_relative_to(root),'CUP_STREAM_PROJECT_ONLY')
   for parent in (pp,*pp.parents):
    need(not parent.is_symlink(),'CUP_STREAM_SYMLINK')
    if parent==root:break
   self.fd=os.fdopen(os.open(pp,os.O_RDONLY|os.O_NOFOLLOW),'rb');self.before=identity(os.fstat(self.fd.fileno()))
   need(self.before[2]==pin['bytes'],'CUP_STREAM_SIZE')
   h=hashlib.sha256()
   for b in iter(lambda:self.fd.read(1<<20),b''):tick();h.update(b)
   need(h.hexdigest()==pin['sha256'],'CUP_STREAM_SAME_FD_SHA')
   self.fd.seek(0);self.z=zipfile.ZipFile(self.fd)
   infos=self.z.infolist();need(len(infos)==len(shapes) and set(self.z.namelist())=={k+'.npy' for k in shapes},'CUP_EXACT_NPZ_MEMBERS')
   for k,shape in shapes.items():
    info=self.z.getinfo(k+'.npy');need(info.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) and not(info.flag_bits&1),'CUP_ZIP_ENCODING')
    s=self.z.open(info);self.members[k]=s;ver=np.lib.format.read_magic(s)
    if ver==(1,0):sh,order,dtype=np.lib.format.read_array_header_1_0(s)
    elif ver==(2,0):sh,order,dtype=np.lib.format.read_array_header_2_0(s)
    else:raise ValueError('CUP_NPY_VERSION')
    need(sh==shape and order is False and dtype==np.dtype('uint8'),'CUP_NPY_EXACT_UINT8_SHAPE')
    need(info.file_size==s.tell()+int(np.prod(shape,dtype=np.int64)),'CUP_NPY_EXACT_BYTES')
   ns={sh[0] for sh in shapes.values()};need(len(ns)==1,'CUP_STREAM_N_MISMATCH');self.n=ns.pop()
  except BaseException:self.close();raise
 def one(self):
  need(self.count<self.n,'CUP_STREAM_OVERREAD');self.tick();out={}
  for k,shape in self.shapes.items():
   out[k]=np.frombuffer(exactread(self.members[k],int(np.prod(shape[1:],dtype=np.int64))),np.uint8).reshape((1,)+shape[1:])
  self.count+=1;return out
 def finish(self):
  need(self.count==self.n,'CUP_STREAM_INCOMPLETE')
  for s in self.members.values():self.tick();need(s.read(1)==b'','CUP_NPY_TRAILING_OR_CRC')
  need(identity(os.fstat(self.fd.fileno()))==self.before==identity(Path(self.pin['path']).stat()),'CUP_STREAM_INPUT_CHANGED')
  self.finished=True
 def close(self):
  for s in self.members.values():
   try:s.close()
   except Exception:pass
  if self.z is not None:self.z.close()
  if self.fd is not None:self.fd.close()
 def __enter__(self):return self
 def __exit__(self,*a):self.close()
class Writer:
 """Writes per-member project-local NPY spool. No whole-N materialization."""
 def __init__(self,path,shapes,tick=lambda:None):
  self.path=Path(path);self.shapes=shapes;self.tick=tick;self.count=0;self.files={}
  self.cache=self.path.parent/(self.path.name+'.stream_cache');self.cache.mkdir()
  try:
   for k,shape in shapes.items():
    f=(self.cache/(k+'.npy')).open('xb');self.files[k]=f
    np.lib.format.write_array_header_2_0(f,dict(descr=np.dtype('uint8').str,fortran_order=False,shape=shape))
  except BaseException:self.close();raise
 def one(self,arrays):
  self.tick();need(set(arrays)==set(self.shapes),'CUP_WRITER_FIELDS')
  for k,shape in self.shapes.items():
   x=arrays[k];need(x.dtype==np.uint8 and x.shape==(1,)+shape[1:],'CUP_WRITER_UINT8_SHAPE');need(self.count<shape[0],'CUP_WRITER_OVERFLOW')
   self.files[k].write(x.tobytes(order='C'))
  self.count+=1
 def finish(self):
  need(all(self.count==s[0] for s in self.shapes.values()),'CUP_WRITER_PARTIAL_NOT_COMPLETE')
  self.close()
  with self.path.open('xb') as fd,zipfile.ZipFile(fd,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
   for k in self.shapes:
    self.tick()
    with (self.cache/(k+'.npy')).open('rb') as src,z.open(k+'.npy','w',force_zip64=True) as dst:
     for b in iter(lambda:src.read(1<<20),b''):self.tick();dst.write(b)
 def close(self):
  for f in self.files.values():
   if not f.closed:f.close()
