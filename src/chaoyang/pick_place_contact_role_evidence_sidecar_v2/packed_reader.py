"""One verified file descriptor; four packed members, one native at a time."""
from __future__ import annotations
import hashlib,os,zipfile
from pathlib import Path
import numpy as np
FIELDS=('sam_core','equipment','protected_visible','cup_observed')
def need(v,s):
 if not v:raise ValueError(s)
def identity(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
def member_header(stream,info,n):
 version=np.lib.format.read_magic(stream)
 if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
 elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
 else:raise ValueError('ROLE_NPY_UNSUPPORTED_VERSION')
 need(shape==(n,38400) and order is False and dtype==np.dtype(np.uint8),'ROLE_PACKED_UINT8_BIG_SDK_DTYPE_SHAPE')
 need(info.file_size==stream.tell()+n*38400,'ROLE_PACKED_MEMBER_EXACT_SIZE')
 return shape
class PackedFrames:
 def __init__(self,ref,n,budget=lambda:None):
  self.ref=ref;self.n=n;self.budget=budget;self.members=[];self.file=None;self.zip=None;self.before=None
  try:
   self.file=Path(ref['path']).open('rb');self.before=identity(os.fstat(self.file.fileno()))
   need(self.before[2]==ref['bytes'],'ROLE_MASK_BYTES')
   h=hashlib.sha256();count=0
   for b in iter(lambda:self.file.read(1<<20),b''):budget();h.update(b);count+=len(b)
   need(count==ref['bytes'] and h.hexdigest()==ref['sha256'],'ROLE_OFFICIAL_GPU_MASK_SHA')
   self.file.seek(0);self.zip=zipfile.ZipFile(self.file)
   names=self.zip.namelist();need(len(names)==len(set(names)),'ROLE_DUPLICATE_ZIP_MEMBER')
   for field in FIELDS:
    info=self.zip.getinfo(field+'.npy');s=self.zip.open(info);self.members.append(s);member_header(s,info,n)
  except BaseException:self.close();raise
 def __iter__(self):
  for i in range(self.n):
   self.budget();out={}
   for k,s in zip(FIELDS,self.members):
    b=s.read(38400);need(len(b)==38400,'ROLE_MASK_FRAME_SHORT')
    out[k]=np.unpackbits(np.frombuffer(b,dtype=np.uint8),count=307200,bitorder='big').reshape(480,640).astype(bool)
   yield i,out
  for s in self.members:need(s.read(1)==b'','ROLE_MASK_CRC_EOF_OR_TAIL')
  need(identity(os.fstat(self.file.fileno()))==self.before and identity(Path(self.ref['path']).stat())==self.before,'ROLE_MASK_SAME_FD_AFTER_DRIFT')
 def close(self):
  for s in self.members:
   try:s.close()
   except Exception:pass
  if self.zip is not None:self.zip.close()
  if self.file is not None:self.file.close()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
