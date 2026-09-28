"""Fail-closed offline admission ledger. No renderer, model or solver execution.

Caller must verify referenced evidence bytes against the explicit verified_refs
read-set; this pure consumer does not claim cryptographic signature authenticity.
"""
from dataclasses import asdict
import re
from .final_frame_disposition_v1 import SideEvidence, MotionSource, decide_frame

SCHEMA = 'MOTION_FOUR_CERTIFICATE_LEDGER_V1'
KINDS = ('motion', 'visibility', 'render', 'compositor')
SIDES = ('left', 'right')

def generate(*, session_id, source_sha256, frame_ids, source_rows, certificates, verified_refs):
    if not isinstance(session_id, str) or not session_id or not re.fullmatch('[0-9a-f]{64}', source_sha256):
        raise ValueError('SESSION_AND_SOURCE_SHA_REQUIRED')
    if not frame_ids or any(type(x) is not int or x < 0 for x in frame_ids) or any(b <= a for a,b in zip(frame_ids,frame_ids[1:])):
        raise ValueError('STRICT_FRAME_TIMELINE_REQUIRED')
    expected = {(f,s) for f in frame_ids for s in SIDES}
    def indexed(rows):
        out = {}
        for r in rows:
            key = (r.get('frame_id'),r.get('side'))
            if key not in expected or key in out: raise ValueError('DUPLICATE_OR_FOREIGN_FRAME_SIDE')
            out[key] = r
        return out
    sources = indexed(source_rows)
    if set(sources) != expected: raise ValueError('SOURCE_TIMELINE_INCOMPLETE')
    if set(certificates) - set(KINDS): raise ValueError('UNKNOWN_CERTIFICATE_KIND')
    tables = {k:indexed(certificates.get(k,[])) for k in KINDS}
    trusted = {(r['path'],r['sha256']) for r in verified_refs if r.get('bytes_verified') is True and r.get('stable_read') is True}
    def check(kind,key):
        r=tables[kind].get(key)
        if r is None: return 'UNKNOWN','CERTIFICATE_MISSING'
        if r.get('session_id') != session_id or r.get('source_sha256') != source_sha256:
            raise ValueError('CERTIFICATE_SOURCE_BINDING_MISMATCH')
        status=r.get('status')
        if status not in ('PASS','FAIL','UNKNOWN'): raise ValueError('UNKNOWN_CERTIFICATE_STATUS')
        refs=r.get('evidence_refs',[])
        if not refs or any((e.get('path'),e.get('sha256')) not in trusted or e.get('sha256') == source_sha256 for e in refs):
            return 'UNKNOWN','INDEPENDENT_VERIFIED_EVIDENCE_MISSING'
        if r.get('reviewed_independent') is not True: return 'UNKNOWN','INDEPENDENT_REVIEW_MISSING'
        if status != 'PASS': return status,r.get('reason') or 'EVIDENCE_NOT_PASS'
        if kind == 'motion' and any(r.get('capabilities',{}).get(c) != 'PASS' for c in ('position','rotation','fingers')):
            return 'UNKNOWN','MOTION_CAPABILITY_NOT_PASSED'
        if kind == 'visibility' and type(r.get('replacement_required')) is not bool:
            return 'UNKNOWN','VISIBILITY_UNDETERMINED'
        return 'PASS','VERIFIED_DECLARED_EVIDENCE'
    rows=[]; counts={'ROBOT':0,'PARTIAL_PRESERVE':0,'ORIGINAL_FRAME_FALLBACK':0}; gaps=[]
    for f in frame_ids:
        evidence=[]; per_side=[]; compositor=True
        for side in SIDES:
            key=(f,side); src=sources[key]
            if type(src.get('valid')) is not bool: raise ValueError('SOURCE_VALID_BOOLEAN_REQUIRED')
            try: source=MotionSource(src['kind'])
            except (KeyError,ValueError): raise ValueError('SOURCE_KIND_REQUIRED')
            checked={k:check(k,key) for k in KINDS}
            required=tables['visibility'].get(key,{}).get('replacement_required',True) if checked['visibility'][0]=='PASS' else True
            compositor=compositor and checked['visibility'][0]=='PASS' and checked['compositor'][0]=='PASS'
            failures=[k+':'+reason for k,(status,reason) in checked.items() if status!='PASS']
            gaps.extend({'frame_id':f,'side':side,'certificate':k,'status':status,'reason':reason} for k,(status,reason) in checked.items() if status!='PASS')
            evidence.append(SideEvidence(f,side,required,source,src['valid'],checked['motion'][0]=='PASS',checked['render'][0]=='PASS',source_sha256,';'.join(failures) or None))
            per_side.append({'side':side,'source_kind':source.value,'source_valid':src['valid'],'certificates':{k:{'status':v[0],'reason':v[1]} for k,v in checked.items()}})
        decision=decide_frame(evidence,compositor_inputs_valid=compositor)
        counts[decision.display.value]+=1
        rows.append({'frame_id':f,'evidence':per_side,'disposition':asdict(decision)})
    return {'schema_version':SCHEMA,'session_id':session_id,'source_sha256':source_sha256,'frame_count':len(frame_ids),'side_row_count':len(expected),'rows':rows,'missing_or_failed_evidence':gaps,'disposition_counts':counts,'ledger_execution':'COMPLETED','product_completed':False,'visual_review_pass':False,'quality_adopted':False,'render_executed':False,'gpu_used':False,'claim_limit':'Admission ledger only; fallback is not Robot replacement delivery. Verified refs attest bytes, not independent scientific truth.'}
