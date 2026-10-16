"""Governed finite original-DEV mask-role sidecar, no physical contact authority."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--role-spec',type=Path,required=True)
 actual=list(sys.argv[1:] if argv is None else argv)
 if len(actual)!=2 or actual[0]!='--role-spec':p.error('exactly --role-spec SPEC is required; no duplicate or extra arguments')
 a=p.parse_args(actual)
 from chaoyang.pick_place_contact_role_evidence_sidecar_v1.contract import require_registered_context
 exact=['--role-spec',str(a.role_spec)]
 ctx=require_registered_context(a.role_spec,exact)
 from chaoyang.pick_place_contact_role_evidence_sidecar_v1.driver import run
 return run(ctx)
if __name__=='__main__':raise SystemExit(main())
