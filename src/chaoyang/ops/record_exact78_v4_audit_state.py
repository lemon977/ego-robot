#!/usr/bin/env python3
"""Attach the immutable Exact78 cohort audit to its V4 lane state."""
from pathlib import Path
from chaoyang.governance.common import REPO_ROOT, atomic_json, artifact_ref, load_json, now_iso
def main():
 root=REPO_ROOT/'_run/current/four_stream_full_pipeline_v4/attempts/attempt_0001/lanes/exact78'; audit=root/'EXACT78_COHORT_AUDIT.json'; state=root/'STATE.json'; s=load_json(state); s.update(status='READY_CPU',current_action='FROZEN_COHORT_VERIFIED_PAIR_PRODUCTION_PENDING',blocker=None,updated_at=now_iso(),latest_artifacts=list({x['path']:x for x in [*s.get('latest_artifacts',[]),artifact_ref(audit)]}.values())); atomic_json(state,s); print({'status':'PASSED','audit':str(audit)})
if __name__=='__main__': main()
