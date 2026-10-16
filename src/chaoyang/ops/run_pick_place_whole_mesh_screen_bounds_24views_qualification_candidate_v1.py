"""Future maintained q-frozen Render qualifier. No implicit Task or Root authority."""
import argparse
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--maintenance-spec',required=True);args=parser.parse_args(argv)
    from chaoyang.whole_mesh_screenbounds_qualification_v1.contract import context_from_spec,ROOT,TASK,read
    from chaoyang.exact78_baseline.stage_reuse import current_producer
    from chaoyang.whole_mesh_screenbounds_qualification_v1.driver import run_qualification
    from pathlib import Path
    from chaoyang.whole_mesh_screenbounds_qualification_v1.contract import OP,need
    need(Path(__file__).resolve()==ROOT/'src/chaoyang/ops'/(OP+'.py'),'SCREEN_CANONICAL_OPERATION_ENTRY_REQUIRED')
    context=context_from_spec(args.maintenance_spec,ROOT)
    producer=current_producer(ROOT)
    context['packet_ref']=producer['task_packet']
    context['packet']=read(producer['task_packet'],ROOT)
    return run_qualification(context,producer,ROOT)
if __name__=='__main__':raise SystemExit(main())
