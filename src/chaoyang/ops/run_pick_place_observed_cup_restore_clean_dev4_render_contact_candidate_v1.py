"""Root-registered finite DEV4 fresh Render/Contact from actual C3; no q/model change."""
import sys
from pathlib import Path
from chaoyang.governance.exact78_operation_gate import authorize_operation_entry
OPERATION='run_pick_place_observed_cup_restore_clean_dev4_render_contact_candidate_v1'
def main(argv=None):
 args=list(sys.argv[1:] if argv is None else argv)
 authorize_operation_entry(OPERATION,args)
 if len(args)!=2 or args[0]!='--render-contact-spec':raise ValueError('CUP_EXACT_TWO_ARGS')
 from chaoyang.pick_place_cup_restore_render_contact_v1.contract import require_registered_context
 from chaoyang.pick_place_cup_restore_render_contact_v1.driver import run
 return run(require_registered_context(Path(args[1]),args))
if __name__=='__main__':raise SystemExit(main())
