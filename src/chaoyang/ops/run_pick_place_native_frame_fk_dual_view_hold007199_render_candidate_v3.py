"""Finite q-bitfrozen renderer operation; the actual CLI main and fence precede the consumer."""
def main(argv=None):
 import sys
 from chaoyang.governance.exact78_operation_gate import authorize_operation_entry
 from chaoyang.governance.native_frame_fk_maintenance_contract_v3 import approved_context,OPS,mode
 args=list(sys.argv[1:] if argv is None else argv)
 operation=OPS[1]
 authorize_operation_entry(operation,args)
 if len(args)!=2 or args[0]!='--maintenance-spec':raise ValueError('FK_EXACT_TWO_ARGUMENTS')
 context=approved_context(args[1])
 if mode(context['spec'])!=1:raise ValueError('FK_EXPLICIT_OPERATION_CONTEXT')
 from chaoyang.native_frame_fk_render_candidate_v3.driver import run
 return run(context)
if __name__=='__main__':raise SystemExit(main())
