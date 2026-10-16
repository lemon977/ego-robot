"""Root-approved finite HOLD Render-only maintenance; existing fence is mandatory."""
def run_operation(argv=None):
 import sys
 from chaoyang.governance.exact78_operation_gate import authorize_operation_entry
 from chaoyang.governance.reference_tint_maintenance_contract_v1 import OPERATION,approved_context,import_consumer
 args=list(sys.argv[1:] if argv is None else argv)
 authorize_operation_entry(OPERATION,args) # Unchanged actual writer/descendant/startticks/epoch/token/signature gate.
 if len(args)!=2 or args[0]!='--maintenance-spec':raise ValueError('TINT_EXACT_MAINTENANCE_SPEC_ARGUMENT')
 context=approved_context(args[1]) # Exact Root descriptor, plan, operation, fixed HOLD scope and code refs.
 consumer=import_consumer(context['plan'])
 return consumer.run_operation(args) # The consumer repeats the formal gate before any candidate writes.
if __name__=='__main__':raise SystemExit(run_operation())
