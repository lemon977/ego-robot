"""SHA/native-frame input and delivery audits for the isolated pick/place candidate."""
from pathlib import Path
import argparse,json
def main(argv=None):
    from chaoyang.governance.exact78_operation_gate import authorize_operation_entry
    authorize_operation_entry("run_exact78_pick_place_audit_candidate_v1",argv)
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-root",type=Path,default=Path("/path/to/private_storage/datasets/ego/pick_place_1008"))
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--delivery-index",type=Path)
    parser.add_argument("--resolve-time-boundaries",action="store_true")
    parser.add_argument("--output-root",type=Path,required=True)
    args=parser.parse_args(argv)
    from chaoyang.pick_place_1008.audit import run_p0_audit,audit_deliveries
    if args.resolve_time_boundaries:
        if not args.manifest:parser.error("--manifest required for time resolution")
        from chaoyang.pick_place_1008.audit import resolve_time_boundaries
        result=resolve_time_boundaries(args.manifest,args.output_root)
    elif args.delivery_index:
        if not args.manifest:parser.error("--manifest required for delivery audit")
        result=audit_deliveries(args.manifest,args.delivery_index,args.output_root)
        from chaoyang.pick_place_1008.pipeline import write
        write(args.output_root/"RESULT.json",result)
    else:result=run_p0_audit(args.source_root,args.output_root)
    print(json.dumps({"status":result["status"],"sessions":result.get("sessions"),"raw_frames":result.get("raw_frames")}),flush=True)
    return 0 if result["status"]=="PASSED" else 2
if __name__=="__main__":raise SystemExit(main())
