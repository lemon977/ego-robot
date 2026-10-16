"""Generate source/result comparison sheets; never auto-claim AI review."""
from pathlib import Path
import json,os
import cv2,numpy as np
from .pipeline import write,read
from chaoyang.exact78_baseline.stage_reuse import ref,verify
def build_review_evidence(delivery_index,output):
    index=read(delivery_index);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);rows=[]
    for row in index["sessions"]:
        sid=row["session_id"];files=row.get("files",{})
        if not files:continue
        for pin in files.values():verify(pin)
        session=read(files["session_result"]["path"]);n=session["frame_count"]
        rf=Path(files["session_result"]["path"]).parent
        contacts=read(rf/"contact_events.json")
        states=[json.loads(line) for line in Path(files["frame_states"]["path"]).read_text().splitlines()]
        fixed=session["ai_review"]["required_fixed_frames"]
        selected=set(fixed);windows=contacts["windows"]
        for window in windows:
            a,b=int(window["start_frame"]),min(n-1,int(window["end_frame_inclusive"]))
            selected.update(range(a,b+1,5));selected.update([a,b,(a+b)//2])
        abnormal=session["ai_review"].get("abnormal_raw_frame_indices",[])
        abnormal_windows=session["ai_review"].get("required_abnormal_windows",[])
        selected.update(abnormal)
        for window in abnormal_windows:
            a,b=int(window["start_frame"]),min(n-1,int(window["end_frame_inclusive"]))
            selected.update(range(a,b+1,5));selected.update([a,b,(a+b)//2])
        selected=sorted(i for i in selected if 0<=i<n)
        folder=output/sid;folder.mkdir()
        import av
        reader=av.open(files["review"]["path"])
        reader.streams.video[0].codec_context.thread_count=2
        tiles=[];sheets=[];frame=0
        for decoded in reader.decode(video=0):
            pixels=decoded.to_ndarray(format="bgr24")
            if frame in selected:
                raw=pixels[:480,:640];final=pixels[480:960,1280:1920]
                tile=np.concatenate([cv2.resize(raw,(480,360)),cv2.resize(final,(480,360))],axis=1)
                cv2.putText(tile,f"session {sid} raw frame {frame} / {n}",(10,345),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
                tiles.append((frame,tile))
                if len(tiles)==6:
                    sheet=np.concatenate([np.concatenate([tiles[j][1],tiles[j+1][1]],axis=1) for j in (0,2,4)],axis=0)
                    path=folder/f"comparison_{len(sheets):03}.jpg";cv2.imwrite(str(path),sheet,[cv2.IMWRITE_JPEG_QUALITY,93])
                    sheets.append({"frames":[v[0] for v in tiles],"image":ref(path)});tiles=[]
            if frame in fixed:
                path=folder/f"anchor_{frame:04}.jpg";cv2.imwrite(str(path),pixels,[cv2.IMWRITE_JPEG_QUALITY,94])
            frame+=1
        reader.close()
        if frame!=n:raise RuntimeError("REVIEW_VIDEO_FRAME_COUNT")
        if tiles:
            ids=[v[0] for v in tiles];tiles+=[(-1,np.zeros_like(tiles[0][1]))]*(6-len(tiles))
            sheet=np.concatenate([np.concatenate([tiles[j][1],tiles[j+1][1]],axis=1) for j in (0,2,4)],axis=0)
            path=folder/f"comparison_{len(sheets):03}.jpg";cv2.imwrite(str(path),sheet,[cv2.IMWRITE_JPEG_QUALITY,93])
            sheets.append({"frames":ids,"image":ref(path)})
        value={"session_id":sid,"session_result":ref(files["session_result"]["path"]),
             "status":"EVIDENCE_READY_AI_INSPECTION_PENDING","fixed_frames":fixed,
             "proposed_inspection_frames":selected,"abnormal_frames":abnormal,
             "candidate_windows":windows,"abnormal_windows":abnormal_windows,"comparison_sheets":sheets,
             "sampling":"ALL_FIXED_ANCHORS_ALL_WINDOW_ENDPOINTS_AND_EVERY_FIFTH_NATIVE_FRAME_ALL_INVALID_FRAMES",
             "ai_inspection_performed":False,"human_review_status":"PENDING"}
        write(folder/"EVIDENCE.json",value);rows.append({"session_id":sid,"evidence":ref(folder/"EVIDENCE.json")})
    result={"status":"COMPLETE","sessions":rows,"source_index":ref(delivery_index),"ai_inspection_performed":False}
    write(output/"RESULT.json",result);return result
