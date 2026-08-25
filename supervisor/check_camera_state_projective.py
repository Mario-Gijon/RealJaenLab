#!/usr/bin/env python3
import json
from pathlib import Path
import cv2, numpy as np

from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    build_projective_camera_state_model,
    evaluate_homography_manual_points,
    homography_from_state,
    projective_state_leave_one_out,
)

P=Path("supervisor/data/reviews/benchmark_easy_2min.json")

def find_cal(C,frame):
    for c in C:
        if isinstance(c,dict) and c.get("source")=="manual" and c.get("purpose")=="match_setup":
            try:
                if int(c.get("frame"))==frame:return c
            except:pass
    return None

def main():
    r=json.loads(P.read_text())
    f=r["field"]; pitch=f["pitch_model"]; setup=f["match_setup"]; C=f.get("calibrations",[])
    video=Path("input")/r["video"]["filename"]
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise SystemExit(f"No se pudo abrir {video}")
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or r["video"].get("width") or 0)
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or r["video"].get("height") or 0); cap.release()
    completed=[int(x) for x in setup.get("completed_frames",[])]
    L=float(pitch["length_m"]); W=float(pitch["width_m"])
    print("\nPROJECTIVE CAMERA STATE V1 · LOO 2D/3D/4D/5D")
    results={}
    for d in (2,3,4,5):
        x=projective_state_leave_one_out(C,completed,image_width=w,image_height=h,pitch_length_m=L,pitch_width_m=W,dimensions=d)
        results[d]=x
        if x.get("status")=="ok":
            print(f"{d}D · feature-z={x['feature_z_median_rmse']:.3f} · grid med={x['grid_median_of_medians_px']:.1f}px · P90 med={x['grid_p90_of_medians_px']:.1f}px · max med={x['grid_max_median_px']:.1f}px · median P90={x['grid_median_p90_px']:.1f}px")
        else: print(d,"D",x)
    for d,x in results.items():
        if x.get("status")!="ok":continue
        print(f"\n{d}D por anchor")
        for q in x["per_anchor"]:
            if q.get("status")!="ok":print(" ",q);continue
            g=q["visible_grid"]; m=q["manual_points"]
            gt=f"{g['median_px']:.1f}/{g['p90_px']:.1f}" if g.get("status")=="ok" else g.get("status")
            mt=f"{m['median_px']:.1f}/{m['p90_px']:.1f}" if m.get("status")=="ok" else m.get("status")
            print(f"  f{q['frame']:>4} · grid={gt}px · HUMAN={mt}px · z={q['feature_z_rmse']:.3f}")
    valid=[(d,x["grid_median_of_medians_px"]) for d,x in results.items() if x.get("status")=="ok"]
    best=min(valid,key=lambda z:z[1])[0]
    print(f"\nFRAME 115 · usando mejor dimensionalidad LOO: {best}D")
    model=build_projective_camera_state_model(C,completed,image_width=w,image_height=h,pitch_length_m=L,pitch_width_m=W,
        config=ProjectiveCameraStateConfig(dimensions=best))
    states={int(a["frame"]):np.asarray(a["state"],float) for a in model["anchors"]}
    alpha=(115-87)/(175-87)
    state=(1-alpha)*states[87]+alpha*states[175]
    decoded=homography_from_state(model,state)
    cal115=find_cal(C,115)
    print("cumulative variance:",[round(float(x),4) for x in model["cumulative_variance"]])
    print("alpha_time:",round(alpha,3))
    print("state:",[round(float(x),3) for x in state])
    if decoded.get("status")!="ok":print(decoded);return
    ev=evaluate_homography_manual_points(decoded["homography"],cal115,model["config"]["maximum_h_condition"])
    print("115 HUMAN evaluation:")
    print(json.dumps(ev,indent=2,ensure_ascii=False))
    print("H PRIOR:")
    print(json.dumps(decoded["homography"],indent=2))
    print("\nNo se ha refinado H ni modificado la review.")

if __name__=="__main__":
    main()
