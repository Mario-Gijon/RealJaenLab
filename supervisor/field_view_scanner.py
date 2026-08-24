
"""Geometry-first FIELD View Scanner V1.

Selects representative raw-video frames for HUMAN field calibration without
using manual/propagated/refined homographies or person detections.

Absolute view identity is based on pitch-like line geometry and coarse pitch
shape, not raw RGB/pixel similarity. Dominant inter-frame motion is used only
for transition diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class FieldViewScannerConfig:
    analysis_width: int = 640
    sample_step_frames: int = 5
    min_line_length_ratio: float = 0.045
    min_white_support: float = 0.16
    min_green_context: float = 0.28
    min_geometry_lines: int = 2
    angle_bins: int = 12
    rho_bins: int = 18
    grid_rows: int = 4
    grid_cols: int = 6
    min_green_fraction: float = 0.12
    min_quality_score: float = 0.16
    minimum_coverage_radius: float = 0.03
    noise_radius_multiplier: float = 2.5
    representative_search_radius_frames: int = 12
    representative_search_step_frames: int = 2
    safety_max_recommendations: int = 60


def _resize(frame, width):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    s = width / float(w)
    return cv2.resize(frame, (width, max(1, round(h*s))), interpolation=cv2.INTER_AREA)


def _green_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    b,g,r = cv2.split(img.astype(np.int16))
    exg = 2*g-r-b
    thr = max(6.0, float(np.percentile(exg,55)))
    mask = (((h>=25)&(h<=100)&(s>=32)&(v>=24)) | (exg>=thr)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    return mask


def _white_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _,s,v = cv2.split(hsv)
    thr = max(105.0, float(np.percentile(v,68)))
    return ((v>=thr)&(s<=125)).astype(np.uint8)


def _segment_support(seg, mask, n=18):
    x1,y1,x2,y2 = seg
    H,W = mask.shape
    t = np.linspace(0,1,n)
    xs = np.clip(np.rint(x1+t*(x2-x1)).astype(int),0,W-1)
    ys = np.clip(np.rint(y1+t*(y2-y1)).astype(int),0,H-1)
    return float(np.mean(mask[ys,xs] > 0))


def _pitch_lines(img, green, white, cfg):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H,W = gray.shape
    diag = math.hypot(W,H)
    found = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if found is None:
        return []

    # OpenCV's Python bindings are not shape-stable across builds:
    # some return LSD segments as (N, 1, 4), others as (N, 4).
    # Normalize both forms to a single (N, 4) array.
    found = np.asarray(found)
    if found.size == 0:
        return []
    if found.ndim not in (2, 3) or found.shape[-1] != 4:
        raise ValueError(
            f"Unexpected LineSegmentDetector output shape: {found.shape}"
        )
    found = found.reshape(-1, 4)

    green_ctx = cv2.dilate(green, np.ones((11,11),np.uint8))
    out=[]
    for raw in found:
        seg = raw.astype(float)
        x1,y1,x2,y2 = seg
        length = math.hypot(x2-x1,y2-y1)
        lr = length/max(diag,1e-9)
        if lr < cfg.min_line_length_ratio or lr > .95:
            continue
        ws = _segment_support(seg, white, 20)
        gs = _segment_support(seg, green_ctx, 20)
        if ws < cfg.min_white_support or gs < cfg.min_green_context:
            continue
        angle = math.atan2(y2-y1,x2-x1) % math.pi
        nx,ny = -math.sin(angle), math.cos(angle)
        mx,my = (x1+x2)/2-W/2,(y1+y2)/2-H/2
        rho = float(np.clip((nx*mx+ny*my)/max(diag,1e-9),-.75,.75))
        weight = lr*(.35+.65*ws)*(.35+.65*gs)
        out.append(dict(segment=seg, angle=angle, rho=rho,
                        length_ratio=lr, weight=float(weight)))
    out.sort(key=lambda x:x["weight"], reverse=True)
    return out[:80]


def _line_hist(lines,cfg):
    hist=np.zeros((cfg.angle_bins,cfg.rho_bins),float)
    for q in lines:
        a=(q["angle"]/math.pi*cfg.angle_bins)%cfg.angle_bins
        r=(q["rho"]+.75)/1.5*(cfg.rho_bins-1)
        a0=int(math.floor(a))%cfg.angle_bins; a1=(a0+1)%cfg.angle_bins
        r0=int(np.clip(math.floor(r),0,cfg.rho_bins-1)); r1=min(r0+1,cfg.rho_bins-1)
        da=a-math.floor(a); dr=r-math.floor(r); w=q["weight"]
        hist[a0,r0]+=w*(1-da)*(1-dr); hist[a0,r1]+=w*(1-da)*dr
        hist[a1,r0]+=w*da*(1-dr); hist[a1,r1]+=w*da*dr
    s=hist.sum()
    if s>1e-12: hist/=s
    return hist.ravel()


def _grid(mask,rows,cols):
    H,W=mask.shape; vals=[]
    for y in range(rows):
        y0=round(y*H/rows); y1=round((y+1)*H/rows)
        for x in range(cols):
            x0=round(x*W/cols); x1=round((x+1)*W/cols)
            c=mask[y0:y1,x0:x1]
            vals.append(float(np.mean(c>0)) if c.size else 0.0)
    return np.asarray(vals,float)



def _line_spatial_grid(lines,W,H,rows=8,cols=12):
    canvas=np.zeros((rows,cols),np.float32)
    if W<=0 or H<=0: return canvas.ravel().astype(float)
    for q in lines:
        x1,y1,x2,y2=q["segment"]
        p1=(int(np.clip(round(x1/max(W,1)*(cols-1)),0,cols-1)),
            int(np.clip(round(y1/max(H,1)*(rows-1)),0,rows-1)))
        p2=(int(np.clip(round(x2/max(W,1)*(cols-1)),0,cols-1)),
            int(np.clip(round(y2/max(H,1)*(rows-1)),0,rows-1)))
        cv2.line(canvas,p1,p2,float(q["weight"]),1,lineType=cv2.LINE_AA)
    canvas=cv2.GaussianBlur(canvas,(3,3),0)
    s=float(canvas.sum())
    if s>1e-12: canvas/=s
    return canvas.ravel().astype(float)

def _intersection_grid(lines,W,H,cfg):
    grid=np.zeros((cfg.grid_rows,cfg.grid_cols),float); count=0
    use=lines[:30]
    for i,a in enumerate(use):
        p=a["segment"]; v1=np.array([p[2]-p[0],p[3]-p[1]],float)
        for b in use[i+1:]:
            da=abs(a["angle"]-b["angle"]); da=min(da,math.pi-da)
            if da<math.radians(18): continue
            q=b["segment"]; v2=np.array([q[2]-q[0],q[3]-q[1]],float)
            M=np.array([[v1[0],-v2[0]],[v1[1],-v2[1]]],float)
            if abs(np.linalg.det(M))<1e-8: continue
            try: t,_=np.linalg.solve(M,np.array([q[0]-p[0],q[1]-p[1]],float))
            except np.linalg.LinAlgError: continue
            px=p[0]+t*v1[0]; py=p[1]+t*v1[1]
            if not(-.15*W<=px<=1.15*W and -.15*H<=py<=1.15*H): continue
            cx=int(np.clip(px/max(W,1)*cfg.grid_cols,0,cfg.grid_cols-1))
            cy=int(np.clip(py/max(H,1)*cfg.grid_rows,0,cfg.grid_rows-1))
            grid[cy,cx]+=math.sqrt(max(a["weight"],0)*max(b["weight"],0)); count+=1
    s=grid.sum()
    if s>1e-12: grid/=s
    return grid.ravel(),count


def _descriptor(frame,cfg):
    img=_resize(frame,cfg.analysis_width)
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(1.6,(8,8)).apply(l)
    img=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
    green=_green_mask(img); white=_white_mask(img)
    lines=_pitch_lines(img,green,white,cfg)
    H,W=green.shape
    lh=_line_hist(lines,cfg)
    ls=_line_spatial_grid(lines,W,H,8,12)
    ig,nint=_intersection_grid(lines,W,H,cfg)
    gg=_grid(green,cfg.grid_rows,cfg.grid_cols)
    edges=cv2.Canny(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),60,150)
    eg=_grid(((edges>0)&(cv2.dilate(green,np.ones((9,9),np.uint8))>0)).astype(np.uint8),
             cfg.grid_rows,cfg.grid_cols)
    groups=[]
    for vec,w in [(lh,.40),(ls,.30),(ig,.12),(gg,.10),(eg,.08)]:
        n=np.linalg.norm(vec); v=vec/(n if n>1e-12 else 1.0)
        groups.append(v*math.sqrt(w))
    d=np.concatenate(groups); n=np.linalg.norm(d); d=d/(n if n>1e-12 else 1.0)
    families=len({int(((q["angle"]/math.pi)*4+.5))%4 for q in lines})
    gf=float(np.mean(green>0))
    sharp=1-math.exp(-float(np.var(cv2.Laplacian(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),cv2.CV_32F)))/180)
    q=(.28*min(1,len(lines)/10)+.22*min(1,families/3)+.18*min(1,nint/8)
       +.12*min(1,gf/.45)+.10*sharp+.10*min(1,sum(x["length_ratio"] for x in lines)/1.2))
    return dict(descriptor=d,quality_score=float(np.clip(q,0,1)),
                geometry=dict(green_fraction=gf,line_count=len(lines),
                              line_family_count=families,intersection_count=nint,
                              sharpness=float(sharp)))


def field_view_distance(a,b):
    a=np.asarray(a,float).ravel(); b=np.asarray(b,float).ravel()
    if a.shape!=b.shape or a.size==0: return float("inf")
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na<1e-12 or nb<1e-12: return float("inf")
    c=float(np.dot(a,b)/(na*nb))
    return float(np.clip((1-c)*.5,0,1))


def _motion(prev_gray,gray):
    pts=cv2.goodFeaturesToTrack(prev_gray,350,.01,7,blockSize=7)
    if pts is None or len(pts)<12: return {"status":"insufficient_features"}
    nxt,st,_=cv2.calcOpticalFlowPyrLK(prev_gray,gray,pts,None,winSize=(21,21),maxLevel=3)
    if nxt is None or st is None: return {"status":"flow_failed"}
    ok=st.ravel()>0; src=pts.reshape(-1,2)[ok]; dst=nxt.reshape(-1,2)[ok]
    if len(src)<10: return {"status":"insufficient_tracks"}
    M,inl=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.5)
    if M is None: return {"status":"ransac_failed"}
    a,c=M[0,0],M[1,0]; scale=math.sqrt(max(a*a+c*c,0))
    return {"status":"ok","translation_px":float(math.hypot(M[0,2],M[1,2])),
            "tx_px":float(M[0,2]),"ty_px":float(M[1,2]),
            "scale":float(scale),"zoom_delta":float(scale-1),
            "inlier_ratio":float(np.mean(inl.ravel()>0)) if inl is not None else None}


def _coverage_radius(samples,cfg):
    vals=[]
    for a,b in zip(samples,samples[1:]):
        if not(a["viable"] and b["viable"]): continue
        dist=field_view_distance(a["descriptor"],b["descriptor"])
        motion=b["motion_from_previous"]
        if np.isfinite(dist) and motion.get("status")=="ok":
            vals.append((motion["translation_px"],dist))
    if not vals:
        return cfg.minimum_coverage_radius
    m=np.array([x[0] for x in vals],float); thr=np.quantile(m,.35)
    ds=np.array([d for mm,d in vals if mm<=thr],float)
    if not len(ds): ds=np.array([d for _,d in vals],float)
    med=float(np.median(ds)); mad=float(np.median(np.abs(ds-med)))
    return float(np.clip(max(cfg.minimum_coverage_radius,(med+2.5*1.4826*mad)*cfg.noise_radius_multiplier),.04,.32))


def _select_states(samples,radius,cfg):
    if not samples: return []
    selected=[max(samples,key=lambda s:s["quality_score"])]
    nearest={s["frame"]:field_view_distance(s["descriptor"],selected[0]["descriptor"]) for s in samples}
    while len(selected)<cfg.safety_max_recommendations:
        cand=max(samples,key=lambda s:nearest[s["frame"]]*(.8+.2*s["quality_score"]))
        if not np.isfinite(nearest[cand["frame"]]) or nearest[cand["frame"]]<=radius: break
        selected.append(cand)
        for s in samples:
            nearest[s["frame"]]=min(nearest[s["frame"]],field_view_distance(s["descriptor"],cand["descriptor"]))
    return selected


def _read(cap,idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok,fr=cap.read()
    return fr if ok else None


def scan_field_view_states(video_path: str|Path, *, config: FieldViewScannerConfig|None=None)->dict[str,Any]:
    cfg=config or FieldViewScannerConfig(); path=Path(video_path)
    if not path.exists(): return {"status":"video_not_found","video":str(path)}
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): return {"status":"video_open_failed","video":str(path)}
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); fps=float(cap.get(cv2.CAP_PROP_FPS) or 0)
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    samples=[]; prev=None
    for idx in range(0,n,max(1,cfg.sample_step_frames)):
        fr=_read(cap,idx)
        if fr is None: continue
        a=_descriptor(fr,cfg)
        small=_resize(fr,cfg.analysis_width); gray=cv2.equalizeHist(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY))
        motion={"status":"first_sample"} if prev is None else _motion(prev,gray)
        g=a["geometry"]
        viable=(a["quality_score"]>=cfg.min_quality_score and g["green_fraction"]>=cfg.min_green_fraction
                and g["line_count"]>=cfg.min_geometry_lines)
        samples.append(dict(frame=idx,**a,viable=viable,motion_from_previous=motion))
        prev=gray
    viable=[s for s in samples if s["viable"]]
    radius=_coverage_radius(samples,cfg)
    seeds=_select_states(viable,radius,cfg)
    reps=[]
    for seed in seeds:
        best=None
        for idx in range(max(0,seed["frame"]-cfg.representative_search_radius_frames),
                         min(n-1,seed["frame"]+cfg.representative_search_radius_frames)+1,
                         max(1,cfg.representative_search_step_frames)):
            fr=_read(cap,idx)
            if fr is None: continue
            a=_descriptor(fr,cfg); d=field_view_distance(seed["descriptor"],a["descriptor"])
            if d>radius*.85: continue
            score=.78*a["quality_score"]+.22*max(0,1-d/max(radius*.85,1e-9))
            c=dict(frame=idx,seed_frame=seed["frame"],state_distance_from_seed=float(d),
                   quality_score=a["quality_score"],geometry=a["geometry"],descriptor=a["descriptor"],score=float(score))
            if best is None or c["score"]>best["score"]: best=c
        reps.append(best or dict(frame=seed["frame"],seed_frame=seed["frame"],state_distance_from_seed=0.0,
                                 quality_score=seed["quality_score"],geometry=seed["geometry"],
                                 descriptor=seed["descriptor"],score=seed["quality_score"]))
    cap.release()
    final=[]
    for c in sorted(reps,key=lambda x:x["score"],reverse=True):
        if any(field_view_distance(c["descriptor"],e["descriptor"])<=radius*.45 for e in final): continue
        final.append(c)
    final.sort(key=lambda x:x["frame"])
    public=[dict(sequence=i+1,frame=c["frame"],seed_frame=c["seed_frame"],
                 state_distance_from_seed=c["state_distance_from_seed"],
                 quality_score=c["quality_score"],geometry=c["geometry"])
            for i,c in enumerate(final)]
    return {"status":"ok","video":path.name,
            "metadata":{"frame_count":n,"fps":fps,"width":W,"height":H,"duration_seconds":n/fps if fps>0 else None},
            "policy":{"uses_raw_video_only":True,"uses_manual_homographies":False,
                      "uses_propagated_homographies":False,"uses_refined_homographies":False,
                      "uses_people_detection":False,"uses_pixel_image_similarity_for_view_identity":False,
                      "view_identity_is_geometry_first":True},
            "scan":{"sample_step_frames":cfg.sample_step_frames,"sample_count":len(samples),
                    "viable_sample_count":len(viable),"coverage_radius":radius},
            "recommendation_count":len(public),"recommendations":public,"config":asdict(cfg)}


def summarize_field_view_scan(result):
    recs=result.get("recommendations",[]) if isinstance(result,dict) else []
    scan=result.get("scan",{}) if isinstance(result,dict) else {}
    return {"status":result.get("status") if isinstance(result,dict) else "invalid_result",
            "video":result.get("video") if isinstance(result,dict) else None,
            "sample_count":scan.get("sample_count",0),
            "viable_sample_count":scan.get("viable_sample_count",0),
            "coverage_radius":scan.get("coverage_radius"),
            "recommendation_count":len(recs),
            "recommended_frames":[r.get("frame") for r in recs if isinstance(r,dict)]}
