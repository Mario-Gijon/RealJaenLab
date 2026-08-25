"""Projective Camera State V1.

Low-dimensional state over the COMPLETE local projective jet of H^-1 at the
image centre. Unlike the previous 6-observable state, this retains enough
information to reconstruct a full homography.

Normalized image: s=(u-cx)/W, t=(v-cy)/H
Normalized world: X=x/L, Y=y/PW

G = image_norm -> world_norm:
X=(a s+b t+c)/(p s+q t+1)
Y=(d s+e t+f)/(p s+q t+1)

Jet = [c,f,J00,J01,J10,J11,p,q], where
J00=a-cp, J01=b-cq, J10=d-fp, J11=e-fq.

The jet determines G exactly:
a=J00+cp, b=J01+cq, d=J10+fp, e=J11+fq.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from typing import Any
import numpy as np

EPS=1e-10

@dataclass(frozen=True)
class ProjectiveCameraStateConfig:
    dimensions:int=3
    minimum_feature_std:float=1e-8
    maximum_h_condition:float=1e14
    evaluation_x_samples:int=17
    evaluation_y_samples:int=13
    evaluation_margin_px:float=12.0

def _as_h(v, maxcond=1e14):
    try: h=np.asarray(v,float)
    except (TypeError,ValueError): return None
    if h.shape!=(3,3) or not np.isfinite(h).all(): return None
    if abs(float(h[2,2]))<EPS: return None
    h=h/float(h[2,2])
    try: det=float(np.linalg.det(h)); cond=float(np.linalg.cond(h))
    except np.linalg.LinAlgError: return None
    if not np.isfinite(det) or abs(det)<1e-12 or not np.isfinite(cond) or cond>maxcond: return None
    return h

def _Ai(w,h):
    return np.array([[w,0,w/2],[0,h,h/2],[0,0,1]],float)

def _Aw(L,W):
    return np.array([[L,0,0],[0,W,0],[0,0,1]],float)

def projective_jet_from_homography(H,*,image_width,image_height,pitch_length_m,pitch_width_m,maximum_condition=1e14):
    H=_as_h(H,maximum_condition)
    if H is None:return None
    try:
        G=np.linalg.inv(_Aw(pitch_length_m,pitch_width_m))@np.linalg.inv(H)@_Ai(image_width,image_height)
    except np.linalg.LinAlgError:return None
    if not np.isfinite(G).all() or abs(float(G[2,2]))<EPS:return None
    G/=G[2,2]
    a,b,c=G[0]; d,e,f=G[1]; p,q,_=G[2]
    feat=np.array([c,f,a-c*p,b-c*q,d-f*p,e-f*q,p,q],float)
    if not np.isfinite(feat).all():return None
    return feat

def homography_from_projective_jet(feature,*,image_width,image_height,pitch_length_m,pitch_width_m,maximum_condition=1e14):
    try:f=np.asarray(feature,float).reshape(-1)
    except (TypeError,ValueError):return None
    if f.shape!=(8,) or not np.isfinite(f).all():return None
    c,y0,j00,j01,j10,j11,p,q=f
    G=np.array([[j00+c*p,j01+c*q,c],[j10+y0*p,j11+y0*q,y0],[p,q,1]],float)
    try:
        pixel_to_world=_Aw(pitch_length_m,pitch_width_m)@G@np.linalg.inv(_Ai(image_width,image_height))
        H=np.linalg.inv(pixel_to_world)
    except np.linalg.LinAlgError:return None
    return _as_h(H,maximum_condition)

def _project(points,H):
    pts=np.asarray(points,float).reshape(-1,2)
    q=np.c_[pts,np.ones(len(pts))]@H.T
    den=q[:,2]
    valid=np.isfinite(q).all(1)&np.isfinite(den)&(np.abs(den)>EPS)
    out=np.full((len(pts),2),np.nan)
    out[valid]=q[valid,:2]/den[valid,None]
    valid&=np.isfinite(out).all(1)
    return out,valid

def _fit(features,k,minstd):
    X=np.asarray(features,float)
    mean=X.mean(0); std=X.std(0); std=np.where(std<minstd,1,std)
    Z=(X-mean)/std; zm=Z.mean(0); C=Z-zm
    try: _,s,vt=np.linalg.svd(C,full_matrices=False)
    except np.linalg.LinAlgError:return None
    rank=min(len(X)-1,len(vt)); k=min(max(1,k),rank)
    var=s*s; ratio=var/var.sum() if var.sum()>EPS else np.zeros_like(var)
    return dict(mean=mean,std=std,zm=zm,components=vt[:k],states=C@vt[:k].T,
                ratio=ratio[:rank],cum=np.cumsum(ratio[:rank]),k=k)

def _anchors(calibrations,completed,*,image_width,image_height,pitch_length_m,pitch_width_m,cfg):
    completed={int(x) for x in completed}
    out=[]
    for c in calibrations:
        if not isinstance(c,dict):continue
        try:f=int(c.get("frame"))
        except:continue
        if f not in completed or c.get("source")!="manual" or c.get("purpose")!="match_setup":continue
        H=_as_h(c.get("homography"),cfg.maximum_h_condition)
        if H is None:continue
        feat=projective_jet_from_homography(H,image_width=image_width,image_height=image_height,
            pitch_length_m=pitch_length_m,pitch_width_m=pitch_width_m,maximum_condition=cfg.maximum_h_condition)
        if feat is not None:out.append(dict(frame=f,H=H,feature=feat,calibration=c))
    return sorted(out,key=lambda x:x["frame"])

def build_projective_camera_state_model(calibrations,completed_frames,*,image_width,image_height,pitch_length_m,pitch_width_m,config=None):
    cfg=config or ProjectiveCameraStateConfig()
    A=_anchors(calibrations,completed_frames,image_width=image_width,image_height=image_height,
        pitch_length_m=pitch_length_m,pitch_width_m=pitch_width_m,cfg=cfg)
    if len(A)<cfg.dimensions+1:return {"status":"not_enough_anchors","anchor_count":len(A)}
    fit=_fit([a["feature"] for a in A],cfg.dimensions,cfg.minimum_feature_std)
    if fit is None:return {"status":"pca_failed"}
    return {"status":"ok","dimensions":fit["k"],"anchor_frames":[a["frame"] for a in A],
        "anchors":[{"frame":a["frame"],"state":s.tolist()} for a,s in zip(A,fit["states"])],
        "feature_mean":fit["mean"].tolist(),"feature_std":fit["std"].tolist(),"pca_mean":fit["zm"].tolist(),
        "pca_components":fit["components"].tolist(),"variance_ratio":fit["ratio"].tolist(),
        "cumulative_variance":fit["cum"].tolist(),"image_width":int(image_width),"image_height":int(image_height),
        "pitch_length_m":float(pitch_length_m),"pitch_width_m":float(pitch_width_m),"config":asdict(cfg)}

def homography_from_state(model,state):
    if model.get("status")!="ok":return {"status":"invalid_model"}
    s=np.asarray(state,float).reshape(-1)
    if s.shape!=(model["dimensions"],):return {"status":"invalid_state"}
    mean=np.asarray(model["feature_mean"]); std=np.asarray(model["feature_std"])
    zm=np.asarray(model["pca_mean"]); comp=np.asarray(model["pca_components"])
    feat=(zm+s@comp)*std+mean
    H=homography_from_projective_jet(feat,image_width=model["image_width"],image_height=model["image_height"],
        pitch_length_m=model["pitch_length_m"],pitch_width_m=model["pitch_width_m"],
        maximum_condition=model["config"]["maximum_h_condition"])
    return {"status":"ok","homography":H.tolist(),"jet_feature":feat.tolist()} if H is not None else {"status":"jet_to_h_failed"}

def evaluate_homography_manual_points(H,calibration,maximum_condition=1e14):
    H=_as_h(H,maximum_condition)
    if H is None:return {"status":"invalid_homography"}
    W=[];I=[];S=[]
    for c in calibration.get("constraints",[]):
        if not isinstance(c,dict) or c.get("type")!="point":continue
        try:
            W.append([float(c["world"]["x"]),float(c["world"]["y"])])
            I.append([float(c["image"]["x"]),float(c["image"]["y"])])
            S.append(c.get("semantic"))
        except:pass
    if not W:return {"status":"no_manual_points"}
    P,v=_project(W,H); I=np.asarray(I)
    if not v.any():return {"status":"projection_failed"}
    E=np.linalg.norm(P[v]-I[v],axis=1); names=[S[i] for i in np.flatnonzero(v)]
    return {"status":"ok","point_count":len(E),"median_px":float(np.median(E)),
        "p90_px":float(np.percentile(E,90)),"max_px":float(E.max()),
        "per_point":[{"semantic":n,"error_px":float(e)} for n,e in zip(names,E)]}

def _visible_eval(trueH,*,image_width,image_height,pitch_length_m,pitch_width_m,cfg):
    xs=np.linspace(-pitch_length_m/2,pitch_length_m/2,cfg.evaluation_x_samples)
    ys=np.linspace(-pitch_width_m/2,pitch_width_m/2,cfg.evaluation_y_samples)
    W=np.array([(x,y) for y in ys for x in xs],float)
    I,v=_project(W,trueH); m=cfg.evaluation_margin_px
    inside=v&(I[:,0]>=m)&(I[:,0]<=image_width-m)&(I[:,1]>=m)&(I[:,1]<=image_height-m)
    return W[inside],I[inside]

def evaluate_homography_visible_grid(predH,trueH,*,image_width,image_height,pitch_length_m,pitch_width_m,config=None):
    cfg=config or ProjectiveCameraStateConfig()
    P=_as_h(predH,cfg.maximum_h_condition); T=_as_h(trueH,cfg.maximum_h_condition)
    if P is None or T is None:return {"status":"invalid_homography"}
    W,I=_visible_eval(T,image_width=image_width,image_height=image_height,pitch_length_m=pitch_length_m,pitch_width_m=pitch_width_m,cfg=cfg)
    if len(W)<4:return {"status":"not_enough_visible_points"}
    J,v=_project(W,P)
    if v.sum()<4:return {"status":"projection_failed"}
    E=np.linalg.norm(J[v]-I[v],axis=1)
    return {"status":"ok","median_px":float(np.median(E)),"p90_px":float(np.percentile(E,90)),"max_px":float(E.max()),"count":int(len(E))}

def projective_state_leave_one_out(calibrations,completed_frames,*,image_width,image_height,pitch_length_m,pitch_width_m,dimensions,config=None):
    base=config or ProjectiveCameraStateConfig()
    cfg=ProjectiveCameraStateConfig(dimensions=dimensions,minimum_feature_std=base.minimum_feature_std,
        maximum_h_condition=base.maximum_h_condition,evaluation_x_samples=base.evaluation_x_samples,
        evaluation_y_samples=base.evaluation_y_samples,evaluation_margin_px=base.evaluation_margin_px)
    A=_anchors(calibrations,completed_frames,image_width=image_width,image_height=image_height,
        pitch_length_m=pitch_length_m,pitch_width_m=pitch_width_m,cfg=cfg)
    if len(A)<dimensions+2:return {"status":"not_enough_anchors"}
    R=[]
    for i,h in enumerate(A):
        train=[a["feature"] for j,a in enumerate(A) if j!=i]
        fit=_fit(train,dimensions,cfg.minimum_feature_std)
        if fit is None:continue
        z=(h["feature"]-fit["mean"])/fit["std"]
        state=(z-fit["zm"])@fit["components"].T
        rec=(fit["zm"]+state@fit["components"])*fit["std"]+fit["mean"]
        pred=homography_from_projective_jet(rec,image_width=image_width,image_height=image_height,
            pitch_length_m=pitch_length_m,pitch_width_m=pitch_width_m,maximum_condition=cfg.maximum_h_condition)
        if pred is None:R.append({"frame":h["frame"],"status":"jet_to_h_failed"});continue
        grid=evaluate_homography_visible_grid(pred,h["H"],image_width=image_width,image_height=image_height,
            pitch_length_m=pitch_length_m,pitch_width_m=pitch_width_m,config=cfg)
        manual=evaluate_homography_manual_points(pred,h["calibration"],cfg.maximum_h_condition)
        ferr=float(np.linalg.norm((rec-h["feature"])/fit["std"])/math.sqrt(8))
        R.append({"frame":h["frame"],"status":"ok","feature_z_rmse":ferr,"visible_grid":grid,"manual_points":manual})
    V=[r for r in R if r.get("status")=="ok" and r["visible_grid"].get("status")=="ok"]
    if not V:return {"status":"no_valid_holdouts","per_anchor":R}
    M=np.array([r["visible_grid"]["median_px"] for r in V]); P90=np.array([r["visible_grid"]["p90_px"] for r in V])
    return {"status":"ok","dimensions":dimensions,"holdout_count":len(V),
        "feature_z_median_rmse":float(np.median([r["feature_z_rmse"] for r in V])),
        "grid_median_of_medians_px":float(np.median(M)),"grid_p90_of_medians_px":float(np.percentile(M,90)),
        "grid_max_median_px":float(M.max()),"grid_median_p90_px":float(np.median(P90)),"per_anchor":R}
