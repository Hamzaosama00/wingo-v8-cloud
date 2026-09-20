from __future__ import annotations

import json, math, os, sqlite3, threading, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

try:
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request as GoogleAuthRequest
except Exception:
    service_account = None
    GoogleAuthRequest = None

APP_VERSION = "9.0.0"
MODEL_VERSION = "v9.0-research"
DB_PATH = os.getenv("DB_PATH", "wingo_v9.db")
INGEST_SECRET = os.getenv("INGEST_SECRET", "")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "").strip()
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
HISTORY_LIMIT = max(300, int(os.getenv("HISTORY_LIMIT", "1200")))
MIN_TRAIN = max(60, int(os.getenv("V9_MIN_TRAIN", "100")))
BACKTEST_BLOCK = max(20, int(os.getenv("V9_BACKTEST_BLOCK", "50")))
EMBARGO = max(0, int(os.getenv("V9_EMBARGO", "20")))
SIGNAL_THRESHOLD = float(os.getenv("V9_SIGNAL_THRESHOLD", "0.60"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
CORS_LIST = ["*"] if CORS_ORIGINS.strip() == "*" else [x.strip() for x in CORS_ORIGINS.split(",") if x.strip()]

DB_LOCK = threading.RLock()
MODEL_LOCK = threading.RLock()
_FIRESTORE = None
_FIRESTORE_LOCK = threading.Lock()
MODEL_CACHE: Dict[str, Any] = {"history_issue": None, "prediction": None, "report": None, "ablation": None}


def size_from_number(n: int) -> str: return "big" if int(n) >= 5 else "small"
def parity_from_number(n: int) -> str: return "even" if int(n) % 2 == 0 else "odd"
def normalize_color(c: str) -> str:
    c = (c or "").lower()
    if "violet" in c: return "violet"
    if "red" in c: return "red"
    if "green" in c: return "green"
    return c.strip() or "unknown"
def increment_issue(issue: str) -> str:
    try: return str(int(issue) + 1)
    except Exception: return issue + "_next"
def clip_prob(p: float) -> float: return float(min(1-1e-6, max(1e-6, p)))
def entropy_binary(p: float) -> float:
    p = clip_prob(p)
    return float(-(p*math.log2(p) + (1-p)*math.log2(1-p)))


def db_conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with DB_LOCK, db_conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS v9_rounds(
            issue TEXT PRIMARY KEY, number INTEGER NOT NULL, color TEXT NOT NULL,
            size TEXT NOT NULL, parity TEXT NOT NULL, seen_at REAL NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS v9_predictions(
            issue TEXT PRIMARY KEY, created_at REAL NOT NULL, model_version TEXT NOT NULL,
            predicted_size TEXT NOT NULL, p_big_raw REAL NOT NULL, p_big_calibrated REAL NOT NULL,
            confidence REAL NOT NULL, signal TEXT NOT NULL, regime TEXT NOT NULL,
            threshold REAL NOT NULL, verified INTEGER NOT NULL DEFAULT 0,
            actual_size TEXT, size_win INTEGER, details_json TEXT)""")
        c.commit()


def round_count() -> int:
    with DB_LOCK, db_conn() as c: return int(c.execute("SELECT COUNT(*) c FROM v9_rounds").fetchone()["c"])


def save_round_local(issue, number, color, seen_at=None):
    with DB_LOCK, db_conn() as c:
        cur = c.execute("INSERT OR IGNORE INTO v9_rounds VALUES(?,?,?,?,?,?)",
            (str(issue), int(number), normalize_color(color), size_from_number(number), parity_from_number(number), float(seen_at or time.time())))
        c.commit(); return cur.rowcount > 0


def load_rounds(limit=HISTORY_LIMIT):
    with DB_LOCK, db_conn() as c:
        rows = c.execute("SELECT * FROM v9_rounds ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?", (int(limit),)).fetchall()
    out = [dict(r) for r in rows]; out.reverse(); return out


def save_prediction_local(p):
    details = dict(p)
    with DB_LOCK, db_conn() as c:
        c.execute("""INSERT OR REPLACE INTO v9_predictions
        (issue,created_at,model_version,predicted_size,p_big_raw,p_big_calibrated,confidence,signal,regime,threshold,verified,actual_size,size_win,details_json)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (p["issue"], float(p.get("created_at", time.time())), p.get("model_version", MODEL_VERSION), p["predicted_size"],
         float(p["p_big_raw"]), float(p["p_big_calibrated"]), float(p["confidence"]), p["signal"], p["regime"], float(p["threshold"]),
         int(p.get("verified",0)), p.get("actual_size"), p.get("size_win"), json.dumps(details,separators=(",",":"))))
        c.commit()


def verify_prediction(issue, actual_size):
    with DB_LOCK, db_conn() as c:
        r = c.execute("SELECT * FROM v9_predictions WHERE issue=?", (str(issue),)).fetchone()
        if not r: return None
        win = 1 if r["predicted_size"] == actual_size else 0
        c.execute("UPDATE v9_predictions SET verified=1,actual_size=?,size_win=? WHERE issue=?", (actual_size,win,str(issue))); c.commit()
        d = dict(r); d.update({"verified":1,"actual_size":actual_size,"size_win":win}); return d


def load_predictions(limit=120):
    with DB_LOCK, db_conn() as c:
        rows = c.execute("SELECT * FROM v9_predictions ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?", (int(limit),)).fetchall()
    out=[]
    for r in rows:
        d=dict(r)
        try:
            j=json.loads(d.pop("details_json") or "{}"); j.update(d); d=j
        except Exception: pass
        out.append(d)
    return out


def fs_enc(v):
    if v is None: return {"nullValue":None}
    if isinstance(v,bool): return {"booleanValue":v}
    if isinstance(v,int) and not isinstance(v,bool): return {"integerValue":str(v)}
    if isinstance(v,float): return {"doubleValue":float(v)} if math.isfinite(v) else {"nullValue":None}
    if isinstance(v,str): return {"stringValue":v}
    if isinstance(v,list): return {"arrayValue":{"values":[fs_enc(x) for x in v]}}
    if isinstance(v,dict): return {"mapValue":{"fields":{k:fs_enc(x) for k,x in v.items()}}}
    return {"stringValue":str(v)}


def fs_dec(o):
    if "nullValue" in o: return None
    if "booleanValue" in o: return bool(o["booleanValue"])
    if "integerValue" in o:
        try: return int(o["integerValue"])
        except Exception: return 0
    if "doubleValue" in o: return float(o["doubleValue"])
    if "stringValue" in o: return str(o["stringValue"])
    if "arrayValue" in o: return [fs_dec(x) for x in o.get("arrayValue",{}).get("values",[])]
    if "mapValue" in o: return {k:fs_dec(v) for k,v in o.get("mapValue",{}).get("fields",{}).items()}
    return None


class FirestoreREST:
    def __init__(self, project_id, cred_path):
        if service_account is None: raise RuntimeError("google-auth missing")
        self.base=f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents"
        self.creds=service_account.Credentials.from_service_account_file(cred_path, scopes=["https://www.googleapis.com/auth/datastore","https://www.googleapis.com/auth/cloud-platform"])
        self.req=GoogleAuthRequest(); self.s=requests.Session(); self.lock=threading.Lock()
    def headers(self):
        with self.lock:
            if not self.creds.valid: self.creds.refresh(self.req)
            tok=self.creds.token
        return {"Authorization":f"Bearer {tok}","Content-Type":"application/json"}
    def set_doc(self,col,doc_id,data):
        body={"fields":{k:fs_enc(v) for k,v in data.items()}}
        r=self.s.patch(f"{self.base}/{col}/{doc_id}",headers=self.headers(),json=body,timeout=15); r.raise_for_status()
    def list_docs(self,col,limit=1000):
        out=[]; token=None
        while len(out)<limit:
            params={"pageSize":min(1000,limit-len(out)),"orderBy":"__name__ desc"}
            if token: params["pageToken"]=token
            r=self.s.get(f"{self.base}/{col}",headers=self.headers(),params=params,timeout=20)
            if r.status_code==404: break
            r.raise_for_status(); p=r.json()
            for doc in p.get("documents",[]):
                d={k:fs_dec(v) for k,v in doc.get("fields",{}).items()}; d["_id"]=doc.get("name","").rsplit("/",1)[-1]; out.append(d)
            token=p.get("nextPageToken")
            if not token: break
        return out


def get_firestore():
    global _FIRESTORE
    if not FIREBASE_PROJECT_ID or not GOOGLE_APPLICATION_CREDENTIALS: return None
    if _FIRESTORE is None:
        with _FIRESTORE_LOCK:
            if _FIRESTORE is None: _FIRESTORE=FirestoreREST(FIREBASE_PROJECT_ID,GOOGLE_APPLICATION_CREDENTIALS)
    return _FIRESTORE


def cloud_save_round(r):
    fs=get_firestore()
    if not fs: return None
    try:
        fs.set_doc("rounds",str(r["issue"]),{"issue":str(r["issue"]),"number":int(r["number"]),"color":normalize_color(r["color"]),"size":size_from_number(int(r["number"])),"parity":parity_from_number(int(r["number"])),"seen_at":float(r.get("seen_at",time.time()))}); return None
    except Exception as e: return str(e)


def cloud_save_prediction(p):
    fs=get_firestore()
    if not fs: return None
    try:
        fs.set_doc("v9_predictions",str(p["issue"]),{k:p.get(k) for k in ["issue","created_at","model_version","predicted_size","p_big_raw","p_big_calibrated","confidence","signal","regime","threshold","verified","actual_size","size_win"]}); return None
    except Exception as e: return str(e)


def hydrate_from_firestore():
    fs=get_firestore()
    if not fs: return 0,0,None
    rr=rp=0
    try:
        docs=fs.list_docs("rounds",HISTORY_LIMIT); docs.reverse()
        for d in docs:
            issue=str(d.get("issue") or d.get("_id") or ""); n=d.get("number"); color=d.get("color")
            if issue and n is not None and color is not None:
                if save_round_local(issue,int(n),str(color),float(d.get("seen_at") or time.time())): rr+=1
        preds=fs.list_docs("v9_predictions",600); preds.reverse()
        for p in preds:
            issue=str(p.get("issue") or p.get("_id") or "")
            if issue and p.get("predicted_size"):
                save_prediction_local({"issue":issue,"created_at":float(p.get("created_at") or time.time()),"model_version":p.get("model_version",MODEL_VERSION),"predicted_size":p["predicted_size"],"p_big_raw":float(p.get("p_big_raw") or .5),"p_big_calibrated":float(p.get("p_big_calibrated") or .5),"confidence":float(p.get("confidence") or .5),"signal":p.get("signal","SKIP"),"regime":p.get("regime","UNKNOWN"),"threshold":float(p.get("threshold") or SIGNAL_THRESHOLD),"verified":int(p.get("verified") or 0),"actual_size":p.get("actual_size"),"size_win":p.get("size_win")}); rp+=1
        return rr,rp,None
    except Exception as e: return rr,rp,str(e)


FEATURE_GROUPS={
"base":["last_is_big","last_number_norm","last_even"],
"streak":["current_streak","run1","run2","run3","run_mean_200","run_std_200","run_max_200"],
"multiscale":["big_ratio_20","big_ratio_50","big_ratio_100","big_ratio_300","dev_20_100","dev_20_300","dev_100_300"],
"transition":["p_big_after_big_100","p_big_after_small_100","p_big_after_big_300","p_big_after_small_300"],
"number":["num_mean_20","num_std_20","num_mean_100","num_std_100","sum_last5_norm","last_distance_45","even_ratio_20","mod3_0_50","mod3_1_50","mod3_2_50"],
"regime":["change_rate_20","change_rate_100","entropy_20","entropy_100","short_long_shift"],
"issue_experimental":["issue_mod_2","issue_mod_3","issue_mod_5","issue_mod_7","issue_mod_11"]}
FEATURE_NAMES=[n for g in FEATURE_GROUPS.values() for n in g]


def big_values(h): return [1 if r["size"]=="big" else 0 for r in h]
def ratio_last(v,w):
    if not v: return .5
    z=v[-min(w,len(v)):]; return float(sum(z)/len(z))
def transition_prob(v,w,prev):
    z=v[-min(w+1,len(v)):]; big=1.; total=2.
    for a,b in zip(z[:-1],z[1:]):
        if a==prev:
            total+=1; big+=1 if b==1 else 0
    return float(big/total)
def recent_runs(v,max_history=200):
    z=v[-min(max_history,len(v)):]
    if not z:return [0]
    runs=[]; cur=z[0]; ln=1
    for x in z[1:]:
        if x==cur:ln+=1
        else:runs.append(ln);cur=x;ln=1
    runs.append(ln); return runs
def change_rate(v,w):
    z=v[-min(w,len(v)):]
    if len(z)<2:return .5
    return float(sum(1 for a,b in zip(z[:-1],z[1:]) if a!=b)/(len(z)-1))


def build_features(history,target_issue=None):
    v=big_values(history); nums=[int(r["number"]) for r in history]
    last_big=v[-1] if v else 0; last_num=nums[-1] if nums else 4
    runs=recent_runs(v); rev=list(reversed(runs))
    r20,r50,r100,r300=[ratio_last(v,w) for w in (20,50,100,300)]
    n20=nums[-min(20,len(nums)):] if nums else [4,5]; n100=nums[-min(100,len(nums)):] if nums else [4,5]; n50=nums[-min(50,len(nums)):] if nums else [4,5]
    even20=sum(1 for n in n20 if n%2==0)/max(1,len(n20)); mod3=[sum(1 for n in n50 if n%3==k)/max(1,len(n50)) for k in range(3)]
    try: issue_int=int(target_issue or increment_issue(history[-1]["issue"]))
    except Exception: issue_int=0
    return {
    "last_is_big":float(last_big),"last_number_norm":last_num/9.,"last_even":float(last_num%2==0),
    "current_streak":min(20,rev[0] if rev else 0)/20.,"run1":min(20,rev[0] if len(rev)>0 else 0)/20.,"run2":min(20,rev[1] if len(rev)>1 else 0)/20.,"run3":min(20,rev[2] if len(rev)>2 else 0)/20.,"run_mean_200":float(np.mean(runs)/10.),"run_std_200":float(np.std(runs)/10.),"run_max_200":min(30,max(runs) if runs else 0)/30.,
    "big_ratio_20":r20,"big_ratio_50":r50,"big_ratio_100":r100,"big_ratio_300":r300,"dev_20_100":r20-r100,"dev_20_300":r20-r300,"dev_100_300":r100-r300,
    "p_big_after_big_100":transition_prob(v,100,1),"p_big_after_small_100":transition_prob(v,100,0),"p_big_after_big_300":transition_prob(v,300,1),"p_big_after_small_300":transition_prob(v,300,0),
    "num_mean_20":float(np.mean(n20)/9.),"num_std_20":float(np.std(n20)/4.5),"num_mean_100":float(np.mean(n100)/9.),"num_std_100":float(np.std(n100)/4.5),"sum_last5_norm":float(sum(nums[-5:])/45. if nums else .5),"last_distance_45":abs(last_num-4.5)/4.5,"even_ratio_20":float(even20),"mod3_0_50":float(mod3[0]),"mod3_1_50":float(mod3[1]),"mod3_2_50":float(mod3[2]),
    "change_rate_20":change_rate(v,20),"change_rate_100":change_rate(v,100),"entropy_20":entropy_binary(r20),"entropy_100":entropy_binary(r100),"short_long_shift":abs(r20-r100),
    "issue_mod_2":float(issue_int%2),"issue_mod_3":float((issue_int%3)/2.) if issue_int else 0.,"issue_mod_5":float((issue_int%5)/4.) if issue_int else 0.,"issue_mod_7":float((issue_int%7)/6.) if issue_int else 0.,"issue_mod_11":float((issue_int%11)/10.) if issue_int else 0.}


def features_to_vector(f): return np.array([float(f.get(k,0.)) for k in FEATURE_NAMES],dtype=float)
def build_dataset(rounds,min_context=20):
    X=[];y=[];idxs=[]
    for i in range(min_context,len(rounds)):
        X.append(features_to_vector(build_features(rounds[:i],rounds[i]["issue"]))); y.append(1 if rounds[i]["size"]=="big" else 0); idxs.append(i)
    return (np.vstack(X),np.asarray(y,dtype=int),idxs) if X else (np.zeros((0,len(FEATURE_NAMES))),np.zeros((0,),dtype=int),[])


@dataclass
class FittedModels:
    scaler:Optional[StandardScaler]; logistic:Optional[LogisticRegression]; tree:Optional[HistGradientBoostingClassifier]; constant:Optional[float]=None

def fit_models(X,y):
    if len(y)==0:return FittedModels(None,None,None,.5)
    u=np.unique(y)
    if len(u)<2:return FittedModels(None,None,None,float(u[0]))
    sc=StandardScaler(); xs=sc.fit_transform(X)
    lr=LogisticRegression(C=.5,penalty="l2",solver="lbfgs",max_iter=500,random_state=42); lr.fit(xs,y)
    tr=HistGradientBoostingClassifier(max_iter=60,learning_rate=.05,max_depth=3,min_samples_leaf=20,l2_regularization=1.,random_state=42); tr.fit(X,y)
    return FittedModels(sc,lr,tr,None)

def model_probs(m,x):
    if m.constant is not None:return float(m.constant),float(m.constant)
    x2=x.reshape(1,-1); return float(m.logistic.predict_proba(m.scaler.transform(x2))[0,1]),float(m.tree.predict_proba(x2)[0,1])

def markov_probability(history):
    v=big_values(history)
    if not v:return .5,0,0
    for order in (3,2,1):
        if len(v)<=order:continue
        ctx=tuple(v[-order:]); big=1.; total=2.; support=0
        for i in range(order,len(v)):
            if tuple(v[i-order:i])==ctx:
                support+=1; total+=1; big+=1 if v[i]==1 else 0
        if support>={3:5,2:8,1:12}[order]:return float(big/total),order,support
    z=v[-min(300,len(v)):]; return float((sum(z)+2)/(len(z)+4)),0,len(z)

def ensemble_probability(m,x,h):
    pl,pt=model_probs(m,x); pm,o,s=markov_probability(h); p=.4*pl+.4*pt+.2*pm; votes=[int(pl>=.5),int(pt>=.5),int(pm>=.5)]
    return {"p_big_raw":clip_prob(p),"p_logistic":clip_prob(pl),"p_tree":clip_prob(pt),"p_markov":clip_prob(pm),"markov_order":o,"markov_support":s,"agreement":max(votes.count(0),votes.count(1))}


def safe_metrics(y,probs):
    if not y:return {"samples":0,"accuracy":None,"brier":None,"log_loss":None}
    yy=np.asarray(y,dtype=int); pp=np.asarray([clip_prob(x) for x in probs]); pred=(pp>=.5).astype(int)
    try:ll=float(log_loss(yy,pp,labels=[0,1]))
    except Exception:ll=None
    return {"samples":len(yy),"accuracy":float(accuracy_score(yy,pred)),"brier":float(brier_score_loss(yy,pp)),"log_loss":ll}

def fit_iso(probs,labels):
    if len(probs)<40 or len(set(labels))<2:return None
    try:
        iso=IsotonicRegression(y_min=.02,y_max=.98,out_of_bounds="clip"); iso.fit(np.asarray(probs),np.asarray(labels)); return iso
    except Exception:return None

def apply_iso(iso,p):
    if iso is None:return clip_prob(p)
    try:return clip_prob(float(iso.predict([p])[0]))
    except Exception:return clip_prob(p)

def confidence_curve(labels,probs):
    out=[]; total=len(labels)
    for t in (.52,.55,.58,.60,.62,.65,.70):
        ys=[];ps=[]
        for y,p in zip(labels,probs):
            if max(p,1-p)>=t:ys.append(y);ps.append(1 if p>=.5 else 0)
        n=len(ys); acc=sum(int(a==b) for a,b in zip(ys,ps))/n if n else None
        out.append({"threshold":t,"samples":n,"coverage":n/total if total else 0.,"accuracy":float(acc) if acc is not None else None})
    return out


def walk_forward_report(rounds):
    X,y,idxs=build_dataset(rounds)
    if len(y)<MIN_TRAIN+EMBARGO+10:return {"status":"insufficient_history","samples":len(y),"minimum_needed":MIN_TRAIN+EMBARGO+10,"raw":safe_metrics([],[]),"calibrated_eval":safe_metrics([],[]),"confidence_curve":[],"calibrator":None,"oos_labels":[],"oos_probs":[]}
    labels=[];probs=[];folds=0; train_end=MIN_TRAIN
    while True:
        vs=train_end+EMBARGO
        if vs>=len(y):break
        ve=min(len(y),vs+BACKTEST_BLOCK)
        m=fit_models(X[:train_end],y[:train_end])
        for j in range(vs,ve):
            hist=rounds[:idxs[j]]; probs.append(ensemble_probability(m,X[j],hist)["p_big_raw"]); labels.append(int(y[j]))
        folds+=1; train_end+=BACKTEST_BLOCK
        if ve>=len(y):break
    raw=safe_metrics(labels,probs); iso=None; cal_eval=safe_metrics([],[]); cal_all=list(probs)
    if len(probs)>=60:
        sp=max(40,int(len(probs)*.70)); first=fit_iso(probs[:sp],labels[:sp])
        if first is not None and labels[sp:]: cal_eval=safe_metrics(labels[sp:],[apply_iso(first,p) for p in probs[sp:]])
        iso=fit_iso(probs,labels)
        if iso is not None:cal_all=[apply_iso(iso,p) for p in probs]
    return {"status":"ok","folds":folds,"samples":len(labels),"raw":raw,"calibrated_eval":cal_eval,"confidence_curve":confidence_curve(labels,cal_all),"calibrator":iso,"oos_labels":labels,"oos_probs":cal_all}


def ablation_report(rounds):
    X,y,_=build_dataset(rounds)
    if len(y)<MIN_TRAIN+EMBARGO+25:return []
    split=max(MIN_TRAIN,int(len(y)*.75)); test_start=min(len(y),split+EMBARGO)
    if len(y)-test_start<15:return []
    order=["base","streak","multiscale","transition","number","regime","issue_experimental"]; n2i={n:i for i,n in enumerate(FEATURE_NAMES)}; selected=[];out=[]
    for g in order:
        selected+=FEATURE_GROUPS[g]; ids=[n2i[n] for n in selected]; xt=X[:split][:,ids]; yt=y[:split]; xv=X[test_start:][:,ids]; yv=y[test_start:]
        if len(set(yt.tolist()))<2:continue
        sc=StandardScaler(); xt=sc.fit_transform(xt); xv=sc.transform(xv); lr=LogisticRegression(C=.5,max_iter=400,random_state=42);lr.fit(xt,yt);p=lr.predict_proba(xv)[:,1]
        out.append({"through_group":g,"feature_count":len(ids),**safe_metrics(yv.tolist(),p.tolist())})
    return out


def current_regime(h):
    v=big_values(h); r20=ratio_last(v,20);r100=ratio_last(v,100);e=entropy_binary(r20);cr=change_rate(v,20)
    if e>=.98 and abs(r20-r100)<.08:return "HIGH_ENTROPY"
    if abs(r20-r100)>=.18:return "SHIFT"
    if cr>=.70:return "CHOPPY"
    if cr<=.30:return "STREAKY"
    return "NEUTRAL"

def feature_contributions(m,x):
    if m.constant is not None or m.logistic is None:return []
    z=(x-m.scaler.mean_)/np.where(m.scaler.scale_==0,1.,m.scaler.scale_); c=z*m.logistic.coef_[0];out=[]
    for n,v,k in zip(FEATURE_NAMES,x,c):out.append({"feature":n,"value":float(v),"contribution":float(k),"direction":"BIG" if k>0 else "SMALL"})
    return sorted(out,key=lambda r:abs(r["contribution"]),reverse=True)[:10]


def build_v9_prediction(rounds):
    if not rounds:return {"ready":False,"reason":"no_history","model_version":MODEL_VERSION}
    target=increment_issue(rounds[-1]["issue"])
    if len(rounds)<max(40,MIN_TRAIN//2):return {"ready":False,"reason":"insufficient_history","history":len(rounds),"minimum":max(40,MIN_TRAIN//2),"issue":target,"model_version":MODEL_VERSION}
    X,y,_=build_dataset(rounds)
    with MODEL_LOCK:
        m=fit_models(X,y); f=build_features(rounds,target); x=features_to_vector(f); e=ensemble_probability(m,x,rounds); report=walk_forward_report(rounds); iso=report.get("calibrator"); p_raw=e["p_big_raw"];p_cal=apply_iso(iso,p_raw)
        predicted="big" if p_cal>=.5 else "small"; conf=max(p_cal,1-p_cal); regime=current_regime(rounds); reasons=[]; signal="PREDICT"
        if conf<SIGNAL_THRESHOLD:signal="SKIP";reasons.append("confidence_below_threshold")
        if int(report.get("samples") or 0)<50:signal="SKIP";reasons.append("insufficient_out_of_sample_support")
        if regime=="HIGH_ENTROPY" and conf<max(.65,SIGNAL_THRESHOLD):signal="SKIP";reasons.append("high_entropy_regime")
        if e["agreement"]<2:signal="SKIP";reasons.append("low_model_agreement")
        pred={"ready":True,"issue":target,"created_at":time.time(),"model_version":MODEL_VERSION,"predicted_size":predicted,"p_big_raw":p_raw,"p_big_calibrated":p_cal,"confidence":conf,"threshold":SIGNAL_THRESHOLD,"signal":signal,"signal_reasons":reasons,"regime":regime,"agreement":e["agreement"],"models":{"logistic":e["p_logistic"],"boosted_tree":e["p_tree"],"markov":e["p_markov"],"markov_order":e["markov_order"],"markov_support":e["markov_support"]},"calibration":{"available":iso is not None,"method":"isotonic" if iso is not None else "none","oos_samples":int(report.get("samples") or 0)},"feature_contributions":feature_contributions(m,x),"verified":0,"actual_size":None,"size_win":None}
        public={k:v for k,v in report.items() if k not in {"calibrator","oos_labels","oos_probs"}}
        MODEL_CACHE.update({"history_issue":rounds[-1]["issue"],"prediction":pred,"report":public,"ablation":ablation_report(rounds)})
        return pred

def latest_or_build(rounds):
    if rounds and MODEL_CACHE.get("history_issue")==rounds[-1]["issue"] and MODEL_CACHE.get("prediction"):return MODEL_CACHE["prediction"]
    return build_v9_prediction(rounds)

def live_accuracy():
    p=[x for x in load_predictions(500) if int(x.get("verified") or 0)==1]; q=[x for x in p if x.get("signal")=="PREDICT"]
    def s(a):
        n=len(a);w=sum(1 for x in a if int(x.get("size_win") or 0)==1);return {"tested":n,"correct":w,"accuracy":w/n if n else None}
    return {"all_verified":s(p),"predict_only":s(q),"coverage":len(q)/len(p) if p else None}


class RoundInput(BaseModel):
    issue:str; number:int; color:str; secret:str

app=FastAPI(title="WinGo V9 Research Engine",version=APP_VERSION)
app.add_middleware(CORSMiddleware,allow_origins=CORS_LIST,allow_credentials=False,allow_methods=["GET","POST","OPTIONS"],allow_headers=["*"])

@app.on_event("startup")
def startup():
    init_db()
    if round_count()==0:hydrate_from_firestore()

@app.get("/")
def root():return {"service":"online","version":APP_VERSION,"model_version":MODEL_VERSION,"purpose":"research/backtesting","target":"big_small"}
@app.get("/health")
def health():return {"service":"online","version":APP_VERSION,"stored_rounds":round_count(),"history_limit":HISTORY_LIMIT,"minimum_train":MIN_TRAIN,"embargo":EMBARGO,"signal_threshold":SIGNAL_THRESHOLD,"firestore_enabled":bool(FIREBASE_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS)}
@app.get("/api/memory")
def memory():
    p=psutil.Process(os.getpid());m=p.memory_info();return {"rss_mb":round(m.rss/1048576,2),"vms_mb":round(m.vms/1048576,2),"threads":p.num_threads(),"history_limit":HISTORY_LIMIT,"firestore_transport":"REST"}
@app.get("/api/history")
def history(limit:int=100):
    r=load_rounds(max(1,min(limit,500)));r.reverse();return {"rounds":r,"count":len(r)}
@app.get("/api/predictions")
def predictions(limit:int=100):
    r=load_predictions(max(1,min(limit,500)));return {"predictions":r,"count":len(r)}
@app.get("/api/backtest")
def backtest():
    r=load_rounds(HISTORY_LIMIT);latest_or_build(r);return {"model_version":MODEL_VERSION,"walk_forward":MODEL_CACHE.get("report"),"ablation":MODEL_CACHE.get("ablation")}
@app.get("/api/dashboard")
def dashboard():
    r=load_rounds(HISTORY_LIMIT);pred=latest_or_build(r) if r else {"ready":False,"reason":"no_history","model_version":MODEL_VERSION}
    return {"version":APP_VERSION,"model_version":MODEL_VERSION,"latest_round":r[-1] if r else None,"prediction":pred,"accuracy":live_accuracy(),"walk_forward":MODEL_CACHE.get("report"),"ablation":MODEL_CACHE.get("ablation"),"recent_rounds":list(reversed(r[-30:])),"recent_predictions":load_predictions(30),"status":{"stored_rounds":len(r),"history_limit":HISTORY_LIMIT,"minimum_train":MIN_TRAIN,"signal_threshold":SIGNAL_THRESHOLD,"firestore_enabled":bool(FIREBASE_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS)}}

@app.post("/api/ingest")
def ingest(item:RoundInput):
    if not INGEST_SECRET or item.secret!=INGEST_SECRET:return {"ok":False,"error":"Unauthorized"}
    issue=str(item.issue).strip();n=int(item.number);color=normalize_color(item.color)
    if not issue:return {"ok":False,"error":"Missing issue"}
    if not 0<=n<=9:return {"ok":False,"error":"Number must be 0..9"}
    if color not in {"red","green","violet"}:return {"ok":False,"error":"Invalid color"}
    rr=rp=0;re=None
    if round_count()==0:rr,rp,re=hydrate_from_firestore()
    inserted=save_round_local(issue,n,color); cre=cloud_save_round({"issue":issue,"number":n,"color":color,"seen_at":time.time()}) if inserted else None
    ver=verify_prediction(issue,size_from_number(n)); cve=cloud_save_prediction(ver) if ver else None
    pred=None;cpe=None
    if inserted:
        rounds=load_rounds(HISTORY_LIMIT);pred=build_v9_prediction(rounds)
        if pred.get("ready"):save_prediction_local(pred);cpe=cloud_save_prediction(pred)
    return {"ok":True,"inserted":inserted,"issue":issue,"stored_rounds":round_count(),"restored_rounds":rr,"restored_predictions":rp,"restore_error":re,"verified_prediction":ver,"prediction":pred,"cloud_errors":{"round":cre,"verified_prediction":cve,"prediction":cpe}}
