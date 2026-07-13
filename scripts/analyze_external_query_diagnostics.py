#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

SYSTEMS=("base_m32","pca_r16_int8","rars_r16_int8")
METRICS=("recall@10","success@10","mrr@10","ndcg@10")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--per-query-metrics",required=True,type=Path)
    p.add_argument("--query-manifest",required=True,type=Path)
    p.add_argument("--qrels",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    return p.parse_args()

def read_json(p:Path)->Any:
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(p:Path,v:Any):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def load_positive_qrels(path:Path):
    out={}
    with path.open(encoding="utf-8") as f:
        for n,raw in enumerate(f,1):
            line=raw.strip()
            if not line or line.startswith("#"): continue
            parts=line.replace(",","\t").split()
            if n==1 and any(x.lower() in {"qid","query_id"} for x in parts): continue
            if len(parts)==3: qid,docid,rel=parts
            elif len(parts)>=4: qid,_,docid,rel=parts[:4]
            else: raise ValueError(f"Unsupported qrels line {n}")
            if float(rel)>0: out.setdefault(str(qid),set()).add(int(docid))
    return out

def classify_difference(v:float,atol:float=1e-12):
    return "win" if v>atol else "loss" if v<-atol else "tie"

def build_query_table(frame,manifest,positives):
    out=frame.copy()
    out["qid"]=out["qid"].astype(str)
    texts=manifest.get("query_texts",[""]*len(out))
    if len(texts)!=len(out): raise ValueError("query_texts length mismatch")
    if out["qid"].tolist()!=[str(x) for x in manifest["query_ids"]]:
        raise ValueError("Per-query CSV order does not match query manifest")
    out.insert(1,"query_text",texts)
    out.insert(2,"query_token_count",[len(str(x).split()) for x in texts])
    out.insert(3,"positive_qrel_count",[len(positives[q]) for q in out["qid"]])
    for m in METRICS:
        s=m.replace("@","_at_")
        out[f"rars_minus_pca_{s}"]=out[f"rars_r16_int8_{m}"]-out[f"pca_r16_int8_{m}"]
        out[f"pca_minus_base_{s}"]=out[f"pca_r16_int8_{m}"]-out[f"base_m32_{m}"]
        out[f"rars_minus_base_{s}"]=out[f"rars_r16_int8_{m}"]-out[f"base_m32_{m}"]
    out["rars_vs_pca_recall_group"]=out["rars_minus_pca_recall_at_10"].map(classify_difference)
    out["rars_mrr_up_recall_down_vs_pca"]=(out["rars_minus_pca_mrr_at_10"]>1e-12)&(out["rars_minus_pca_recall_at_10"]<-1e-12)
    out["rars_ndcg_up_recall_down_vs_pca"]=(out["rars_minus_pca_ndcg_at_10"]>1e-12)&(out["rars_minus_pca_recall_at_10"]<-1e-12)
    return out

def leave_one_query_out(frame):
    rows=[]
    for name,left,right in [("rars_minus_pca","rars_r16_int8","pca_r16_int8"),("pca_minus_base","pca_r16_int8","base_m32"),("rars_minus_base","rars_r16_int8","base_m32")]:
        for m in METRICS:
            d=(frame[f"{left}_{m}"]-frame[f"{right}_{m}"]).to_numpy(float)
            total=float(d.sum()); n=len(d)
            for i,qid in enumerate(frame["qid"].astype(str)):
                rows.append({"qid":qid,"contrast":name,"metric":m,"full_mean_difference":float(d.mean()),"leave_one_out_mean_difference":(total-float(d[i]))/(n-1),"removed_query_difference":float(d[i])})
    return pd.DataFrame(rows)

def summarize(frame,loo):
    r={"query_count":int(len(frame)),"analysis_type":"post_hoc_diagnostic_only","retrieval_performed":False,"fitting_performed":False,"selection_performed":False,"retuning_performed":False,"rars_vs_pca":{}}
    for m in METRICS:
        s=m.replace("@","_at_"); x=frame[f"rars_minus_pca_{s}"]; c=x.map(classify_difference).value_counts()
        r["rars_vs_pca"][m]={"mean_difference":float(x.mean()),"win_tie_loss":{"win":int(c.get("win",0)),"tie":int(c.get("tie",0)),"loss":int(c.get("loss",0))}}
    r["tradeoffs"]={"mrr_up_recall_down_count":int(frame["rars_mrr_up_recall_down_vs_pca"].sum()),"ndcg_up_recall_down_count":int(frame["rars_ndcg_up_recall_down_vs_pca"].sum())}
    p=loo[(loo["contrast"]=="rars_minus_pca")&(loo["metric"]=="recall@10")]
    r["leave_one_query_out_primary"]={"minimum_difference":float(p["leave_one_out_mean_difference"].min()),"maximum_difference":float(p["leave_one_out_mean_difference"].max()),"all_negative":bool((p["leave_one_out_mean_difference"]<0).all()),"sign_changes":int((np.sign(p["leave_one_out_mean_difference"])!=np.sign(p["full_mean_difference"])).sum())}
    return r

def main():
    a=parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(a.per_query_metrics,dtype={"qid":str})
    m=read_json(a.query_manifest); q=load_positive_qrels(a.qrels)
    missing=[x for x in f["qid"].astype(str) if not q.get(x)]
    if missing: raise ValueError(f"Missing positive qrels: {missing[:5]}")
    t=build_query_table(f,m,q); loo=leave_one_query_out(t); s=summarize(t,loo)
    t.to_csv(a.output_dir/"query_outcome_groups.csv",index=False)
    t[["qid","query_text","query_token_count","positive_qrel_count","rars_minus_pca_recall_at_10","rars_minus_pca_success_at_10","rars_minus_pca_mrr_at_10","rars_minus_pca_ndcg_at_10","rars_vs_pca_recall_group","rars_mrr_up_recall_down_vs_pca","rars_ndcg_up_recall_down_vs_pca"]].to_csv(a.output_dir/"metric_tradeoffs.csv",index=False)
    loo.to_csv(a.output_dir/"leave_one_query_out.csv",index=False)
    write_json(a.output_dir/"external_diagnostic_summary.json",s)
    print(json.dumps(s,indent=2))
if __name__=="__main__": main()
