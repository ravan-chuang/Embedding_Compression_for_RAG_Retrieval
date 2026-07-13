from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd
SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"analyze_external_query_diagnostics.py"
SPEC=importlib.util.spec_from_file_location("diag",SCRIPT); MODULE=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)

def test_classify_difference():
    assert MODULE.classify_difference(1)=="win"
    assert MODULE.classify_difference(-1)=="loss"
    assert MODULE.classify_difference(0)=="tie"

def test_leave_one_query_out():
    d={"qid":["a","b","c"]}
    for s in MODULE.SYSTEMS:
        for m in MODULE.METRICS: d[f"{s}_{m}"]=[0.,0.,0.]
    d["rars_r16_int8_recall@10"]=[1.,0.,0.]
    d["pca_r16_int8_recall@10"]=[0.,.5,0.]
    r=MODULE.leave_one_query_out(pd.DataFrame(d))
    p=r[(r["contrast"]=="rars_minus_pca")&(r["metric"]=="recall@10")]
    assert np.isclose(p.iloc[0]["leave_one_out_mean_difference"],-.25)

def test_tradeoff_flags():
    d={"qid":["q1"]}
    vals={"base_m32":[.5,1,.5,.5],"pca_r16_int8":[.5,1,.5,.5],"rars_r16_int8":[.25,1,1,.8]}
    for s,v in vals.items():
        for m,x in zip(MODULE.METRICS,v): d[f"{s}_{m}"]=[x]
    out=MODULE.build_query_table(pd.DataFrame(d),{"query_ids":["q1"],"query_texts":["example query"]},{"q1":{1,2,3,4}})
    assert out.iloc[0]["rars_vs_pca_recall_group"]=="loss"
    assert bool(out.iloc[0]["rars_mrr_up_recall_down_vs_pca"])
