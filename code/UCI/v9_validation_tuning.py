#!/usr/bin/env python3
"""V9 exploratory equal-candidate-budget validation-only tuning.
Never select parameters or feature groups using test labels. Historical prior test reuse
makes ALL test comparisons exploratory, not independently confirmatory.
"""
from __future__ import annotations
import argparse, hashlib, io, json, platform, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sklearn

SENSORS = [x for i in range(1,10) for x in (f'T{i}',f'RH_{i}')]
CAL=['hour_sin','hour_cos','dow_sin','dow_cos']
WX=['T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint']
CANDIDATES={
 'Ridge':[
  {'alpha':0.1},{'alpha':1.0},{'alpha':10.0},{'alpha':100.0}],
 'RandomForest':[
  {'n_estimators':60,'max_depth':8,'min_samples_leaf':8},
  {'n_estimators':60,'max_depth':12,'min_samples_leaf':4},
  {'n_estimators':60,'max_depth':16,'min_samples_leaf':2},
  {'n_estimators':60,'max_depth':None,'min_samples_leaf':8}],
 'HistGradientBoosting':[
  {'max_iter':80,'max_leaf_nodes':15,'learning_rate':0.05,'l2_regularization':1.0},
  {'max_iter':110,'max_leaf_nodes':20,'learning_rate':0.07,'l2_regularization':1.0},
  {'max_iter':150,'max_leaf_nodes':15,'learning_rate':0.03,'l2_regularization':5.0},
  {'max_iter':80,'max_leaf_nodes':31,'learning_rate':0.1,'l2_regularization':0.1}]
}

def create(name,params):
    if name=='Ridge':return make_pipeline(StandardScaler(),Ridge(**params))
    if name=='RandomForest':return RandomForestRegressor(**params,random_state=2026,n_jobs=2)
    if name=='HistGradientBoosting':return HistGradientBoostingRegressor(**params,random_state=2026)
    raise ValueError(name)

def metric(y,p):
    e=np.abs(np.asarray(y)-np.asarray(p))
    return {'MAE_Wh':float(e.mean()),'RMSE_Wh':float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2))),
        'P95_AE_Wh':float(np.percentile(e,95))}

def boot_ci(d,block=144,B=2000,seed=20261109):
    d=np.asarray(d,float); n=len(d);k=int(np.ceil(n/block));rng=np.random.default_rng(seed)
    means=np.empty(B)
    for i in range(B):
        starts=rng.integers(0,n-block+1,k)
        sample=np.concatenate([np.arange(st,st+block) for st in starts])[:n]
        means[i]=d[sample].mean()
    return [float(x) for x in np.quantile(means,[.025,.975])]

def run(path,out):
    out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        blob=z.read('energydata_complete.csv')
    sha=hashlib.sha256(blob).hexdigest()
    raw=pd.read_csv(io.BytesIO(blob));raw['date']=pd.to_datetime(raw['date']);raw=raw.sort_values('date').reset_index(drop=True)
    if len(raw)!=19735 or raw['date'].duplicated().any() or raw.isna().sum().sum()!=0:raise ValueError('Unexpected dataset')
    date=raw.date;features=SENSORS+WX
    X=raw[features].copy()
    X['hour_sin']=np.sin(2*np.pi*(date.dt.hour*60+date.dt.minute)/1440)
    X['hour_cos']=np.cos(2*np.pi*(date.dt.hour*60+date.dt.minute)/1440)
    X['dow_sin']=np.sin(2*np.pi*date.dt.dayofweek/7)
    X['dow_cos']=np.cos(2*np.pi*date.dt.dayofweek/7)
    y=raw.Appliances.to_numpy(float)
    a,b=int(.6*len(X)),int(.8*len(X))
    feat={'calendar_only':CAL,'weather_calendar':WX+CAL,'full_sensor_weather_calendar':features+CAL}
    allc=[]; selections=[]; preds={}
    start=time.time()
    for name in CANDIDATES:
        for group,cols in feat.items():
            group_results=[]
            for index,params in enumerate(CANDIDATES[name],1):
                model=create(name,params)
                tic=time.perf_counter();model.fit(X.iloc[:a][cols],y[:a]);fit_s=time.perf_counter()-tic
                pred_v=model.predict(X.iloc[a:b][cols]);v=metric(y[a:b],pred_v)
                record=dict(model=name,feature_group=group,candidate=index,params=json.dumps(params,sort_keys=True),
                            validation_MAE_Wh=v['MAE_Wh'],validation_RMSE_Wh=v['RMSE_Wh'],fit_seconds=fit_s)
                allc.append(record);group_results.append((v['MAE_Wh'],index,params,model))
            # Ties are broken by candidate index, never test data.
            best=min(group_results,key=lambda z:(z[0],z[1]))
            p=best[3].predict(X.iloc[b:][cols]);test=metric(y[b:],p)
            selections.append(dict(model=name,feature_group=group,selected_candidate=best[1],
                   parameters=json.dumps(best[2],sort_keys=True),validation_MAE_Wh=best[0],**{'test_'+k:v for k,v in test.items()},
                   training_n=a,validation_n=b-a,test_n=len(y)-b,selection_rule='lowest validation MAE among four candidates'))
            preds[(name,group)]=p
    tab=pd.DataFrame(allc);sel=pd.DataFrame(selections)
    tab.to_csv(out/'V9_validation_candidate_grid.csv',index=False)
    sel.to_csv(out/'V9_tuned_feature_groups.csv',index=False)
    family=sel.sort_values(['model','validation_MAE_Wh','feature_group'],kind='stable').groupby('model',sort=False).head(1).sort_values('model')
    family.to_csv(out/'V9_validation_selected_families.csv',index=False)
    baseline=[]
    for label,value in [('Training median',float(np.median(y[:a]))),('Training mean',float(y[:a].mean()))]:
        baseline.append(dict(baseline=label,validation_MAE_Wh=metric(y[a:b],np.full(b-a,value))['MAE_Wh'],
                **{'test_'+k:v for k,v in metric(y[b:],np.full(len(y)-b,value)).items()}))
    pd.DataFrame(baseline).to_csv(out/'V9_constant_baselines.csv',index=False)
    med=np.median(y[:a]);me=np.abs(y[b:]-med)
    diffs=[]
    for row in family.itertuples():
        d=np.abs(y[b:]-preds[(row.model,row.feature_group)])-me
        diffs.append(dict(model=row.model,feature_group=row.feature_group,
            tuned_minus_median_MAE_Wh=float(d.mean()),CI95_low_Wh=boot_ci(d)[0],CI95_high_Wh=boot_ci(d)[1],
            direction='positive means training-median control has lower test MAE'))
    pd.DataFrame(diffs).to_csv(out/'V9_exploratory_paired_block_bootstrap.csv',index=False)
    meta=dict(source_sha256=sha,n_rows=len(raw),train=a,validation=b-a,test=len(y)-b,
            train_end=str(date.iloc[a-1]),validation_end=str(date.iloc[b-1]),test_end=str(date.iloc[-1]),
            feature_groups={k:cols for k,cols in feat.items()},candidates_per_model=4,
            selection='Four fixed configurations for every family x feature-group; validation MAE only; no test selection',
            training='Only first 60 percent used to fit; no train+validation refit after selection',
            information='All are contemporaneous estimation without target lag; baseline target-median trained only on first 60 percent',
            CAVEAT='Final test segment previously consulted extensively in V1-V8. All V9 test results are exploratory, NOT a new independent holdout.',
            python=platform.python_version(),sklearn=sklearn.__version__,elapsed_s=round(time.time()-start,2))
    (out/'V9_protocol_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print('GRID candidates',len(tab),'group winners',len(sel),'\n',sel[['model','feature_group','selected_candidate','validation_MAE_Wh','test_MAE_Wh','test_RMSE_Wh']].round(3).to_string(index=False))
    print('FAMILY BEST VALIDATION\n',family[['model','feature_group','validation_MAE_Wh','test_MAE_Wh']].round(3).to_string(index=False))
    print('CONSTANTS\n',pd.DataFrame(baseline).round(3).to_string(index=False))
    print('PAIRED\n',pd.DataFrame(diffs).round(3).to_string(index=False));print('METADATA',json.dumps(meta)[:300])
    return sel

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--source',type=Path,default=Path('/mnt/data/appliances+energy+prediction.zip'))
    pa.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'outputs')
    a=pa.parse_args();run(a.source,a.out)
