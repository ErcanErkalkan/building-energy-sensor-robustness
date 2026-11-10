#!/usr/bin/env python3
"""V12 *exploratory* chronological-origin robustness of V11 within same UCI home.

No new building; previous research had access to the entire dataset. All V12
results are post-hoc temporal sensitivity analyses, not external validation.
No final 80-100% test interval used anywhere in these additional folds.
"""
from __future__ import annotations
import io,zipfile,json,hashlib,argparse
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

S=[q for j in range(1,10) for q in (f'T{j}',f'RH_{j}')]
WX=['T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint']
CAL=['hour_sin','hour_cos','dow_sin','dow_cos']
SEEDS=[101,102,103]
FOLDS=[(.35,.40,.50),(.45,.50,.60),(.55,.60,.70),(.65,.70,.80)]
CANDIDATES={
'Ridge':[{'alpha':1.0},{'alpha':100.0}],
'HistGradientBoosting':[{'max_iter':80,'max_leaf_nodes':15,'learning_rate':0.05,'l2_regularization':1.0},
                        {'max_iter':110,'max_leaf_nodes':20,'learning_rate':0.07,'l2_regularization':1.0}]
}

def model(name,p):
 if name=='Ridge':return make_pipeline(StandardScaler(),Ridge(**p))
 return HistGradientBoostingRegressor(**p,random_state=2026)

def mae(y,p):return float(np.abs(y-p).mean())
def rmse(y,p):return float(np.sqrt(((y-p)**2).mean()))

def masks(n,seed,blocks=15,block_size=36):
 """Match affected timestamps, six absent channels each, and each channel margin.
 Coherent missing channels fixed during 6h windows; rotating swaps every 10min.
 """
 rg=np.random.default_rng(seed)
 slots=rg.permutation(n//block_size)
 chosen=[]
 for idx in slots:
  idx=int(idx)
  if all(abs(idx-k)>1 for k in chosen):
   chosen.append(idx)
   if len(chosen)==blocks:break
 if len(chosen)<blocks:raise ValueError('Could not space outage blocks')
 labels=np.repeat(np.arange(3),blocks//3);rg.shuffle(labels)
 coh=np.zeros((n,18),bool);rot=coh.copy()
 for j,idx in enumerate(sorted(chosen)):
  a=idx*block_size; group=int(labels[j]);coh[a:a+block_size,6*group:6*group+6]=True
  order=rg.permutation(3)
  for offset in range(block_size):
   g=int(order[offset%3]);rot[a+offset,6*g:6*g+6]=True
 assert np.array_equal(coh.any(1),rot.any(1))
 assert np.array_equal(coh.sum(1),rot.sum(1))
 assert np.array_equal(coh.sum(0),rot.sum(0))
 assert (coh.sum(1)[coh.any(1)]==6).all()
 assert np.all(coh.sum(0)==blocks//3*block_size)
 return coh,rot

def fill(x,mask,method,median,prev):
 z=x.copy();a=z[S].to_numpy(copy=True);a[mask]=np.nan;z.loc[:,S]=a
 if method=='median':return z.fillna(median)
 if method=='causal_ffill':
  zz=pd.concat([prev.to_frame().T,z],ignore_index=True).ffill().iloc[1:]
  zz.index=z.index
  return zz.fillna(median)
 raise ValueError(method)

def run(archive,out):
 out.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive) as z:csv=z.read('energydata_complete.csv')
 sha=hashlib.sha256(csv).hexdigest()
 raw=pd.read_csv(io.BytesIO(csv));raw['date']=pd.to_datetime(raw['date']);raw=raw.sort_values('date').reset_index(drop=True)
 assert len(raw)==19735 and not raw.isna().any().any() and raw.date.is_monotonic_increasing
 X=raw[S+WX].copy();minutes=raw.date.dt.hour*60+raw.date.dt.minute
 X['hour_sin']=np.sin(2*np.pi*minutes/1440);X['hour_cos']=np.cos(2*np.pi*minutes/1440)
 X['dow_sin']=np.sin(2*np.pi*raw.date.dt.dayofweek/7);X['dow_cos']=np.cos(2*np.pi*raw.date.dt.dayofweek/7)
 y=raw.Appliances.to_numpy(dtype=float);N=len(y)
 groups={'weather_calendar':WX+CAL,'full_sensor_weather_calendar':S+WX+CAL}
 scores=[];contrast=[];cand_audit=[];mask_audit=[]
 for fi,(trp,vp,ep) in enumerate(FOLDS,1):
  train=int(N*trp);val=int(N*vp);end=int(N*ep)
  assert 0<train<val<end<=int(N*.8)
  fold_y=y[val:end];med=float(np.median(y[:train]));base=mae(fold_y,np.full(len(fold_y),med))
  scores.append(dict(fold=fi,model='TrainingMedian',features='none',train_end=train,validation_end=val,eval_end=end,
    n_eval=end-val,validation_MAE_Wh=mae(y[train:val],np.full(val-train,med)),eval_MAE_Wh=base,
    eval_RMSE_Wh=rmse(fold_y,np.full(len(fold_y),med)),selected_params='{}'))
  print('fold',fi,'train/val/eval',train,val,end,'median_mae',round(base,3),flush=True)
  selected={}
  for name in CANDIDATES:
   for group,cols in groups.items():
    candidates=[]
    for ci,params in enumerate(CANDIDATES[name],1):
     m=model(name,params);m.fit(X.iloc[:train][cols],y[:train]);pred=m.predict(X.iloc[train:val][cols]);score=mae(y[train:val],pred)
     candidates.append((score,ci,params,m))
     cand_audit.append(dict(fold=fi,model=name,features=group,candidate=ci,params=json.dumps(params,sort_keys=True),validation_MAE_Wh=score))
    score,ci,params,best=min(candidates,key=lambda x:(x[0],x[1]))
    p=best.predict(X.iloc[val:end][cols]);ev=mae(fold_y,p)
    scores.append(dict(fold=fi,model=name,features=group,train_end=train,validation_end=val,eval_end=end,
       n_eval=end-val,validation_MAE_Wh=score,eval_MAE_Wh=ev,eval_RMSE_Wh=rmse(fold_y,p),selected_params=json.dumps(params,sort_keys=True)))
    if group=='full_sensor_weather_calendar':selected[name]=(score,ci,params,best)
  for seed in SEEDS:
   coh,rot=masks(end-val,seed)
   affected=coh.any(1)
   mask_audit.append(dict(fold=fi,seed=seed,n_eval=end-val,affected_timestamps=int(affected.sum()),
      missing_cells=int(coh.sum()),missing_fraction=float(coh.mean()),
      missing_per_sensor=int(coh.sum(0)[0]),missing_per_affected_t=6,
      equal_timestamp_exposure=bool(np.array_equal(coh.any(1),rot.any(1))),
      equal_sensor_margins=bool(np.array_equal(coh.sum(0),rot.sum(0)))))
   for name,(vscore,ci,params,m) in selected.items():
    for method in ('median','causal_ffill'):
     med_feature=X.iloc[:train].median()
     p1=m.predict(fill(X.iloc[val:end],coh,method,med_feature,X.iloc[val-1]))
     p2=m.predict(fill(X.iloc[val:end],rot,method,med_feature,X.iloc[val-1]))
     e1=np.abs(fold_y-p1);e2=np.abs(fold_y-p2)
     contrast.append(dict(fold=fi,seed=seed,model=name,imputation=method,
        coherent_MAE_Wh=float(e1.mean()),rotating_MAE_Wh=float(e2.mean()),
        diff_MAE_Wh=float((e1-e2).mean()),
        diff_affected_MAE_Wh=float((e1[affected]-e2[affected]).mean()),
        coherent_RMSE_Wh=rmse(fold_y,p1),rotating_RMSE_Wh=rmse(fold_y,p2)))
 scores=pd.DataFrame(scores);con=pd.DataFrame(contrast);ca=pd.DataFrame(cand_audit);mask=pd.DataFrame(mask_audit)
 scores.to_csv(out/'V12_rolling_origin_clean_controls.csv',index=False)
 con.to_csv(out/'V12_matched_topology_all_seeds.csv',index=False)
 ca.to_csv(out/'V12_validation_candidate_audit.csv',index=False)
 mask.to_csv(out/'V12_mask_checks.csv',index=False)
 group=con.groupby(['fold','model','imputation']).agg(mean_diff=('diff_MAE_Wh','mean'),
  min_diff=('diff_MAE_Wh','min'),max_diff=('diff_MAE_Wh','max'),mean_affected_diff=('diff_affected_MAE_Wh','mean')).reset_index()
 group.to_csv(out/'V12_fold_topology_summary.csv',index=False)
 meta={'status':'post-hoc exploratory chronological sensitivity, not independent external validation',
  'source_sha256_csv':sha,'n_data':N,'fold_proportions':FOLDS,'seeds':SEEDS,'n_folds':4,
  'models':list(CANDIDATES.keys()),'n_candidates_per_family_and_group':2,'n_baseline_rows':len(scores),
  'n_topology_run_rows':len(con),'affected_per_fold_and_seed':540,
  'per_sensor_missing_per_seed':180,'n_channels':18,
  'validation':'parameters fit on earlier training and chosen via immediately next validation segment only; no future leakage',
  'no_test_80_100':'Original 80-100 percent terminal test segment not used in this V12 check',
  'limits':['Same house and known source; folds not independently acquired','Overlapping expanding training periods; no fold-level independent sampling assumptions',
  'No retrospective hypothesis or model-selection revision can make this confirmatory','Masks simulated, 6 channel groups not 18-channel common failure',
  'Cross-dataset candidate SmartlivingEPC not downloaded/validated']}
 (out/'V12_protocol.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
 print('\nCLEAN',scores[['fold','model','features','validation_MAE_Wh','eval_MAE_Wh','eval_RMSE_Wh']].round(2).to_string(index=False),flush=True)
 print('\nTOPOLOGY',group.round(3).to_string(index=False),flush=True)
 return scores,con,group

if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--source',type=Path,default=Path(__file__).resolve().parent/'data/uci_appliances_energy_original.zip')
 pa.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'outputs')
 args=pa.parse_args();run(args.source,args.out)
