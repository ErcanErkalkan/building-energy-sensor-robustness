#!/usr/bin/env python3
"""Exploratory V11 exact-marginal, paired-timestamp sensor-failure experiment.

No independent external holdout is available. All test-set contrasts remain
post-hoc, single-home sensitivity analyses and must not be called confirmatory.
Input is the unchanged original UCI archive supplied by the user.
"""
from __future__ import annotations
import argparse, hashlib, io, json, platform, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SENSORS=[x for i in range(1,10) for x in (f'T{i}',f'RH_{i}')]
WX=['T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint']
CAL=['hour_sin','hour_cos','dow_sin','dow_cos']
SEEDS=[101,102,103,104,105]
N_BLOCKS=33
BLOCK_LEN=36
K_CHANNELS=6


def build_features(source):
    with zipfile.ZipFile(source) as z:
        raw_bytes=z.read('energydata_complete.csv')
    raw=pd.read_csv(io.BytesIO(raw_bytes));raw['date']=pd.to_datetime(raw['date'])
    raw=raw.sort_values('date').reset_index(drop=True)
    assert len(raw)==19735 and not raw.date.duplicated().any() and not raw.isna().any().any()
    X=raw[SENSORS+WX].copy()
    minutes=raw.date.dt.hour*60+raw.date.dt.minute
    X['hour_sin']=np.sin(2*np.pi*minutes/1440)
    X['hour_cos']=np.cos(2*np.pi*minutes/1440)
    X['dow_sin']=np.sin(2*np.pi*raw.date.dt.dayofweek/7)
    X['dow_cos']=np.cos(2*np.pi*raw.date.dt.dayofweek/7)
    return X,raw.Appliances.to_numpy(float),hashlib.sha256(raw_bytes).hexdigest()


def two_masks(n,seed):
    """Same affected rows, same 6 missing channels per affected row,
    same total missing count per sensor; only temporal membership differs.
    """
    rng=np.random.default_rng(seed)
    slots=rng.permutation(n//BLOCK_LEN)
    chosen=[]
    for s in slots:
        if all(abs(int(s)-x)>1 for x in chosen):
            chosen.append(int(s))
            if len(chosen)==N_BLOCKS:break
    if len(chosen)!=N_BLOCKS:raise AssertionError('Cannot select spaced blocks')
    chosen=sorted(chosen)
    # Uniform group count: each six-channel group appears in 11 complete blocks.
    labels=np.repeat(np.arange(3),N_BLOCKS//3)
    rng.shuffle(labels)
    coherent=np.zeros((n,len(SENSORS)),dtype=bool)
    rotating=np.zeros_like(coherent)
    for ix,slot in enumerate(chosen):
        start=slot*BLOCK_LEN
        coherent[start:start+BLOCK_LEN,labels[ix]*6:(labels[ix]+1)*6]=True
        order=rng.permutation(3)
        for t in range(BLOCK_LEN):
            g=order[t%3]
            rotating[start+t,g*6:(g+1)*6]=True
    assert np.array_equal(coherent.any(axis=1),rotating.any(axis=1))
    assert np.array_equal(coherent.sum(axis=1),rotating.sum(axis=1))
    assert np.array_equal(coherent.sum(axis=0),rotating.sum(axis=0))
    assert (coherent.sum(axis=1)[coherent.any(axis=1)]==K_CHANNELS).all()
    assert np.all(coherent.sum(axis=0)==N_BLOCKS//3*BLOCK_LEN)
    return coherent,rotating,chosen


def impute(testX,mask,method,train_median,preceding):
    d=testX.copy()
    arr=d[SENSORS].to_numpy(copy=True)
    arr[mask]=np.nan
    d.loc[:,SENSORS]=arr
    if method=='train_median':return d.fillna(train_median)
    if method=='causal_ffill':
        a=pd.concat([preceding.to_frame().T,d],ignore_index=True).ffill().iloc[1:]
        a.index=d.index
        return a.fillna(train_median)
    raise ValueError(method)


def model_from_selection(name,params):
    if name=='Ridge':return make_pipeline(StandardScaler(),Ridge(**params))
    if name=='RandomForest':return RandomForestRegressor(**params,random_state=2026,n_jobs=2)
    if name=='HistGradientBoosting':return HistGradientBoostingRegressor(**params,random_state=2026)
    raise ValueError(name)


def bootstrap(d,B=2000,block=144,seed=260911):
    d=np.asarray(d,float);n=len(d);k=int(np.ceil(n/block))
    rg=np.random.default_rng(seed); vals=np.empty(B)
    for i in range(B):
        starts=rg.integers(0,n-block+1,size=k)
        ids=np.concatenate([np.arange(s,s+block) for s in starts])[:n]
        vals[i]=np.mean(d[ids])
    return np.percentile(vals,[2.5,97.5]).tolist()


def run(source,selected_path,out):
    out.mkdir(parents=True,exist_ok=True)
    X,y,sha=build_features(source)
    a=int(.6*len(X));b=int(.8*len(X))
    trX,teX=X.iloc[:a],X.iloc[b:]
    trY,teY=y[:a],y[b:]
    selected=pd.read_csv(selected_path)
    full=selected[selected.feature_group=='full_sensor_weather_calendar']
    assert len(full)==3
    models={}
    for item in full.itertuples():
        m=model_from_selection(item.model,json.loads(item.parameters));m.fit(trX,trY)
        clean=m.predict(teX)
        assert abs(np.mean(np.abs(clean-teY))-float(item.test_MAE_Wh))<1e-6, item.model
        models[item.model]=m
    tm=trX.median();prev=X.iloc[b-1]
    rows=[]; masks=[]; paired_series={}
    for seed in SEEDS:
        coh,rot,slots=two_masks(len(teY),seed)
        affected=coh.any(axis=1)
        masks.append({'seed':seed,'affected_readings':int(affected.sum()),'affected_fraction':float(affected.mean()),
             'missing_cells':int(coh.sum()),'missing_cell_fraction':float(coh.mean()),
             'per_channel_missing_readings':int(coh.sum(axis=0)[0]),'missing_per_affected_timestamp':K_CHANNELS,
             'all_18_absent_rows':int(coh.all(axis=1).sum()),'block_count':N_BLOCKS,'nominal_block_readings':BLOCK_LEN,
             'coherent_max_contiguous_per_sensor':BLOCK_LEN,'rotating_max_contiguous_per_sensor':1,
             'slots':json.dumps(slots)})
        for meth in ['train_median','causal_ffill']:
            dat={'coherent_six_channel_block':impute(teX,coh,meth,tm,prev),
                 'rotating_six_channel_group':impute(teX,rot,meth,tm,prev)}
            for name,mdl in models.items():
                errors={}
                for design,filled in dat.items():
                    e=np.abs(teY-mdl.predict(filled))
                    errors[design]=e
                    rows.append(dict(seed=seed,model=name,imputation=meth,design=design,
                              MAE_all_Wh=float(e.mean()),MAE_affected_Wh=float(e[affected].mean()),
                              MAE_unaffected_Wh=float(e[~affected].mean()),
                              RMSE_all_Wh=float(np.sqrt(np.mean(e**2))),
                              P95_AE_all_Wh=float(np.quantile(e,.95)),n_affected=int(affected.sum())))
                key=(name,meth)
                delta=errors['coherent_six_channel_block']-errors['rotating_six_channel_group']
                paired_series.setdefault(key,[]).append(delta)
    row_df=pd.DataFrame(rows);row_df.to_csv(out/'V11_matched_exposure_all_runs.csv',index=False)
    pd.DataFrame(masks).to_csv(out/'V11_exact_mask_audit.csv',index=False)
    contrasts=[]
    for (model,meth),deltas in paired_series.items():
        d=np.mean(deltas,axis=0)
        low,hi=bootstrap(d)
        sub=row_df[(row_df.model==model)&(row_df.imputation==meth)]
        vals=sub.groupby('design').agg(MAE=('MAE_all_Wh','mean'),MAE_affected=('MAE_affected_Wh','mean'),
                    RMSE=('RMSE_all_Wh','mean'),P95=('P95_AE_all_Wh','mean'))
        a1=vals.loc['coherent_six_channel_block'];a2=vals.loc['rotating_six_channel_group']
        contrasts.append({'model':model,'imputation':meth,
            'coherent_MAE_Wh':a1.MAE,'rotating_MAE_Wh':a2.MAE,
            'coherent_affected_MAE_Wh':a1.MAE_affected,'rotating_affected_MAE_Wh':a2.MAE_affected,
            'coherent_RMSE_Wh':a1.RMSE,'rotating_RMSE_Wh':a2.RMSE,
            'coherent_minus_rotating_MAE_Wh':float(d.mean()),
            'CI95_low_Wh':low,'CI95_high_Wh':hi,
            'CI_convention':'Exploratory 24h paired moving-block percentile interval; masks fixed'})
    cdf=pd.DataFrame(contrasts).sort_values(['model','imputation']);cdf.to_csv(out/'V11_paired_topology_contrasts.csv',index=False)
    protocol={'analysis':'V11 exact-marginal paired time-exposure sensitivity','source_sha256':sha,
      'n_total':len(X),'n_train':a,'n_validation':b-a,'n_test':len(teY),
      'time_step_minutes':10,'affected_blocks':N_BLOCKS,'block_len':BLOCK_LEN,'n_affected_per_seed':N_BLOCKS*BLOCK_LEN,
      'sensors':len(SENSORS),'missing_channels_per_affected_timestamp':K_CHANNELS,'per_channel_missing':(N_BLOCKS//3)*BLOCK_LEN,
      'mask_seed_list':SEEDS,'fit_parameters':'V9 validation-only full-feature group winners, no test-based tuning',
      'n_models':len(models),'n_imputation_methods':2,'n_run_rows':len(row_df),'n_pairs':len(cdf),
      'bootstrap_blocks_readings':144,'bootstrap_repetitions':2000,
      'limitations':['One residential series; no new independent holdout','Masks simulated; matched marginal rates do not emulate all-channel common outage',
       'Results are post-hoc exploratory; confidence limits conditional on masks and data','Outdoor T6/RH_6 are counted among local channels'],
      'python':platform.python_version(),'sklearn':sklearn.__version__}
    (out/'V11_protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+'\n')
    print('RESULTS\n',cdf.round(3).to_string(index=False));print('MASKS\n',pd.DataFrame(masks).iloc[:,:9].to_string(index=False))
    return cdf

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path(__file__).resolve().parent/'data/uci_appliances_energy_original.zip')
    p.add_argument('--selected',type=Path,default=Path(__file__).resolve().parent/'outputs/V9_tuned_feature_groups.csv')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'outputs')
    a=p.parse_args();run(a.source,a.selected,a.out)
