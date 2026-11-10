#!/usr/bin/env python3
"""Post-hoc sensor grouping sensitivity on unchanged UCI residence.

Reuses V12 validation-selected full-feature models for four earlier time windows.
The final 80%-100% historical holdout is not touched. For each timestamp mask,
permutes which physical sensors belong to three 6-channel groups, matching
row exposure, number missing per row, and total missing count per channel.
Results are descriptive sensitivity tests, NOT independent validation or p-values.
"""
from __future__ import annotations
import argparse, io, json, hashlib, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import v12_rolling_origin as old

SEEDS=[101,102,103]
PARTITIONS=10  # Original arrangement + 9 distinct permutations
MODEL_NAMES=['Ridge','HistGradientBoosting']
METHODS=['median','causal_ffill']


def load_data(source):
    with zipfile.ZipFile(source) as z: raw_bytes=z.read('energydata_complete.csv')
    raw=pd.read_csv(io.BytesIO(raw_bytes));raw['date']=pd.to_datetime(raw['date'])
    raw=raw.sort_values('date').reset_index(drop=True)
    assert len(raw)==19735 and raw.date.is_monotonic_increasing and not raw.date.duplicated().any()
    assert not raw.isna().any().any()
    X=raw[old.S+old.WX].copy()
    minute=raw.date.dt.hour*60+raw.date.dt.minute
    X['hour_sin']=np.sin(2*np.pi*minute/1440)
    X['hour_cos']=np.cos(2*np.pi*minute/1440)
    X['dow_sin']=np.sin(2*np.pi*raw.date.dt.dayofweek/7)
    X['dow_cos']=np.cos(2*np.pi*raw.date.dt.dayofweek/7)
    return X,raw.Appliances.to_numpy(float),hashlib.sha256(raw_bytes).hexdigest()


def randomize_grouping(mask, seed, partition):
    # The baseline grouping is exactly the V12 arrangement; others are fixed
    # pseudo-random remappings, not additional independently sampled buildings.
    perm=np.arange(18) if partition==0 else np.random.default_rng(
        seed + 100_000*partition + 13).permutation(18)
    return mask[:,perm],perm


def run(source,selection_csv,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    X,y,sha=load_data(source)
    selections=pd.read_csv(selection_csv)
    rows=[];audit=[]
    n=len(X)
    for fold,(train_fraction,val_fraction,end_fraction) in enumerate(old.FOLDS,1):
        a,b,end=[int(n*p) for p in (train_fraction,val_fraction,end_fraction)]
        assert end<=int(.8*n)
        selected=selections[(selections.fold==fold) & (selections.features=='full_sensor_weather_calendar')]
        assert set(selected.model)==set(MODEL_NAMES)
        fit={}
        for rec in selected.itertuples():
            params=json.loads(rec.selected_params)
            estimator=old.model(rec.model,params)
            estimator.fit(X.iloc[:a],y[:a]);fit[rec.model]=estimator
            clean=old.mae(y[b:end],estimator.predict(X.iloc[b:end]))
            assert abs(clean-float(rec.eval_MAE_Wh))<1e-7,(fold,rec.model,clean,rec.eval_MAE_Wh)
        train_median=X.iloc[:a].median();last_seen=X.iloc[b-1]
        local=X.iloc[b:end];truth=y[b:end]
        for seed in SEEDS:
            original_coh,original_rot=old.masks(end-b,seed)
            for partition in range(PARTITIONS):
                coh,permutation=randomize_grouping(original_coh,seed,partition)
                rot,_=randomize_grouping(original_rot,seed,partition)
                assert np.array_equal(coh.any(axis=1),rot.any(axis=1))
                assert np.array_equal(coh.sum(axis=1),rot.sum(axis=1))
                assert np.array_equal(coh.sum(axis=0),rot.sum(axis=0))
                assert (coh.sum(0)==180).all() and (coh.sum(1)[coh.any(1)]==6).all()
                assert int(coh.sum())==3240 and int(coh.any(1).sum())==540
                affected=coh.any(1)
                audit.append(dict(fold=fold,seed=seed,partition=partition,
                    n_eval=len(truth),affected_rows=int(affected.sum()),
                    missing_cells=int(coh.sum()),per_channel_missing=180,
                    channels_missing_per_affected_row=6,mask_equal_margins=True,
                    channel_order=';'.join(map(str,permutation))))
                for method in METHODS:
                    inp_coh=old.fill(local,coh,method,train_median,last_seen)
                    inp_rot=old.fill(local,rot,method,train_median,last_seen)
                    for name,estimator in fit.items():
                        p_coh=estimator.predict(inp_coh)
                        p_rot=estimator.predict(inp_rot)
                        errors_coh=np.abs(truth-p_coh)
                        errors_rot=np.abs(truth-p_rot)
                        d=errors_coh-errors_rot
                        rows.append(dict(fold=fold,seed=seed,partition=partition,model=name,
                            imputation=method,coherent_MAE_Wh=float(errors_coh.mean()),
                            rotating_MAE_Wh=float(errors_rot.mean()),
                            delta_MAE_Wh=float(d.mean()),
                            delta_affected_MAE_Wh=float(d[affected].mean()),
                            abs_delta_MAE_Wh=float(abs(d.mean())),
                            affected_rows=int(affected.sum()),n_eval=len(truth)))
        print('finished fold',fold,'rows',len(rows),flush=True)
    df=pd.DataFrame(rows);audit=pd.DataFrame(audit)
    assert len(df)==4*3*PARTITIONS*2*2 and len(audit)==4*3*PARTITIONS
    df.to_csv(out/'V13_all_groupings.csv',index=False)
    audit.to_csv(out/'V13_mask_audit.csv',index=False)
    summary=(df.groupby(['model','imputation']).agg(
        n_masks=('delta_MAE_Wh','size'),mean_delta=('delta_MAE_Wh','mean'),
        median_delta=('delta_MAE_Wh','median'),
        min_delta=('delta_MAE_Wh','min'),max_delta=('delta_MAE_Wh','max'),
        frac_positive=('delta_MAE_Wh',lambda v:float(np.mean(np.array(v)>0))),
        mean_abs_delta=('abs_delta_MAE_Wh','mean'),
        median_affected_delta=('delta_affected_MAE_Wh','median')).reset_index())
    summary.to_csv(out/'V13_grouping_summary.csv',index=False)
    folds=(df.groupby(['fold','model','imputation']).agg(
        mean_delta=('delta_MAE_Wh','mean'),
        min_delta=('delta_MAE_Wh','min'),
        max_delta=('delta_MAE_Wh','max'),
        positive_fraction=('delta_MAE_Wh',lambda v:float(np.mean(np.array(v)>0)))).reset_index())
    folds.to_csv(out/'V13_fold_summary.csv',index=False)
    meta={'label':'post-hoc exploratory grouping sensitivity only','source_sha256':sha,
      'n_measurements':n,'n_folds':4,'seeds':SEEDS,'n_groupings':PARTITIONS,
      'n_masks':len(audit),'n_paired_contrasts':len(df),
      'grouping':'identity + 9 fixed pseudo-random channel permutations for each seed',
      'same_masks':'each pair matches affected timestamps, missing channels per affected row, and per-channel missingness',
      'source_models':'V12 selected full-feature regressors; selected using earlier validation only',
      'no_final_80_100_pct':'No observation in original final 20% is used in this extension',
      'uncertainty':'No inferential confidence intervals computed; permutations and rolling folds are not independent research sites',
      'external_validation':'NOT PERFORMED; SmartlivingEPC raw data not available in runtime'}
    (out/'V13_protocol.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    print(summary.round(4).to_string(index=False),flush=True)
    return df,summary,folds

if __name__=='__main__':
    p=argparse.ArgumentParser()
    base=Path(__file__).resolve().parent
    p.add_argument('--data',type=Path,default=base/'data/uci_appliances_energy_original.zip')
    p.add_argument('--selection',type=Path,default=base/'outputs/V12_rolling_origin_clean_controls.csv')
    p.add_argument('--out',type=Path,default=base/'outputs')
    args=p.parse_args();run(args.data,args.selection,args.out)
