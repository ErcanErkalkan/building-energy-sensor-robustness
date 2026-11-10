#!/usr/bin/env python3
"""V14 external-site replication. Observational sites differ from UCI; no zero-shot claim.

Source: SmartlivingEPC 10.5281/zenodo.15781077; supplied archives.
Frozen analysis decisions: original meter diff for exact one-hour intervals; earliest two
co-sited temp/humidity device groups selected by TRAIN PERIOD COVERAGE, then by ID;
full-case evaluation, 2/4 missing and hour-block common vs rotating missingness.
"""
from pathlib import Path
import zipfile, hashlib, json, argparse
from io import BytesIO
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

SITE_NAMES = ["Single Family House", "Private flat", "School", "Sports Center"]
ALL_ARCHIVES = ["Single Family House", "Private flat", "School", "Sports Center", "Mixed-use building", "City hall", "Public lighting"]
SEEDS = [101, 102, 103]
BLOCK_H = 6
MISSING_PROPORTION = .30
MODELS = ("Ridge", "HGB")
METHODS = ("training_median", "causal_forward_fill")
BASE_COLUMNS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]


def audit_files(input_dir):
    rows=[]
    for site in ALL_ARCHIVES:
        path=input_dir/(site + '.zip')
        if not path.exists():
            rows.append({'building':site,'archive_available':False});continue
        with zipfile.ZipFile(path) as z:
            d=defaultdict(int)
            for n in z.namelist():
                if n.endswith('.csv'):d[n.rsplit('_',1)[-1].replace('.csv','')]+=1
            rows.append({'building':site,'archive_available':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'csv_files':sum(d.values()),'energy_meters':d['ENERGYMETER'],'temp_channels':d['TEMPERATURESENSOR'],'humidity_channels':d['HUMIDITYSENSOR'],'co2_channels':d['CO2SENSOR'],'pm25_channels':d['PM25SENSOR'],'eligible':site in SITE_NAMES})
    return pd.DataFrame(rows)


def hour_aggregate_sensor(z,filename,idx):
    s=pd.read_csv(z.open(filename),usecols=['timestamp','value'])
    s['timestamp']=pd.to_datetime(s['timestamp'],errors='coerce')
    s['value']=pd.to_numeric(s['value'],errors='coerce')
    s=s.dropna(subset=['timestamp']).sort_values('timestamp').drop_duplicates('timestamp',keep='last').set_index('timestamp')['value']
    # Predeclared physical sanity limits, applied without seeing target or test errors.
    if filename.endswith('TEMPERATURESENSOR.csv'): s=s.where(s.between(5,45))
    elif filename.endswith('HUMIDITYSENSOR.csv'): s=s.where(s.between(1,100))
    s=s.resample('h',label='right',closed='right').mean()
    return s.reindex(idx)


def load_site(path):
    with zipfile.ZipFile(path) as z:
        meters=[n for n in z.namelist() if n.endswith('_ENERGYMETER.csv')]
        if len(meters)!=1:raise ValueError(f'{path.stem}: expected exactly one unambiguous electricity meter, observed {len(meters)}')
        md=pd.read_csv(z.open(meters[0]))
        md['timestamp']=pd.to_datetime(md.timestamp,errors='coerce')
        md['value']=pd.to_numeric(md.value,errors='coerce')
        if md.unit.dropna().nunique()!=1 or md.unit.dropna().iloc[0]!='kWh':raise ValueError('Inconsistent meter units')
        met=(md.dropna(subset=['timestamp','value']).sort_values('timestamp').drop_duplicates('timestamp',keep='last').set_index('timestamp')['value'])
        diff=met.diff(); dt=met.index.to_series().diff().dt.total_seconds()/3600
        # Difference only exactly consecutive hours; reject negative resets. Zero retained.
        good=(dt==1)&(diff>=0)&np.isfinite(diff)
        data=pd.DataFrame({'y_kwh':diff.loc[good].astype(float)},index=met.index[good])
        if len(data)<500:raise ValueError('Not enough consecutive hourly meter readings')
        grouping=defaultdict(dict)
        for n in z.namelist():
            if n.endswith(('_TEMPERATURESENSOR.csv','_HUMIDITYSENSOR.csv')):
                dev,typ=n.rsplit('_',1)
                grouping[dev][typ.replace('SENSOR.csv','')]=n
        valid_devices={sid:files for sid,files in grouping.items() if {'TEMPERATURE','HUMIDITY'}.issubset(files)}
        if len(valid_devices)<2:raise ValueError('Less than 2 co-sited T/RH device groups')
        panel={}
        for sid,files in valid_devices.items():
            for typ in ['TEMPERATURE','HUMIDITY']:
                panel[f'{sid}_{typ}']=hour_aggregate_sensor(z,files[typ],data.index)
        raw_features=pd.DataFrame(panel,index=data.index)
        cutoff=int(.60*len(data))
        ranked=sorted(valid_devices,key=lambda k:(-raw_features[[k+'_TEMPERATURE',k+'_HUMIDITY']].iloc[:cutoff].notna().all(axis=1).sum(),k))
        ids=ranked[:2]
        cols=[f'{id_}_{feature}' for id_ in ids for feature in ['TEMPERATURE','HUMIDITY']]
        data=data.join(raw_features[cols])
        return data,ids,cols,{'meter_file':meters[0],'meter_readings':len(met),'meter_nonconsecutive_intervals':int((dt!=1).sum()-1 if len(dt)>0 else 0),'meter_negative_one_hour_increments':int(((dt==1)&(diff<0)).sum()),'target_valid_hours':len(data)}


def calendar_features(index):
    hour=index.hour + index.minute/60
    day=index.dayofweek
    month=index.month
    return pd.DataFrame({'hour_sin':np.sin(2*np.pi*hour/24),'hour_cos':np.cos(2*np.pi*hour/24),
      'dow_sin':np.sin(2*np.pi*day/7),'dow_cos':np.cos(2*np.pi*day/7),
      'month_sin':np.sin(2*np.pi*(month-1)/12),'month_cos':np.cos(2*np.pi*(month-1)/12)},index=index)


def model(name):
    if name=='Ridge':return make_pipeline(StandardScaler(),Ridge(alpha=10))
    if name=='HGB':return HistGradientBoostingRegressor(max_iter=110,max_leaf_nodes=20,learning_rate=.07,l2_regularization=1.,random_state=2026)
    raise ValueError(name)


def errors(y,p):
    return {'mae_kwh':float(mean_absolute_error(y,p)),'rmse_kwh':float(np.sqrt(mean_squared_error(y,p))),'p95ae_kwh':float(np.quantile(abs(np.asarray(y)-np.asarray(p)),.95))}


def consecutive_candidates(idx,all_good):
    # All-good is a complete-case boolean on chronological, hourly meter rows.
    flag=all_good.astype(bool)
    breaks=(~flag)|(idx.to_series().diff().ne(pd.Timedelta(hours=1)).to_numpy())
    runs=breaks.cumsum()
    candidates=[]
    # Place 6-hour candidates with a 6-hour gap, no overlap between candidates.
    i=0
    while i<len(idx):
        if not flag[i]:i+=1;continue
        j=i+1
        while j<len(idx) and flag[j] and idx[j]-idx[j-1]==pd.Timedelta(hours=1):j+=1
        for start in range(i,j-BLOCK_H+1,BLOCK_H*2):candidates.append(np.arange(start,start+BLOCK_H))
        i=j
    return candidates


def make_masks(df,cols,seed):
    idx=df.index
    good=df[cols].notna().all(axis=1).to_numpy()
    candidates=consecutive_candidates(idx,good)
    n_complete=int(good.sum())
    requested_blocks=int(np.floor(MISSING_PROPORTION*n_complete/BLOCK_H/2))*2
    nblocks=min(requested_blocks,(len(candidates)//2)*2)
    if nblocks<4:raise ValueError(f'Insufficient observed complete consecutive windows: {nblocks} blocks')
    rng=np.random.default_rng(seed)
    choice=rng.choice(len(candidates),size=nblocks,replace=False)
    # Sort selected blocks chronologically so assignment does not follow random choice order.
    blocks=sorted([candidates[c] for c in choice],key=lambda a:a[0])
    coherent=np.zeros((len(df),4),dtype=bool)
    rotating=np.zeros((len(df),4),dtype=bool)
    for k,b in enumerate(blocks):
        # co-sited T/RH pair per device: (0,1) or (2,3)
        cg=k%2
        coherent[np.ix_(b,[cg*2,cg*2+1])]=True
        for h,pos in enumerate(b):
            rg=h%2
            rotating[pos,rg*2:rg*2+2]=True
    assert not (coherent & ~good[:,None]).any()
    assert not (rotating & ~good[:,None]).any()
    assert (coherent.sum(axis=1)==rotating.sum(axis=1)).all()
    assert (coherent.sum(axis=0)==rotating.sum(axis=0)).all()
    assert coherent.sum()==rotating.sum()==nblocks*BLOCK_H*2
    assert np.array_equal(coherent.any(axis=1),rotating.any(axis=1))
    return coherent,rotating,blocks,candidates


def fit_and_evaluate(site,path,output):
    data,ids,cols,meta=load_site(path)
    n=len(data);i=int(.60*n);j=int(.80*n)
    # Chronological split BEFORE dropping incomplete sensor hours.
    subsets={'train':data.iloc[:i],'validation':data.iloc[i:j],'test':data.iloc[j:]}
    cal=calendar_features(data.index)
    complete=data[cols].notna().all(axis=1)
    train_idx=subsets['train'].index[complete.iloc[:i]]
    val_idx=subsets['validation'].index[complete.iloc[i:j]]
    test_idx=subsets['test'].index[complete.iloc[j:]]
    if min(len(train_idx),len(val_idx),len(test_idx))<100:raise ValueError(f'{site}: insufficient complete case train/val/test')
    sensor_medians=data.loc[train_idx,cols].median()
    ytrain=data.loc[train_idx,'y_kwh']
    clean=[]; masked=[]; maskrows=[];contrasts=[]
    fitted={}
    for kind in MODELS:
        for setup,feat in [('calendar',[]),('full',cols)]:
            xfit=pd.concat([cal,data[feat]],axis=1) if feat else cal
            mod=model(kind)
            mod.fit(xfit.loc[train_idx],ytrain)
            fitted[(kind,setup)]=mod
            for split,ii in [('validation',val_idx),('test',test_idx)]:
                pred=mod.predict(xfit.loc[ii]); e=errors(data.loc[ii,'y_kwh'],pred)
                clean.append({'building':site,'model':kind,'feature_group':setup,'period':split,'n':len(ii),**e})
    baseline=float(ytrain.median())
    for split,ii in [('validation',val_idx),('test',test_idx)]:
        e=errors(data.loc[ii,'y_kwh'],np.repeat(baseline,len(ii)))
        clean.append({'building':site,'model':'TrainMedian','feature_group':'none','period':split,'n':len(ii),**e})
    valclean=clean
    full_data=data[cols].copy()
    # For affected test samples only, all unmasked original observed values are retained.
    for seed in SEEDS:
        t=data.iloc[j:]; coh,rot,blocks,candidates=make_masks(t,cols,seed)
        affected=np.any(coh,axis=1)
        timestamps=t.index[affected]
        check={'building':site,'seed':seed,'test_all_valid_hour_count':len(t),'test_complete_hours':len(test_idx),'affected_hours':int(affected.sum()),'total_missing_cells':int(coh.sum()),'fraction_all_test_hours_affected':float(affected.mean()),'fraction_complete_test_hours_affected':float(affected.sum()/len(test_idx)),'coherent_channel_loss_counts':coh.sum(axis=0).tolist(),'rotating_channel_loss_counts':rot.sum(axis=0).tolist(),'selected_six_hour_windows':len(blocks),'candidate_six_hour_windows':len(candidates),'equal_timestamp_exposure':bool(np.array_equal(coh.any(axis=1),rot.any(axis=1))),'equal_channel_marginal':bool(np.array_equal(coh.sum(axis=0),rot.sum(axis=0)))}
        maskrows.append(check)
        preds={}
        for top,mask in [('coherent',coh),('rotating',rot)]:
            repaired=data[cols].copy()
            arr=repaired.loc[t.index,cols].copy()
            arr=arr.mask(mask)
            repaired.loc[t.index,cols]=arr
            for fill in METHODS:
                if fill=='training_median':completed=repaired.fillna(sensor_medians)
                else:completed=repaired.ffill().fillna(sensor_medians)
                features=pd.concat([cal,completed],axis=1)
                for kind in MODELS:
                    p=fitted[(kind,'full')].predict(features.loc[test_idx])
                    yp=data.loc[test_idx,'y_kwh']
                    e=errors(yp,p)
                    ea=errors(data.loc[timestamps,'y_kwh'],fitted[(kind,'full')].predict(features.loc[timestamps]))
                    masked.append({'building':site,'seed':seed,'model':kind,'imputation':fill,'mask_topology':top,'n_test':len(test_idx),'n_affected':int(affected.sum()),**e,'affected_mae_kwh':ea['mae_kwh']})
                    preds[(kind,fill,top)]=p
        for kind in MODELS:
            for fill in METHODS:
                p1=preds[(kind,fill,'coherent')];p2=preds[(kind,fill,'rotating')]
                index_lookup=pd.Index(test_idx).get_indexer(timestamps)
                e1=np.abs(data.loc[test_idx,'y_kwh'].values-p1)
                e2=np.abs(data.loc[test_idx,'y_kwh'].values-p2)
                contrasts.append({'building':site,'seed':seed,'model':kind,'imputation':fill,'delta_coherent_minus_rotating_mae_kwh':float((e1-e2).mean()),'delta_affected_only_mae_kwh':float((e1-e2)[index_lookup].mean()),'n_test':len(test_idx),'n_affected':len(index_lookup)})
    qc={'building':site,'selected_co_sited_device_ids':ids,'selected_local_channels':cols,'train_hours_total':len(subsets['train']),'validation_hours_total':len(subsets['validation']),'test_hours_total':len(subsets['test']),'train_complete_hours':len(train_idx),'validation_complete_hours':len(val_idx),'test_complete_hours':len(test_idx),'train_missing_rate':float(1-len(train_idx)/len(subsets['train'])),'validation_missing_rate':float(1-len(val_idx)/len(subsets['validation'])),'test_missing_rate':float(1-len(test_idx)/len(subsets['test'])),'train_start':str(subsets['train'].index[0]),'train_end':str(subsets['train'].index[-1]),'validation_start':str(subsets['validation'].index[0]),'validation_end':str(subsets['validation'].index[-1]),'test_start':str(subsets['test'].index[0]),'test_end':str(subsets['test'].index[-1]),'training_target_median_kwh':baseline,**meta}
    return qc,clean,masked,maskrows,contrasts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',default=str(Path(__file__).resolve().parent/'data_external'));ap.add_argument('--out-dir',default=str(Path(__file__).resolve().parent/'results'));args=ap.parse_args()
    input_dir=Path(args.input_dir);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    meta=audit_files(input_dir);meta.to_csv(out/'V14_all_seven_archive_inventory.csv',index=False)
    quality=[];allclean=[];allmask=[];audit=[];contr=[]
    for site in SITE_NAMES:
        qc,cc,mm,aa,dd=fit_and_evaluate(site,input_dir/(site+'.zip'),out)
        quality.append(qc);allclean+=cc;allmask+=mm;audit+=aa;contr+=dd
        print('SITE',site,'train',qc['train_complete_hours'],'valid',qc['validation_complete_hours'],'test',qc['test_complete_hours'],'hours affected',aa[0]['affected_hours'])
    pd.DataFrame(quality).to_csv(out/'V14_site_quality.csv',index=False)
    pd.DataFrame(allclean).to_csv(out/'V14_clean_models.csv',index=False)
    pd.DataFrame(allmask).to_csv(out/'V14_all_fault_runs.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'V14_mask_integrity.csv',index=False)
    pd.DataFrame(contr).to_csv(out/'V14_paired_contrasts.csv',index=False)
    summary=pd.DataFrame(contr).groupby(['building','model','imputation']).agg(mean_contrast_kwh=('delta_coherent_minus_rotating_mae_kwh','mean'),min_contrast_kwh=('delta_coherent_minus_rotating_mae_kwh','min'),max_contrast_kwh=('delta_coherent_minus_rotating_mae_kwh','max'),mean_affected_contrast_kwh=('delta_affected_only_mae_kwh','mean'),n_seeds=('seed','nunique')).reset_index()
    summary.to_csv(out/'V14_contrast_summary.csv',index=False)
    info={'date':'2026-11-09','dataset':'SmartlivingEPC 10.5281/zenodo.15781077','research_status':'Post-V13 exploratory external site replication with explicit schema-driven protocol amendment, not model transfer','target':'positive or zero first difference of cumulative kWh on exactly consecutive 1-h energy readings','sensor_aggregation':'mean sensor observations (t - 1 hour, t] ; no future sensor data','clock':'naive provided timestamp; no documented timezone harmonization','model':'each site trained independently; not zero-shot','features':'calendar, plus two co-sited T/RH sensor devices (four channels)','selection':'two devices by train availability; no outcome data','split':'first 60%, next 20%, last 20% of valid hourly target readings; complete-case subset each partition','mask':'identical complete-case test timestamps; exactly 2/4 channels per affected timestamp; 6-h coherent group vs hourly rotating group; equal per-device marginals; 3 random seeds','limitations':['cumulative electricity target is not same as UCI contemporaneous 10-minute appliance use','post-hoc exploratory protocol adaptation after inspecting schema, no untouched test preregistration','incomplete natural sensor coverage and complete-case selection','small number of sites, no population inference','mixed-use and city hall excluded due multi-meter ambiguity; public lighting no environmental sensors','naive timezone with DST ambiguity','meter negative differences discarded; positive spikes not clipped','two co-sited devices chosen based on training availability only']}
    (out/'V14_protocol.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print('saved',len(allclean),'clean',len(allmask),'fault',len(contr),'paired',len(audit),'masks')
    print(pd.DataFrame(allclean).query('period=="test"')[['building','model','feature_group','n','mae_kwh','rmse_kwh']].to_string(index=False))
    print(summary.to_string(index=False))

if __name__=='__main__':main()
