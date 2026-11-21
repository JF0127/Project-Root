"""Finalize only a complete Phase 1 production run. No network; never modify crop directories."""
from collections import Counter
import json
import os
from pathlib import Path
import re
import tempfile

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
M=ROOT/'metadata'
TIERS=('tier_1_strict','tier_2_relaxed','excluded_phase1')
BUCKETS=('short','core','long')
PROV=('conversation_id','vendor_id','session_id','speaker_id','listener_id','prompt_hash','prompt_id_unique')
TOP={'turn_id','candidate_tier','source_provenance','model_inputs'}
INPUTS={'speaker_video','speaker_audio','transcript'}
MODEL='gpt-6.2-sol'


def read(path): return json.loads(path.read_text(encoding='utf8'))


def require(ok,msg):
    if not ok: raise ValueError(msg)


def classify(row):
    face,complete,dep=row.pass2_speaker_face_visible,row.pass2_is_semantically_complete,row.pass2_context_dependency
    require(face in ('clear','partial','occluded_or_missing') and type(complete) is bool and dep in ('low','medium','high'),'Invalid pass2 verdict')
    reasons=[]
    if face!='clear': reasons.append('face_'+face)
    if not complete: reasons.append('semantically_incomplete')
    if dep=='high': reasons.append('high_context_dependency')
    return (TIERS[2] if reasons else TIERS[0] if dep=='low' else TIERS[1]),reasons


def manifest_check(obj,row):
    tid=row.turn_id; require(set(obj)==TOP and obj['candidate_tier']==row.candidate_tier and row.candidate_tier in TIERS[:2] and obj['turn_id']==tid,'manifest row mismatch')
    p=obj['source_provenance']; x=obj['model_inputs']
    require(set(p)==set(PROV) and all(isinstance(p[k],str) and p[k] and p[k]==getattr(row,k) for k in PROV),'bad provenance')
    require(set(x)==INPUTS and isinstance(x['transcript'],str) and x['transcript']==row.text and x['transcript'].strip(),'bad transcript or input fields')
    for key,name in [('speaker_video','speaker_A.mp4'),('speaker_audio','speaker_A.wav')]:
        path=ROOT/f'cropped_turns/{tid}/{name}'
        require(x[key]==str(path.relative_to(ROOT)) and path.resolve(strict=True)==path and path.is_file() and path.stat().st_size>0,'bad speaker media reference')


def coverage(df):
    return {'sessions':int(df.session_id.nunique()),'speakers':int(df.speaker_id.nunique()),'prompts':int(df.prompt_id_unique.nunique()),
            'vendor_session_pairs':len(df[['vendor_id','session_id']].drop_duplicates()),'vendor_speaker_pairs':len(df[['vendor_id','speaker_id']].drop_duplicates())}


def audit_metrics(records,timings):
    keys={(r['turn_id'],r['pass_name']) for r in records}; good={(r['turn_id'],r['pass_name']) for r in records if r['status']=='success'}
    attempts=[a for r in records for a in r['attempts']]; runs={r['run_id'] for r in records}
    measured=[r for r in timings if r['run_id'] in runs]; timed={r['run_id'] for r in measured}
    return dict(unique_tasks=len(keys),successful_tasks=len(good),unresolved_tasks=len(keys-good),attempts=len(attempts),retry_attempts=len(attempts)-len(keys),
                successful_by_pass=dict(Counter(n for _,n in good)),usage_unknown_attempts=sum('usage' not in a for a in attempts),
                tokens={k:sum(a.get('usage',{}).get(k,0) for a in attempts) for k in ('prompt_tokens','completion_tokens','total_tokens')},
                measured_wall_sec=sum(r['wall_sec'] for r in measured),untimed_runs=sorted(runs-timed))


def main():
    ids=set(pd.read_parquet(M/'turns_candidates.parquet',columns=['turn_id','triage_status']).query('triage_status=="pass"').turn_id)
    require(len(ids)==1561 and all(re.fullmatch(r'T_\d{6}',i) for i in ids),'Expected the 1561 validated pass IDs')
    folders=list((ROOT/'cropped_turns').iterdir()); require({p.name for p in folders}==ids and all(p.is_dir() and not p.is_symlink() for p in folders),'Crop directory set incomplete or contaminated')
    for stem in ['selection','index','l3_features','l3_complete']:
        data=pd.read_parquet(M/f'pass_1561_{stem}.parquet'); require(len(data)==len(ids) and set(data.turn_id)==ids and data.triage_status.eq('pass').all(),f'Invalid {stem} join')
    df=data.sort_values('turn_id').copy(); require(df.pass1_status.eq('success').all() and df.pass2_status.eq('success').all(),'Unresolved review tasks')
    classified=[classify(row) for row in df.itertuples()]; df['candidate_tier']=[x[0] for x in classified];df['phase1_exclusion_reasons']=[x[1] for x in classified]
    eligible=df[df.candidate_tier.isin(TIERS[:2])]; rows=[]
    for r in df.itertuples():
        folder=ROOT/'cropped_turns'/r.turn_id; obj=read(folder/'meta.json'); profile=obj['l3_profile']; review=profile['multimodal_review']
        require(bool(profile['feature_based']['normalized']['participant_z']) and bool(profile['feature_based']['normalized']['participant_session_z']),'Missing L3 normalization blocks')
        for n,mods in [('pass1',['text']),('pass2',['text','video_frames'])]:
            rec=review[n]; require(rec['status']=='success' and rec['model']==MODEL and rec['observed_modalities']==mods,'Unexpected review state')
            for k,v in rec['result'].items():
                field=getattr(r,f'{n}_{k}'); require((list(field)==v if isinstance(v,list) else field==v),'Meta/complete table mismatch')
        p=obj['source_provenance']; require(p['turn_id']==r.turn_id and all(p[k]==getattr(r,k) for k in PROV),'Provenance mismatch')
        if r.candidate_tier==TIERS[2]: continue
        inputs=obj['model_inputs']; require(inputs['speaker_video']=='speaker_A.mp4' and inputs['speaker_audio']=='speaker_A.wav','Unexpected media alias')
        entry={'turn_id':r.turn_id,'candidate_tier':r.candidate_tier,'source_provenance':{k:p[k] for k in PROV},
               'model_inputs':{'speaker_video':r.speaker_video_path,'speaker_audio':r.speaker_wav_path,'transcript':inputs['transcript']}}
        manifest_check(entry,r);rows.append(entry)
    tiers={}
    for tier in TIERS:
        g=df[df.candidate_tier==tier];tiers[tier]={'n':len(g),'duration':{k:int(g.duration_bucket.eq(k).sum()) for k in BUCKETS},
            'tag_counts':{t:sum(t in a for a in g.pass2_pragmatic_tags) for t in TAGS},'coverage':coverage(g)}
    clean=read(M/'phase1_cleanup_report.json');crop=read(M/'step3_pass_1561_crop_report.json'); local=read(M/'step4_1_pass_1561_report.json')
    require(crop['selected_count']==len(df)==crop['success_count'] and not crop['failed_count'] and local['samples']==len(df),'Stage counts disagree')
    idx=pd.read_parquet(M/'pass_1561_index.parquet'); require((idx[['speaker_video_start_pts','listener_ref_video_start_pts']]==0).all().all(),'Nonzero start PTS')
    ck=[json.loads(l) for l in (M/'step4_2_openai_checkpoints.jsonl').read_text().splitlines()]; split=clean['checkpoint_lines_before_production']
    times=[json.loads(l) for l in (M/'step4_2_openai_timings.jsonl').read_text().splitlines()]; keys=lambda xs:{(r['turn_id'],r['pass_name']) for r in xs}
    require(len(keys(ck[:split]))==220 and not(keys(ck[:split]) & keys(ck[split:])) and keys(ck)=={(i,n) for i in ids for n in ('pass1','pass2')},'Unexpected audit task set/repeated pilot call')
    manifest='phase1_model_inputs_manifest.jsonl'
    with tempfile.TemporaryDirectory(dir=M) as tmp:
        path=Path(tmp)/manifest
        with path.open('w',encoding='utf8') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        # Reparse the actual disk payload: exact field lists and paths, types, values,
        # eligibility and a one-to-one exhaustive match with retained ledger rows.
        with path.open(encoding='utf8') as f:
            for line,r in zip(f,eligible.itertuples(),strict=True): manifest_check(json.loads(line),r)
        for ext in ('.parquet','.csv'): table(M/('phase1_final_candidates'+ext),df.set_index('turn_id'))
        path.replace(M/manifest)
    report={'population':len(df),'cleanup':clean,'crop':crop,'feature_scope_n':local['samples'],'audit':{'model':MODEL,'modalities':{'pass1':['text'],'pass2':['text','video_frames']},
            'reused_pilot_tasks':split,'production_only':audit_metrics(ck[split:],times),'including_pilot':audit_metrics(ck,times)},'tiers':tiers,
            'excluded_reason_counts':dict(Counter(x for arr in df.phase1_exclusion_reasons for x in arr)),
            'retained_combined_coverage':coverage(eligible),'manifest_validation':{'rows':len(rows),'passed':True,'unexpected_fields':0,'paths_relative_to':'project root','excluded_in_manifest':0}}
    with tempfile.NamedTemporaryFile(mode='w',dir=M,delete=False) as f: json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    Path(f.name).replace(M/'phase1_final_report.json'); print('Phase 1 complete:',{k:v['n'] for k,v in tiers.items()},'isolated manifest rows:',len(rows))


# Offline calibration and table export shared with the batch audit.
META=M
TAGS=['empathy_seeking','question_cognitive','positive_sharing','neutral_statement','conflict_challenge','other']
FAU='speaker_stimulus_visual_fau_frame_delta_l2_p90'


def ratio(n,d): return n/d if d else None


def join(df, source, prefix):
    src=source.set_index('turn_id').reindex(df.index)
    for c in df.columns.intersection(src.columns):
        if df[c].equals(src[c]): src=src.drop(columns=c)
        else: src=src.rename(columns={c:prefix+c})
    return df.join(src,validate='one_to_one')


def counts(values): return {str(k):int(v) for k,v in values.value_counts().items()}


def tags(frame):
    arr=frame['pass2_pragmatic_tags'].dropna().tolist()
    return dict(n=len(arr),counts={k:sum(k in a for a in arr) for k in TAGS})


def table(path, df):
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f: tmp=f.name
    try:
        if path.suffix=='.csv':
            csv=df.copy()
            for k in csv:
                if any(isinstance(v,(list,tuple)) or hasattr(v,'tolist') and getattr(v,'ndim',0)>0 for v in csv[k]):
                    csv[k]=csv[k].map(lambda v:json.dumps(v.tolist() if hasattr(v,'tolist') else v,ensure_ascii=False) if isinstance(v,(list,tuple)) or getattr(v,'ndim',0)>0 else v)
            csv.to_csv(tmp,index=True)
        else: df.reset_index().to_parquet(tmp,index=False)
        os.replace(tmp,path)
    finally: Path(tmp).unlink(missing_ok=True)


def export(frame,reviews,history,timings,args,pilot):
    candidates=pd.read_parquet(META/'turns_candidates.parquet'); population=candidates.query('triage_status=="pass"').duration_bucket.value_counts().to_dict()
    features=pd.read_parquet(args.features)
    if not set(frame.index)<=set(features.turn_id): raise ValueError('Supply matching --features covering the full input')
    merged=join(join(frame,candidates,'step2_'),features,'step4_1_')
    rows=[]
    for tid,rev in reviews.items():
        item={'turn_id':tid}
        for p in ('pass1','pass2'):
            rec=rev[p]; item.update({p+'_'+k:v for k,v in rec.items() if k!='result'})
            item.update({p+'_'+k:v for k,v in rec.get('result',{}).items()})
        item.update({k:rev[k] for k in ('d1_tag_changed','d1_intensity_shifted')}); rows.append(item)
    df=merged.join(pd.DataFrame(rows).set_index('turn_id'),validate='one_to_one')
    for c in ['pass2_speaker_face_visible','pass2_is_semantically_complete','pass2_context_dependency','pass2_pragmatic_tags','pass2_speaker_expressive_intensity','pass2_modality_vs_text_cue']:
        if c not in df: df[c]=None
    base=(df.pass2_speaker_face_visible=='clear') & df.pass2_is_semantically_complete.eq(True)
    df['keep_strict']=base & df.pass2_context_dependency.eq('low'); df['keep_relaxed']=base & df.pass2_context_dependency.isin(['low','medium'])
    stem='pilot_pass_110_l3_complete' if pilot else f"{args.input_index.stem.removesuffix('_index')}_l3_complete"
    for suffix in ('.parquet','.csv'): table(META/(stem+suffix),df)
    keep={}; exp={'strict':0.,'relaxed':0.}
    for b in BUCKETS:
        sub=df[df.duration_bucket==b]; n=len(sub); row={'pilot_n':n,'population_n':int(population[b]),'pass2_success':int(sub.pass2_status.eq('success').sum())}
        for definition in exp:
            k=int(sub['keep_'+definition].sum()); rate=ratio(k,n); row[definition]={'n':k,'rate':rate}
            if rate is not None: exp[definition]+=rate*population[b]
        keep[b]=row
    paired=df[df.d1_tag_changed.notna()]; n=len(paired)
    d1={}
    for k in ('d1_tag_changed','d1_intensity_shifted'):
        hits=int(paired[k].sum()); d1[k]={'n':hits,'paired_n':n,'rate':ratio(hits,n)}
    d1['cue_counts']=counts(df.pass2_modality_vs_text_cue)
    d1['either_n']=int((paired.d1_tag_changed.eq(True)|paired.d1_intensity_shifted.eq(True)).sum())
    stats={}
    for l in ('low','medium','high'):
        g=df[df.pass2_speaker_expressive_intensity==l]; vals=g[FAU].dropna(); stats[l]={'rated_n':len(g),'finite_n':len(vals),'median':float(vals.median()) if len(vals) else None}
    med=[s['median'] for s in stats.values()]; monotone=None if None in med else med[0]<med[1]<med[2]
    h=pd.DataFrame([dict(turn_id=r['turn_id'],pass_name=r['pass_name'],run_id=r['run_id'],elapsed_sec=r['elapsed_sec'],
            duration_bucket=df.at[r['turn_id'],'duration_bucket'],attempts=len(r['attempts']),
            usage_missing=sum('usage' not in a for a in r['attempts']),
            **{k:sum(a.get('usage',{}).get(k,0) for a in r['attempts']) for k in ('total_tokens','prompt_tokens','completion_tokens')}) for r in history])
    run_rows=[r for r in timings if r['run_id'] in set(h.run_id)]; total_sec=sum(r['wall_sec'] for r in run_rows)
    health={}
    for p in ('pass1','pass2'):
        sub=h[h.pass_name==p]; ok=int(df[p+'_status'].eq('success').sum());health[p]=dict(success=ok,failed=len(df)-ok,api_attempts=int(sub.attempts.sum()),
            total_tokens=int(sub.total_tokens.sum()),mean_task_sec=float(sub.elapsed_sec.mean()))
    # Empirical concurrency efficiency, including worker scheduling, retries and crop prep.
    # Exclude 2-worker trial from throughput calibration when predicting at 6 workers.
    comparable=[r for r in run_rows if r['workers']==args.workers and r['tasks']==len(h[h.run_id==r['run_id']])]
    capacity=ratio(sum(h[h.run_id==r['run_id']].elapsed_sec.sum() for r in comparable),sum(r['wall_sec'] for r in comparable))
    totals={'total_tokens':0.,'elapsed_sec':0.}
    for b in BUCKETS:
        sub=h[h.duration_bucket==b];
        for key in totals:
            if keep[b]['pilot_n']: totals[key]+=float(sub[key].sum())*population[b]/keep[b]['pilot_n']
    complete_buckets=all(keep[b]['pilot_n'] for b in BUCKETS)
    report=dict(model=history[-1]['model'],pass2_modalities=next(r['observed_modalities'] for r in history if r['pass_name']=='pass2'),
        n=len(df),population_n=sum(population.values()),health=health,execution_runs=run_rows,wall_sec=total_sec,
        usage={k:int(h[k].sum()) for k in ('prompt_tokens','completion_tokens','total_tokens','usage_missing')},
        projection=dict(method='Within each duration bucket scale pilot total tokens and summed task seconds per sample to the population; divide task seconds by measured worker throughput. Excludes probe, no-op resume and offline export.',
            workers=args.workers,effective_parallelism=capacity,token_estimate=round(totals['total_tokens']) if complete_buckets else None,
            minutes=totals['elapsed_sec']/capacity/60 if capacity and complete_buckets else None),
        retention=dict(definition='Pass2 clear face AND semantically complete AND context low (strict) or low/medium (relaxed); denominator all selected, failed rows not retained.',
            buckets=keep,pilot_strict=int(df.keep_strict.sum()),pilot_relaxed=int(df.keep_relaxed.sum()),weighted_population_estimate={k:round(v) if complete_buckets else None for k,v in exp.items()}),
        d1=d1,d2={'all_pass2':tags(df),'strict_retained':tags(df[df.keep_strict]),'relaxed_retained':tags(df[df.keep_relaxed])},
        local_fau=dict(feature=FAU,by_intensity=stats,strictly_increasing=monotone),
        caveats=['Single-model judgments from four still frames; no ground truth labels.', 'Pilot is stratified, population counts are extrapolations, not audited counts.', 'D2 is multi-label: counts can exceed sample count.'])
    name='step4_2_pass110_report.json' if pilot else stem+'_report.json'
    with tempfile.NamedTemporaryFile(dir=META,mode='w',delete=False) as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
    os.replace(f.name,META/name)


if __name__=='__main__': main()
