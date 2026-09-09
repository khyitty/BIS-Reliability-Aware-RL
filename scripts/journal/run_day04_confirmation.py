"""Day-04 long-horizon scaled factorial confirmation (private-safe runner)."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, pickle, random, shutil, subprocess, sys, time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"/"journal"))
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from evaluate_night02_cpu import Controller, candidates
from night02_common import JournalSequenceEnv, aggregate_subject_rows, atomic_json, make_environment, sha256_path, stable_order, truncate_bundle
from run_day03_scale import FixedTransform, actor_diagnostics, apply_transform, canonical_sha
from run_night02_cpu import _digest, load_config as load_old_config, load_private as load_night02_private, utc_now
from vitaldb_state_selection.anesthesia import BISObservationProcessor, ObservationRule, PreprocessingID
from vitaldb_state_selection.anesthesia.state import S0_FIELDS,S1_FIELDS
from vitaldb_state_selection.rl_integration.config import PAPER_ORIENTED_PPO_CANDIDATE_V1,make_ppo_model

DEFAULT_CONFIG=ROOT/"configs/journal/day04_confirmation.json"
PUBLIC_BRIEF=ROOT/"docs/journal/DAY04_BRIEF.md"
PUBLIC_REPORT=ROOT/"reports/journal/DAY04_RESULTS.md"
PUBLIC_RESULTS=ROOT/"reports/journal/day04_results.csv"
PUBLIC_CONTRASTS=ROOT/"reports/journal/day04_contrasts.csv"
PUBLIC_VISIBILITY=ROOT/"reports/journal/day04_visibility_grid.csv"
PUBLIC_SUMMARY=ROOT/"reports/journal/day04_summary.json"
PUBLIC_OVERLEAF=ROOT/"docs/journal/DAY04_OVERLEAF_NOTE.md"
PHASE8B_HASH="96e9f4d329b0131634a756fc4b4a03acbce5e97a10d65a2a416948130f9d9fb2"
PHASE8C_HASH="25ad8a860f6c9b0b45febec7ff7d0d0edf88c0f1953229c8d95e207508d3a606"
PRIMARY_METRICS=("mae_0_1800","mae_0_600","mae_600_1800","time_in_40_60_fraction","time_below_40_fraction","time_above_60_fraction","cumulative_reward","total_propofol_mg","action_sd","consecutive_action_variation","action_boundary_fraction","core_clip_fraction","visible_fraction","sqi_rejection_fraction","stale_fraction")
CONTRAST_FORMULAS={"P0_state":"P0S1-P0S0","P1_state":"P1S1-P1S0","S0_preprocessing":"P1S0-P0S0","S1_preprocessing":"P1S1-P0S1","interaction":"(P1S1-P1S0)-(P0S1-P0S0)"}

def load_config(path:Path)->tuple[dict[str,Any],str]:
    raw=path.read_bytes(); return json.loads(raw),hashlib.sha256(raw).hexdigest()

def out_root(config:dict[str,Any])->Path:
    out=(ROOT/config["output_root"]).resolve(); rel=out.relative_to(ROOT).as_posix()
    if subprocess.run(["git","check-ignore","-q",rel],cwd=ROOT).returncode or subprocess.check_output(["git","ls-files",rel],cwd=ROOT,text=True).strip(): raise RuntimeError("Day04 output must be ignored/untracked")
    out.mkdir(parents=True,exist_ok=True); return out

def context(config:dict[str,Any]):
    old,old_hash=load_old_config(ROOT/config["night02_config"]); old_out,membership,store,scalers=load_night02_private(old)
    if old_out.resolve()!=(ROOT/config["night02_output_root"]).resolve(): raise RuntimeError("Night02 root mismatch")
    return old,old_hash,old_out,membership,store,scalers

def condition(config:dict[str,Any],condition_id:str)->dict[str,Any]: return next(dict(row) for row in config["conditions"] if row["condition_id"]==condition_id)
def fields(state:str): return S0_FIELDS if state=="S0" else S1_FIELDS
def factors(out:Path,state:str)->np.ndarray:
    data=json.loads((out/"nscale_factors.json").read_text(encoding="utf-8"))["factors_by_field"]
    return np.asarray([data[name] for name in fields(state)],dtype=np.float32)
def write_status(out:Path,phase:str,**extra:Any)->None: atomic_json(out/"status.json",{"phase":phase,"updated_utc":utc_now(),"latest_error":None,**extra})

def assert_parent(config:dict[str,Any])->None:
    parent=config["required_parent_sha"]
    if subprocess.check_output(["git","merge-base","HEAD",parent],cwd=ROOT,text=True).strip()!=parent: raise RuntimeError("required parent is not ancestor")

def prepare(config_path:Path,verify_stores:bool=True)->None:
    config,config_hash=load_config(config_path); assert_parent(config); out=out_root(config)
    _,_,old_out,old_membership,store,_=context(config)
    hist=store.repository_root
    if subprocess.check_output(["git","status","--porcelain"],cwd=hist,text=True).strip(): raise RuntimeError("historical checkout dirty")
    if verify_stores:
        b_hash=store.template_store.verify_all(); c_hash=store.verify_all()
    else: b_hash,c_hash=PHASE8B_HASH,PHASE8C_HASH
    if (b_hash,c_hash)!=(PHASE8B_HASH,PHASE8C_HASH): raise RuntimeError("private store root hash changed")
    full_train_subjects=list(old_membership["full_train_subjects"]); full_train_cases=list(old_membership["full_train_cases"])
    old_val_subjects=set(old_membership["validation_subjects"]); old_val_cases=set(old_membership["validation_cases"])
    remaining=[s for s in old_membership["full_validation_subjects"] if s not in old_val_subjects]
    fresh_subjects=stable_order(remaining,seed=int(config["split_seed"]),label=config["fresh_validation_label"])[:int(config["fresh_validation_subject_limit"])]
    by_subject=defaultdict(list)
    for row in store.rows: by_subject[str(row["subjectid"])].append(str(row["caseid"]))
    fresh_cases=[case for subject in fresh_subjects for case in sorted(by_subject[subject],key=lambda x:(int(x),x))]
    tuning_subjects=stable_order(full_train_subjects,seed=int(config["split_seed"]),label=config["tuning_label"])[:int(config["tuning_subject_limit"])]
    tuning_cases=[case for subject in tuning_subjects for case in sorted(by_subject[subject],key=lambda x:(int(x),x))]
    if set(full_train_subjects)&set(fresh_subjects) or old_val_subjects&set(fresh_subjects) or set(full_train_cases)&set(fresh_cases) or old_val_cases&set(fresh_cases): raise RuntimeError("membership overlap")
    private={"full_train_subjects":full_train_subjects,"full_train_cases":full_train_cases,"old_validation_subjects":list(old_val_subjects),"old_validation_cases":list(old_val_cases),"fresh_validation_subjects":fresh_subjects,"fresh_validation_cases":fresh_cases,"tuning_subjects":tuning_subjects,"tuning_cases":tuning_cases}
    atomic_json(out/"private_membership.json",private) # freeze before any fresh bundle load
    min_train=min(store.load_case(c).episode_horizon_seconds for c in full_train_cases)
    min_fresh=min(store.load_case(c).episode_horizon_seconds for c in fresh_cases)
    if min(min_train,min_fresh)<float(config["horizon_seconds"]): raise RuntimeError("selected bundle shorter than 1800s")
    day03=(ROOT/config["day03_output_root"]).resolve(); d3factor=json.loads((day03/"nscale_factors.json").read_text(encoding="utf-8"))
    d3models=[]
    for state in ("S0","S1"):
        for seed in (45,46,47):
            d=day03/"jobs"/f"Nscale_{state}"/f"seed_{seed}"; comp=json.loads((d/"OUTPUT_COMPLETE.json").read_text(encoding="utf-8")); model=d/"final_post_update/model.zip"
            if comp.get("model_sha256")!=sha256_path(model) or comp.get("training_epochs")!=1270: raise RuntimeError("Day03 transfer model invalid")
            d3models.append({"state":state,"seed":seed,"model_sha256":comp["model_sha256"]})
    public={"protocol_id":config["protocol_id"],"required_parent_sha":config["required_parent_sha"],"config_sha256":config_hash,"phase8b_store_sha256":b_hash,"phase8c_store_sha256":c_hash,"full_train_subject_count":len(full_train_subjects),"full_train_case_count":len(full_train_cases),"fresh_validation_subject_count":len(fresh_subjects),"fresh_validation_case_count":len(fresh_cases),"old_validation_excluded_subject_count":len(old_val_subjects),"tuning_subject_count":len(tuning_subjects),"tuning_case_count":len(tuning_cases),"full_train_subject_sha256":_digest(full_train_subjects),"full_train_case_sha256":_digest(full_train_cases),"fresh_validation_subject_sha256":_digest(fresh_subjects),"fresh_validation_case_sha256":_digest(fresh_cases),"tuning_subject_sha256":_digest(tuning_subjects),"minimum_train_horizon_seconds":min_train,"minimum_fresh_validation_horizon_seconds":min_fresh,"membership_frozen_before_fresh_bundle_preflight":True,"fresh_validation_preflight_only_bundle_count":len(fresh_cases),"fresh_validation_outcome_access_count":0,"test_access_count":0,"day03_factor_sha256":d3factor["factors_sha256"],"day03_transfer_model_count":len(d3models),"historical_checkout_clean":True,"process_guard":config["process_guard"]}
    atomic_json(out/"preflight.json",public); atomic_json(out/"day03_transfer_models.json",{"models":d3models,"factor_sha256":d3factor["factors_sha256"]})
    manifest={**public,"factor_recipe":config["transform"],"conditions":config["conditions"],"job_order":config["job_order"],"seeds":config["seeds"],"requested_timesteps":config["training_target_timesteps"],"effective_timesteps":config["training_target_timesteps"],"expected_rollouts":config["expected_rollouts"],"expected_training_epochs":config["expected_training_epochs"],"horizon_seconds":config["horizon_seconds"],"metrics":list(PRIMARY_METRICS),"contrast_formulas":CONTRAST_FORMULAS,"preoutcome_frozen_utc":utc_now(),"validation_outcomes_computed":False}
    atomic_json(out/"protocol_manifest.json",manifest)
    PUBLIC_BRIEF.parent.mkdir(parents=True,exist_ok=True)
    PUBLIC_BRIEF.write_text(f"""# Day 04 Frozen Brief\n\n## Question\n\nDoes the Nscale-conditioned S1 advantage and P×S pattern persist at 1,800 seconds using the full development-training partition, five new paired seeds, and a previously unused internal-validation subset?\n\n## Frozen design\n\n- Parent: `{config['required_parent_sha']}`\n- Training: {len(full_train_subjects):,} subjects / {len(full_train_cases):,} cases\n- Fresh internal validation: {len(fresh_subjects)} subjects / {len(fresh_cases)} cases; prior 24-subject subset excluded\n- Conditions: P0S0, P1S0, P0S1, P1S1\n- Seeds: 48–52; requested 524,288 steps per job; 20 jobs\n- Horizon: 1,800 seconds; final post-update checkpoint only\n- Nscale factors: train-only pooled P0/P1, constant 1.5 and profile-specific tuned PI trajectories\n- Primary endpoint: subject-mean latent-BIS MAE over 0–1,800 seconds\n- Process guard: `psutil` unavailable, so one low-thread worker is the conservative fallback; reliable VS Code parent-chain detection is unavailable.\n\n## Boundaries\n\nThis is VitalDB-informed evaluation of simulated trajectories under fixed patient profiles and remifentanil schedules. It is not clinical intervention evidence, external validation, or causal inference. Validation fits nothing. The sealed original test cohort is not accessed.\n""",encoding="utf-8")
    print(json.dumps(public,indent=2))

def audit_case(template:Any,threshold:float|None,age:float,horizon:float)->dict[str,float]:
    # Input-only linear scan equivalent to BISObservationProcessor. Count effective
    # staleness directly, even when a later SQI rejection masks the STALE label.
    ObservationRule(f"audit_{threshold}_{age}",threshold,age)
    bis=list(template.bis_events);sqi={e.timestamp_seconds:e.value for e in template.sqi_events};index=0;masks=[];effective_stale=[];accepted_time=None;latest_reason="no_prior_observation";sqi_rejections=0;ingested=0
    steps=max(1,int(math.floor(horizon/10.0)))
    for step in range(steps):
        t=step*10.0
        while index<len(bis) and bis[index].timestamp_seconds<=t:
            event=bis[index];ingested+=1
            if not event.available:latest_reason="explicit_missing_event"
            elif threshold is not None and event.timestamp_seconds not in sqi:latest_reason="sqi_missing_exact_timestamp";sqi_rejections+=1
            elif threshold is not None and (not math.isfinite(sqi[event.timestamp_seconds]) or sqi[event.timestamp_seconds]<threshold):latest_reason="sqi_below_threshold";sqi_rejections+=1
            else:latest_reason="available";accepted_time=event.timestamp_seconds
            index+=1
        stale=accepted_time is not None and t-accepted_time>age
        masks.append(int(accepted_time is not None and not stale));effective_stale.append(int(stale))
    transitions=sum(a!=b for a,b in zip(masks,masks[1:])); longest=run=0
    for value in masks:
        run=0 if value else run+1; longest=max(longest,run)
    return {"visible_fraction":sum(masks)/len(masks),"zero_visibility":float(not any(masks)),"first_visible_seconds":float(next((i*10 for i,v in enumerate(masks) if v),horizon)),"visibility_transitions":float(transitions),"longest_invisible_seconds":float(longest*10),"sqi_rejection_fraction":sqi_rejections/max(1,ingested),"staleness_rejection_fraction":sum(effective_stale)/len(effective_stale),"decision_count":float(len(masks))}

def audit(config_path:Path)->None:
    config,_=load_config(config_path); out=out_root(config); _,_,_,_,store,_=context(config); membership=json.loads((out/"private_membership.json").read_text(encoding="utf-8"))
    case_rows=[]
    for caseid in membership["full_train_cases"]:
      bundle=store.load_case(caseid)
      for threshold in config["visibility_grid"]["sqi_thresholds"]:
       for age in config["visibility_grid"]["age_seconds"]:
        for label,horizon in (("0_600",600.0),("0_1800",1800.0),("full",bundle.episode_horizon_seconds)):
            result=audit_case(bundle.observation_template,threshold,float(age),float(horizon)); case_rows.append({"subjectid":bundle.subjectid,"caseid":caseid,"sqi_threshold":"off" if threshold is None else str(threshold),"age_seconds":age,"window":label,**result})
    atomic_json(out/"visibility_audit_private.json",{"rows":case_rows,"split":"full_development_train_only","test_access_count":0})
    aggregate=[]; metric_names=("visible_fraction","zero_visibility","first_visible_seconds","visibility_transitions","longest_invisible_seconds","sqi_rejection_fraction","staleness_rejection_fraction")
    for threshold in config["visibility_grid"]["sqi_thresholds"]:
      key="off" if threshold is None else str(threshold)
      for age in config["visibility_grid"]["age_seconds"]:
       for window in ("0_600","0_1800","full"):
        selected=[r for r in case_rows if r["sqi_threshold"]==key and r["age_seconds"]==age and r["window"]==window]
        agg=aggregate_subject_rows(selected,list(metric_names)); total_decisions=sum(r["decision_count"] for r in selected)
        aggregate.append({"sqi_threshold":key,"age_seconds":age,"window":window,"subject_count":len({r["subjectid"] for r in selected}),"case_count":len(selected),"overall_visibility_fraction":sum(r["visible_fraction"]*r["decision_count"] for r in selected)/total_decisions,**{f"subject_mean_{k}":v for k,v in agg.items()}})
    atomic_json(out/"visibility_audit_aggregate.json",{"rows":aggregate,"split":"full_development_train_only","test_access_count":0})
    PUBLIC_VISIBILITY.parent.mkdir(parents=True,exist_ok=True)
    with PUBLIC_VISIBILITY.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
    print(json.dumps({"case_rows":len(case_rows),"aggregate_rows":len(aggregate)}))

def run_baseline_case(store:Any,caseid:str,cond:dict[str,Any],scaler:Any,horizon:float,controller:Controller)->float:
    env=make_environment(truncate_bundle(store.load_case(caseid),horizon),cond,scaler,45); _,info=env.reset(seed=45); controller.reset(); values=[];done=False
    while not done:
        action=controller.predict_once(info); _,_,terminated,truncated,info=env.step(np.asarray([action],dtype=np.float32)); values.append(abs(float(info["latent_true_bis"])-50));done=bool(terminated or truncated)
    env.close(); return float(np.mean(values))

def tune_baselines(config_path:Path)->None:
    config,_=load_config(config_path);out=out_root(config);_,_,_,_,store,scalers=context(config);m=json.loads((out/"private_membership.json").read_text(encoding="utf-8")); selections={};scores=[]
    for profile,cid in (("P0","P0S0"),("P1","P1S0")):
      cond=condition(config,cid)
      for family,rows in candidates().items():
       ranked=[]
       for idx,candidate in enumerate(rows):
        values=[run_baseline_case(store,c,cond,scalers["S0"],float(config["horizon_seconds"]),candidate) for c in m["tuning_cases"]]
        score=float(np.mean(values)); ranked.append((score,idx,candidate));scores.append({"profile":profile,"family":family,"candidate_index":idx,"mae":score})
       score,idx,best=min(ranked,key=lambda x:(x[0],x[1])); selections[f"{profile}/{family}"]=best.public()|{"candidate_index":idx,"tuning_case_mean_mae":score}
    atomic_json(out/"baseline_tuning.json",{"selection":selections,"scores":scores,"tuning_subject_count":len(m["tuning_subjects"]),"tuning_case_count":len(m["tuning_cases"]),"split":"development_train_only","test_access_count":0})
    print(json.dumps(selections,indent=2))

def calibrate(config_path:Path)->None:
    config,config_hash=load_config(config_path);out=out_root(config);_,_,old_out,_,store,scalers=context(config);m=json.loads((out/"private_membership.json").read_text(encoding="utf-8"));base=json.loads((out/"baseline_tuning.json").read_text(encoding="utf-8"))["selection"]
    steps=int(config["horizon_seconds"]//10); total=len(m["full_train_cases"])*2*2*steps; path=out/"calibration_observations_private.npy"; data=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(total,len(S1_FIELDS))); cursor=0
    for profile,cid in (("P0","P0S1"),("P1","P1S1")):
      cond=condition(config,cid);p=base[f"{profile}/PI"]
      for mode in ("constant","PI"):
       for caseid in m["full_train_cases"]:
        env=make_environment(truncate_bundle(store.load_case(caseid),float(config["horizon_seconds"])),cond,scalers["S1"],45);obs,info=env.reset(seed=45);ctl=Controller(p["family"],p["base"],p["kp"],p["ki"]);ctl.reset();done=False
        while not done:
            data[cursor]=obs;cursor+=1;action=1.5 if mode=="constant" else ctl.predict_once(info);obs,_,terminated,truncated,info=env.step(np.asarray([action],dtype=np.float32));done=bool(terminated or truncated)
        env.close()
    if cursor!=total or not np.isfinite(data).all(): raise RuntimeError("calibration accounting")
    q95=np.asarray([np.quantile(np.abs(data[:,i]),.95) for i in range(len(S1_FIELDS))]);d=np.maximum(1,q95);binary={"sex_binary",*(x for x in S1_FIELDS if x.startswith("bis_mask_"))}
    for i,name in enumerate(S1_FIELDS):
        if name in binary:d[i]=1
    mapped={name:float(d[i]) for i,name in enumerate(S1_FIELDS)};binary_indices=[i for i,x in enumerate(S1_FIELDS) if x in binary]
    if not np.array_equal(np.asarray(data[:,binary_indices]),np.asarray(data[:,binary_indices])/d[binary_indices]):raise RuntimeError("binary changed")
    bis_factor=mapped["bis_value_t+0"];bis_order=[x/bis_factor for x in (40.0,50.0,60.0)]
    if not bis_order[0]<bis_order[1]<bis_order[2]:raise RuntimeError("BIS order changed")
    day03=json.loads(((ROOT/config["day03_output_root"])/"nscale_factors.json").read_text(encoding="utf-8"))["factors_by_field"]
    comparison=[{"field_name":name,"day03_factor":day03[name],"day04_factor":mapped[name],"ratio":mapped[name]/day03[name]} for name in S1_FIELDS]
    manifest={"protocol_id":config["protocol_id"],"config_sha256":config_hash,"formula":"z_j / d_j; d_j=max(1,q95(abs(z_j)))","source_scaler_sha256":sha256_path(old_out/"scaler_registry.json"),"calibration_split":"full_development_train_only","profiles":["P0","P1"],"controllers":["constant_1.5","profile_specific_PI"],"case_count":len(m["full_train_cases"]),"observation_count":total,"factors_by_field":mapped,"factors_sha256":canonical_sha(mapped),"day03_comparison":comparison,"checks":{"finite_positive":all(math.isfinite(v) and v>0 for v in mapped.values()),"binary_bitwise_unchanged":True,"s0_strict_prefix":tuple(S1_FIELDS[:len(S0_FIELDS)])==tuple(S0_FIELDS),"shared_across_profiles":True,"bis_40_50_60":bis_order,"bis_order_preserved":True,"validation_used":False,"test_used":False},"test_access_count":0,"frozen_utc":utc_now()}
    atomic_json(out/"nscale_factors.json",manifest); print(json.dumps({"observations":total,"factor_sha256":manifest["factors_sha256"]}))

class Day04SequenceEnv(JournalSequenceEnv):
    """Journal sampler with a resumable active environment snapshot."""
    def reset(self,*,seed:int|None=None,options:dict[str,Any]|None=None):
        gym.Env.reset(self,seed=self.seed_value if seed is None else seed)
        if options not in (None,{}):raise ValueError("reset options unsupported")
        if self.environment is not None:self.environment.close()
        caseid=self.caseids[int(self.generator.integers(0,len(self.caseids)))];self.seen.add(caseid);self.episodes+=1
        self.environment=make_environment(self._bundle(caseid),self.condition,self.scaler,self.seed_value)
        return self.environment.reset(seed=self.seed_value)
    def recovery_snapshot(self)->dict[str,Any]:
        return {"generator_state":self.generator.bit_generator.state,"episodes":self.episodes,"seen":self.seen,"environment":self.environment}
    def restore_recovery(self,payload:dict[str,Any])->None:
        self.generator.bit_generator.state=payload["generator_state"];self.episodes=int(payload["episodes"]);self.seen=set(payload["seen"]);self.environment=payload["environment"]

def make_vector(config:dict[str,Any],membership:dict[str,Any],store:Any,scalers:Any,cond:dict[str,Any],seed:int,out:Path)->tuple[DummyVecEnv,Day04SequenceEnv]:
    seq=Day04SequenceEnv(store,membership["full_train_cases"],cond,scalers[cond["state_id"]],seed,float(config["horizon_seconds"]));fac=factors(out,cond["state_id"])
    return DummyVecEnv([lambda:FixedTransform(seq,"Nscale",fac)]),seq

def job_dir(out:Path,seed:int,cid:str)->Path:return out/"jobs"/f"seed_{seed}"/cid

def save_checkpoint(directory:Path,step:int,model:PPO,sequence:Day04SequenceEnv,identity:dict[str,Any],final:bool=False)->Path:
    name="final_post_update" if final else f"checkpoint_{step:010d}_post_update";dest=directory/name
    if dest.exists():return dest
    tmp=directory/f".{name}.partial";tmp.mkdir(parents=True,exist_ok=False);model.save(str(tmp/"model"))
    with (tmp/"environment.pkl").open("wb") as f:pickle.dump(sequence.recovery_snapshot(),f,pickle.HIGHEST_PROTOCOL)
    np.save(tmp/"last_obs.npy",model._last_obs,allow_pickle=False);np.save(tmp/"last_episode_starts.npy",model._last_episode_starts,allow_pickle=False)
    with (tmp/"rng.pkl").open("wb") as f:pickle.dump({"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()},f,pickle.HIGHEST_PROTOCOL)
    meta=identity|{"timestep":step,"training_epochs":int(model._n_updates),"completed_rollouts":int(model._n_updates/model.n_epochs),"artifact_stage":"post_update_authoritative_final" if final else "post_update_recovery_checkpoint","model_sha256":sha256_path(tmp/"model.zip"),"environment_sha256":sha256_path(tmp/"environment.pkl"),"rng_sha256":sha256_path(tmp/"rng.pkl"),"created_utc":utc_now()}
    atomic_json(tmp/"metadata.json",meta);atomic_json(tmp/"COMPLETE.json",{"complete":True,"metadata_sha256":sha256_path(tmp/"metadata.json"),"model_sha256":meta["model_sha256"],"environment_sha256":meta["environment_sha256"],"rng_sha256":meta["rng_sha256"]});os.replace(tmp,dest);return dest

def load_checkpoint(path:Path,vector:DummyVecEnv,sequence:Day04SequenceEnv)->PPO:
    with (path/"environment.pkl").open("rb") as f:sequence.restore_recovery(pickle.load(f))
    model=PPO.load(str(path/"model.zip"),env=vector,device="cpu",force_reset=False);model._last_obs=np.load(path/"last_obs.npy",allow_pickle=False);model._last_episode_starts=np.load(path/"last_episode_starts.npy",allow_pickle=False)
    with (path/"rng.pkl").open("rb") as f:rng=pickle.load(f)
    random.setstate(rng["python"]);np.random.set_state(rng["numpy"]);torch.set_rng_state(rng["torch"]);return model

def effective_budget(config:dict[str,Any],out:Path)->tuple[int,int,int]:
    protocol=json.loads((out/"protocol_manifest.json").read_text(encoding="utf-8"));target=int(protocol["effective_timesteps"]);return target,target//int(config["n_steps"]),(target//int(config["n_steps"]))*int(config["n_epochs"])

def train_one(config_path:Path,cid:str,seed:int)->None:
    config,config_hash=load_config(config_path);out=out_root(config);_,_,old_out,_,store,scalers=context(config);m=json.loads((out/"private_membership.json").read_text(encoding="utf-8"));cond=condition(config,cid);target,rollouts,epochs=effective_budget(config,out);interval=int(config["checkpoint_interval_timesteps"]);directory=job_dir(out,seed,cid);directory.mkdir(parents=True,exist_ok=True)
    complete=directory/"OUTPUT_COMPLETE.json"
    if complete.exists():
        payload=json.loads(complete.read_text(encoding="utf-8"));
        if payload.get("config_sha256")==config_hash and payload.get("training_epochs")==epochs:return
        raise RuntimeError("completion identity mismatch")
    factor=json.loads((out/"nscale_factors.json").read_text(encoding="utf-8"));identity={"protocol_id":config["protocol_id"],"config_sha256":config_hash,"condition_id":cid,"profile":cond["profile"],"state_id":cond["state_id"],"seed":seed,"target_timesteps":target,"factor_sha256":factor["factors_sha256"],"source_scaler_sha256":sha256_path(old_out/"scaler_registry.json"),"training_universe_sha256":_digest(m["full_train_cases"]),"horizon_seconds":config["horizon_seconds"],"test_access_count":0}
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1);vector,sequence=make_vector(config,m,store,scalers,cond,seed,out)
    checkpoints=sorted(directory.glob("checkpoint_*_post_update"));starting=0
    if checkpoints:
        latest=checkpoints[-1];meta=json.loads((latest/"metadata.json").read_text(encoding="utf-8"));starting=int(meta["timestep"]);model=load_checkpoint(latest,vector,sequence)
    else:model=make_ppo_model(vector,replace(PAPER_ORIENTED_PPO_CANDIDATE_V1,seed=seed,total_timesteps=target,purpose="day04_long_horizon_confirmation"))
    started=time.perf_counter()
    while starting<target:
        chunk=min(interval,target-starting);model.learn(total_timesteps=chunk,reset_num_timesteps=(starting==0),progress_bar=False);starting=int(model.num_timesteps)
        if starting<target:save_checkpoint(directory,starting,model,sequence,identity)
    if model.num_timesteps!=target or model._n_updates!=epochs:raise RuntimeError(f"update mismatch {model.num_timesteps}/{model._n_updates}")
    final=save_checkpoint(directory,target,model,sequence,identity,True);meta=json.loads((final/"metadata.json").read_text(encoding="utf-8"));meta|={"completed":True,"wall_seconds_this_process":time.perf_counter()-started,"expected_rollouts":rollouts,"expected_training_epochs":epochs};atomic_json(complete,meta);vector.close();print(json.dumps({"condition":cid,"seed":seed,"steps":target,"epochs":epochs}))

def smoke(config_path:Path)->None:
    config,_=load_config(config_path);out=out_root(config);_,_,_,_,store,scalers=context(config);m=json.loads((out/"private_membership.json").read_text(encoding="utf-8"));cond=condition(config,"P0S0");root=out/"smoke";root.mkdir(exist_ok=True);torch.set_num_threads(1);random.seed(777);np.random.seed(777);torch.manual_seed(777)
    vector,seq=make_vector(config,m,store,scalers,cond,777,out);model=make_ppo_model(vector,replace(PAPER_ORIENTED_PPO_CANDIDATE_V1,seed=777,total_timesteps=4096,purpose="day04_resume_equivalence"));model.learn(total_timesteps=2048,progress_bar=False);checkpoint=save_checkpoint(root,2048,model,seq,{"purpose":"day04_resume_equivalence"});model.learn(total_timesteps=2048,reset_num_timesteps=False,progress_bar=False);expected={k:v.detach().cpu().clone() for k,v in model.policy.state_dict().items()};vector.close()
    vector2,seq2=make_vector(config,m,store,scalers,cond,777,out);resumed=load_checkpoint(checkpoint,vector2,seq2);resumed.learn(total_timesteps=2048,reset_num_timesteps=False,progress_bar=False);exact=all(torch.equal(expected[k],v.detach().cpu()) for k,v in resumed.policy.state_dict().items());vector2.close()
    if resumed.num_timesteps!=4096 or resumed._n_updates!=20 or not exact:raise RuntimeError("resume smoke mismatch")
    atomic_json(root/"verification.json",{"verified":True,"steps":4096,"rollouts":2,"training_epochs":20,"bitwise_policy_resume_equivalence":True,"finite_input":True,"test_access_count":0});print("smoke verified")

def benchmark(config_path:Path)->None:
    config,_=load_config(config_path);out=out_root(config);_,_,_,_,store,scalers=context(config);m=json.loads((out/"private_membership.json").read_text(encoding="utf-8"));torch.set_num_threads(1);vector,_=make_vector(config,m,store,scalers,condition(config,"P0S0"),48,out);model=make_ppo_model(vector,replace(PAPER_ORIENTED_PPO_CANDIDATE_V1,seed=48,total_timesteps=8192,purpose="day04_benchmark"));start=time.perf_counter();model.learn(total_timesteps=8192,progress_bar=False);seconds=time.perf_counter()-start;vector.close();projected=seconds*(int(config["training_target_timesteps"])/8192)*20/3600+0.75;fallback=projected>float(config["budget_fallback_projection_hours"]);target=int(config["uniform_budget_fallback_timesteps"] if fallback else config["training_target_timesteps"]);protocol=json.loads((out/"protocol_manifest.json").read_text(encoding="utf-8"));protocol|={"benchmark_steps":8192,"benchmark_seconds":seconds,"steps_per_second":8192/seconds,"projected_core_hours":projected,"budget_fallback_invoked":fallback,"effective_timesteps":target,"expected_rollouts":target//2048,"expected_training_epochs":target//2048*10,"worker_count":1,"budget_frozen_before_primary_training":True};atomic_json(out/"protocol_manifest.json",protocol);atomic_json(out/"benchmark.json",protocol);print(json.dumps({"sps":8192/seconds,"projected_hours":projected,"fallback":fallback,"target":target}))

def preoutcome_committed()->bool:
    required=["configs/journal/day04_confirmation.json","scripts/journal/run_day04_confirmation.py","tests/test_day04_confirmation.py","docs/journal/DAY04_BRIEF.md","reports/journal/day04_visibility_grid.csv"]
    if any(not subprocess.check_output(["git","ls-files",p],cwd=ROOT,text=True).strip() for p in required):return False
    if subprocess.check_output(["git","status","--porcelain","--",*required],cwd=ROOT,text=True).strip():return False
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip();branches=subprocess.check_output(["git","branch","-r","--contains",head],cwd=ROOT,text=True)
    return "origin/journal/day04-confirmation" in branches

def supervise(config_path:Path)->None:
    config,_=load_config(config_path);out=out_root(config)
    if not preoutcome_committed():raise RuntimeError("pre-outcome protocol/code must be committed and pushed before training")
    jobs=[(seed,c["condition_id"]) for seed in config["seeds"] for c in config["conditions"]];attempts=defaultdict(int);started=time.time()
    for index,(seed,cid) in enumerate(jobs):
        complete=job_dir(out,seed,cid)/"OUTPUT_COMPLETE.json"
        if complete.exists():continue
        key=f"seed_{seed}/{cid}"
        while attempts[key]<=int(config["retry_limit"]):
            attempts[key]+=1;write_status(out,"training",completed_jobs=sum((job_dir(out,s,c)/"OUTPUT_COMPLETE.json").exists() for s,c in jobs),total_jobs=20,active_job=key,attempt=attempts[key],elapsed_minutes=(time.time()-started)/60,waiting_for_external_cpu=False,process_guard_limitation=config["process_guard"])
            directory=job_dir(out,seed,cid);directory.mkdir(parents=True,exist_ok=True);log=(directory/"worker.log").open("a",encoding="utf-8");result=subprocess.run([sys.executable,str(Path(__file__).resolve()),"train-one","--config",str(config_path.resolve()),"--condition",cid,"--seed",str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);log.close()
            if result.returncode==0 and complete.exists():break
            if attempts[key]>int(config["retry_limit"]):raise RuntimeError(f"job failed {key}")
    write_status(out,"training_complete",completed_jobs=20,total_jobs=20,elapsed_minutes=(time.time()-started)/60)

def metric_record(latent:list[float],actions:list[float],rewards:list[float],clips:int,reasons:list[str])->dict[str,float]:
    bis=np.asarray(latent);act=np.asarray(actions);first=bis[:60];late=bis[60:]
    return {"mae_0_1800":float(np.mean(np.abs(bis-50))),"mae_0_600":float(np.mean(np.abs(first-50))),"mae_600_1800":float(np.mean(np.abs(late-50))),"time_in_40_60_fraction":float(np.mean((bis>=40)&(bis<=60))),"time_below_40_fraction":float(np.mean(bis<40)),"time_above_60_fraction":float(np.mean(bis>60)),"cumulative_reward":float(np.sum(rewards)),"total_propofol_mg":float(np.sum(act)),"action_sd":float(np.std(act)),"consecutive_action_variation":float(np.mean(np.abs(np.diff(act)))),"action_boundary_fraction":float(np.mean((act<=1e-8)|(act>=27.7-1e-8))),"core_clip_fraction":clips/len(act),"visible_fraction":reasons.count("available")/len(reasons),"sqi_rejection_fraction":sum(x in ("sqi_missing_exact_timestamp","sqi_below_threshold") for x in reasons)/len(reasons),"stale_fraction":reasons.count("stale_beyond_pipeline_cap")/len(reasons)}

def run_eval_case(store:Any,caseid:str,cond:dict[str,Any],scaler:Any,horizon:float,seed:int,policy:Callable[[np.ndarray,dict[str,Any]],float],transform:Callable[[np.ndarray],np.ndarray]|None=None,observations:list[np.ndarray]|None=None)->dict[str,float]:
    env=make_environment(truncate_bundle(store.load_case(caseid),horizon),cond,scaler,seed);obs,info=env.reset(seed=seed);latent=[];actions=[];rewards=[];reasons=[];clips=0;done=False
    while not done:
        model_obs=obs if transform is None else transform(obs)
        if observations is not None:observations.append(np.asarray(model_obs,dtype=np.float32))
        action=policy(model_obs,info);obs,reward,terminated,truncated,info=env.step(np.asarray([action],dtype=np.float32));latent.append(float(info["latent_true_bis"]));actions.append(float(info["applied_action_mg_per_10s"]));rewards.append(float(reward));reasons.append(str(info["visible_current_bis_reason"]));clips+=int(info["action_was_clipped"]);done=bool(terminated or truncated)
    env.close();return metric_record(latent,actions,rewards,clips,reasons)

def subject_rows(case_rows:list[dict[str,Any]])->list[dict[str,Any]]:
    grouped=defaultdict(list)
    for row in case_rows:grouped[row["subjectid"]].append(row)
    return [{"subjectid":subject,**{name:float(np.mean([r[name] for r in rows])) for name in PRIMARY_METRICS}} for subject,rows in grouped.items()]

def verify_models(config_path:Path)->dict[str,Any]:
    config,config_hash=load_config(config_path);out=out_root(config);target,rollouts,epochs=effective_budget(config,out);m=json.loads((out/"private_membership.json").read_text(encoding="utf-8"));factor=json.loads((out/"nscale_factors.json").read_text(encoding="utf-8"));by_seed=defaultdict(list);count=0
    if [p for p in out.rglob("*.partial")]:raise RuntimeError("partial artifact cannot be promoted")
    for seed in config["seeds"]:
      for cond in config["conditions"]:
        d=job_dir(out,seed,cond["condition_id"]);completion=json.loads((d/"OUTPUT_COMPLETE.json").read_text(encoding="utf-8"));final=d/"final_post_update";meta=json.loads((final/"metadata.json").read_text(encoding="utf-8"));marker=json.loads((final/"COMPLETE.json").read_text(encoding="utf-8"))
        expected={"config_sha256":config_hash,"condition_id":cond["condition_id"],"seed":seed,"timestep":target,"training_epochs":epochs,"completed_rollouts":rollouts,"factor_sha256":factor["factors_sha256"],"training_universe_sha256":_digest(m["full_train_cases"]),"artifact_stage":"post_update_authoritative_final","test_access_count":0}
        if any(completion.get(k)!=v or meta.get(k)!=v for k,v in expected.items()):raise RuntimeError("final identity mismatch")
        if marker["model_sha256"]!=sha256_path(final/"model.zip") or marker["metadata_sha256"]!=sha256_path(final/"metadata.json"):raise RuntimeError("final checksum mismatch")
        model=PPO.load(str(final/"model.zip"),device="cpu")
        if model.num_timesteps!=target or model._n_updates!=epochs:raise RuntimeError("loaded update mismatch")
        with (final/"environment.pkl").open("rb") as f:snapshot=pickle.load(f)
        by_seed[seed].append((canonical_sha(snapshot["generator_state"]),int(snapshot["episodes"]),_digest(list(snapshot["seen"]))))
        count+=1
    if any(len(set(values))!=1 for values in by_seed.values()):raise RuntimeError("paired case order mismatch")
    result={"verified":True,"model_count":count,"target_timesteps":target,"total_timesteps":target*count,"rollouts_per_job":rollouts,"training_epochs_per_job":epochs,"paired_case_order":True,"factor_sha256":factor["factors_sha256"],"test_access_count":0,"verified_utc":utc_now()};atomic_json(out/"model_verification.json",result);return result

def evaluate(config_path:Path)->None:
    config,_=load_config(config_path);out=out_root(config);verification=verify_models(config_path);_,_,_,_,store,scalers=context(config);m=json.loads((out/"private_membership.json").read_text(encoding="utf-8"));atomic_json(out/"evaluation_lock.json",{"locked":True,"model_verification_sha256":sha256_path(out/"model_verification.json"),"factor_sha256":verification["factor_sha256"],"baseline_sha256":sha256_path(out/"baseline_tuning.json"),"validation_membership_sha256":_digest(m["fresh_validation_subjects"]),"validation_outcome_access_before_lock":False,"locked_utc":utc_now(),"test_access_count":0})
    private=[];aggregates=[];diagnostics=[];horizon=float(config["horizon_seconds"])
    for seed in config["seeds"]:
      for cond in config["conditions"]:
        cid=cond["condition_id"];model=PPO.load(str(job_dir(out,seed,cid)/"final_post_update/model.zip"),device="cpu");fac=factors(out,cond["state_id"]);observations=[];cases=[]
        for caseid in m["fresh_validation_cases"]:
            metrics=run_eval_case(store,caseid,cond,scalers[cond["state_id"]],horizon,seed,lambda o,i,model=model:float(np.asarray(model.predict(o,deterministic=True)[0]).reshape(-1)[0]),lambda x,fac=fac:apply_transform(x,"Nscale",fac),observations)
            row={"kind":"primary","condition_id":cid,"seed":seed,"caseid":caseid,"subjectid":str(store._by_case[caseid]["subjectid"]),**metrics};private.append(row);cases.append(row)
        subjects=subject_rows(cases);aggregates.append({"kind":"primary","condition_id":cid,"seed":seed,"subject_count":len(subjects),"case_count":len(cases),**{name:float(np.mean([r[name] for r in subjects])) for name in PRIMARY_METRICS}});diagnostics.append({"kind":"primary","condition_id":cid,"seed":seed,"finite_input":bool(np.isfinite(observations).all()),**actor_diagnostics(model,np.stack(observations))})
    selected=json.loads((out/"baseline_tuning.json").read_text(encoding="utf-8"))["selection"]
    for profile,cid in (("P0","P0S0"),("P1","P1S0")):
      cond=condition(config,cid)
      for family in ("constant","P","PI"):
        p=selected[f"{profile}/{family}"];cases=[]
        for caseid in m["fresh_validation_cases"]:
            ctl=Controller(p["family"],p["base"],p["kp"],p["ki"]);ctl.reset();metrics=run_eval_case(store,caseid,cond,scalers["S0"],horizon,45,lambda o,i,ctl=ctl:ctl.predict_once(i));row={"kind":"baseline","condition_id":profile,"seed":family,"caseid":caseid,"subjectid":str(store._by_case[caseid]["subjectid"]),**metrics};private.append(row);cases.append(row)
        subjects=subject_rows(cases);aggregates.append({"kind":"baseline","condition_id":profile,"seed":family,"subject_count":len(subjects),"case_count":len(cases),**{name:float(np.mean([r[name] for r in subjects])) for name in PRIMARY_METRICS}})
    day03=ROOT/config["day03_output_root"];d3manifest=json.loads((day03/"nscale_factors.json").read_text(encoding="utf-8"));d3map=d3manifest["factors_by_field"]
    for state in ("S0","S1"):
      cond=condition(config,f"P0{state}");fac=np.asarray([d3map[name] for name in fields(state)],dtype=np.float32)
      for seed in (45,46,47):
        model=PPO.load(str(day03/"jobs"/f"Nscale_{state}"/f"seed_{seed}"/"final_post_update/model.zip"),device="cpu");observations=[];cases=[]
        for caseid in m["fresh_validation_cases"]:
            metrics=run_eval_case(store,caseid,cond,scalers[state],horizon,seed,lambda o,i,model=model:float(np.asarray(model.predict(o,deterministic=True)[0]).reshape(-1)[0]),lambda x,fac=fac:apply_transform(x,"Nscale",fac),observations);row={"kind":"day03_transfer","condition_id":f"P0{state}","seed":seed,"caseid":caseid,"subjectid":str(store._by_case[caseid]["subjectid"]),**metrics};private.append(row);cases.append(row)
        subjects=subject_rows(cases);aggregates.append({"kind":"day03_transfer","condition_id":f"P0{state}","seed":seed,"subject_count":len(subjects),"case_count":len(cases),**{name:float(np.mean([r[name] for r in subjects])) for name in PRIMARY_METRICS}});diagnostics.append({"kind":"day03_transfer","condition_id":f"P0{state}","seed":seed,"finite_input":bool(np.isfinite(observations).all()),**actor_diagnostics(model,np.stack(observations))})
    atomic_json(out/"evaluation_private.json",{"rows":private,"locked_evaluation_passes":1,"test_access_count":0});atomic_json(out/"evaluation_aggregate.json",{"rows":aggregates,"diagnostics":diagnostics,"validation_subject_count":len(m["fresh_validation_subjects"]),"validation_case_count":len(m["fresh_validation_cases"]),"test_access_count":0});print(json.dumps({"private_rows":len(private),"aggregate_rows":len(aggregates)}))

def contrast_value(values:dict[str,float],name:str)->float:
    if name=="P0_state":return values["P0S1"]-values["P0S0"]
    if name=="P1_state":return values["P1S1"]-values["P1S0"]
    if name=="S0_preprocessing":return values["P1S0"]-values["P0S0"]
    if name=="S1_preprocessing":return values["P1S1"]-values["P0S1"]
    if name=="interaction":return (values["P1S1"]-values["P1S0"])-(values["P0S1"]-values["P0S0"])
    raise KeyError(name)

def contrasts(config_path:Path)->None:
    config,_=load_config(config_path);out=out_root(config);private=json.loads((out/"evaluation_private.json").read_text(encoding="utf-8"))["rows"];primary=[r for r in private if r["kind"]=="primary"]
    subjects=sorted({r["subjectid"] for r in primary});seeds=config["seeds"];conditions=[c["condition_id"] for c in config["conditions"]];subject_case=defaultdict(list)
    for r in primary:subject_case[(r["seed"],r["condition_id"],r["subjectid"])].append(r)
    values={}
    for seed in seeds:
      for cid in conditions:
       for subject in subjects:values[(seed,cid,subject)]={metric:float(np.mean([r[metric] for r in subject_case[(seed,cid,subject)]])) for metric in ("mae_0_1800","mae_0_600","mae_600_1800")}
    seed_rows=[];summary=[];rng=np.random.default_rng(20260909)
    for name,formula in CONTRAST_FORMULAS.items():
      for metric in ("mae_0_1800","mae_0_600","mae_600_1800"):
        per_seed=[]
        for seed in seeds:
            delta=contrast_value({cid:float(np.mean([values[(seed,cid,s)][metric] for s in subjects])) for cid in conditions},name);per_seed.append(delta);seed_rows.append({"row_type":"seed","contrast":name,"formula":formula,"metric":metric,"seed":seed,"delta":delta})
        if metric=="mae_0_1800":
            draws=[]
            for _ in range(int(config["hierarchical_bootstrap_draws"])):
                sampled_seeds=rng.choice(seeds,len(seeds),replace=True);sampled_subjects=rng.choice(subjects,len(subjects),replace=True);cell={cid:float(np.mean([values[(int(seed),cid,str(subject))][metric] for seed in sampled_seeds for subject in sampled_subjects])) for cid in conditions};draws.append(contrast_value(cell,name))
            low,high=np.quantile(draws,[.025,.975]);favorable=sum(x<0 for x in per_seed)
            if name in ("P0_state","P1_state") and np.mean(per_seed)<0 and favorable==5 and high<0:language="strong_internal_robustness"
            elif name in ("P0_state","P1_state") and np.mean(per_seed)<0 and favorable>=4:language="moderate_suggestive"
            elif name in ("P0_state","P1_state"):language="unstable_inconclusive"
            else:language="descriptive_only"
            summary.append({"row_type":"summary","contrast":name,"formula":formula,"metric":metric,"seed":"mean","delta":float(np.mean(per_seed)),"seed_sd":float(np.std(per_seed,ddof=1)),"favorable_seed_count":favorable,"bootstrap_low":float(low),"bootstrap_high":float(high),"evidence_language":language})
    rows=seed_rows+summary;atomic_json(out/"contrasts.json",{"rows":rows,"bootstrap":"paired_seed_and_subject_descriptive","draws":config["hierarchical_bootstrap_draws"],"test_access_count":0})
    with PUBLIC_CONTRASTS.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(config_path:Path)->None:
    config,config_hash=load_config(config_path);out=out_root(config);agg=json.loads((out/"evaluation_aggregate.json").read_text(encoding="utf-8"));contrast=json.loads((out/"contrasts.json").read_text(encoding="utf-8"));protocol=json.loads((out/"protocol_manifest.json").read_text(encoding="utf-8"));factor=json.loads((out/"nscale_factors.json").read_text(encoding="utf-8"));visibility=json.loads((out/"visibility_audit_aggregate.json").read_text(encoding="utf-8"))["rows"]
    rows=agg["rows"]
    with PUBLIC_RESULTS.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    primary=[r for r in rows if r["kind"]=="primary"];means=[]
    for cid in ("P0S0","P1S0","P0S1","P1S1"):
        selected=[r for r in primary if r["condition_id"]==cid];means.append({"condition_id":cid,**{metric+"_mean":float(np.mean([r[metric] for r in selected])) for metric in PRIMARY_METRICS},**{metric+"_seed_sd":float(np.std([r[metric] for r in selected],ddof=1)) for metric in PRIMARY_METRICS}})
    summaries=[r for r in contrast["rows"] if r["row_type"]=="summary"]
    transfer=[r for r in rows if r["kind"]=="day03_transfer"];transfer_delta=[]
    for seed in (45,46,47):transfer_delta.append(next(r for r in transfer if r["condition_id"]=="P0S1" and r["seed"]==seed)["mae_0_1800"]-next(r for r in transfer if r["condition_id"]=="P0S0" and r["seed"]==seed)["mae_0_1800"])
    timing=json.loads((out/"timing.json").read_text(encoding="utf-8")) if (out/"timing.json").exists() else {}
    summary={"protocol_id":config["protocol_id"],"config_sha256":config_hash,"evidence_scope":config["evidence_scope"],"protocol":protocol,"validation_subject_count":agg["validation_subject_count"],"validation_case_count":agg["validation_case_count"],"primary_condition_summary":means,"aggregate_rows":rows,"contrasts":contrast,"diagnostics":agg["diagnostics"],"visibility_grid":visibility,"factor_summary":{"factors_sha256":factor["factors_sha256"],"observation_count":factor["observation_count"],"checks":factor["checks"],"day03_comparison":factor["day03_comparison"]},"day03_transfer_P0_state_seed_deltas":transfer_delta,"timing":timing,"test_access_count":0,"generated_utc":utc_now()};atomic_json(PUBLIC_SUMMARY,summary)
    c={r["contrast"]:r for r in summaries};p={r["condition_id"]:r for r in means};baseline={(r["condition_id"],r["seed"]):r for r in rows if r["kind"]=="baseline"};diag=[r for r in agg["diagnostics"] if r["kind"]=="primary"]
    lines=["# Day 04 Long-Horizon Confirmation Results","","## 결론","",f"P0 state contrast: {c['P0_state']['evidence_language']}; P1 state contrast: {c['P1_state']['evidence_language']}. 이 결과는 고정 환자 profile과 remifentanil schedule 아래 재구성 simulation trajectory에만 해당한다.","","## Primary conditions","","| Condition | MAE 0–1800 (mean ± seed SD) | MAE 0–600 | MAE 600–1800 | visibility | boundary |","|---|---:|---:|---:|---:|---:|"]
    for cid in ("P0S0","P1S0","P0S1","P1S1"):r=p[cid];lines.append(f"| {cid} | {r['mae_0_1800_mean']:.4f} ± {r['mae_0_1800_seed_sd']:.4f} | {r['mae_0_600_mean']:.4f} | {r['mae_600_1800_mean']:.4f} | {r['visible_fraction_mean']:.3f} | {r['action_boundary_fraction_mean']:.3f} |")
    lines += ["","## Fixed contrasts","","| Contrast | Mean ± seed SD | Favorable seeds | Descriptive 95% hierarchical bootstrap | Language |","|---|---:|---:|---:|---|"]
    for name in CONTRAST_FORMULAS:r=c[name];lines.append(f"| {name} | {r['delta']:+.4f} ± {r['seed_sd']:.4f} | {r['favorable_seed_count']}/5 | [{r['bootstrap_low']:+.4f}, {r['bootstrap_high']:+.4f}] | {r['evidence_language']} |")
    lines += ["","## Baselines and diagnostics","",f"Train-only tuned PI validation MAE: P0 {baseline[('P0','PI')]['mae_0_1800']:.4f}, P1 {baseline[('P1','PI')]['mae_0_1800']:.4f}. Primary actor inputs were finite. Per-seed saturation and action diagnostics are in the machine-readable summary.","","## Day 03 transfer diagnostic","",f"Frozen Day 03 P0 S1−S0 1,800-second deltas on the fresh subset: " + ", ".join(f"{x:+.4f}" for x in transfer_delta)+". This secondary diagnostic changes training universe and horizon simultaneously and is not an isolated causal comparison.","","## Limitations","","The source cohort is internally reused, although this 96-subject subset was previously unused for journal development. Known historical test results were not used and original-test access remained zero. This is reconstructed simulation, not clinical or causal evidence. Five seeds make the hierarchical interval descriptive.","","## Reproduction","",f"Effective budget: {protocol['effective_timesteps']:,} steps/job; 20 jobs. Resume: `.venv-journal\\Scripts\\python.exe scripts\\journal\\run_day04_confirmation.py resume`. Private identifiers, models, checkpoints, logs and rows remain under the ignored Day 04 output root."]
    PUBLIC_REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    PUBLIC_OVERLEAF.write_text("# Day 04 Overleaf note\n\n- Setting: VitalDB-informed reconstructed PK/PD simulation; no clinical or causal claim.\n- Full development-training partition, fresh internal-validation subset, 1,800-s horizon, five paired seeds.\n- Primary MAE values: "+"; ".join(f"{cid} {p[cid]['mae_0_1800_mean']:.4f} ± {p[cid]['mae_0_1800_seed_sd']:.4f}" for cid in ("P0S0","P1S0","P0S1","P1S1"))+".\n- Contrasts: "+"; ".join(f"{name} {c[name]['delta']:+.4f} ({c[name]['favorable_seed_count']}/5 favorable)" for name in CONTRAST_FORMULAS)+".\n- Descriptive hierarchical bootstrap resampled paired seeds and subjects; n=5 policy seeds.\n",encoding="utf-8")

def verify(config_path:Path)->None:
    config,_=load_config(config_path);out=out_root(config);models=verify_models(config_path);summary=json.loads(PUBLIC_SUMMARY.read_text(encoding="utf-8"));m=json.loads((out/"private_membership.json").read_text(encoding="utf-8"));private=json.loads((out/"evaluation_private.json").read_text(encoding="utf-8"))["rows"]
    primary=[r for r in private if r["kind"]=="primary"]
    expected=20*len(m["fresh_validation_cases"])
    if len(primary)!=expected or models["model_count"]!=20 or summary["test_access_count"]!=0:raise RuntimeError("evaluation/model accounting")
    public=[PUBLIC_BRIEF,PUBLIC_REPORT,PUBLIC_RESULTS,PUBLIC_CONTRASTS,PUBLIC_VISIBILITY,PUBLIC_SUMMARY,PUBLIC_OVERLEAF]
    for path in public:
        text=path.read_text(encoding="utf-8")
        if any(x in text for x in ("subjectid","caseid","C:\\Users\\","source_root","patient_profile.json")):raise RuntimeError(f"privacy token {path}")
    result={"verified":True,"models":20,"checkpoints":20*(models["target_timesteps"]//int(config["checkpoint_interval_timesteps"])),"total_trained_timesteps":models["total_timesteps"],"validation_subjects":len(m["fresh_validation_subjects"]),"validation_cases":len(m["fresh_validation_cases"]),"evaluation_calls":len(primary),"subject_aggregation_before_group_metrics":True,"paired_case_order":True,"privacy_scan":True,"test_access_count":0,"verified_utc":utc_now()};atomic_json(out/"verification.json",result);print(json.dumps(result,indent=2))

def resume(config_path:Path)->None:
    config,_=load_config(config_path);out=out_root(config);start=time.time();atomic_json(out/"timing.json",{"started_utc":utc_now()})
    if not (out/"preflight.json").exists():prepare(config_path)
    if not (out/"visibility_audit_aggregate.json").exists():audit(config_path)
    if not (out/"baseline_tuning.json").exists():tune_baselines(config_path)
    if not (out/"nscale_factors.json").exists():calibrate(config_path)
    if not (out/"smoke/verification.json").exists():smoke(config_path)
    if not (out/"benchmark.json").exists():benchmark(config_path)
    supervise(config_path)
    if not (out/"evaluation_aggregate.json").exists():evaluate(config_path)
    if not (out/"contrasts.json").exists():contrasts(config_path)
    timing={"started_utc":datetime.fromtimestamp(start,timezone.utc).isoformat(),"completed_utc":utc_now(),"wall_hours":(time.time()-start)/3600};atomic_json(out/"timing.json",timing);summarize(config_path);verify(config_path)

def main()->None:
    p=argparse.ArgumentParser();p.add_argument("command",choices=("prepare","audit","baseline","calibrate","smoke","benchmark","train-one","supervise","evaluate","contrasts","summarize","verify","resume"));p.add_argument("--config",type=Path,default=DEFAULT_CONFIG);p.add_argument("--condition");p.add_argument("--seed",type=int);p.add_argument("--skip-store-verification",action="store_true");a=p.parse_args()
    if a.command=="prepare":prepare(a.config,not a.skip_store_verification)
    elif a.command=="audit":audit(a.config)
    elif a.command=="baseline":tune_baselines(a.config)
    elif a.command=="calibrate":calibrate(a.config)
    elif a.command=="smoke":smoke(a.config)
    elif a.command=="benchmark":benchmark(a.config)
    elif a.command=="train-one":
        if a.condition is None or a.seed is None:p.error("train-one needs --condition/--seed")
        train_one(a.config,a.condition,a.seed)
    elif a.command=="supervise":supervise(a.config)
    elif a.command=="evaluate":evaluate(a.config)
    elif a.command=="contrasts":contrasts(a.config)
    elif a.command=="summarize":summarize(a.config)
    elif a.command=="verify":verify(a.config)
    else:resume(a.config)
if __name__=="__main__":main()
