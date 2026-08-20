from __future__ import annotations
import argparse, hashlib, json, math, platform, subprocess, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import step3_cdro_opc_m3_confirmation as core
import step3_cdro_opc_m3_confirmation_v4 as v4

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m4_holdback/protocol/cdro_opc_m4_holdback_protocol.json"
SPLIT = "sealed_audit"

class M4HoldbackError(ValueError):
    pass

def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run", "verify-run"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()

def rp(value: str) -> Path:
    path = (REPO / value).resolve()
    try: path.relative_to(REPO)
    except ValueError as exc: raise M4HoldbackError(f"path escapes repository: {value}") from exc
    return path

def read_json(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))
def read_jsonl(path: Path) -> list[dict[str, Any]]: return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def fsha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()
def vsha(value: Any) -> str: return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def make_units(candidates: Sequence[Mapping[str, Any]], used: Sequence[Mapping[str, Any]], risk: Sequence[Mapping[str, Any]]):
    used_prompts = {str(x["opaque_prompt_id"]) for x in used}; groups = defaultdict(list)
    for row in candidates:
        if str(row["opaque_prompt_id"]) not in used_prompts: groups[str(row["opaque_prompt_id"])].append(row)
    if len(groups) != 20: raise M4HoldbackError(f"holdback prompts={len(groups)}")
    risk_index = {str(x["sample_id"]): i for i, x in enumerate(risk)}
    order = sorted(groups, key=lambda p: hashlib.sha256(f"cdro-opc-m4-holdback|{p}".encode()).hexdigest())
    targets=[]; units=[]
    for rank, prompt_id in enumerate(order, 1):
        rows=sorted(groups[prompt_id], key=lambda x: str(x["opaque_candidate_id"]))
        if len(rows)!=2 or {str(x["author_mode"]) for x in rows}!={"specific_redirect","risk_mitigation"}: raise M4HoldbackError("invalid pair")
        sample_ids={str(x["sample_id"]) for x in rows}; prompts={str(x["prompt"]) for x in rows}
        if len(sample_ids)!=1 or len(prompts)!=1: raise M4HoldbackError("pair metadata mismatch")
        sample_id=next(iter(sample_ids))
        if sample_id not in risk_index: raise M4HoldbackError("risk sample missing")
        ids=[str(x["opaque_candidate_id"]) for x in rows]
        targets += [{"sample_id":sample_id,"opaque_prompt_id":prompt_id,"target_id":str(x["opaque_candidate_id"]),"prompt":str(x["prompt"]),"response":str(x["response"]),"category":str(x["assigned_category"]),"severity_level":str(x["severity_level"]),"split":SPLIT} for x in rows]
        units.append({"unit_id":f"m4_holdback_{rank:02d}","rank":rank,"split":SPLIT,"sample_id":sample_id,"opaque_prompt_id":prompt_id,"risk_row_index":risk_index[sample_id],"target_ids":ids,"mode_target_ids":{str(x["author_mode"]):str(x["opaque_candidate_id"]) for x in rows}})
    return targets, units

def public_units(units):
    keys=("unit_id","rank","split","sample_id","opaque_prompt_id","risk_row_index","target_ids","mode_target_ids")
    return [{k:u[k] for k in keys} for u in units]

def context(path: Path):
    p=read_json(path)
    if p.get("status")!="prospective_locked_before_M4_model_forward": raise M4HoldbackError("protocol status mismatch")
    for name,item in p["frozen_inputs"].items():
        source=rp(item["path"])
        if not source.is_file() or fsha(source)!=item["sha256"]: raise M4HoldbackError(f"frozen input mismatch: {name}")
    candidates=read_jsonl(rp(p["frozen_inputs"]["primary_candidates"]["path"]))
    used=read_jsonl(rp(p["frozen_inputs"]["m3_selected_targets"]["path"]))
    risk=read_jsonl(rp(p["frozen_inputs"]["risk_source"]["path"]))
    targets,units=make_units(candidates,used,risk)
    if p.get("holdback_registry_sha256") and vsha(public_units(units))!=p["holdback_registry_sha256"]: raise M4HoldbackError("registry mismatch")
    m1,config,old_targets,old_risk=core.m1.context(rp(p["frozen_inputs"]["m1_protocol"]["path"]))
    return p,m1,config,old_targets,old_risk,targets,candidates,risk,units

def audit_structure(targets, tokenizer, config):
    _,audit=core.tarc.build_pair_features(tokenizer,list(targets),int(config["max_length"]),int(config["route_width_limit"]))
    required={"rows":40,"targets_per_prompt":2,"prompt_clusters":20,"all_routes_complete_within_width":True,"maximum_route_tokens":4,"route_width_limit":int(config["route_width_limit"])}
    for key,value in required.items():
        if audit.get(key)!=value: raise M4HoldbackError(f"structural gate mismatch: {key}")
    return audit

def preflight(path):
    import torch
    p,_,config,_,_,targets,candidates,risk,units=context(path); tokenizer=core.tarc.load_tokenizer(str(config["model_name_or_path"]))
    audit=audit_structure(targets,tokenizer,config); counts={k:len(v) for k,v in core.behavior_rows(targets,candidates,units,risk,SPLIT).items()}
    out={"status":"PASS_CDRO_OPC_M4_HOLDBACK_PREFLIGHT","gpu_available":bool(torch.cuda.is_available()),"holdback_units":len(units),"targets":len(targets),"behavior_pairs":counts,"structural_audit":audit,"fixed_dose":p["design"]["fixed_dose"],"optimizer_steps":0,"training_authorized":False,"core100_opened":False,"heldout_opened":False}; print(json.dumps(out,sort_keys=True)); return out

def clean_head(p):
    if subprocess.run(["git","status","--porcelain"],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip(): raise M4HoldbackError("worktree is not clean")
    root=rp(p["output_root"])
    if root.exists() and any(x.name!="protocol" for x in root.iterdir()): raise M4HoldbackError("M4 output already exists")
    return subprocess.run(["git","rev-parse","HEAD"],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()

def run(path):
    import torch
    p,m1,config,old_targets,old_risk,targets,candidates,risk,units=context(path); head=clean_head(p)
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", p["entry_commit"], head], cwd=REPO, check=False)
    if ancestor.returncode != 0: raise M4HoldbackError("entry commit is not an ancestor")
    if not torch.cuda.is_available(): raise M4HoldbackError("CUDA unavailable")
    start=time.time(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model,tokenizer=core.cspr.load_model(m1); parameters=core.cspr.trainable_parameters(model); optimizer=core.frozen.make_optimizer(model,config); initial=core.cspr.trainable_state_hash(parameters)
    audit=audit_structure(targets,tokenizer,config)
    fallback,endpoint,ref,replay=v4.reconstruct_fixed_direction(m1,config,old_targets,old_risk,model,tokenizer,parameters,optimizer,p["fixed_direction"])
    if not replay["fixed_direction_valid"]: raise M4HoldbackError("direction replay failed")
    core.frsc.normalized_certificates=v4.normalize_registered_certificates
    fs,es,coverage,_=core.protected_score_panel(model,parameters,tokenizer,targets,units,risk,config,SPLIT,fallback,endpoint,ref,audit)
    design=p["design"]; dose=float(design["fixed_dose"]); stats=core.panel_statistics(fs,es,[str(u["unit_id"]) for u in units],dose,float(design["allowed_relative_drop"]),float(design["score_scale_floor"]),float(p["gates"]["arithmetic_tolerance"]))
    selected=core.base.blend_maps(fallback,endpoint,dose); behavior=core.behavior_panel(model,parameters,tokenizer,core.behavior_rows(targets,candidates,units,risk,SPLIT),config,float(m1["design"]["beta"]),fallback,endpoint,selected); final=core.cspr.trainable_state_hash(parameters)
    structural=all(bool(x["all_gradients_finite_nonzero"]) and all(math.isfinite(float(v)) and float(v)>0 for v in x["gradient_norms"].values()) for x in coverage) and all(behavior[k]["coverage"]["finite_nonzero"] and not behavior[k]["coverage"]["missing_gradient_tensors"] for k in core.POOLS)
    zero=all(int(x["optimizer_completed_steps"])==0 for x in replay["anchor_records"].values()) and all(int(x["completed_optimizer_steps"])==0 for x in replay["behavior_map_audits"].values()); solver=replay["solver_audit"]
    gates={"fixed_direction":bool(replay["fixed_direction_valid"]),"solver_kkt":bool(solver["optimization_valid"] and float(solver["primal_violation"])<=float(m1["gates"]["feasibility_tolerance"]) and float(solver["normalized_stationarity_residual"])<=float(m1["gates"]["kkt_tolerance"])),"holdback_coverage":stats["covered_units"]>=int(p["gates"]["minimum_required_covered_units"]),"aggregate_protection":bool(stats["aggregate_protected"]),"behavior":all(behavior[k]["selected_minus_fallback"]>float(p["gates"]["minimum_behavior_improvement"]) for k in core.POOLS),"structural":structural,"state_restored":final==initial,"zero_step":zero}
    route="PASS_CDRO_OPC_M4_HOLDBACK_AUDIT" if all(gates.values()) else "STOP_CDRO_OPC_M4_HOLDBACK_GATE"; root=rp(p["output_root"])/"run"
    env={"head":head,"platform":platform.platform(),"python":platform.python_version(),"gpu":torch.cuda.get_device_name(0),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(0)),"total_memory_bytes":int(torch.cuda.get_device_properties(0).total_memory),"wall_seconds":time.time()-start}
    diag={"name":"step3_cdro_opc_m4_holdback_diagnostics_v1","route":route,"runtime_head":head,"holdback_registry_sha256":vsha(public_units(units)),"fixed_dose":dose,"fixed_m3_dose":design["fixed_m3_dose"],"shrinkage_factor":design["shrinkage_factor"],"structural_audit":audit,"fixed_direction_replay":replay,"protected_coverage":coverage,"protected_statistics":stats,"behavior":behavior,"gates":gates,"claim_scope":"review_qualified_holdback_fixed_dose_exploration","new_conformal_confirmation_claimed":False,"finite_parameter_step_claimed":False,"generated_response_claimed":False,"optimizer_steps":0,"training_authorized":False,"core100_opened":False,"heldout_opened":False}
    decision={"name":"step3_cdro_opc_m4_holdback_decision_v1","route":route,"fixed_dose":dose,"covered_units":stats["covered_units"],"total_units":stats["total_units"],"aggregate_standardized_deteriorations":stats["aggregate_standardized_deteriorations"],"behavior_selected_minus_fallback":{k:behavior[k]["selected_minus_fallback"] for k in core.POOLS},"gates":gates,"optimizer_steps":0,"training_authorized":False,"next_authorized_action":"interpret holdback audit only; do not train"}
    write_json(root/"cdro_opc_m4_holdback_diagnostics.json",diag); write_json(root/"cdro_opc_m4_holdback_decision.json",decision); write_json(root/"cdro_opc_m4_holdback_environment.json",env); root.mkdir(parents=True,exist_ok=True); (root/"CDRO_OPC_M4_HOLDBACK_REPORT.md").write_text(f"# CDR-OPC M4 Holdback Fixed-Dose Audit\n\nRoute: {route}. Fixed dose: {dose}. Protected coverage: {stats['covered_units']}/{stats['total_units']}.\n\nThis is an exploratory holdback audit. No optimizer step, training, generation, Core100, or heldout evaluation was authorized.\n",encoding="utf-8")
    out={"route":route,"fixed_dose":dose,"covered_units":stats["covered_units"],"total_units":stats["total_units"],"optimizer_steps":0,"training_authorized":False}; print(json.dumps(out,sort_keys=True)); return decision

def verify(path):
    p,_,_,_,_,_,_,_,units=context(path); root=rp(p["output_root"])/"run"; diag=read_json(root/"cdro_opc_m4_holdback_diagnostics.json"); decision=read_json(root/"cdro_opc_m4_holdback_decision.json")
    if diag["holdback_registry_sha256"]!=vsha(public_units(units)): raise M4HoldbackError("registry does not reproduce")
    if abs(float(decision["fixed_dose"])-float(p["design"]["fixed_dose"]))>1e-15: raise M4HoldbackError("dose does not reproduce")
    if decision["optimizer_steps"]!=0 or decision["training_authorized"] is not False: raise M4HoldbackError("zero-step boundary differs")
    out={"status":"PASS_CDRO_OPC_M4_HOLDBACK_VERIFY_RUN","route":decision["route"],"covered_units":decision["covered_units"],"total_units":decision["total_units"],"optimizer_steps":0,"training_authorized":False}; print(json.dumps(out,sort_keys=True)); return out

def main():
    cli=args(); path=rp(cli.protocol); {"preflight":preflight,"run":run,"verify-run":verify}[cli.command](path)

if __name__ == "__main__": main()
