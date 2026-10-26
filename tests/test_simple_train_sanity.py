"""CPU checks for the separate, fixed actor-training execution diagnostic."""
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_sanity", PROJECT / "scripts/evaluate_simple_train_sanity.py")
sanity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sanity)


def test_fixed_train_cases_do_not_select_by_outcome():
    metadata = {"split":{"train":[0,10,20,30,40],"val":[1,24,35,43,52]}}
    full, selected = sanity.make_cases(metadata)
    assert len(full) == len(selected) == 5
    assert [case["episode_id"] for case in full] == [0,10,20,30,40]
    assert all(case["frame"] == 0 and case["object_xy_offset"] == [0.,0.] for case in full)
    assert [case["seed"] for case in full] == [43,53,63,73,83]
    metadata["split"]["val"].append(20)
    with pytest.raises(ValueError, match="actor training"):
        sanity.make_cases(metadata)


def test_replay_keeps_original_index_then_holds_last(tmp_path):
    metadata = {"complete":True,"native_tactile_report":{"velocity_dt":.01},"action_semantics":"source action[s], unchanged"}
    (tmp_path/"manifest.json").write_text(json.dumps(metadata))
    arrays = {"episode_index":np.repeat(sanity.EPISODES,3),"frame_index":np.tile(np.arange(3),5),
              "state":np.zeros((15,22),np.float32),"action":np.arange(15*22,dtype=np.float32).reshape(15,22)}
    for name,value in arrays.items(): np.save(tmp_path/f"{name}.npy",value)
    _,_,rows,frames,commands=sanity.load_replay_source(tmp_path,20)
    np.testing.assert_array_equal(rows,[6,7,8])
    np.testing.assert_array_equal(frames[:5],[0,1,2,2,2])
    np.testing.assert_array_equal(commands[:3],arrays["action"][6:9])
    np.testing.assert_array_equal(commands[-1],arrays["action"][8])
    assert len(commands)==600


def test_recorded_endpoint_is_scoring_only_not_a_selection_rule(tmp_path):
    raw=tmp_path/"raw"/"episode"/"prioception";raw.mkdir(parents=True)
    initial=np.zeros(29);initial[22:25]=[.5,0,.04]
    terminal=initial.copy();terminal[22:25]=[.5,-.1,.04]
    np.save(raw/"first.npy",initial);np.save(raw/"last.npy",terminal)
    timestamps=tmp_path/"timestamps.csv"
    with timestamps.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["prioception_path"]);writer.writeheader()
        writer.writerows([{"prioception_path":"prioception/first.npy"},{"prioception_path":"prioception/last.npy"}])
    nominal={"timestamps_path":str(timestamps),"raw_episode":"episode"}
    audit=sanity.recorded_endpoint_audit(tmp_path/"raw",nominal)
    assert not audit["terminal_legacy_goal_predicate"]
    assert audit["terminal_cube_xyz"]==[.5,-.1,.04]
    terminal[22:25]=[.5,-.3,.04];np.save(raw/"last.npy",terminal)
    assert sanity.recorded_endpoint_audit(tmp_path/"raw",nominal)["terminal_legacy_goal_predicate"]


def test_fake15_diagnostics_run600_ticks_and_do_not_enter_test203(tmp_path,monkeypatch):
    other_spec=importlib.util.spec_from_file_location("simple_eval_fixture",PROJECT/"tests/test_simple_token_evaluation.py")
    other=importlib.util.module_from_spec(other_spec);other_spec.loader.exec_module(other)
    previous,metadata,policies,probe=other.policy_fixture(tmp_path)
    metadata["split"]["train"]=list(sanity.EPISODES)
    full,selected=sanity.make_cases(metadata)
    sources={str(ep):{"qpos_sha256":f"qpos-{ep}","recorded_endpoints_scoring_only":{"terminal_legacy_goal_predicate":False}} for ep in sanity.EPISODES}
    schedules={ep:{"commands":np.full((600,22),float(ep)),"source_frames":np.minimum(np.arange(600),2),
                  "source_global_rows":np.minimum(np.arange(600),2),"held_last_command_ticks":597} for ep in sanity.EPISODES}
    args=sanity.parser().parse_args(["--cache",str(previous.cache),"--output",str(tmp_path/"sanity"),
        "--visual-probe-checkpoint",str(probe),"--weights",str(previous.weights),"--device","cpu","--amp","float32",
        *[item for mode,path in policies.items() for item in ["--policy-checkpoint",f"{mode}={path}"]]])
    resets,steps,encoded=[],[],[]
    raw_pressure=np.zeros(44,np.float32)
    observation={"rgb":np.zeros((2,224,224,3),np.uint8),"state":np.zeros(22),"tactile":raw_pressure,
        "anchors_world":np.zeros((44,3)),"known_cameras_cv":np.repeat(np.eye(4)[None],2,0),"base_to_world":np.eye(4),"cube_GT":np.zeros(3)}
    class Client:
        def __init__(self,*unused): self.tick=0;self.episode=0
        def request(self,command,**values):
            if command=="reset":
                self.tick=0;self.episode=values["episode_id"];resets.append(values)
            elif command=="step":self.tick+=1;steps.append(values)
            info={"cube_center":[.5,0,.04] if self.tick==0 else [.5,-.3,.04],"lifted":False,"inside_goal":self.tick>0,
                  "strict_stable_goal":self.tick>=100,"control_tick":self.tick,"robot_cube_contact_count":0,
                  "initialization":{"episode_id":self.episode,"frame":0,"qpos_sha256":f"qpos-{self.episode}"}}
            return {"info":info,"provenance":other.worker_provenance(),"sim_time":self.tick*.01}
        def close(self):pass
    class Runtime:
        def encode_observation(self,obs):
            assert "cube_GT" not in obs and "info" not in obs
            encoded.append(1);return np.zeros(1088)
    class Policy:
        def reset(self):self.calls=0
        @property
        def needs_observation(self):return self.calls%8==0
        def select_action(self,state=None,features=None,pressure=None):
            if self.needs_observation:assert state.shape==(22,) and features.shape==(1088,) and pressure.shape==(44,)
            else:assert state is features is pressure is None
            self.calls+=1;return np.zeros(22)
    monkeypatch.setattr(sanity,"prepare_sources",lambda unused:(metadata,full,selected,sources,schedules))
    monkeypatch.setattr(sanity,"SimulationClient",Client)
    monkeypatch.setattr(sanity,"read_observation",lambda reply:observation)
    monkeypatch.setattr(sanity,"make_runtime",lambda unused:Runtime())
    monkeypatch.setattr(sanity,"create_policy",lambda *unused:Policy())
    sanity.run(args)
    assert len(resets)==15 and len(steps)==9000 and len(encoded)==750
    assert all(not row["observe"] for row in resets[:5])
    summary=json.loads((args.output/"summary.json").read_text())
    assert summary["complete"] and summary["actor_split"]=="train" and summary["excluded_from_test203"]
    assert set(summary["models"])=={"replay","frozen_zero","frozen_pressure"}
    assert all(model["cases"]==5 for model in summary["models"].values())
    with np.load(args.output/"replay/train_ep000_nominal/trajectory.npz") as trajectory:
        assert trajectory["actions"].shape==(600,22) and trajectory["source_frames"][-1]==2
    args.resume=True;sanity.run(args)
    assert len(resets)==15 and len(steps)==9000
