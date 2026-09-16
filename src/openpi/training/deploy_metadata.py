"""deploy_metadata.json for openpi checkpoints: written by train.py at every save, read by serving.

The sidecar makes a checkpoint self-describing for YAM Eval, the registry ingest and the runners: shapes,
camera keys, chunk length, the openpi TrainConfig it was trained with (name, asset_id, model variants), the
task instructions the dataset carried, the wandb run and the code version. `config_from_sidecar` rebuilds a
TrainConfig from it, so a checkpoint can be served on a host whose config registry never had its name.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import subprocess
from typing import Any

CAMERA_KEYS = ["scene_camera", "left_wrist_camera", "right_wrist_camera"]
IMAGE_OBS_KEYS = ["observation/image_head", "observation/image_left_wrist", "observation/image_right_wrist"]
REPACK = {
    "observation/image_head": "observation.images.scene_camera",
    "observation/image_left_wrist": "observation.images.left_wrist_camera",
    "observation/image_right_wrist": "observation.images.right_wrist_camera",
    "observation/state": "observation.state",
    "actions": "action",
    "prompt": "prompt",
}


def git_sha(repo_root: pathlib.Path | None = None) -> str | None:
    root = repo_root or pathlib.Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def wandb_info(run) -> dict | None:
    if run is None:
        return None
    try:
        return {"id": run.id, "url": run.url, "project": run.project, "entity": run.entity}
    except Exception:  # noqa: BLE001
        return None


def tasks_from_dataset(config) -> list[str] | None:
    """The distinct task strings of the LeRobot dataset the config trains on (None when unavailable)."""
    try:
        data_config = config.data.create(config.assets_dirs, config.model)
        repo_id = data_config.repo_id
        if not repo_id or repo_id == "fake":
            return None
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
        tasks = lerobot_dataset.LeRobotDatasetMetadata(repo_id).tasks
        values = list(tasks.values()) if isinstance(tasks, dict) else list(tasks)
        out, seen = [], set()
        for t in values:
            t = str(t).strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out or None
    except Exception:  # noqa: BLE001
        return None


def build_metadata(config, *, wandb_run=None, tasks: list[str] | None = None, task_instruction: str | None = None,
                   dataset_snapshot_id: str | None = None, code_sha: str | None = None) -> dict:
    model = config.model
    repo_id = getattr(config.data, "repo_id", None)
    asset_id = getattr(getattr(config.data, "assets", None), "asset_id", None)
    horizon = int(model.action_horizon)
    if task_instruction is None and tasks:
        task_instruction = " | ".join(tasks)
    return {
        "schema_version": 1,
        "action_dim": 14,
        "state_dim": 14,
        "action_dim_padded": int(getattr(model, "action_dim", 32)),
        "images": {c: [224, 224, 3] for c in CAMERA_KEYS},
        "image_keys": list(CAMERA_KEYS),
        "image_obs_keys": list(IMAGE_OBS_KEYS),
        "proprio_keys": ["follower_l_joint_pos_7d", "follower_r_joint_pos_7d"],
        "act_steps": horizon,
        "action_horizon": horizon,
        "control_hz": None,
        "config_name": config.name,
        "openpi_config": config.name,
        "exp_name": getattr(config, "exp_name", None),
        "asset_id": asset_id,
        "model": {"pi05": bool(getattr(model, "pi05", False)),
                  "paligemma_variant": getattr(model, "paligemma_variant", None),
                  "action_expert_variant": getattr(model, "action_expert_variant", None)},
        "task_instruction": task_instruction,
        "tasks": tasks,
        "language_conditioned": True,
        "model_family": "pi05",
        "wandb": wandb_info(wandb_run),
        "git_sha": code_sha or git_sha(),
        "dataset": {"repo_id": repo_id, "local_path": None,
                    "s3_uri": f"s3://tri-ml-datasets-uw2/raiden_datasets/lerobot/{repo_id.split('/')[-1]}" if repo_id else None,
                    "snapshot_id": dataset_snapshot_id or os.environ.get("YAM_DATASET_SNAPSHOT_ID")},
    }


def write_sidecar(config, step_dir: pathlib.Path | str, **kw: Any) -> pathlib.Path:
    step_dir = pathlib.Path(str(step_dir))
    step_dir.mkdir(parents=True, exist_ok=True)
    out = step_dir / "deploy_metadata.json"
    out.write_text(json.dumps(build_metadata(config, **kw), indent=2))
    return out


def config_from_sidecar(step_dir: pathlib.Path | str):
    """A TrainConfig equivalent to the one the checkpoint was trained with, from its deploy_metadata.json.
    Used by serve_policy when the config name is not registered on the serving host."""
    import openpi.models.pi0_config as pi0_config
    import openpi.policies.yam_policy as yam_policy
    import openpi.training.config as _config
    import openpi.transforms as _transforms

    meta = json.loads((pathlib.Path(str(step_dir)) / "deploy_metadata.json").read_text())
    name = meta.get("openpi_config") or meta.get("config_name")
    asset_id = meta.get("asset_id") or name
    m = meta.get("model") or {}
    if not name or not asset_id:
        raise ValueError(f"deploy_metadata.json in {step_dir} lacks openpi_config/asset_id")
    model = pi0_config.Pi0Config(
        pi05=bool(m.get("pi05", True)), action_horizon=int(meta.get("act_steps") or meta.get("action_horizon") or 10),
        paligemma_variant=m.get("paligemma_variant") or "gemma_2b",
        action_expert_variant=m.get("action_expert_variant") or "gemma_300m")
    data = _config.SimpleDataConfig(
        repo_id=(meta.get("dataset") or {}).get("repo_id") or f"local/{name}",
        assets=_config.AssetsConfig(asset_id=asset_id),
        data_transforms=lambda model: _transforms.Group(
            inputs=[yam_policy.YAMInputs(model_type=model.model_type)], outputs=[yam_policy.YAMOutputs()]),
        base_config=_config.DataConfig(
            repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform(dict(REPACK))]),
            prompt_from_task=True, action_sequence_keys=("action",)),
    )
    return dataclasses.replace(_config.TrainConfig(name=name, model=model, data=data), name=name)
