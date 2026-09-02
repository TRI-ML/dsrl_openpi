"""Write the deploy_metadata.json sidecar next to a trained openpi checkpoint.

openpi's train.py writes params/ train_state/ assets/ _CHECKPOINT_METADATA but NOT the self-describing
sidecar that YAM Eval / serving / the yam-registry ingest bridge read. This backfills it from the
TrainConfig, so a pi05 checkpoint registers like rfm_rl and foundry do.

  uv run scripts/write_deploy_metadata.py \
      --checkpoint-dir <checkpoint_base>/<exp>/<step> \
      --config-name pi05_yam_flippinkcup \
      --task-instruction "flip pink cup upside down and then right side up."

Shapes are the YAM bimanual contract the YAMInputs transform enforces (14-D action/state padded to 32,
3 cameras @224); action_horizon comes from the config. This is intentionally standalone (argparse + the
config registry) so it runs on any host that can import openpi, after training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_metadata(config_name: str, task_instruction: str | None,
                   dataset_snapshot_id: str | None = None,
                   wandb: dict | None = None) -> dict:
    from openpi.training import config as _config

    cfg = _config.get_config(config_name)
    repo_id = getattr(cfg.data, "repo_id", None)
    asset_id = getattr(getattr(cfg.data, "assets", None), "asset_id", None)
    horizon = int(cfg.model.action_horizon)
    return {
        "schema_version": 1,
        "action_dim": 14,
        "state_dim": 14,
        "action_dim_padded": 32,
        "images": {"scene_camera": [224, 224, 3],
                   "left_wrist_camera": [224, 224, 3],
                   "right_wrist_camera": [224, 224, 3]},
        "image_keys": ["scene_camera", "left_wrist_camera", "right_wrist_camera"],
        "image_obs_keys": ["observation/image_head", "observation/image_left_wrist",
                           "observation/image_right_wrist"],
        "proprio_keys": ["follower_l_joint_pos_7d", "follower_r_joint_pos_7d"],
        "act_steps": horizon,
        "action_horizon": horizon,
        "control_hz": None,
        "config_name": config_name,
        "openpi_config": config_name,
        "asset_id": asset_id,
        "task_instruction": task_instruction,
        "language_conditioned": True,
        "model_family": "pi05",
        "wandb": wandb,   # {id,url,project,entity} of the training run; None if not passed
        "dataset": {"repo_id": repo_id, "local_path": None,
                    "s3_uri": f"s3://tri-ml-datasets-uw2/raiden_datasets/lerobot/{repo_id.split('/')[-1]}"
                              if repo_id else None,
                    # the registered DatasetSnapshot this trained on, so `yam model register` links
                    # checkpoint→dataset machine-read instead of by name lookup (None until provided).
                    "snapshot_id": dataset_snapshot_id},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True, help="the final step dir (…/<exp>/<step>)")
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--task-instruction", default=None)
    ap.add_argument("--dataset-snapshot-id", default=None,
                    help="the registered DatasetSnapshot this trained on (links checkpoint→dataset)")
    ap.add_argument("--wandb-url", default=None, help="training wandb run URL (links checkpoint→run)")
    ap.add_argument("--wandb-id", default=None)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-entity", default=None)
    args = ap.parse_args()
    # This sidecar is written after training, so there's no live wandb run to read — the launcher passes
    # the run it created. Build the block only from what was given; else None (no fabricated entity).
    wandb = {k: v for k, v in (("id", args.wandb_id), ("url", args.wandb_url),
                               ("project", args.wandb_project), ("entity", args.wandb_entity)) if v} or None
    meta = build_metadata(args.config_name, args.task_instruction, args.dataset_snapshot_id, wandb=wandb)
    out = Path(args.checkpoint_dir) / "deploy_metadata.json"
    out.write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
