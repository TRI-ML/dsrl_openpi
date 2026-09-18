"""Backfill deploy_metadata.json next to an existing openpi checkpoint.

train.py writes this sidecar at every save (openpi.training.deploy_metadata); this CLI is for checkpoints trained
before that, or on a host that can import openpi but not the dataset. Task strings come from the dataset when it
is reachable, else from --task-instruction; the wandb run from --wandb-* (there is no live run after training).

  uv run scripts/write_deploy_metadata.py --checkpoint-dir <exp>/<step> --config-name pi05_yam_flippinkcup \
      --task-instruction "flip pink cup upside down and then right side up." --wandb-url https://wandb.ai/tri/rfm_rl/runs/<id>
"""
from __future__ import annotations

import argparse
import json

from openpi.training import config as _config
from openpi.training import deploy_metadata as _dm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True, help="the step dir (<exp>/<step>)")
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--task-instruction", default=None, help="override; default = the dataset's task strings")
    ap.add_argument("--dataset-snapshot-id", default=None)
    ap.add_argument("--dataset-s3-uri", default=None, help="canonical S3 prefix of the training dataset (overrides the raiden_datasets guess)")
    ap.add_argument("--materialization-id", default=None, help="registry materialization id of that dataset")
    ap.add_argument("--action-space", default=None, help="deploy action-space contract, e.g. delta_joint_abs_gripper")
    ap.add_argument("--image-preprocess", default=None, help="deploy image preprocess contract, e.g. resize_with_pad_224")
    ap.add_argument("--wandb-url", default=None)
    ap.add_argument("--wandb-id", default=None)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--git-sha", default=None, help="code version the checkpoint was trained with")
    args = ap.parse_args()
    cfg = _config.get_config(args.config_name)
    tasks = None if args.task_instruction else _dm.tasks_from_dataset(cfg)
    meta = _dm.build_metadata(cfg, tasks=tasks, task_instruction=args.task_instruction,
                              dataset_snapshot_id=args.dataset_snapshot_id, code_sha=args.git_sha)
    wandb = {k: v for k, v in (("id", args.wandb_id), ("url", args.wandb_url), ("project", args.wandb_project),
                               ("entity", args.wandb_entity)) if v}
    meta["wandb"] = wandb or None
    if args.dataset_s3_uri:
        meta["dataset"]["s3_uri"] = args.dataset_s3_uri
    if args.materialization_id:
        meta["dataset"]["materialization_id"] = args.materialization_id
    if args.action_space:
        meta["action_space"] = args.action_space
    if args.image_preprocess:
        meta["image_preprocess"] = args.image_preprocess
    out = _dm.write_sidecar(cfg, args.checkpoint_dir, tasks=tasks, task_instruction=args.task_instruction,
                            dataset_snapshot_id=args.dataset_snapshot_id, code_sha=args.git_sha)
    out.write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}: task_instruction={meta['task_instruction']!r} wandb={meta['wandb']}")


if __name__ == "__main__":
    main()
