"""yam_train_config_from_sidecar (migrated from rfm_rl.utils.pi05_sidecar_config): the pure pieces.

* the builder + sidecar helpers must not pin JAX to the CPU at import (the DSRL rollout runs the base on the GPU);
* it refuses a non-pi05 sidecar and a sidecar missing the keys it needs, and stamps the config identically.
"""
import json
import os
import subprocess
import sys
import types

import pytest

from openpi.training import deploy_metadata as dm

SIDECAR = {
    "model_family": "pi05",
    "config_name": "pi05_yam_placeteabag_full30k",
    "asset_id": "pi05_yam_placeteabag_full30k",
    "action_horizon": 10,
    "action_dim_padded": 32,
    "action_dim": 14,
    "task_instruction": "Place the teabag into the teapot",
}


def test_sidecar_helpers_identify_pi05():
    assert dm.sidecar_path_for("/x/ckpt/5000") == "/x/ckpt/5000/deploy_metadata.json"
    assert dm.sidecar_is_pi05(SIDECAR)
    assert dm.sidecar_is_pi05({"openpi_config": "pi05_yam_x"})
    assert not dm.sidecar_is_pi05({"model_family": "vla_foundry", "config_name": "p8_multitask6"})


def test_importing_deploy_metadata_does_not_pin_jax_to_cpu():
    # the builder lazily imports the openpi model/config modules; importing the module itself must not touch JAX
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import openpi.training.deploy_metadata; print(os.environ.get('JAX_PLATFORMS', '<unset>'))",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"},
    )
    assert out.stdout.strip() == "<unset>"


def test_builder_refuses_non_pi05_and_missing_keys(tmp_path, monkeypatch):
    # stub the four imports the builder makes so the checks run without a full JAX/openpi model stack
    _stub_openpi(monkeypatch)
    bad = tmp_path / "deploy_metadata.json"
    bad.write_text(
        json.dumps({"model_family": "vla_foundry", "asset_id": "a", "action_horizon": 10, "action_dim_padded": 32})
    )
    with pytest.raises(ValueError, match="not a pi05 checkpoint"):
        dm.yam_train_config_from_sidecar(str(bad), "x")
    bad.write_text(json.dumps({**SIDECAR, "asset_id": ""}))
    with pytest.raises(ValueError, match="lacks 'asset_id'"):
        dm.yam_train_config_from_sidecar(str(bad), "x")
    bad.write_text(json.dumps(SIDECAR))
    cfg = dm.yam_train_config_from_sidecar(str(bad), "pi05_yam_placeteabag_full30k")
    assert cfg.name == "pi05_yam_placeteabag_full30k"
    assert cfg.model.kwargs == {"pi05": True, "action_horizon": 10, "action_dim": 32}
    assert cfg.data.assets.asset_id == "pi05_yam_placeteabag_full30k"


def _stub_openpi(monkeypatch):
    """Minimal stand-ins for the openpi symbols yam_train_config_from_sidecar touches."""

    class Pi0Config:
        def __init__(self, **kw):
            self.kwargs = kw

    class AssetsConfig:
        def __init__(self, asset_id):
            self.asset_id = asset_id

    class DataConfig:
        def __init__(self, **kw):
            self.kw = kw

    class SimpleDataConfig:
        def __init__(self, assets, data_transforms, base_config):
            self.assets, self.tf, self.base = assets, data_transforms, base_config

    class TrainConfig:
        def __init__(self, name, project_name, model, data):
            self.name, self.project_name, self.model, self.data = name, project_name, model, data

    class Group:
        def __init__(self, inputs=(), outputs=()):
            self.inputs, self.outputs = inputs, outputs

    mods = {
        "openpi.models": types.ModuleType("openpi.models"),
        "openpi.models.pi0_config": types.SimpleNamespace(Pi0Config=Pi0Config),
        "openpi.policies": types.ModuleType("openpi.policies"),
        "openpi.policies.yam_policy": types.SimpleNamespace(
            YAMInputs=lambda model_type: ("in", model_type), YAMOutputs=lambda: "out"
        ),
        "openpi.training.config": types.SimpleNamespace(
            SimpleDataConfig=SimpleDataConfig, AssetsConfig=AssetsConfig, DataConfig=DataConfig, TrainConfig=TrainConfig
        ),
        "openpi.transforms": types.SimpleNamespace(Group=Group),
    }
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
