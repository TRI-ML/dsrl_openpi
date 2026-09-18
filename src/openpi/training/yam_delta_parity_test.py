"""Train/serve parity for the YAM delta-action fix (proprio-copycat).

Guards the invariant that AbsoluteActions is the exact inverse of DeltaActions for the
make_bool_mask(6,-1,6,-1) mask (joints delta, grippers absolute), and that the YAM data
transforms plus pi05 model transforms wire the delta pair + resize_with_pad image path.
"""

import numpy as np

from openpi import transforms as _transforms
from openpi.models import pi0_config
from openpi.training import config as _config


def _mask():
    return _transforms.make_bool_mask(6, -1, 6, -1)


def test_delta_absolute_roundtrip_14d():
    rng = np.random.default_rng(0)
    mask = _mask()
    assert len(mask) == 14
    # (horizon, 14) actions + (14,) state
    state = rng.normal(size=(14,)).astype(np.float32)
    actions = rng.normal(size=(10, 14)).astype(np.float32)

    delta = _transforms.DeltaActions(mask)({"state": state, "actions": actions.copy()})["actions"]
    recon = _transforms.AbsoluteActions(mask)({"state": state, "actions": delta.copy()})["actions"]

    np.testing.assert_allclose(recon, actions, atol=1e-5)


def test_gripper_dims_stay_absolute():
    mask = np.asarray(_mask())
    rng = np.random.default_rng(1)
    state = rng.normal(size=(14,)).astype(np.float32)
    actions = rng.normal(size=(10, 14)).astype(np.float32)
    delta = _transforms.DeltaActions(_mask())({"state": state, "actions": actions.copy()})["actions"]
    # Where mask is False (gripper dims 6 and 13), delta == absolute (unchanged).
    for d in np.where(~mask)[0]:
        np.testing.assert_allclose(delta[..., d], actions[..., d], atol=1e-6)
    # Where mask is True (joint dims), delta == abs - state.
    for d in np.where(mask)[0]:
        np.testing.assert_allclose(delta[..., d], actions[..., d] - state[d], atol=1e-5)


def test_helper_pushes_delta_pair_when_enabled():
    mc = pi0_config.Pi0Config(pi05=True, action_horizon=10)
    off = _config._yam_data_transforms(mc, use_delta_joint_actions=False)
    on = _config._yam_data_transforms(mc, use_delta_joint_actions=True)
    assert not any(isinstance(t, _transforms.DeltaActions) for t in off.inputs)
    assert any(isinstance(t, _transforms.DeltaActions) for t in on.inputs)
    assert any(isinstance(t, _transforms.AbsoluteActions) for t in on.outputs)


def test_pi05_model_transforms_use_resize_with_pad_224():
    mc = pi0_config.Pi0Config(pi05=True, action_horizon=10)
    grp = _config.ModelTransformFactory()(mc)
    resizes = [t for t in grp.inputs if isinstance(t, _transforms.ResizeImages)]
    assert resizes, "pi05 model transforms must resize images"
    assert (resizes[0].height, resizes[0].width) == (224, 224)
    # ResizeImages uses image_tools.resize_with_pad (not a stretch); guard the source contract.
    import inspect

    from openpi import transforms as T

    assert "resize_with_pad" in inspect.getsource(T.ResizeImages.__call__)


def test_delta_configs_registered_with_delta_enabled():
    for name in ["pi05_yam_makecoffee_full30k_delta", "pi05_yam_placeteabag_full30k_delta"]:
        cfg = _config.get_config(name)
        mc = cfg.model
        dt = cfg.data.data_transforms(mc)
        assert any(isinstance(t, _transforms.DeltaActions) for t in dt.inputs), name
        assert any(isinstance(t, _transforms.AbsoluteActions) for t in dt.outputs), name
