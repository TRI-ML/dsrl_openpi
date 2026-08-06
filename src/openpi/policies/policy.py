from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        rtc: dict[str, Any] | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        # Real-Time Chunking (RTC): when enabled, route inference through model.sample_actions_rtc,
        # guiding the new chunk's prefix toward the still-executing tail of the previous chunk for
        # smooth transitions. Off by default → identical to the plain sampler (safe A/B). Only pi05
        # (JAX Pi0 with sample_actions_rtc) supports it; a pytorch model or missing method disables it.
        rtc = rtc or {}
        self._rtc_enabled = bool(rtc.get("enabled")) and not is_pytorch and hasattr(model, "sample_actions_rtc")
        self._rtc_kwargs = {k: rtc[k] for k in
                            ("inference_delay", "prefix_attention_horizon", "prefix_attention_schedule",
                             "max_guidance_weight", "sigma", "num_steps") if k in rtc}
        # last emitted chunk in NORMALIZED model space (what sample_actions returns, pre-output-transform)
        # — exactly the space sample_actions_rtc's prev_action_chunk must live in. None until first infer.
        self._rtc_prev_chunk = None

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            if self._rtc_enabled:
                self._sample_actions_rtc = nnx_utils.module_jit(model.sample_actions_rtc)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None, rtc_shift: int | None = None,
              rtc_reset: bool = False) -> dict:  # type: ignore[misc]
        if rtc_reset:
            self.reset()
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        # RTC: guide this chunk by the previous one IF we have a cached prev chunk. `rtc_shift` = how
        # many actions were executed since the last query; we roll the prev chunk left by that so its
        # index k aligns with the new chunk's world-time k, zero-padding the exposed tail (the soft
        # mask zeros it anyway). The FIRST query (no prev) falls back to the plain sampler.
        use_rtc = self._rtc_enabled and self._rtc_prev_chunk is not None
        if use_rtc:
            shift = int(rtc_shift) if rtc_shift is not None else 0
            prev = jnp.roll(self._rtc_prev_chunk, -shift, axis=1)
            if shift > 0:
                prev = prev.at[:, -shift:, :].set(0.0)
            actions = self._sample_actions_rtc(sample_rng_or_pytorch_device, observation, prev, **{
                **{k: v for k, v in self._rtc_kwargs.items()},
                **({"noise": sample_kwargs["noise"]} if "noise" in sample_kwargs else {})})
        else:
            actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        # cache the raw (normalized, model-space) chunk for the NEXT query's prefix guidance
        if self._rtc_enabled and not self._is_pytorch_model:
            self._rtc_prev_chunk = actions
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def reset(self) -> None:
        """Clear per-episode RTC state (the cached previous chunk) so a new rollout starts fresh —
        the first query of the next episode then has no prefix and falls back to the plain sampler."""
        self._rtc_prev_chunk = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict, **kwargs) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs, **kwargs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
