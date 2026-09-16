import dataclasses
import enum
import logging
import os
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def _rtc_from_env() -> dict | None:
    """Read Real-Time Chunking config from env so the launcher can enable it without a CLI-schema
    change (matches how the eval service passes other server opts). Returns None (→ plain sampler,
    unchanged behavior) unless OPENPI_RTC=1. Blog defaults: sigma=0.2, max_guidance_weight=num_steps."""
    if os.environ.get("OPENPI_RTC", "0") != "1":
        return None
    def _f(k, d):
        v = os.environ.get(k)
        return float(v) if v not in (None, "") else d
    def _i(k, d):
        v = os.environ.get(k)
        return int(v) if v not in (None, "") else d
    num_steps = _i("OPENPI_RTC_NUM_STEPS", 10)
    return {
        "enabled": True,
        "num_steps": num_steps,
        "inference_delay": _i("OPENPI_RTC_INFERENCE_DELAY", 1),
        "prefix_attention_horizon": _i("OPENPI_RTC_PREFIX_HORIZON", 4),
        "prefix_attention_schedule": os.environ.get("OPENPI_RTC_SCHEDULE", "exp"),
        "sigma": _f("OPENPI_RTC_SIGMA", 0.2),
        # β=n rule of thumb (blog): default the clip to num_steps unless explicitly set.
        "max_guidance_weight": _f("OPENPI_RTC_MAX_GUIDANCE", float(num_steps)),
    }


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    rtc = _rtc_from_env()
    if rtc:
        logging.info("RTC enabled: %s", rtc)
    match args.policy:
        case Checkpoint():
            try:
                train_config = _config.get_config(args.policy.config)
            except ValueError as e:
                # not registered on this host: rebuild the config from the checkpoint's own sidecar
                import openpi.training.deploy_metadata as _deploy_metadata
                train_config = _deploy_metadata.config_from_sidecar(args.policy.dir)
                logging.warning("config %r not registered here (%s); serving from %s/deploy_metadata.json",
                                args.policy.config, e, args.policy.dir)
            return _policy_config.create_trained_policy(
                train_config, args.policy.dir, default_prompt=args.default_prompt, rtc=rtc
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
