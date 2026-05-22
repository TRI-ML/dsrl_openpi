"""Sim-improvement configs and HDF5/MDS/WDS data loading support."""

import dataclasses
import pathlib
from typing import Protocol, TypeAlias

from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
# import openpi.policies.ur_policy as ur_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.lbm_policy as lbm_policy
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

try:
    from openpi.training.hdf5_dataset import ActionSpace
except ImportError:
    from enum import Enum, auto
    class ActionSpace(Enum):
        JOINT_POSITION = auto()
        EE_POSE = auto()

try:
    from openpi.training.mds_dataset import ActionSpace as MDSActionSpace
except ImportError:
    from enum import Enum, auto
    class MDSActionSpace(Enum):
        JOINT_POSITION = auto()
        EE_POSE = auto()

try:
    from openpi.training.wds_dataset import WDSDataset, WDSDatasetConfig, ConfiguredWDSDataset
except ImportError:
    WDSDataset = None
    WDSDatasetConfig = None
    ConfiguredWDSDataset = None

ModelType: TypeAlias = _model.ModelType


@dataclasses.dataclass(frozen=True)
class MDSDataConfig:
    """
    Config for training on MDS (MosaicML Streaming) datasets.

    MDS provides:
    - Smart S3 streaming with local caching
    - Deterministic shuffling (reproducible across runs)
    - Elastic resume (restart from exact sample position)

    Memory modes:
    - S3 streaming: Automatic caching, minimal local storage
    - Local: Fast access, full dataset on disk
    """
    # The LeRobot repo id (used for asset lookup).
    repo_id: str = tyro.MISSING

    # Path to MDS dataset (local dir or s3://bucket/prefix)
    mds_path: str | None = None

    # Local cache directory for S3 streaming
    local_cache: str = "/tmp/mds_cache"

    # MDS column names
    external_cam_key: str = "external_cam"
    wrist_cam_key: str | None = "wrist_cam"
    state_key: str | None = "state"
    arm_joint_pos_key: str | None = None
    gripper_joint_pos_key: str | None = None
    action_key: str = "action_chunk"

    # Default prompt if not stored in MDS
    default_prompt: str | None = None
    prompt_key: str | None = "prompt"

    # Action space configuration
    action_space: MDSActionSpace = MDSActionSpace.JOINT_POSITION

    # Cache settings
    cache_limit: str | None = None  # e.g., "10gb"
    predownload: int = 6000  # ~3 shards ahead (assuming ~2000 samples/shard)

    # DataLoader settings
    num_workers: int = 0  # Single-threaded to avoid fork() conflict with JAX

    # Repack transform customization
    repack_transforms: tyro.conf.Suppress[_transforms.Group | None] = None

    # Assets configuration
    assets_dir: str | None = None
    asset_id: str | None = None

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig):
        """Create a DataConfig from this MDSDataConfig."""
        from openpi.training.config import DataConfig, ModelTransformFactory
        from openpi.training.mds_dataset import MDSDataset, MDSDatasetConfig
        import openpi.shared.download as _download
        import openpi.shared.normalize as _normalize
        import etils.epath as epath
        import logging

        # Load norm stats
        norm_stats = None
        asset_id = self.asset_id or self.repo_id
        if asset_id:
            data_assets_dir = str(epath.Path(self.assets_dir or assets_dirs) / asset_id)
            try:
                norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
                logging.info(f"Loaded norm stats from {data_assets_dir}")
            except FileNotFoundError:
                logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")

        # Default repack transform
        repack_transform = self.repack_transforms
        if repack_transform is None:
            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/exterior_image_1_left": "observation/image",
                            "observation/wrist_image_left": "observation/wrist_image",
                            "observation/joint_position": "observation/arm_jp",
                            "observation/gripper_position": "observation/gripper_jp",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )

        # Data transforms
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == MDSActionSpace.JOINT_POSITION:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        assert self.mds_path is not None, "Need to set mds_path for MDS data loader."

        # Create MDS config
        mds_config = MDSDatasetConfig(
            external_cam_key=self.external_cam_key,
            wrist_cam_key=self.wrist_cam_key,
            arm_joint_pos_key=self.arm_joint_pos_key,
            gripper_joint_pos_key=self.gripper_joint_pos_key,
            action_key=self.action_key,
            default_prompt=self.default_prompt,
            prompt_key=self.prompt_key,
        )

        # Capture config values for closure
        mds_path = self.mds_path
        local_cache = self.local_cache
        cache_limit = self.cache_limit
        predownload = self.predownload
        num_workers = self.num_workers

        # Create a dataset class that uses StreamingDataLoader internally
        # for optimal shard-sequential access with parallel workers
        class ConfiguredMDSDataset(MDSDataset):
            def __init__(
                self,
                data_dir: str,
                batch_size: int,
                *,
                shuffle: bool = False,
                action_chunk_size: int = 16,
                **kwargs
            ):
                super().__init__(
                    remote=mds_path,
                    local=local_cache,
                    shuffle=False,  # Data pre-shuffled, sequential for prefetching
                    action_chunk_size=action_chunk_size,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    config=mds_config,
                    cache_limit=cache_limit,
                    predownload=predownload,
                )

        return DataConfig(
            repo_id=self.repo_id if self.repo_id is not tyro.MISSING else None,
            asset_id=asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type != ModelType.PI0,
            # Use RLDS path since MDSDataset is now iterable with internal StreamingDataLoader
            rlds_data_dir=self.mds_path,
            dataset_class=ConfiguredMDSDataset,
        )


@dataclasses.dataclass(frozen=True)
class WDSDataConfig:
    """Config for training on WebDataset (WDS) tar shards.

    WDS provides:
    - Streaming from S3 via ``pipe:aws s3 cp``
    - Fast OpenCV JPEG decoding (only for configured camera keys)
    - Sequence cropping around anchor timestep
    - Multi-worker parallel loading via WebLoader

    """

    # The LeRobot repo id (used for asset lookup).
    repo_id: str = tyro.MISSING

    # Path to WDS manifest JSONL (local or s3://)
    manifest_path: str | None = None

    # --- WDSDatasetConfig fields ---
    action_fields: list[str] = dataclasses.field(default_factory=list)
    proprioception_fields: list[str] = dataclasses.field(default_factory=list)
    camera_names: list[str] = dataclasses.field(default_factory=list)
    image_indices: list[int] = dataclasses.field(default_factory=list)
    language_instruction_types: list[str] = dataclasses.field(default_factory=lambda: ["original"])
    lowdim_past_timesteps: int = 1
    lowdim_future_timesteps: int = 14

    # Default prompt if not stored in the dataset
    default_prompt: str | None = None

    # DataLoader settings
    num_workers: int = 2
    shuffle_buffer: int = 2000

    # Assets / normalization
    assets_dir: str | None = None
    asset_id: str | None = None

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: tyro.conf.Suppress[_transforms.Group | None] = None

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig):
        """Create a DataConfig from this WDSDataConfig."""
        from openpi.training.config import DataConfig, ModelTransformFactory
        import openpi.shared.download as _download
        import openpi.shared.normalize as _normalize
        import etils.epath as epath
        import logging

        # Load norm stats
        norm_stats = None
        asset_id = self.asset_id or self.repo_id
        if asset_id:
            data_assets_dir = str(epath.Path(self.assets_dir or assets_dirs) / asset_id)
            try:
                norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
                logging.info(f"Loaded norm stats from {data_assets_dir}")
            except FileNotFoundError:
                logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")

        # Build repack transforms
        repack_transform = self.repack_transforms
        # assert repack_transform is not None, "Need to set repack_transform for WDS data loader."

        # Data transforms (droid policy)
        data_transforms = self.data_transforms
        if data_transforms is None:
            data_transforms = _transforms.Group(
                inputs=[lbm_policy.LBMInputs(model_type=model_config.model_type)],
                outputs=[lbm_policy.LBMOutputs()],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        # Build WDSDatasetConfig (only needed for training, not inference)
        dataset_cls = None
        if WDSDatasetConfig is not None and self.manifest_path is not None:
            wds_config = WDSDatasetConfig(
                action_fields=self.action_fields,
                proprioception_fields=self.proprioception_fields,
                camera_names=self.camera_names,
                image_indices=self.image_indices,
                language_instruction_types=self.language_instruction_types,
                lowdim_past_timesteps=self.lowdim_past_timesteps,
                lowdim_future_timesteps=self.lowdim_future_timesteps,
            )
            dataset_cls = ConfiguredWDSDataset.configure(
                manifest_path=self.manifest_path,
                num_workers=self.num_workers,
                shuffle_buffer=self.shuffle_buffer,
                config=wds_config,
            )

        return DataConfig(
            repo_id=self.repo_id if self.repo_id is not tyro.MISSING else None,
            asset_id=asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )


def get_sim_improvement_configs():
    """Return sim-improvement training configs."""
    # Import here to avoid circular imports.
    from openpi.training.config import TrainConfig, FakeDataConfig, AssetsConfig
    # from openpi.training.misc.sim_improvement_config import HDF5DataConfig
    from openpi.training.config import LeRobotLiberoDataConfig, DataConfig
    
    return [

        # Example MDS (MosaicML Streaming) dataset config
        TrainConfig(
            name="pi0_droid_jointpos_cubeonplate",
            model=pi0_config.Pi0Config(action_horizon=10),
            data=MDSDataConfig(
                repo_id="cube_on_plate",
                # MDS path can be local or S3
                mds_path="s3://tri-ml-datasets-uw2/arhanjain/rollout_datasets/cube_to_plate/mds/",
                local_cache="/tmp/mds_cache",

                # Assets for normalization stats
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid_jointpos/assets",
                asset_id="droid",

                default_prompt="Put the green cube on the plate",

                # MDS column names (matching the schema from convert_npz_to_mds.py)
                external_cam_key="external_cam",
                wrist_cam_key="wrist_cam",
                # state_key="state",
                arm_joint_pos_key="obs.vision.arm_joint_pos",
                gripper_joint_pos_key="obs.vision.gripper_pos",
                action_key="action_chunk",
                prompt_key="prompt",

                action_space=MDSActionSpace.JOINT_POSITION,

                # Cache settings for S3 streaming
                cache_limit="100gb",
                num_workers=4,  # Single-threaded to avoid fork() with JAX
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_droid_jointpos/params",
            ),
            batch_size=256,
            num_train_steps=25_000,
            log_interval=100,
            save_interval=2500,
            keep_period=500,
            save_train_state=False,
            overwrite=True,
            exp_name="pi0_cubeonplate_mds",
            wandb_enabled=True,
            remote_checkpoint_dir="s3://tri-ml-datasets-uw2/arhanjain/openpi",
        ),

        # Test MDS config with pi0_fast (small batch for quick testing)
        TrainConfig(
            name="pi05_dummy_mds_test",
            model=pi0_config.Pi0Config(
                action_horizon=15,
                pi05=True,
                paligemma_variant="dummy",
                action_expert_variant="dummy",
            ),
            data=MDSDataConfig(
                repo_id="cube_to_plate_test",
                mds_path="s3://tri-ml-datasets-uw2/arhanjain/rollout_datasets/cube_to_plate/mds/",
                local_cache="/tmp/mds_cache",

                assets_dir="gs://openpi-assets/checkpoints/pi05_droid_jointpos/assets",
                asset_id="droid",

                default_prompt="Put the cube on the plate",

                # MDS column names
                external_cam_key="external_cam",
                wrist_cam_key="wrist_cam",
                arm_joint_pos_key="obs.vision.arm_joint_pos",
                gripper_joint_pos_key="obs.vision.gripper_pos",
                action_key="action_chunk",

                prompt_key="prompt",

                action_space=MDSActionSpace.JOINT_POSITION,
                cache_limit="100gb",
                num_workers=0,  # Single-threaded to avoid fork() with JAX
            ),
            # weight_loader=weight_loaders.CheckpointWeightLoader(
            #     "gs://openpi-assets/checkpoints/pi0_fast_droid_jointpos/params",
            # ),
            batch_size=64,
            num_train_steps=500,
            log_interval=10,
            save_interval=1000,
            keep_period=1000,
            save_train_state=False,
            overwrite=True,
            exp_name="mds_test",
            wandb_enabled=False,
        ),
        
        TrainConfig(
            name="pi05_lbm_sim",
            model=pi0_config.Pi0Config(
                action_horizon=15,
                pi05=True,
                # paligemma_variant="dummy",
                # action_expert_variant="dummy",
            ),
            data=WDSDataConfig(
                repo_id="lbm_sim",
                manifest_path="s3://tri-ml-datasets-uw2/vla_foundry_datasets/v0.4.2-sim/manifest.jsonl",

                # Assets for normalization stats
                assets_dir="s3://tri-ml-datasets-uw2/arhanjain/openpi/assets/pi05_lbm_sim",
                asset_id="lbm_sim",

                # default_prompt="Pick up the object",

                # WDS shard layout
                camera_names=["wrist_right_minus", "wrist_left_plus", "scene_left_0", "scene_right_0"],
                image_indices=[0],

                # Lowdim fields (must match keys inside lowdim.npz)
                action_fields=[
                    "robot__action__poses__left::panda__xyz_relative",
                    "robot__action__poses__right::panda__xyz_relative",
                    "robot__action__poses__left::panda__rot_6d_relative",
                    "robot__action__poses__right::panda__rot_6d_relative",
                    "robot__action__grippers__left::panda_hand",
                    "robot__action__grippers__right::panda_hand",
                ],
                proprioception_fields=[
                    "robot__actual__poses__left::panda__xyz",
                    "robot__actual__poses__right::panda__xyz",
                    "robot__actual__poses__left::panda__rot_6d",
                    "robot__actual__poses__right::panda__rot_6d",
                ],
                lowdim_past_timesteps=0,
                lowdim_future_timesteps=14,

                num_workers=16,
                shuffle_buffer=2000,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_droid_jointpos/params",
            ),
            batch_size=256,
            num_train_steps=100_000,
            log_interval=100,
            save_interval=10_000,
            keep_period=10_000,
            save_train_state=False,
            overwrite=True,
            exp_name="pi05_lbm_sim_wds",
            wandb_enabled=True,
            remote_checkpoint_dir="s3://tri-ml-datasets-uw2/arhanjain/openpi",
        ),

        TrainConfig(
            name="pi05_lbm_placecuponcoaster",
            model=pi0_config.Pi0Config(
                action_horizon=15,
                pi05=True,
            ),
            data=WDSDataConfig(
                repo_id="lbm_placecuponcoaster",
                manifest_path="s3://tri-ml-datasets-uw2/vla_foundry_datasets_test/v0.4.3.9/sim/PlaceCupOnCoaster/shards/manifest.jsonl",

                # Assets for normalization stats — will be computed and cached on first run
                assets_dir="s3://tri-ml-datasets-uw2/arhanjain/openpi/assets/pi05_lbm_placecuponcoaster",
                asset_id="lbm_placecuponcoaster",

                # WDS shard layout
                camera_names=["wrist_right_minus", "wrist_left_plus", "scene_left_0", "scene_right_0"],
                image_indices=[0],

                action_fields=[
                    "robot__action__poses__left::panda__xyz_relative",
                    "robot__action__poses__right::panda__xyz_relative",
                    "robot__action__poses__left::panda__rot_6d_relative",
                    "robot__action__poses__right::panda__rot_6d_relative",
                    "robot__action__grippers__left::panda_hand",
                    "robot__action__grippers__right::panda_hand",
                ],
                proprioception_fields=[
                    "robot__actual__poses__left::panda__xyz",
                    "robot__actual__poses__right::panda__xyz",
                    "robot__actual__poses__left::panda__rot_6d",
                    "robot__actual__poses__right::panda__rot_6d",
                ],
                lowdim_past_timesteps=0,
                lowdim_future_timesteps=14,

                num_workers=16,
                shuffle_buffer=2000,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_droid_jointpos/params",
            ),
            batch_size=256,
            num_train_steps=100_000,
            log_interval=100,
            save_interval=10_000,
            keep_period=10_000,
            save_train_state=False,
            overwrite=True,
            exp_name="pi05_lbm_placecuponcoaster_wds",
            wandb_enabled=True,
            remote_checkpoint_dir="s3://tri-ml-datasets-uw2/arhanjain/openpi",
        ),

        TrainConfig(
            name="pi05_lbm_putmugonsaucer",
            model=pi0_config.Pi0Config(
                action_horizon=15,
                pi05=True,
            ),
            data=WDSDataConfig(
                repo_id="lbm_putmugonsaucer",
                manifest_path="s3://tri-ml-datasets-uw2/vla_foundry_datasets_test/v0.4.3.9/sim/PutMugOnSaucer/shards/manifest.jsonl",

                # Assets for normalization stats — will be computed and cached on first run
                assets_dir="s3://tri-ml-datasets-uw2/arhanjain/openpi/assets/pi05_lbm_putmugonsaucer",
                asset_id="lbm_putmugonsaucer",

                # WDS shard layout
                camera_names=["wrist_right_minus", "wrist_left_plus", "scene_left_0", "scene_right_0"],
                image_indices=[0],

                action_fields=[
                    "robot__action__poses__left::panda__xyz_relative",
                    "robot__action__poses__right::panda__xyz_relative",
                    "robot__action__poses__left::panda__rot_6d_relative",
                    "robot__action__poses__right::panda__rot_6d_relative",
                    "robot__action__grippers__left::panda_hand",
                    "robot__action__grippers__right::panda_hand",
                ],
                proprioception_fields=[
                    "robot__actual__poses__left::panda__xyz",
                    "robot__actual__poses__right::panda__xyz",
                    "robot__actual__poses__left::panda__rot_6d",
                    "robot__actual__poses__right::panda__rot_6d",
                ],
                lowdim_past_timesteps=0,
                lowdim_future_timesteps=14,

                num_workers=16,
                shuffle_buffer=2000,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_droid_jointpos/params",
            ),
            batch_size=256,
            num_train_steps=100_000,
            log_interval=100,
            save_interval=10_000,
            keep_period=10_000,
            save_train_state=False,
            overwrite=True,
            exp_name="pi05_lbm_putmugonsaucer_wds",
            wandb_enabled=True,
            remote_checkpoint_dir="s3://tri-ml-datasets-uw2/arhanjain/openpi",
        ),

        TrainConfig(
            name="pi05_lbm_hangmug",
            model=pi0_config.Pi0Config(
                action_horizon=15,
                pi05=True,
            ),
            data=WDSDataConfig(
                repo_id="lbm_hangmug",
                manifest_path="s3://tri-ml-datasets-uw2/arhanjain/runs/HangMugFromDryingRack_processed/wds/shards/manifest.jsonl",

                # Assets for normalization stats
                assets_dir="s3://tri-ml-datasets-uw2/arhanjain/openpi/assets/pi05_lbm_hangmug",
                asset_id="lbm_hangmug",

                # WDS shard layout — camera names from actual data
                camera_names=["wrist_camera_right", "wrist_camera_left", "external_camera_right", "external_camera_left"],
                image_indices=[0],

                # Lowdim fields (match existing LBM format)
                action_fields=[
                    "robot__action__poses__left::panda__xyz_relative",
                    "robot__action__poses__right::panda__xyz_relative",
                    "robot__action__poses__left::panda__rot_6d_relative",
                    "robot__action__poses__right::panda__rot_6d_relative",
                    "robot__action__grippers__left::panda_hand",
                    "robot__action__grippers__right::panda_hand",
                ],
                proprioception_fields=[
                    "robot__actual__poses__left::panda__xyz",
                    "robot__actual__poses__right::panda__xyz",
                    "robot__actual__poses__left::panda__rot_6d",
                    "robot__actual__poses__right::panda__rot_6d",
                ],
                lowdim_past_timesteps=0,
                lowdim_future_timesteps=14,

                num_workers=16,
                shuffle_buffer=2000,

                # Custom data transforms with correct camera names for this dataset
                data_transforms=_transforms.Group(
                    inputs=[lbm_policy.LBMInputs(
                        model_type=ModelType.PI05,
                        camera_names=("wrist_camera_right", "wrist_camera_left", "external_camera_right", "external_camera_left"),
                    )],
                    outputs=[lbm_policy.LBMOutputs()],
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_droid_jointpos/params",
            ),
            batch_size=256,
            num_train_steps=100_000,
            log_interval=100,
            save_interval=10_000,
            keep_period=10_000,
            save_train_state=False,
            overwrite=True,
            exp_name="pi05_lbm_hangmug_wds",
            wandb_enabled=True,
            remote_checkpoint_dir="s3://tri-ml-datasets-uw2/arhanjain/openpi",
        ),

        TrainConfig(
            name="pi05_ur",
            model=pi0_config.Pi0Config(
                action_horizon=15,
                pi05=True,
            ),
            data=MDSDataConfig(
                repo_id="ur_cubeonplate_10k",
                mds_path="s3://tri-ml-datasets-uw2/arhanjain/rollout_datasets/ur_cubestack_10k/",
                local_cache="/tmp/mds_cache",

                assets_dir="gs://openpi-assets/checkpoints/pi05_droid_jointpos/assets",
                asset_id="droid",

                default_prompt="Put the cube on the plate",

                # MDS column names
                external_cam_key="external_cam",
                wrist_cam_key="wrist_cam",
                arm_joint_pos_key="obs.vision.arm_joint_pos",
                gripper_joint_pos_key="obs.vision.gripper_pos",
                action_key="action_chunk",

                prompt_key="prompt",

                action_space=MDSActionSpace.JOINT_POSITION,
                cache_limit="100gb",
                num_workers=0,  # Single-threaded to avoid fork() with JAX
            ),
            # weight_loader=weight_loaders.CheckpointWeightLoader(
            #     "gs://openpi-assets/checkpoints/pi0_fast_droid_jointpos/params",
            # ),
            batch_size=64,
            num_train_steps=500,
            log_interval=10,
            save_interval=1000,
            keep_period=1000,
            save_train_state=False,
            overwrite=True,
            exp_name="mds_test",
            wandb_enabled=False,
        ),
    ]

