"""
MosaicML Streaming (MDS) data loader for VLA training.

This loader supports streaming from S3 or local MDS datasets with:
- Smart local caching for S3 data
- Deterministic shuffling
- Elastic resume (exact sample position)
- Automatic shard handling
- Sequential shard access for optimal prefetching

See: https://docs.mosaicml.com/projects/streaming/en/stable/
"""

import dataclasses
import logging
from enum import Enum, auto
from typing import Any

import numpy as np

from PIL import Image
from io import BytesIO
from streaming.base.format.mds.encodings import Encoding, _encodings


class ActionSpace(Enum):
    """Action space for MDS dataset."""
    JOINT_POSITION = auto()


class JPEG95(Encoding):
    """Store PIL image as JPEG with quality 95."""

    def encode(self, obj: Any) -> bytes:
        if isinstance(obj, Image.Image):
            img = obj
        else:
            img = Image.fromarray(obj)
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=95)
        return buf.getvalue()

    def decode(self, data: bytes) -> Any:
        return Image.open(BytesIO(data))


# Register encoding at module level (for main process)
_encodings["jpeg95"] = JPEG95


def _worker_init_fn(worker_id: int) -> None:
    """Initialize worker process by registering custom encodings."""
    # Re-register JPEG95 encoding in spawned worker process
    _encodings["jpeg95"] = JPEG95



@dataclasses.dataclass
class MDSDatasetConfig:
    """Configuration for MDS dataset structure.

    Defines the column names within the MDS dataset.
    """
    # Image column names
    external_cam_key: str = "external_cam"
    wrist_cam_key: str = "wrist_cam"

    # State/proprioception column names
    arm_joint_pos_key: str = "arm_joint_pos"
    gripper_joint_pos_key: str = "gripper_pos"

    # Action column name
    action_key: str = "action_chunk"

    # Prompt/language instruction
    prompt_key: str | None = "prompt"
    default_prompt: str | None = None

    # Action space configuration
    action_space: ActionSpace = ActionSpace.JOINT_POSITION


class MDSDataset:
    """MDS dataset loader for robot demonstration data.

    Uses MosaicML's StreamingDataLoader for optimal shard-sequential access
    with parallel workers. Each worker gets assigned shards and iterates
    sequentially within those shards, enabling efficient prefetching.

    Features:
    - Sequential shard access per worker (optimal for S3 prefetching)
    - Multi-worker parallel processing
    - S3 streaming with intelligent local caching
    """

    def __init__(
        self,
        remote: str,
        *,
        local: str = "/tmp/mds_cache",
        shuffle: bool = False,  # Data should be pre-shuffled; sequential access enables prefetching
        action_chunk_size: int = 16,
        batch_size: int = 256,
        num_workers: int = 0,  # Single-threaded to avoid fork() conflict with JAX
        config: MDSDatasetConfig | None = None,
        predownload: int | None = 6000,  # ~3 shards ahead (assuming ~2000 samples/shard)
        cache_limit: str | None = None,
        num_canonical_nodes: int | None = None,
    ):
        """Initialize the MDS dataset.

        Args:
            remote: Path to MDS dataset (local dir or s3://bucket/prefix).
            local: Local cache directory for downloaded shards.
            shuffle: Whether to shuffle samples (deterministically).
            action_chunk_size: Expected action chunk size (for truncation).
            batch_size: Batch size for StreamingDataLoader.
            num_workers: Number of DataLoader workers.
            config: Configuration for MDS structure. If None, uses defaults.
            predownload: Number of samples to predownload. None for auto.
            cache_limit: Max cache size (e.g., "10gb"). None for unlimited.
            num_canonical_nodes: Number of canonical nodes for distributed training.
        """
        try:
            from streaming import StreamingDataset, StreamingDataLoader
        except ImportError:
            raise ImportError(
                "MosaicML Streaming is required for MDS datasets. "
                "Install with: pip install mosaicml-streaming"
            )

        self.remote = remote
        self.local = local
        self.shuffle = shuffle
        self.action_chunk_size = action_chunk_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config = config or MDSDatasetConfig()

        logging.info(f"Loading MDS dataset from: {remote}")
        logging.info(f"Local cache: {local}")
        logging.info(f"Predownload: {predownload}, shuffle: {shuffle}, batch_size: {batch_size}, num_workers: {num_workers}")

        # Create streaming dataset
        streaming_kwargs = {
            "remote": remote,
            "local": local,
            "shuffle": shuffle,
            "batch_size": batch_size,
        }

        if predownload is not None:
            streaming_kwargs["predownload"] = predownload
        if cache_limit is not None:
            streaming_kwargs["cache_limit"] = cache_limit
        if num_canonical_nodes is not None:
            streaming_kwargs["num_canonical_nodes"] = num_canonical_nodes

        self._streaming_dataset = StreamingDataset(**streaming_kwargs)
        self._total_samples = len(self._streaming_dataset)

        # Custom collate function - converts PIL images to numpy and stacks
        def collate_fn(batch):
            """Collate batch into stacked numpy arrays."""
            result = {}
            for key in batch[0].keys():
                values = [sample[key] for sample in batch]
                # Convert PIL images to numpy
                if isinstance(values[0], Image.Image):
                    values = [np.array(v) for v in values]
                # Stack numpy arrays
                if isinstance(values[0], np.ndarray):
                    result[key] = np.stack(values, axis=0)
                else:
                    result[key] = values
            return result

        # Create StreamingDataLoader
        # Using num_workers=0 to avoid fork() which conflicts with JAX's multithreading
        self._data_loader = StreamingDataLoader(
            self._streaming_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )

        logging.info(f"MDS dataset loaded with {self._total_samples} samples")

    def _process_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Process a single sample from the streaming dataset."""
        config = self.config

        # Images are already numpy arrays from cv2 decode
        external_image = sample[config.external_cam_key]
        wrist_image = sample.get(config.wrist_cam_key)

        # Load actions and truncate to action_chunk_size
        actions = sample[config.action_key]
        if isinstance(actions, np.ndarray):
            actions = actions.astype(np.float32)
        else:
            actions = np.array(actions, dtype=np.float32)

        # Truncate to model's action horizon if needed
        if actions.shape[0] > self.action_chunk_size:
            actions = actions[:self.action_chunk_size]

        # Load joint positions
        arm_jp = sample[config.arm_joint_pos_key]
        gripper_jp = sample[config.gripper_joint_pos_key]

        if isinstance(arm_jp, np.ndarray):
            arm_jp = arm_jp.astype(np.float32)
        else:
            arm_jp = np.array(arm_jp, dtype=np.float32)

        if isinstance(gripper_jp, np.ndarray):
            gripper_jp = gripper_jp.astype(np.float32)
        else:
            gripper_jp = np.array(gripper_jp, dtype=np.float32)

        # Get prompt
        prompt = config.default_prompt or ""
        if config.prompt_key and config.prompt_key in sample:
            prompt_val = sample[config.prompt_key]
            if isinstance(prompt_val, bytes):
                prompt = prompt_val.decode("utf-8")
            else:
                prompt = str(prompt_val)

        return {
            "actions": actions,
            "observation": {
                "image": external_image,
                "wrist_image": wrist_image,
                "arm_jp": arm_jp,
                "gripper_jp": gripper_jp,
            },
            "prompt": prompt,
        }

    def _process_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Process a batch of samples."""
        batch_size = len(batch[self.config.external_cam_key])
        processed = []
        for i in range(batch_size):
            sample = {k: v[i] for k, v in batch.items()}
            processed.append(self._process_sample(sample))
        return self._collate_batch(processed)

    def _collate_batch(self, samples: list[dict]) -> dict:
        """Collate processed samples into a batch."""
        batch = {
            "actions": np.stack([s["actions"] for s in samples], axis=0),
            "observation": {
                "image": np.stack([s["observation"]["image"] for s in samples], axis=0),
            },
            "prompt": np.array([s["prompt"] for s in samples]),
        }

        if samples[0]["observation"]["wrist_image"] is not None:
            batch["observation"]["wrist_image"] = np.stack(
                [s["observation"]["wrist_image"] for s in samples], axis=0
            )

        batch["observation"]["arm_jp"] = np.stack(
            [s["observation"]["arm_jp"] for s in samples], axis=0
        )
        batch["observation"]["gripper_jp"] = np.stack(
            [s["observation"]["gripper_jp"] for s in samples], axis=0
        )

        return batch

    def __iter__(self):
        """Iterate over batches using StreamingDataLoader."""
        for batch in self._data_loader:
            yield self._process_batch(batch)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self._total_samples
