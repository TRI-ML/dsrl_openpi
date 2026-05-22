"""
WebDataset (WDS) data loader for VLA training.

Self-contained loader with zero vla_foundry dependencies. Requires only:
  webdataset, numpy, torch, opencv-python, Pillow

Handles:
- JSONL manifest parsing and shard selection
- Fast JPEG decoding via OpenCV
- Robotics field extraction (images, lowdim, language, metadata)
- Sequence cropping around anchor timestep
- Batching and collation into tensors

Usage:
    from wds_dataset import WDSDataset, WDSDatasetConfig

    config = WDSDatasetConfig(
        action_fields=["robot__action__poses__left::panda__xyz_relative", ...],
        proprioception_fields=["robot__actual__poses__left::panda__xyz", ...],
        camera_names=["wrist_camera", "scene_left_0"],
        image_indices=[-1, 0],
    )
    dataset = WDSDataset(
        manifest_path="/path/to/manifest.jsonl",
        config=config,
        batch_size=4,
    )
    for batch in dataset:
        print(batch["actions"].shape)          # [B, T, action_dim]
        print(batch["observation"]["images"])   # dict of camera_name -> [B, H, W, 3]
        print(batch["prompt"])                  # list[str]
"""

import dataclasses
import io
import json
import logging
import os
import random
from dataclasses import field
from typing import Any

import cv2
import numpy as np
import torch
import webdataset as wds
from PIL import Image
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# Disable OpenCV internal threading to avoid contention with DataLoader workers.
cv2.setNumThreads(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ImageDecoder:
    """Picklable WebDataset image decoder that only decodes specified keys.

    Args:
        image_names: If provided, only decode images whose stem (e.g. "wrist_camera_t0")
            is in this list. Other images are returned as raw bytes (skipping the
            expensive decode). If *None*, all images are decoded.
    """

    def __init__(self, image_names: list[str] | None = None):
        self.allowed: set[str] | None = set(image_names) if image_names else None

    def __call__(self, key: str, data: bytes):
        if not key.endswith((".jpg", ".jpeg", ".png", ".webp")):
            return None
        if self.allowed is not None:
            stem = key.rsplit(".", 1)[0].lstrip(".")
            if stem not in self.allowed:
                return data
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _log_and_continue(exn: BaseException) -> bool:
    """Ignore WebDataset errors and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def _crop_sequence(data, anchor_idx: int, past: int, future: int):
    """Crop array [T, ...] to [past + 1 + future, ...] centred on anchor."""
    start = anchor_idx - past
    end = anchor_idx + future + 1
    return data[start:end]


def _load_manifest(path: str) -> list[dict]:
    """Load a JSONL manifest. Each line: {"shard": "...", "num_sequences": N}.

    Supports both local paths and s3:// URIs (via fsspec).
    """
    import fsspec

    with fsspec.open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_datastring(manifest_path: str, shard_names: list[str]) -> str:
    """Build a WebDataset-compatible datastring from shard names."""
    root = os.path.dirname(manifest_path)
    if root:
        root += "/"
    # Strip .tar suffix if already present to avoid double extension
    clean_names = [n[:-4] if n.endswith(".tar") else n for n in shard_names]
    if len(clean_names) > 1:
        ds = root + "{" + ",".join(clean_names) + "}.tar"
    else:
        ds = root + clean_names[0] + ".tar"
    if manifest_path.startswith("s3"):
        ds = f"pipe:aws s3 cp {ds} -"
    return ds


def _select_shards(
    manifest: list[dict],
    num_workers: int,
    seed: int,
    shuffle: bool,
) -> tuple[list[str], int]:
    """Select shards from a manifest ensuring divisibility by num_workers.

    Returns (shard_names, total_samples).
    """
    entries = list(manifest)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(entries)

    # Ensure shard count is divisible by num_workers for balanced loading.
    total_workers = max(1, num_workers)
    usable = (len(entries) // total_workers) * total_workers
    entries = entries[:usable] if usable > 0 else entries[:total_workers]

    shard_names = [e["shard"] for e in entries]
    total_samples = sum(e["num_sequences"] for e in entries)
    return shard_names, total_samples


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class WDSDatasetConfig:
    """Configuration for WDS dataset structure.

    Defines field names, camera layout, and sequence parameters.
    """

    # Lowdim field names (must match keys inside lowdim.npz in the tar shards)
    action_fields: list[str] = field(default_factory=list)
    proprioception_fields: list[str] = field(default_factory=list)

    # Camera / image layout
    camera_names: list[str] = field(default_factory=list)
    image_indices: list[int] = field(default_factory=list)

    # Language instruction types to sample from: "original", "randomized", "verbose", "alternative"
    language_instruction_types: list[str] = field(default_factory=lambda: ["original"])

    # Sequence window around anchor timestep
    lowdim_past_timesteps: int = 1
    lowdim_future_timesteps: int = 14


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class WDSDataset:
    """WebDataset loader for robot demonstration data.

    Self-contained -- no vla_foundry imports required.

    Reads tar shards produced by vla_foundry preprocessing, each containing:
      - ``*.{camera}_{timestep}.jpg``  camera images
      - ``*.lowdim.npz``               low-dimensional sensor data
      - ``*.metadata.json``            anchor timestep info
      - ``*.language_instructions.json`` language annotations

    Features:
    - Sequential shard access per worker (optimal for S3 prefetching)
    - Multi-worker parallel processing
    - S3 streaming via ``pipe:aws s3 cp``
    - Fast OpenCV JPEG decoding with PIL fallback
    - Sequence cropping and batching
    """

    def __init__(
        self,
        manifest_path: str,
        *,
        batch_size: int = 4,
        num_workers: int = 2,
        shuffle: bool = True,
        shuffle_buffer: int = 2000,
        seed: int = 42,
        config: WDSDatasetConfig | None = None,
        manifest_entries: list[dict] | None = None,
    ):
        """Initialize the WDS dataset.

        Args:
            manifest_path: Path to dataset manifest JSONL file. Each line is
                ``{"shard": "shard_00000", "num_sequences": N}``.
                The tar shards must be in the same directory.
            batch_size: Batch size.
            num_workers: Number of DataLoader workers.
            shuffle: Whether to shuffle shards and samples.
            shuffle_buffer: Buffer size for within-shard sample shuffling.
            seed: Random seed for shuffling.
            config: Configuration for dataset fields. Uses defaults if None.
            manifest_entries: Pre-loaded manifest entries. If provided, skips
                loading from manifest_path (but still uses manifest_path to
                derive the shard root directory).
        """
        self.config = config or WDSDatasetConfig()
        self.batch_size = batch_size

        logging.info(f"Loading WDS dataset from: {manifest_path}")

        # Load manifest and select shards
        manifest = manifest_entries if manifest_entries is not None else _load_manifest(manifest_path)
        shard_names, self._total_samples = _select_shards(manifest, num_workers, seed, shuffle)
        datastring = _build_datastring(manifest_path, shard_names)
        logging.info(f"Selected {len(shard_names)} shards, {self._total_samples} samples")

        # Derive image names from camera_names x image_indices
        self._image_names: list[str] = []
        if self.config.camera_names and self.config.image_indices:
            self._image_names = [
                f"{cam}_t{idx}" for idx in self.config.image_indices for cam in self.config.camera_names
            ]
        print(f"[WDS] _image_names={self._image_names}")
        print(f"[WDS] action_fields={self.config.action_fields}")
        print(f"[WDS] proprioception_fields={self.config.proprioception_fields}")

        # Build pipeline
        pipeline = self._build_pipeline(datastring, shuffle, shuffle_buffer, seed)

        # Wrap in WebLoader
        prefetch = 8 if num_workers > 0 else None
        self._dataloader = wds.WebLoader(
            pipeline,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch,
        )

        logging.info(f"WDS dataset ready: {self._total_samples} samples, batch_size={batch_size}")

    # ---- pipeline construction ------------------------------------------------

    def _build_pipeline(self, datastring: str, shuffle: bool, shuffle_buffer: int, seed: int) -> wds.DataPipeline:
        stages: list[Any] = [
            wds.SimpleShardList(datastring),
        ]
        if shuffle:
            stages.append(
                wds.shuffle(shuffle_buffer, initial=min(500, shuffle_buffer), rng=random.Random(seed)),
            )
        stages.extend(
            [
                wds.split_by_node,
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=_log_and_continue),
                wds.decode(_ImageDecoder(self._image_names or None), handler=_log_and_continue),
                wds.select(self._filter_sample),
                wds.map(self._extract_fields, handler=_log_and_continue),
                self._batch_and_collate,
            ]
        )
        return wds.DataPipeline(*stages)

    # ---- batching -------------------------------------------------------------

    def _batch_and_collate(self, src):
        """Collect samples into batches and collate, skipping batches with shape mismatches."""
        batch: list[dict] = []
        yielded = 0
        skipped = 0
        for sample in src:
            batch.append(sample)
            if len(batch) >= self.batch_size:
                try:
                    yield self._collate_batch(batch)
                    yielded += 1
                except (ValueError, RuntimeError) as e:
                    skipped += 1
                    logging.warning(
                        f"Skipping batch {yielded + skipped} ({skipped} skipped so far): {e}"
                    )
                batch = []
        if skipped > 0:
            logging.warning(f"WDS batching: {skipped}/{yielded + skipped} batches skipped due to shape mismatches")

    # ---- per-sample processing ------------------------------------------------

    @staticmethod
    def _filter_sample(sample: dict) -> bool:
        """Require images + lowdim + metadata."""
        has_lowdim = any(k.endswith("lowdim.npz") for k in sample)
        has_metadata = any(k.endswith("metadata.json") for k in sample)
        has_images = any(k.endswith(".jpg") for k in sample)
        return has_lowdim and has_metadata and has_images

    def _extract_fields(self, sample: dict) -> dict:
        """Extract images, lowdim, metadata, and language from a decoded tar sample."""
        cfg = self.config

        # Only keep configured images (if specified) to handle heterogeneous multi-task data.
        wanted_images: set[str] | None = set(self._image_names) if self._image_names else None

        images: dict[str, np.ndarray] = {}
        data: dict[str, Any] = {}
        for key, value in sample.items():
            if key.endswith(".jpg"):
                # Key format: {sample_id}.{camera}_{timestep}.jpg
                # Skip images that were not decoded (still raw bytes).
                if isinstance(value, bytes):
                    continue
                img_key = key.split(".")[-2]
                if wanted_images is not None and img_key not in wanted_images:
                    continue
                images[img_key] = np.asarray(value)
            else:
                for suffix in ("lowdim.npz", "metadata.json", "language_instructions.json"):
                    if key.endswith(suffix):
                        data[suffix] = value

        # Language instruction
        lang_data = data.get("language_instructions.json", {})
        instruction = self._select_instruction(lang_data, cfg.language_instruction_types)

        lowdim = data.get("lowdim.npz", {})
        metadata = data.get("metadata.json", {})
        anchor = metadata.get("anchor_relative_idx")

        # Only extract configured lowdim fields (or all if none configured).
        wanted_lowdim = set(cfg.action_fields) | set(cfg.proprioception_fields)

        extracted: dict[str, np.ndarray | None] = {}
        for key in lowdim.keys():
            if wanted_lowdim and key not in wanted_lowdim:
                continue
            field_data = lowdim.get(key)
            if field_data is not None and anchor is not None:
                field_data = _crop_sequence(field_data, anchor, cfg.lowdim_past_timesteps, cfg.lowdim_future_timesteps)
            extracted[key] = field_data

        # Update anchor to reflect new position after cropping
        if anchor is not None:
            metadata = {**metadata, "anchor_relative_idx": cfg.lowdim_past_timesteps}

        return {
            "images": images,
            "lowdim": extracted,
            "metadata": metadata,
            "language_instruction": instruction,
        }

    @staticmethod
    def _select_instruction(lang_data: dict, types: list[str]) -> str:
        """Pick a random language instruction from the specified types."""
        available: list[str] = []
        for t in types:
            if t in lang_data:
                v = lang_data[t]
                if isinstance(v, str):
                    v = [v]
                available.extend(v)
        return random.choice(available) if available else ""

    # ---- batch collation ------------------------------------------------------

    def _collate_batch(self, samples: list[dict]) -> dict:
        """Collate a list of sample dicts into stacked tensors / arrays."""
        cfg = self.config
        anchor = cfg.lowdim_past_timesteps
        batch_size = len(samples)

        # Images: dict of name -> [B, H, W, 3] numpy
        # Use configured names, or fall back to intersection of all samples' keys.
        if self._image_names:
            image_names = self._image_names
        else:
            key_sets = [set(s["images"].keys()) for s in samples if s.get("images")]
            image_names = sorted(set.intersection(*key_sets)) if key_sets else []
        collated_images: dict[str, np.ndarray] = {}
        for name in image_names:
            imgs = [s["images"].get(name) for s in samples]
            none_count = sum(1 for x in imgs if x is None)
            if none_count == batch_size:
                continue
            if none_count > 0:
                print(f"[WDS] Image '{name}': {none_count}/{batch_size} samples missing, skipping image key")
                continue
            shapes = set(img.shape for img in imgs)
            dtypes = set(img.dtype for img in imgs)
            if len(shapes) > 1:
                logging.warning(f"Image '{name}': shape mismatch {shapes}")
                raise ValueError(f"Image '{name}': shape mismatch {shapes}")
            if len(dtypes) > 1:
                logging.warning(f"Image '{name}': dtype mismatch {dtypes}")
                raise ValueError(f"Image '{name}': dtype mismatch {dtypes}")
            try:
                collated_images[name] = np.stack(imgs, axis=0)
            except Exception as e:
                logging.warning(f"Image '{name}': np.stack failed: {e}")
                raise

        # Lowdim fields -> torch tensors
        lowdim: dict[str, torch.Tensor] = {}
        needed_keys = set(cfg.action_fields) | set(cfg.proprioception_fields)
        if not needed_keys:
            # Fall back to intersection of all samples' lowdim keys.
            key_sets = [set(s["lowdim"].keys()) for s in samples]
            needed_keys = set.intersection(*key_sets) if key_sets else set()
        for key in sorted(needed_keys):
            values = [s["lowdim"].get(key) for s in samples]
            none_count = sum(1 for v in values if v is None)
            if none_count == batch_size:
                continue
            if none_count > 0:
                print(f"[WDS] Lowdim '{key}': missing in {none_count}/{batch_size} samples, skipping field")
                continue
            shapes = set(v.shape for v in values)
            if len(shapes) > 1:
                logging.warning(f"Lowdim '{key}': shape mismatch {shapes}")
                raise ValueError(f"Lowdim '{key}': shape mismatch {shapes}")
            try:
                lowdim[key] = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in values])
            except Exception as e:
                logging.warning(f"Lowdim '{key}': torch.stack failed: {e}")
                raise

        # Actions: only configured action fields -> {field: [B, T, dim]}
        if cfg.action_fields:
            actions = {k: lowdim[k] for k in cfg.action_fields if k in lowdim}
        else:
            actions = lowdim

        # Proprioception: only configured proprioception fields, truncated to past+1
        if cfg.proprioception_fields:
            proprioception = {k: lowdim[k][:, : anchor + 1] for k in cfg.proprioception_fields if k in lowdim}
        else:
            proprioception = {k: v[:, : anchor + 1] for k, v in lowdim.items()}

        return {
            "actions": actions,
            "observation": {
                "images": collated_images,
                "proprioception": proprioception,
            },
            "prompt": np.array([s["language_instruction"] for s in samples]),
        }

    # ---- iteration ------------------------------------------------------------

    def __iter__(self):
        """Iterate over processed batches."""
        for batch in self._dataloader:
            yield batch

    def __len__(self) -> int:
        """Return total number of samples across selected shards."""
        return self._total_samples


class ConfiguredWDSDataset(WDSDataset):
    """Picklable WDSDataset adapter for the ``dataset_class`` interface.

    Call ``configure()`` to set class-level defaults, then pass the class
    as ``dataset_class`` to ``DataConfig``.  Because the class itself is
    defined at module level, ``spawn`` multiprocessing can pickle it.
    """

    _manifest_path: str = ""
    _wds_num_workers: int = 0
    _shuffle_buffer: int = 2000
    _wds_config: WDSDatasetConfig = WDSDatasetConfig()

    @classmethod
    def configure(
        cls,
        manifest_path: str,
        num_workers: int,
        shuffle_buffer: int,
        config: WDSDatasetConfig,
    ) -> type["ConfiguredWDSDataset"]:
        """Set class-level config and return the class itself."""
        cls._manifest_path = manifest_path
        cls._wds_num_workers = num_workers
        cls._shuffle_buffer = shuffle_buffer
        cls._wds_config = config
        return cls

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,
        shuffle: bool = False,
        action_chunk_size: int = 16,
        **kwargs,
    ):
        super().__init__(
            manifest_path=self._manifest_path,
            batch_size=batch_size,
            num_workers=self._wds_num_workers,
            shuffle=shuffle,
            shuffle_buffer=self._shuffle_buffer,
            config=self._wds_config,
        )


# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Test WDS dataset loading")
    # parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=20)
    args = parser.parse_args()

    manifest_path = "s3://tri-ml-datasets-uw2/vla_foundry_datasets/v0.4.2-sim/manifest.jsonl"

    config = WDSDatasetConfig(
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
    )

    import time

    dataset = WDSDataset(
        manifest_path=manifest_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        config=config,
    )

    print(f"\nDataset: {len(dataset)} samples\n")

    t0 = time.time()
    num_batches = args.num_batches
    for i, batch in enumerate(dataset):
        print(f"--- Batch {i} ---")
        for field_name, field_tensor in batch["actions"].items():
            print(f"  actions/{field_name}: {field_tensor.shape}")
        obs = batch["observation"]
        for cam, img in obs["images"].items():
            print(f"  image/{cam}: {img.shape}")
        for field_name, field_tensor in obs["proprioception"].items():
            print(f"  proprio/{field_name}: {field_tensor.shape}")
        prompts = batch["prompt"]
        if len(prompts) > 0:
            print(f"  prompt[0]: {prompts[0][:80]}...")
        if i + 1 >= num_batches:
            break
    t1 = time.time()
    elapsed = t1 - t0
    bps = num_batches / elapsed if elapsed > 0 else float("inf")
    print(f"\nProcessed {num_batches} batches in {elapsed:.3f} seconds")
    print(f"Batches per second: {bps:.2f}")
