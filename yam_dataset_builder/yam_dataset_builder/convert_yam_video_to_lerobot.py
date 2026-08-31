"""
Convert raiden "videos"-format YAM data to a LeRobot v2.1 dataset for openpi fine-tuning.

Sibling of convert_yam_data_to_lerobot.py, which reads the per-frame layout
(rgb/<cam>/<i>.png + lowdim/<i>.pkl). The raiden processed export is different:

  <episode>/
    rgb/<cam>.mp4                 already-encoded H.264 video, one file per camera
    rgb/<cam>_timestamps.npy
    lowdim/<cam>.npz              joints (N,14), action_joints (N,14), action (N,26 cartesian)
    metadata.json                 num_frames, language.prompt, cameras (4: incl ego)

Only the frame SOURCE (decode the mp4 to frames) and the joint LOADER (read the npz) differ; the
LeRobot dataset schema, the joint convention (state=joints[:-1], action=joints[1:] absolute target),
and save_episode are identical to the banana converter — so the resulting dataset trains with the same
pi05 config. ego_camera is dropped to keep the model's 3 cameras.

Usage:
  uv run yam_dataset_builder/yam_dataset_builder/convert_yam_video_to_lerobot.py \
      --args.raw-dir /home/robot-lab/raiden/data/processed/flip_pink_cup \
      --args.repo-id local/flippinkcup_yam --args.overwrite
"""

import dataclasses
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

# Reuse the exact schema constants + ffmpeg encoder from the per-frame converter so the two datasets are
# byte-for-byte compatible consumers of the same pi05 config. Robust to being run as a script (path) or
# imported as part of the package.
try:
    from yam_dataset_builder.convert_yam_data_to_lerobot import CAMERAS, FPS, MOTORS, encode_video_ffmpeg
except ModuleNotFoundError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from convert_yam_data_to_lerobot import CAMERAS, FPS, MOTORS, encode_video_ffmpeg


@dataclasses.dataclass(frozen=True)
class Args:
    raw_dir: Path
    """Path to the raiden processed "videos" dataset directory (episodes are 0000, 0001, ...)."""
    repo_id: str
    """LeRobot repo ID (e.g., local/flippinkcup_yam)."""
    push_to_hub: bool = False
    private: bool = True
    overwrite: bool = False


def _frames_from_mp4(mp4_path: Path, dst_dir: Path, num_frames: int) -> None:
    """Decode the first `num_frames` frames of an mp4 into dst_dir/frame_%06d.png (LeRobot-style),
    so the downstream encode + stats-sampling path is identical to the per-frame converter."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(mp4_path),
        "-frames:v", str(num_frames),
        "-start_number", "0",
        "-loglevel", "error",
        str(dst_dir / "frame_%06d.png"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed decoding {mp4_path}:\n{r.stderr}")
    got = len(list(dst_dir.glob("frame_*.png")))
    if got < num_frames:
        raise RuntimeError(f"{mp4_path.name}: decoded {got} frames < expected {num_frames}")


def _load_joints(ep_dir: Path) -> np.ndarray:
    """(N,14) measured joints from the lowdim npz. Robot signals are camera-independent; read the first
    camera npz that exists. Matches the per-frame converter's `joints` (14-D bimanual)."""
    lowdim = ep_dir / "lowdim"
    for cam in ("scene_camera", *CAMERAS):
        npz = lowdim / f"{cam}.npz"
        if npz.exists():
            arr = np.load(npz, allow_pickle=True)
            if "joints" not in arr.files:
                raise RuntimeError(f"{npz} has no 'joints' (keys: {list(arr.files)})")
            return np.asarray(arr["joints"], dtype=np.float32).reshape(-1, 14)
    raise FileNotFoundError(f"no lowdim/<cam>.npz in {ep_dir}")


def process_episode(dataset: LeRobotDataset, ep_dir: Path, output_dir: Path) -> None:
    with open(ep_dir / "metadata.json") as f:
        metadata = json.load(f)
    task = metadata["language"]["prompt"][0]

    joints = _load_joints(ep_dir)
    # action[t] = joints[t+1] (absolute joint target), drop last frame — same as the banana converter.
    actions = joints[1:]
    joints = joints[:-1]
    num_frames = int(joints.shape[0])
    if num_frames < 2:
        raise RuntimeError(f"{ep_dir.name}: only {num_frames} usable frames")

    episode_index = dataset.meta.total_episodes
    tmp_frames_root = output_dir / "_tmp_frames"

    def _decode_cam(cam: str) -> None:
        img_key = f"observation.images.{cam}"
        dst_dir = tmp_frames_root / img_key / f"episode_{episode_index:06d}"
        _frames_from_mp4(ep_dir / "rgb" / f"{cam}.mp4", dst_dir, num_frames)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(_decode_cam, CAMERAS))

    def _encode_cam(cam: str) -> None:
        img_key = f"observation.images.{cam}"
        imgs_dir = tmp_frames_root / img_key / f"episode_{episode_index:06d}"
        video_path = output_dir / dataset.meta.get_video_file_path(episode_index, img_key)
        encode_video_ffmpeg(imgs_dir, video_path, num_frames, FPS)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(_encode_cam, CAMERAS))

    ep_buffer = dataset.create_episode_buffer(episode_index)
    ep_buffer["size"] = num_frames
    ep_buffer["task"] = [task] * num_frames
    ep_buffer["frame_index"] = list(range(num_frames))
    ep_buffer["timestamp"] = [i / FPS for i in range(num_frames)]
    ep_buffer["observation.state"] = [joints[i] for i in range(num_frames)]
    ep_buffer["action"] = [actions[i] for i in range(num_frames)]
    for cam in CAMERAS:
        img_key = f"observation.images.{cam}"
        imgs_dir = tmp_frames_root / img_key / f"episode_{episode_index:06d}"
        ep_buffer[img_key] = [str(imgs_dir / f"frame_{i:06d}.png") for i in range(num_frames)]

    dataset.episode_buffer = ep_buffer
    dataset.save_episode()

    for cam in CAMERAS:
        img_key = f"observation.images.{cam}"
        ep_frames_dir = tmp_frames_root / img_key / f"episode_{episode_index:06d}"
        if ep_frames_dir.exists():
            shutil.rmtree(ep_frames_dir)


def main(args: Args):
    ep_dirs = sorted(d for d in args.raw_dir.iterdir()
                     if d.is_dir() and d.name.isdigit() and (d / "metadata.json").exists())
    print(f"Found {len(ep_dirs)} episodes in {args.raw_dir}")

    output_dir = HF_LEROBOT_HOME / args.repo_id
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            print(f"Output dir {output_dir} exists; pass --args.overwrite to rebuild.")
            sys.exit(1)

    features = {
        "observation.state": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (3, 720, 1280), "names": ["channels", "height", "width"]}

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=FPS, robot_type="yam_bimanual", features=features,
        use_videos=True, tolerance_s=0.0001, image_writer_processes=0, image_writer_threads=0)

    skipped = []
    for ep_dir in tqdm.tqdm(ep_dirs, desc="Converting episodes"):
        try:
            process_episode(dataset, ep_dir, output_dir)
        except Exception as e:  # noqa: BLE001 - a bad episode shouldn't kill the run
            print(f"\nSKIPPING {ep_dir.name}: {e}")
            skipped.append((ep_dir.name, str(e)))

    tmp_frames_root = output_dir / "_tmp_frames"
    if tmp_frames_root.exists():
        shutil.rmtree(tmp_frames_root)
    if skipped:
        print(f"Skipped {len(skipped)}/{len(ep_dirs)} episodes: {[s[0] for s in skipped]}")
    print(f"Dataset saved to {output_dir}")
    print(f"Total episodes: {dataset.num_episodes}, Total frames: {dataset.num_frames}")

    if args.push_to_hub:
        dataset.push_to_hub(private=args.private)


if __name__ == "__main__":
    tyro.cli(main)
