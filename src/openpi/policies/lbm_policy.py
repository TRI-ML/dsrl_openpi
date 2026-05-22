import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_lbm_example() -> dict:
    """Creates a random input example for the LBM policy."""
    return {
        "observation": {
            "proprioception": {
                "robot__actual__poses__left::panda__xyz": np.random.rand(3),
                "robot__actual__poses__right::panda__xyz": np.random.rand(3),
                "robot__actual__poses__left::panda__rot_6d": np.random.rand(6),
                "robot__actual__poses__right::panda__rot_6d": np.random.rand(6),
            },
            "images": {
                "wrist_right_minus_t0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
                "wrist_left_plus_t0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
                "scene_left_0_t0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
                "scene_right_0_t0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            },
        },
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LBMInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    # Camera name keys (without _t0 suffix). Defaults to original LBM sim names.
    camera_names: tuple[str, ...] = ("wrist_right_minus", "wrist_left_plus", "scene_left_0", "scene_right_0")

    def __call__(self, data: dict) -> dict:
        STATE_KEYS = [
            "robot__actual__poses__left::panda__xyz",
            "robot__actual__poses__right::panda__xyz",
            "robot__actual__poses__left::panda__rot_6d",
            "robot__actual__poses__right::panda__rot_6d",
        ]
        state = np.concatenate([data["observation"]["proprioception"][key].squeeze() for key in STATE_KEYS])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        image_keys = tuple(f"{cam}_t0" for cam in self.camera_names)
        parsed_images = tuple(_parse_image(data["observation"]["images"][k]) for k in image_keys)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = image_keys
                images = parsed_images
                image_masks = tuple(np.True_ for _ in self.camera_names)
            case _model.ModelType.PI0_FAST:
                names = image_keys
                images = parsed_images
                image_masks = tuple(np.True_ for _ in self.camera_names)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        ACTION_KEYS = [
            "robot__action__poses__left::panda__xyz_relative",
            "robot__action__poses__right::panda__xyz_relative",
            "robot__action__poses__left::panda__rot_6d_relative",
            "robot__action__poses__right::panda__rot_6d_relative",
            "robot__action__grippers__left::panda_hand",
            "robot__action__grippers__right::panda_hand",
        ]
        if "actions" in data:
            actions = np.concatenate(
                [data["actions"][key].reshape(data["actions"][key].shape[0], -1) for key in ACTION_KEYS]
                , axis=-1
            )
            inputs["actions"] = actions

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]
            print(f"Prompt: {data['prompt']}")

        return inputs


@dataclasses.dataclass(frozen=True)
class LBMOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Action dims: left_xyz(3), right_xyz(3), left_rot6d(6), right_rot6d(6), left_grip(1), right_grip(1) = 20
        return {"actions": np.asarray(data["actions"][:, :20])}
