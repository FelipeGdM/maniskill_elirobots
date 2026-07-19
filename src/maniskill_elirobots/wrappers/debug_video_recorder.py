from typing import override

import numpy as np
from gymnasium import logger
from mani_skill.utils import common, gym_utils
from PIL import Image, ImageDraw, ImageFont

# from mani_skill.utils.visualization.misc import put_info_on_image
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

from maniskill_elirobots.assets import ASSETS_PATH


def put_text_on_image(image: np.ndarray, lines: list[str]):
    # global TEXT_FONT
    assert image.dtype == np.uint8, image.dtype
    image = image.copy()
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    text_font_path = ASSETS_PATH / "fonts" / "UbuntuSansMono-Regular.ttf"
    text_font = ImageFont.truetype(str(text_font_path), size=16)

    y = -10
    for line in lines:
        bbox = draw.textbbox((0, 0), text=line)
        textheight = bbox[3] - bbox[1]
        y += textheight + 10
        x = 10
        draw.text((x, y), text=line, fill=(0, 0, 0), font=text_font)
    return np.array(image)


def put_info_on_image(image, info: dict[str, float], extras=None, overlay=True):
    lines = [f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in info.items()]
    if extras is not None:
        lines.extend(extras)
    return put_text_on_image(image, lines)


class DebugVecVideoRecorder(VecVideoRecorder):
    @override
    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()

        self.step_id += 1
        if self.recording:
            scalar_info = gym_utils.extract_scalars_from_info(
                infos[0],
                batch_size=self.num_envs,
            )

            scalar_info["reward"] = common.to_numpy(rewards)

            if np.size(scalar_info["reward"]) > 1:
                scalar_info["reward"] = [float(rew) for rew in scalar_info["reward"]]
            else:
                scalar_info["reward"] = float(scalar_info["reward"][0])

            self._capture_info_frame(scalar_info)
            if len(self.recorded_frames) > self.video_length:
                print(f"Saving video to {self.video_path}")
                self._stop_recording()
        elif self._video_enabled():
            self._start_video_recorder()

        return obs, rewards, dones, infos

    def _capture_info_frame(self, info: dict[str, float]):
        assert self.recording, "Cannot capture a frame, recording wasn't started."  # noqa: S101

        frame = self.env.render()

        info_frame = put_info_on_image(frame, info)

        if isinstance(frame, np.ndarray):
            self.recorded_frames.append(info_frame)
        else:
            self._stop_recording()
            logger.warn(f"Recording stopped: expected type of frame returned by render to be a numpy array, got instead {type(frame)}.")

    @override
    def _capture_frame(self) -> None:
        return self._capture_info_frame({})
