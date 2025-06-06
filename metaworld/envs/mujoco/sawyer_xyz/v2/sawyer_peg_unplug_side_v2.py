from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.envs.mujoco.utils import reward_utils
from metaworld.types import InitConfigDict


class SawyerPegUnplugSideEnvV2(SawyerXYZEnv):
    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        model_name: str | None = None,
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.25, 0.6, -0.001)
        obj_high = (-0.15, 0.8, 0.001)
        goal_low = obj_low + np.array([0.194, 0.0, 0.131])
        goal_high = obj_high + np.array([0.194, 0.0, 0.131])

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            model_name=model_name
        )

        self.init_config: InitConfigDict = {
            "obj_init_pos": np.array([-0.225, 0.6, 0.05]),
            "hand_init_pos": np.array((0, 0.6, 0.2)),
        }
        self.goal = np.array([-0.225, 0.6, 0.0])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_v2_path_for("sawyer_xyz/sawyer_peg_unplug_side.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        # obj = obs[4:7]

        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward,
            grasp_success,
        ) = self.compute_reward(action, obs)
        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)

        info = {
            "success": success,
            "near_object": near_object,
            "grasp_success": grasp_success,
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self._get_site_pos("pegEnd")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("plug1").xquat

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos
        qpos[12:16] = np.array([1.0, 0.0, 0.0, 0.0])
        qvel[9:12] = 0
        self.set_state(qpos, qvel)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        pos_box = self._get_state_rand_vec()
        self.model.body("box").pos = pos_box
        pos_plug = pos_box + np.array([0.044, 0.0, 0.131])
        self._set_obj_xyz(pos_plug)
        self.obj_init_pos = self._get_site_pos("pegEnd")

        self._target_pos = pos_plug + np.array([0.15, 0.0, 0.0])
        self.model.site("goal").pos = self._target_pos
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        tcp = self.tcp_center
        obj = obs[4:7]
        tcp_opened: float = obs[3]
        target = self._target_pos
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        obj_to_target = float(np.linalg.norm(obj - target))
        pad_success_margin = 0.05
        object_reach_radius = 0.01
        x_z_margin = 0.005
        obj_radius = 0.025

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=object_reach_radius,
            obj_radius=obj_radius,
            pad_success_thresh=pad_success_margin,
            xz_thresh=x_z_margin,
            desired_gripper_effort=0.8,
            high_density=True,
        )
        in_place_margin = float(np.linalg.norm(self.obj_init_pos - target))

        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.05),
            margin=in_place_margin,
            sigmoid="long_tail",
        )
        grasp_success = tcp_opened > 0.5 and (obj[0] - self.obj_init_pos[0] > 0.015)

        reward = 2 * object_grasped

        if grasp_success and tcp_to_obj < 0.035:
            reward = 1 + 2 * object_grasped + 5 * in_place

        if obj_to_target <= 0.05:
            reward = 10.0

        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            obj_to_target,
            object_grasped,
            in_place,
            float(grasp_success),
        )
