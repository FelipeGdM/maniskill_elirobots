import copy
from importlib import resources
from pathlib import Path
from typing import override

import numpy as np

# from sapien import Pose
import sapien
from mani_skill.agents.base_agent import Actor, BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, PDJointPosMimicControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Link, Pose
from torch import Tensor


@register_agent()
class EC63(BaseAgent):
    """ManiSkill representation of EC63."""

    uid = "ec63"
    urdf_path = str(resources.files("maniskill_elirobots") / "assets/ec63/ec63_description.urdf")
    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]

    gripper_joint_names = [
        "finger_1_joint",
        "finger_2_joint",
    ]

    ee_link_name = "claw_tcp_link"

    """Reference values taken from panda robot"""
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    keyframes = {
        "rest": Keyframe(
            qpos=np.array([0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04]),
            pose=sapien.Pose(),
        ),
    }

    @override
    def _after_init(self):
        # Tool Center Point
        self.tcp: list[Link] | Link | None = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)  # pyright: ignore[reportUninitializedInstanceVariable]

    @property
    @override
    def _controller_configs(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
        )

        controller_configs = {
            "pd_joint_delta_pos": {
                "arm": arm_pd_joint_delta_pos,
                "gripper": gripper_pd_joint_pos,
            },
            "pd_joint_pos": {
                "arm": arm_pd_joint_pos,
                "gripper": gripper_pd_joint_pos,
            },
        }

        # Make a deepcopy in case users modify any config
        return copy.deepcopy(controller_configs)

    @property
    @override
    def _sensor_configs(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return [
            CameraConfig(
                uid="hand_camera",
                pose=Pose.create_from_pq(p=Tensor([-1, 0, 0]), q=Tensor([1, 0, 0, 0])),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                # mount=self.robot.links_map["camera_link"],
            ),
        ]

    @override
    def is_grasping(self, object: Actor | None = None) -> bool:
        return False

    @override
    def is_static(self, threshold: float) -> bool:
        return False

    @property
    def tcp_pos(self):
        if self.tcp is not Link:
            return None
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        if self.tcp is None:
            return None
        return self.tcp.pose


if __name__ == "__main__":
    import mani_skill.examples.demo_robot as demo_robot_script

    args = demo_robot_script.Args(
        robot_uid="ec63",
        random_actions=True,
        control_mode="pd_joint_delta_pos",
        shader="minimal",
    )

    demo_robot_script.main(args)
