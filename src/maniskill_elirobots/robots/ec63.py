import copy
from importlib import resources
from pathlib import Path
from typing import Any, cast, override

import numpy as np

# from sapien import Pose
import sapien
import torch
from mani_skill.agents.base_agent import Actor, BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, PDJointPosMimicControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Link, Pose
from mani_skill.utils.structs.link import Link
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
            qpos=[
                0.0,
                -3 * np.pi / 8,
                3 * np.pi / 8,
                -4 * np.pi / 8,
                4 * np.pi / 8,
                0,
                0,
                0,
            ],
            pose=sapien.Pose(),
        ),
    }

    @property
    def finger_right_link(self):
        link = sapien_utils.get_obj_by_name(self.robot.get_links(), "claw_finger_1")
        if not isinstance(link, Link):
            msg = "claw_finger_1 should return exactly one link"
            raise TypeError(msg)
        return link

    @property
    def finger_left_link(self):
        link = sapien_utils.get_obj_by_name(self.robot.get_links(), "claw_finger_2")
        if not isinstance(link, Link):
            msg = "claw_finger_2 should return exactly one link"
            raise TypeError(msg)
        return link

    @property
    def tcp(self):
        link = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        if not isinstance(link, Link):
            msg = f"{self.ee_link_name} should return exactly one link"
            raise TypeError(msg)
        return link

    @override
    def _after_init(self):
        pass

    @property
    @override
    def _controller_configs(self):
        controller_configs = {
            "pd_joint_pos": {
                "arm": PDJointPosControllerConfig(
                    joint_names=self.arm_joint_names,
                    lower=None,
                    upper=None,
                    stiffness=self.arm_stiffness,
                    damping=self.arm_damping,
                    force_limit=self.arm_force_limit,
                    normalize_action=False,
                ),
                "gripper": PDJointPosMimicControllerConfig(
                    joint_names=self.gripper_joint_names,
                    lower=None,  # a trick to have force when the object is thin
                    upper=None,
                    stiffness=self.gripper_stiffness,
                    damping=self.gripper_damping,
                    force_limit=self.gripper_force_limit,
                    mimic={"finger_2_joint": {"joint": "finger_1_joint"}},
                ),
            },
            "pd_joint_delta_pos": {
                "arm": PDJointPosControllerConfig(
                    joint_names=self.arm_joint_names,
                    # lower=None,
                    # upper=None,
                    lower=-0.5,
                    upper=0.5,
                    stiffness=self.arm_stiffness,
                    damping=self.arm_damping,
                    force_limit=self.arm_force_limit,
                    normalize_action=True,
                    use_delta=True,
                    use_target=True,
                    # drive_mode="force",
                    # drive_mode="acceleration",
                ),
                "gripper": PDJointPosMimicControllerConfig(
                    joint_names=self.gripper_joint_names,
                    lower=None,  # a trick to have force when the object is thin
                    upper=None,
                    stiffness=self.gripper_stiffness,
                    damping=self.gripper_damping,
                    force_limit=self.gripper_force_limit,
                    mimic={"finger_2_joint": {"joint": "finger_1_joint"}},
                ),
            },
        }

        # Make a deepcopy in case users modify any config
        return copy.deepcopy(controller_configs)

    @property
    @override
    def _sensor_configs(self):
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
    def is_grasping(self, obj: Actor | None, min_force: float = 1e-3, max_angle: float = 80) -> Tensor:
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        if obj is None:
            return torch.Tensor([False])
        l_contact_forces = cast("Tensor", self.scene.get_pairwise_contact_forces(self.finger_left_link, obj))
        r_contact_forces = cast("Tensor", self.scene.get_pairwise_contact_forces(self.finger_right_link, obj))

        lforce = cast("float", torch.linalg.norm(l_contact_forces, axis=1))
        rforce = cast("float", torch.linalg.norm(r_contact_forces, axis=1))

        left_finger_pos = self.finger_left_link.pose.p
        right_finger_pos = self.finger_right_link.pose.p

        travel_dir = left_finger_pos - right_finger_pos

        langle = common.compute_angle_between(x1=travel_dir, x2=l_contact_forces)
        rangle = common.compute_angle_between(x1=-travel_dir, x2=r_contact_forces)

        lflag = torch.logical_and(lforce >= min_force, torch.rad2deg(langle) <= max_angle)
        rflag = torch.logical_and(rforce >= min_force, torch.rad2deg(rangle) <= max_angle)
        return torch.logical_and(lflag, rflag)

    @override
    def is_static(self, threshold: float = 0.2):
        qvel = cast("Tensor", self.robot.get_qvel()[..., :6])
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose


def _script():
    import mani_skill.examples.demo_robot as demo_robot_script  # noqa: PLC0415

    args = demo_robot_script.Args(
        robot_uid="ec63",
        random_actions=True,
        control_mode="pd_joint_delta_pos",
        shader="minimal",
    )

    demo_robot_script.main(args)


if __name__ == "__main__":
    _script()
