"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by what agents/actors are
loaded, how agents/actors are randomly initialized during env resets, how goals are randomized and parameterized in observations, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the poses of all actors, articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do. If followed correctly you can easily build a
task that can simulate on the CPU and be parallelized on the GPU without having to manage GPU memory and parallelization apart from some
code that need to be written in batched mode (e.g. reward, success conditions)

For a minimal implementation of a simple task, check out
mani_skill /envs/tasks/push_cube.py which is annotated with comments to explain how it is implemented
"""

from typing import Any, cast, override

import numpy as np
import sapien
import torch
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils

# from mani_skill.utils.building import actors
from mani_skill.utils.building.actors.common import build_cube, build_cylinder, build_red_white_target, build_sphere, build_twocolor_peg
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from quatorch import Quaternion, quaternion
from torch.nn import functional
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qconjugate, quat2axangle

from maniskill_elirobots.robots.ec63 import EC63
from maniskill_elirobots.tasks.scene_builder.actors import build_transparent_sphere, build_twocolor_cylinder
from maniskill_elirobots.utils.math import coin_angle


# register the environment by a unique ID and specify a max time limit. Now once this file is imported you can do gym.make("FlipCoin-v0")
@register_env("FlipCoin-v1", max_episode_steps=50)
class FlipCoinEnv(BaseEnv):
    """
    Task Description
    ----------------
    Simple task where the objective is to flip a two sided coin over the table

    Randomizations
    --------------
    - how is it randomized?
    - how is that randomized?

    Success Conditions
    ------------------
    The coin is in the same place as it started, but with orientation flipped

    Visualization: link to a video/gif of the task being solved
    """

    # here you can define a list of robots that this task is built to support and be solved by. This is so that
    # users won't be permitted to use robots not predefined here. If SUPPORTED_ROBOTS is not defined then users can do anything
    SUPPORTED_ROBOTS = ["ec63"]
    # if you want to say you support multiple robots you can use SUPPORTED_ROBOTS = [["panda", "panda"], ["panda", "fetch"]] etc.

    # to help with programming, you can assert what type of agents are supported like below, and any shared properties of self.agent
    # become available to typecheckers and auto-completion. E.g. Panda and Fetch both share a property called .tcp (tool center point).
    agent: EC63

    initial_agent_pose = sapien.Pose(p=[-0.4, 0, 0])

    coin_half_length = 5e-3
    coin_radius = 18e-3

    coin_max_height = 200e-3

    coin_normal_axis = torch.tensor([1.0, 0.0, 0.0])
    coin_desired_axis = torch.tensor([0.0, 0.0, 1.0])

    initial_coin_pose = sapien.Pose(
        p=[-0.1, 0, 2 * coin_half_length],
        q=euler2quat(0, np.pi / 2, 0),
    )

    initial_goal_pose = sapien.Pose(
        p=[0.1, 0, 1e-5],
        q=euler2quat(0, np.pi / 2, 0),
    )

    goal_thresh: float = 25e-3

    init_qpos = {
        "ec63": torch.tensor([0.0, -6 * np.pi / 8, 5 * np.pi / 8, -3 * np.pi / 8, 4 * np.pi / 8, 0, 0, 0]),
        "panda": torch.tensor([0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04]),
    }

    @property
    def goal_sphere(self) -> Actor:
        return self.scene_elements["goal_sphere"]

    @property
    def goal_region(self) -> Actor:
        return self.scene_elements["goal_region"]

    @property
    def coin(self) -> Actor:
        return self.scene_elements["coin"]

    # if you want to do typing for multi-agent setups, use this below and specify what possible tuples of robots are permitted by typing
    # this will then populate agent.agents (list of the instantiated agents) with the right typing
    # agent: MultiAgent[Union[Tuple[Panda, Panda], Tuple[Panda, Panda, Panda]]]

    # in the __init__ function you can pick a default robot your task should use e.g. the panda robot by setting a default for robot_uids argument
    # note that if robot_uids is a list of robot uids, then we treat it as a multi-agent setup and load each robot separately.
    def __init__(self, *args, robot_uids: str = "ec63", robot_init_qpos_noise: float = 0.02, qvel_penalty=1.0, qvel_tolerance=0.2, angle_penalty=1.0, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.scene_elements = {}
        self.goal_radius = kwargs.get("goal_radius", 0.1)

        self.qvel_penalty = qvel_penalty
        self.qvel_tolerance = qvel_tolerance

        self.angle_penalty = angle_penalty

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations. Note that tasks need to tune their GPU memory configurations accordingly
    # in order to save memory while also running with no errors. In general you can start with low values and increase them
    # depending on the messages that show up when you try to run more environments in parallel. Since this is a python property
    # you can also check self.num_envs to dynamically set configurations as well
    @property
    @override
    def _default_sim_config(self):
        return SimConfig(gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18))

    """
    Reconfiguration Code

    below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general. In CPU simulation,
    for some tasks these may need to be called multiple times if you need to swap out object assets. In GPU simulation these will only ever be called once.
    """

    @override
    def _load_agent(self, options: dict):
        # this code loads the agent into the current scene. You should use it to specify the initial pose(s) of the agent(s)
        # such that they don't collide with other objects initially
        super()._load_agent(options, self.initial_agent_pose)

    @override
    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.scene_elements["table_scene"] = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.scene_elements["table_scene"].build()

        # self.scene_elements["coin"] = build_cube(
        #     self.scene,
        #     half_size=self.coin_radius,
        #     color=[1, 0, 0, 1],
        #     name="coin",
        #     initial_pose=self.initial_coin_pose,
        # )

        self.scene_elements["coin"] = build_twocolor_peg(
            self.scene,
            length=2 * self.coin_half_length,
            width=self.coin_radius,
            color_1=[1, 0, 0, 1],
            color_2=[0, 0, 1, 1],
            name="coin",
            initial_pose=self.initial_coin_pose,
        )

        # self.scene_elements["coin"] = build_twocolor_cylinder(
        #     self.scene,
        #     radius=self.coin_radius,
        #     color_1=[1, 0, 0, 1],
        #     color_2=[0, 0, 1, 1],
        #     half_length=self.coin_half_length,
        #     name="coin",
        #     initial_pose=self.initial_coin_pose,
        # )

        # we then add the cube that we want to push and give it a color and size using a convenience build_cube function
        # we specify the body_type to be "dynamic" as it should be able to move when touched by other objects / the robot
        # finally we specify an initial pose for the cube so that it doesn't collide with other objects initially

        # we also add in red/white target to visualize where we want the cube to be pushed to
        # we specify add_collisions=False as we only use this as a visual for videos and do not want it to affect the actual physics
        # we finally specify the body_type to be "kinematic" so that the object stays in place
        self.scene_elements["goal_region"] = build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=self.initial_goal_pose,
        )

        self.scene_elements["goal_sphere"] = build_transparent_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 0.5],
            name="goal_sphere",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

        # optionally you can automatically hide some Actors from view by appending to the self._hidden_objects list. When visual observations
        # are generated or env.render_sensors() is called or env.render() is called with render_mode="sensors", the actor will not show up.
        # This is useful if you intend to add some visual goal sites as e.g. done in PickCube that aren't actually part of the task
        # and are there just for generating evaluation videos.
        # self._hidden_objects.append(self.goal_region)
        self._hidden_objects.append(self.scene_elements["goal_sphere"])

    @property
    @override
    def _default_sensor_configs(self):
        # To customize the sensors that capture images/pointclouds for the environment observations,
        # simply define a CameraConfig as done below for Camera sensors. You can add multiple sensors by returning a list
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])  # sapien_utils.look_at is a utility to get the pose of a camera that looks at a target

        # to see what all the sensors capture in the environment for observations, run env.render_sensors() which returns an rgb array you can visualize
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    @override
    def _default_human_render_camera_configs(self):
        # this is just like _sensor_configs, but for adding cameras used for rendering when you call env.render()
        # when render_mode="rgb_array" or env.render_rgb_array()
        # Another feature here is that if there is a camera called render_camera, this is the default view shown initially when a GUI is opened
        pose = sapien_utils.look_at([0.15, 0.0, 0.5], [-0.2, 0.0, 0.2])
        # pose = sapien_utils.look_at([0.2, 0.25, 0.35], [-0.2, 0.0, 0.2])
        return CameraConfig(
            "render_camera",
            pose,
            512,
            512,
            1.5,
            0.01,
            100,
            shader_pack="default",
        )

    @override
    def _setup_sensors(self, options: dict):
        # default code here will setup all sensors. You can add additional code to change the sensors e.g.
        # if you want to randomize camera positions
        return super()._setup_sensors(options)

    @override
    def _load_lighting(self, options: dict):
        # default code here will setup all lighting. You can add additional code to change the lighting e.g.
        # if you want to randomize lighting in the scene
        return super()._load_lighting(options)

    """
    Episode Initialization Code

    below are all functions involved in episode initialization during environment reset called in the same order. As a user
    you can change these however you want for your desired task. Note that these functions are given a env_idx variable.

    `env_idx` is a torch Tensor representing the indices of the parallel environments that are being initialized/reset. This is used
    to support partial resets where some parallel envs might be reset while others are still running (useful for faster RL and evaluation).
    Generally you only need to really use it to determine batch sizes via len(env_idx). ManiSkill helps handle internally a lot of masking
    you might normally need to do when working with GPU simulation. For specific details check out the push_cube.py code
    """

    @override
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size

            env_count = len(env_idx)

            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.

            init_qpos = self.init_qpos[self.robot_uids] if options.get("init_qpos") is None else cast("torch.Tensor", options.get("init_qpos"))

            self.scene_elements["table_scene"].initialize(env_idx)
            self.agent.reset(
                init_qpos=init_qpos,
            )

            # here we write some randomization code that randomizes the x, y position of the cube we are pushing in the range [-0.1, -0.1] to [0.1, 0.1]
            xyz = torch.zeros((env_count, 3))
            xyz[..., :2] = torch.rand((env_count, 2)) * self.coin_radius * 4 - 2 * self.coin_radius

            coin_xyz = xyz + torch.tensor(self.initial_coin_pose.get_p()) if options.get("coin_xyz") is None else cast("torch.Tensor", options.get("coin_xyz"))

            angle_mult = torch.randint(0, 4, size=(env_count,))
            angle_axis = 2 * torch.pi * torch.rand((env_count, 1))

            axis = torch.stack([torch.sin(angle_axis), torch.cos(angle_axis), torch.zeros((env_count, 1))], dim=1).reshape((env_count, 3))

            coin_q = Quaternion.from_axis_angle(
                axis,
                angle=angle_mult * np.pi / 2,
            )

            self.coin.set_pose(
                Pose.create_from_pq(
                    p=coin_xyz,
                    q=self.initial_coin_pose.get_q(),
                ),
            )

            # we can then create a pose object using Pose.create_from_pq to then set the cube pose with. Note that even though our quaternion
            # is not batched, Pose.create_from_pq will automatically batch p or q accordingly
            # furthermore, notice how here we do not even use env_idx as a variable to say set the pose for objects in desired
            # environments. This is because internally any calls to set data on the GPU buffer (e.g. set_pose, set_linear_velocity etc.)
            # automatically are masked so that you can only set data on objects in environments that are meant to be initialized

            # here we set the location of that red/white target (the goal region). In particular here, we set the position to be in front of the cube
            goal_region_xyz = coin_xyz.clone()

            theta = 2 * torch.pi * torch.rand((env_count, 1))

            goal_region_xyz[..., :2] += 4 * self.coin_radius * torch.stack([torch.sin(theta), torch.cos(theta)], dim=1).reshape((env_count, 2))

            goal_region_xyz[..., 2] = 1e-5

            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=goal_region_xyz,
                    q=self.initial_goal_pose.get_q(),
                ),
            )

            sphere_xyz = goal_region_xyz.clone()

            sphere_xyz[..., 2] = self.coin_half_length + torch.rand((env_count,)) * self.coin_max_height + 200e-3

            self.goal_sphere.set_pose(
                Pose.create_from_pq(
                    p=sphere_xyz,
                ),
            )

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """

    @override
    def evaluate(self):
        # this function is used primarily to determine success and failure of a task, both of which are optional. If a dictionary is returned
        # containing "success": bool array indicating if the env is in success state or not, that is used as the terminated variable returned by
        # self.step. Likewise if it contains "fail": bool array indicating the opposite (failure state or not) the same occurs. If both are given
        # then a logical OR is taken so terminated = success | fail. If neither are given, terminated is always all False.
        #
        # You may also include additional keys which will populate the info object returned by self.step and that will be given to
        # `_get_obs_extra` and `_compute_dense_reward`. Note that as everything is batched, you must return a batched array of
        # `self.num_envs` booleans (or 0/1 values) for success an dfail as done in the example below

        obj_displacement_vect = self.goal_sphere.pose.p - self.coin.pose.p

        obj_to_goal_dist = torch.linalg.norm(obj_displacement_vect, axis=1)
        is_obj_placed = obj_to_goal_dist <= self.goal_thresh
        # is_robot_static = self.agent.is_static(self.qvel_tolerance)

        qvel_mod = torch.linalg.norm(self.agent.robot.get_qvel()[..., :6], axis=1)
        is_obj_static = torch.linalg.norm(self.coin.get_linear_velocity(), axis=1) <= self.qvel_tolerance

        coin_q = Quaternion(self.coin.pose.get_q())
        coin_normal_axis_self_reference = self.coin_normal_axis.to(device=self.device)
        coin_normal_axis_world_reference = cast("torch.Tensor", coin_q.rotate_vector(coin_normal_axis_self_reference))

        is_grasped = self.agent.is_grasping(self.coin) & (torch.abs(functional.cosine_similarity(coin_normal_axis_world_reference, self.agent.gripper_travel_dir)) < 0.1)  # noqa: PLR2004

        obj_angle_dist = coin_angle(
            coin_q,
            coin_normal_axis_self_reference,
            self.coin_desired_axis.to(device=self.device),
        )

        is_angle_zero = obj_angle_dist < 45  # noqa: PLR2004

        return {
            # "success": is_obj_placed & is_obj_static & is_grasped,
            "success": is_obj_placed & is_obj_static & is_angle_zero,
            "is_obj_placed": is_obj_placed,
            "is_obj_static": is_obj_static,
            "is_grasped": is_grasped,
            "is_angle_zero": is_angle_zero,
            "obj_to_goal_dist": obj_to_goal_dist,
            "obj_displacement_vect": obj_displacement_vect,
            "obj_angle_dist": obj_angle_dist,
            "qvel_mod": qvel_mod,
        }

    @override
    def _get_obs_extra(self, info: dict):
        # should return an dict of additional observation data for your tasks
        # this will be included as part of the observation in the "extra" key when obs_mode="state_dict" or any of the visual obs_modes
        # and included as part of a flattened observation when obs_mode="state". Moreover, you have access to the info object
        # which is generated by the `evaluate` function above
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_sphere.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.coin.pose.raw_pose,
                tcp_to_obj_pos=self.coin.pose.p - self.agent.tcp_pose.p,
                obj_displacement_vect=info["obj_displacement_vect"],
                obj_angle_dist=info["obj_angle_dist"] * torch.pi / 180,
            )
        return obs

    @override
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # you can optionally provide a dense reward function by returning a scalar value here. This is used when reward_mode="dense"
        # note that as everything is batched, you must return a batch of self.num_envs rewards as done in the example below.
        # Moreover, you have access to the info object which is generated by the `evaluate` function above

        tcp_to_obj_dist = torch.linalg.norm(self.coin.pose.p - self.agent.tcp_pose.p, axis=1)
        # reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reaching_reward = torch.exp(-5 * tcp_to_obj_dist)

        obj_to_goal_dist = info["obj_to_goal_dist"]
        place_reward = torch.exp(-5 * obj_to_goal_dist)

        # static_reward = 1 - torch.tanh(self.qvel_penalty * info["qvel_mod"])
        # static_reward = torch.exp(-self.qvel_penalty * info["qvel_mod"])

        # angle_reward = torch.exp(-self.angle_penalty * info["obj_angle_dist"])
        angle_reward = torch.cos(info["obj_angle_dist"] / 180 * torch.pi)

        is_grasped = info["is_grasped"]
        # is_obj_placed = info["is_obj_placed"]
        is_angle_zero = info["is_angle_zero"]

        # reward = reaching_reward + is_grasped + place_reward * is_grasped + static_reward * is_obj_placed + angle_reward * is_obj_placed
        reward = reaching_reward + is_grasped + angle_reward * is_grasped + 2 * place_reward * is_grasped * is_angle_zero  # + static_reward * is_obj_placed

        reward[info["success"]] = 10.0

        return reward

    @override
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # this should be equal to compute_dense_reward / max possible reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0

    @override
    def get_state_dict(self):
        # this function is important in order to allow accurate replaying of trajectories. Make sure to specify any
        # non simulation state related data such as a random 3D goal position you generated
        # alternatively you can skip this part if the environment's rewards, observations, eval etc. are dependent on simulation data only
        # e.g. self.your_custom_actor.pose.p will always give you your actor's 3D position
        state = super().get_state_dict()
        # state["goal_pos"] = add_your_non_sim_state_data_here
        return state

    @override
    def set_state_dict(self, state, env_idx: torch.Tensor = None):
        # this function complements get_state and sets any non simulation state related data correctly so the environment behaves
        # the exact same in terms of output rewards, observations, success etc. should you reset state to a given state and take the same actions
        # self.goal_pos = state["goal_pos"]
        super().set_state_dict(state, env_idx)

    def get_attr(self, attr, *args):
        return [self.__dict__.get(attr)]


if __name__ == "__main__":
    from pprint import pprint

    import gymnasium as gym

    envs = gym.make(
        "FlipCoin-v1",
        robot_uids="ec63",
        num_envs=1,
        reward_mode="dense",
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
    )

    obs, info = envs.reset()

    pprint(obs, sort_dicts=False)  # noqa: T203
