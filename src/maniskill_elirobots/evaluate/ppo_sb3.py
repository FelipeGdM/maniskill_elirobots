import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder

video_folder = "logs/videos/"
video_length = 50

exp_name = "flipcoin-ec63-1784343291"
exp_name = "flipcoin-ec63-1784343958"

model = PPO.load(exp_name)

env_kwargs = {
    "obs_mode": "state",
    "render_mode": "rgb_array",
    "sim_backend": "physx_cpu",
    "render_backend": "sapien_cpu",
    "control_mode": "pd_joint_delta_pos",
    "sim_config": {
        "sim_freq": 100,
        "control_freq": 20,
    },
}

# envs = gym.make(
#     "maniskill_elirobots:FlipCoin-v1",
#     robot_uids="ec63",
#     # num_envs=1,
#     **env_kwargs,
# )

envs = ManiSkillVectorEnv(
    "maniskill_elirobots:FlipCoin-v1",
    robot_uids="ec63",
    num_envs=1,
    **env_kwargs,
)

print(f"{envs.render_mode=}")

sb3_env = ManiSkillSB3VectorEnv(envs)

print(f"{sb3_env.render_mode=}")

vec_env = VecVideoRecorder(
    sb3_env,
    video_folder,
    record_video_trigger=lambda x: x == 0,
    video_length=video_length,
    name_prefix=exp_name,
)

obs = vec_env.reset()
for _ in range(video_length + 1):
    action, info = model.predict(obs)
    print(action)
    obs, info, reward, _ = vec_env.step(action)
# Save the video
vec_env.close()


# print(evaluate_policy(model, ManiSkillSB3VectorEnv(envs), n_eval_episodes=50))
