import gymnasium as gym
from gymnasium.envs.registration import register


register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
register(
    id="rearrange",
    entry_point="env.rearrange.miniworld.envs.jeparoom:RearrangeOneRoom",
    kwargs={"size": 12},
    max_episode_steps=250,
)

env_id = "rearrange"

if env_id in gym.envs.registry:
    print(f"✅ '{env_id}' is registered.")
else:
    print(f"❌ '{env_id}' is NOT registered.")
