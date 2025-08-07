import numpy as np
from miniworld.envs.jeparoom import RearrangeOneRoom  # Adjust import path
from utils import aggregate_dct  # Assumes this exists like in pusht
import math
import gymnasium as gym


if not hasattr(gym.wrappers.TimeLimit, "__getattr__"):
    def _forward_to_inner(self, name):
        return getattr(self.env, name)        # delegate to wrapped env
    gym.wrappers.TimeLimit.__getattr__ = _forward_to_inner

class RearrangeOneRoomWrapper(RearrangeOneRoom):
    def __init__(self, size=2, seed=0, max_entities=4, **kwargs):
        super().__init__(size=size, seed=seed, max_entities=max_entities, **kwargs)
        self.action_dim = 1

    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as init, one as goal.
        You need to define what "state" means for your task.
        This example uses agent (x, z, yaw) only.
        """

    def update_env(self, env_info):
        """
        Optional: Update any metadata if needed
        """
        print("Updating environment with info:", env_info)

    def eval_state(self, goal_state, cur_state):
        """
        Evaluates success: agent (x,z) close and orientation aligned.
        """
        pass

    def get_agent_state(self):
        """
        Extracts agent position and yaw as a flat vector.
        """
        pass

    def reset(self, seed=None):
        """
        Overridden to return image obs and state vector
        """
        pass

    def prepare(self, seed, init_state):
        """
        Resets to a specific agent state
        """
        pass

    def step(self, action):
       pass

    def step_multiple(self, actions):
        print("Stepping with actions:", actions)

    def rollout(self, seed, init_state, actions):
        print("Rolling out with seed:", seed)
        print("Initial state:", init_state)
        print("Actions shape:", actions)