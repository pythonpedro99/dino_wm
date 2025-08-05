import numpy as np
from miniworld.envs.jeparoom import RearrangeOneRoom  # Adjust import path
from utils import aggregate_dct  # Assumes this exists like in pusht
import math

class RearrangeOneRoomWrapper(RearrangeOneRoom):
    def __init__(self, max_entities=4):
        super().__init__(max_entities=max_entities)
        self.action_dim = self.action_space.shape[0]

    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as init, one as goal.
        You need to define what "state" means for your task.
        This example uses agent (x, z, yaw) only.
        """
        rs = np.random.RandomState(seed)

        def random_state():
            x = rs.uniform(0.5, self.size_a - 0.5)
            z = rs.uniform(0.5, self.size_b - 0.5)
            yaw = rs.uniform(-np.pi, np.pi)
            return np.array([x, z, yaw], dtype=np.float32)

        return random_state(), random_state()

    def update_env(self, env_info):
        """
        Optional: Update any metadata if needed
        """
        self.shape = env_info.get("shape", None)

    def eval_state(self, goal_state, cur_state):
        """
        Evaluates success: agent (x,z) close and orientation aligned.
        """
        pos_diff = np.linalg.norm(goal_state[:2] - cur_state[:2])
        angle_diff = np.abs(goal_state[2] - cur_state[2])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        success = pos_diff < 0.4 and angle_diff < (np.pi / 10)
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {"success": success, "state_dist": state_dist}

    def get_agent_state(self):
        """
        Extracts agent position and yaw as a flat vector.
        """
        pos = self.agent.pos
        yaw = self.agent.dir
        return np.array([pos[0], pos[2], yaw], dtype=np.float32)

    def reset(self, seed=None):
        """
        Overridden to return image obs and state vector
        """
        obs, _ = super().reset(seed=seed)
        state = self.get_agent_state()
        return {"rgb": obs}, state

    def prepare(self, seed, init_state):
        """
        Resets to a specific agent state
        """
        self.seed(seed)
        self.reset_to_state = (
            init_state  # ← needs to be implemented in base env if you want full control
        )
        obs, state = self.reset(seed)
        return obs, state

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        state = self.get_agent_state()
        info["state"] = state
        return {"rgb": obs}, reward, done, info

    def step_multiple(self, actions):
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            obs, r, d, info = self.step(action)
            obses.append(obs)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.array(rewards)
        dones = np.array(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)

        # Add first frame
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        return obses, states
