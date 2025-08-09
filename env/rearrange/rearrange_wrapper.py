import numpy as np
from miniworld.envs.jeparoom import RearrangeOneRoom  # Adjust import path
from utils import aggregate_dct  # Assumes this exists like in pusht
import math
import gymnasium as gym
import random


if not hasattr(gym.wrappers.TimeLimit, "__getattr__"):
    def _forward_to_inner(self, name):
        return getattr(self.env, name)        # delegate to wrapped env
    gym.wrappers.TimeLimit.__getattr__ = _forward_to_inner

class RearrangeOneRoomWrapper(RearrangeOneRoom):
    def __init__(self, size=2, seed=0, max_entities=4, **kwargs):
        super().__init__(size=size, seed=seed, max_entities=max_entities, **kwargs)
        self.action_dim = 1
        self.proprio_dim = 1
        self.target_name = "object"
        self._target_idx = None

    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as init, one as goal.
        You need to define what "state" means for your task.
        This example uses agent (x, z, yaw) only.
        """
        pass
    
    def reset(self, *, seed=None, options=None):
        # keep seed behavior explicit & deterministic
        if seed is None:
            seed = getattr(self, "seed", None)

        if seed is not None:
            seed = int(seed)
            # your custom RNGs used by _gen_world()
            self.rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)

        # helpful debug line; for subprocess workers use flush or file logging
        print(f"[Wrapper.reset] seed={seed}", flush=True)

        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def _resolve_target_index(self):
      ents = list(getattr(self.unwrapped, "entities", []))
      self._target_idx = None
      if not ents or not isinstance(self.target_name, str):
          return

      # Parse "Box_3" -> cls_hint="Box", idx_hint=3 (global index)
      parts = self.target_name.split("_", 1)
      cls_hint = parts[0]
      idx_hint = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else None
      cls_hint_l = cls_hint.lower()

      # 1) Prefer exact global index if provided and valid
      if idx_hint is not None and 0 <= idx_hint < len(ents):
          ent = ents[idx_hint]
          if type(ent).__name__.lower() == cls_hint_l:
              self._target_idx = idx_hint
              return

      # 2) Otherwise, pick the first entity of that class
      for i, e in enumerate(ents):
          if type(e).__name__.lower() == cls_hint_l:
              self._target_idx = i
              return

      # 3) Last resort: first non-Agent
      for i, e in enumerate(ents):
          if type(e).__name__ != "Agent":
              self._target_idx = i
              return



    def update_env(self, env_info):
        """Reset env using the dataset seed (prefer master_seed, fallback to seed)."""
        self.target_name = env_info.get("object")
        self.seed = env_info.get("seed")
        
    def eval_state(self, goal_state, cur_state):
      """
      Evaluate success based on 3D position closeness.
      goal_state, cur_state: (3,) or (T, 3) arrays/lists with (x, y, z)
      """
      import numpy as np

      goal_state = np.array(goal_state, dtype=np.float32)
      cur_state = np.array(cur_state, dtype=np.float32)

      # If inputs are sequences over time, take the last frame
      if goal_state.ndim > 1:
          goal_state = goal_state[-1]
      if cur_state.ndim > 1:
          cur_state = cur_state[-1]

      # Distance threshold in meters
      threshold = 0.2

      dist = np.linalg.norm(goal_state - cur_state)
      success = dist <= threshold

      return {"success": success, "distance": dist}


    def get_agent_state(self):
        """
        Extracts agent position and yaw as a flat vector.
        """
        pass


    def prepare(self, seed, init_state):
        """
        Resets to a specific agent state
        """
        pass

    def step_multiple(self, actions):
        print("Stepping with actions:", actions)

    def rollout(self, _seed_unused, _init_state_unused, actions):
        """Step given actions, capture obs only. No reset, no states."""
        states = []
        print(self.seed)
        first, _ = self.reset(seed=self.seed)
        for idx, ent in enumerate(self.unwrapped.entities):
            print(f"Entity {idx}: {ent}  pos={ent.pos}")
        self._resolve_target_index
        def to_chw01(img):
            # assume HWC uint8; convert to CHW float32 in [0,1]
            img = np.asarray(img)
            if img.ndim == 2:                      # H,W -> H,W,1
                img = img[..., None]
            if img.shape[-1] in (1, 3, 4):         # H,W,C
                img = img.astype(np.float32) / 255.0
                img = img.transpose(2, 0, 1)       # C,H,W
            return img

        # initial frame (no reset here)
                         # fallback
        frames = [to_chw01(first)]
        states.append(np.array(self.unwrapped.entities[0].pos, dtype=np.float32))

        A = np.asarray(actions)
        T = A.shape[0]

        for t in range(T):
            a = A[t]
            if isinstance(self.action_space, gym.spaces.Discrete):
                a = int(a) if np.ndim(a) == 0 else int(np.asarray(a).reshape(-1)[0])
            else:
                a = np.asarray(a, dtype=np.float32)

            obs, reward, terminated, truncated, info = self.step(a)
            frames.append(to_chw01(obs))
            print(self.target_name)
            #print(self.unwrapped.entities[self._target_idx])
            states.append(np.array(self.unwrapped.entities[0].pos, dtype=np.float32))
            if terminated or truncated:
                break

        visual = np.stack(frames, axis=0)  # (T+1, C, H, W)
        proprio = np.zeros((visual.shape[0], getattr(self, "proprio_dim", 0)), dtype=np.float32)
        states = np.stack(states, axis=0)   
        
        return {"visual": visual, "proprio": proprio}, states