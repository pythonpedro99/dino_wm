import gymnasium as gym

def _fwd(self, name):
    """Forward unknown attributes down through .env links."""
    inner = object.__getattribute__(self, "env")      
    while True:
        if hasattr(inner, name):
            return getattr(inner, name)                 
        if hasattr(inner, "env"):
            inner = object.__getattribute__(inner, "env")
            continue
        raise AttributeError(
            f"'{type(self).__name__}' and its inner envs have no attribute '{name}'"
        )

if not hasattr(gym.wrappers.TimeLimit, "__getattr__"):
    gym.wrappers.TimeLimit.__getattr__ = _fwd