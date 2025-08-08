import hydra
from omegaconf import OmegaConf
import hydra
print(hydra.__file__)
print(dir(hydra))

@hydra.main(config_path=None)
def register_resolvers(cfg):
    pass

# Define the resolver function
# def replace_slash(value: str) -> str:
#     return value.replace('/', '_')

def replace_slash(value):
    if value is None:
        return "none"
    return value.replace('/', '_')

# Register the resolver with Hydra
OmegaConf.register_new_resolver("replace_slash", replace_slash)

if __name__ == "__main__":
    register_resolvers()

