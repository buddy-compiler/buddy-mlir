import os

config.excludes = list(getattr(config, "excludes", [])) + ["set_model_env.sh"]

config.excludes.extend(
    name
    for name in os.listdir(os.path.dirname(__file__))
    if name.endswith(".py")
)
