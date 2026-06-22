import os

config.excludes = list(getattr(config, "excludes", []))

config.excludes.extend(
    name
    for name in os.listdir(os.path.dirname(__file__))
    if name.endswith(".py")
)
