from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.install_lib import install_lib

class CustomBuildPy(build_py):

    def initialize_options(self):
        super().initialize_options()
        self.build_lib = "build/python_packages"


setup(
    name="buddy",
    version="0.0.1",
    packages=["buddy", "buddy.graph", "buddy.graph.transform", "buddy.ops"],
    package_dir={
        "buddy": "frontend/Python",
        "buddy.graph": "frontend/Python/graph",
        "buddy.graph.transform": "frontend/Python/graph/transform",
        "buddy.ops": "frontend/Python/ops"
    },
    include_package_data=False,
    # cmdclass={"build_py": CustomBuildPy}
)
