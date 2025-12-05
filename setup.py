from setuptools import find_packages, setup

setup(
    name="psrl",
    version="0.0.1",
    package_dir={"": "."},
    packages=find_packages(where="."),
    package_data={
        "psrl": ["trainer/config/*.yaml"],
    },
    include_package_data=True,
)
