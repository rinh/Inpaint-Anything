from setuptools import find_packages, setup

setup(
    name="inpaint_anything",
    version="0.1",
    author="rinh",
    description="inpaint_anything package",
    packages=find_packages(exclude="segment_anything"),
    install_requires=[]
)