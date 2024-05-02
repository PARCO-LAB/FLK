from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="flk",
    version="0.0.2",
    description="Low-Latency Biomechanics-Aware Filter for Real-time 3D Human Pose Estimation",
    long_description=readme,
    author="Enrico Martini",
    author_email="enrico.martini@univr.it",
    url="https://github.com/PARCO-LAB/FLK.git",
    license=license,
    packages=find_packages(exclude=("test", "doc")),
    install_requires=[],
    setup_requires=[],
    tests_require=[],
    include_package_data=True,
)