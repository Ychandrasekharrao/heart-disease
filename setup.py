from setuptools import setup, find_packages
from typing import List

hypen_e_dot = '-e .'

def get_requirements(file_path) -> List[str]:

    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    return requirements

setup(
    name="diabetes-prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)