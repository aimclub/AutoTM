import os

import pkg_resources
from setuptools import setup


def deploy_data_files():
    templates = [os.path.join("deploy", f) for f in os.listdir("deploy") if f.endswith(".j2")]
    files = templates + ["deploy/test-datasets-config.yaml"]
    return files


with open('requirements.txt') as f:
    strs = f.readlines()

install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(strs)]

setup(
    name='kube_fitness',
    version='0.1.0',
    description='kube fitness',
    package_dir={"": "kube_fitness"},
    packages=["kube_fitness"],
    install_requires=install_requires,
    include_package_data=True,
    data_files=[
        ('share/kube-fitness-data/deploy', deploy_data_files())
    ],
    scripts=['bin/kube-fitnessctl', 'bin/fitnessctl'],
    entry_points = {
        'console_scripts': ['deploy-config-generator=kube_fitness.deploy_config_generator:main'],
    }
)