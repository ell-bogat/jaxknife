from setuptools import setup, find_packages

# copied from orbitize
def get_requires():
    reqs = []
    for line in open('requirements.txt', 'r').readlines():
        reqs.append(line)
    return reqs

setup(
    name = 'jaxknife',
    version = '0.0',
    packages = find_packages(),
    install_requires = get_requires()
)