from os import path

from setuptools import find_packages, setup


def read_requirements_file(filename):
    file = '%s/%s' % (path.dirname(path.realpath(__file__)), filename)
    with open(file) as f:
        return [line.strip() for line in f]


with open("deepshut/__version__.py") as version_file:
    exec(version_file.read())


with open("README.rst") as readme_file:
    long_description = readme_file.read()

install_requires = read_requirements_file('requirements.txt')

setup(
    name='DeepShutdown',
    version=__version__,
    author='lccasagrande',
    author_email='lcamelocasagrande@gmail.com',
    url='https://github.com/lccasagrande/DeepShutdown',
    license='MIT',
    description="Scheduling Servers Shutdown in Grid Computing with Deep Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    python_requires='>=3.8',
    extras_require={
        'tf': ['tensorflow>=2.0.0'],
        'tf_gpu': ['tensorflow-gpu>=2.0.0'],
    },
    install_requires=install_requires,
    packages=find_packages(),
    keywords=["Deep Reinforcement Learning", "Cluster",
              "Scheduler", "Resource and Job Management"],
)
