from setuptools import setup

setup(name='DeepShutdown',
      author='lccasagrande',
      version='0.1',
      python_requires='>=3.6',
      extras_require={
          'tf': ['tensorflow==1.14'],
          'tf_gpu': ['tensorflow-gpu==1.14'],
      },
      install_requires=[
          'gym',
          'tqdm',
          'numpy',
          'pandas',
          'joblib',
          'seaborn',
          'tqdm',
          'matplotlib',
          'cloudpickle'
      ])
