from setuptools import setup

setup(name='DeepShutdown',
      author='lccasagrande',
      version='0.2',
      python_requires='>=3.7',
      extras_require={
          'tf': ['tensorflow>=2.0.0'],
          'tf_gpu': ['tensorflow-gpu>=2.0.0'],
      },
      install_requires=[
          'gym',
          'tqdm',
          'numpy',
          'pandas',
          'joblib',
          'seaborn',
          'matplotlib',
          'cloudpickle'
      ])
