from setuptools import setup

setup(name='DRLGrid',
      author='lccasagrande',
      version='0.0.1',
      python_requires='>=3.6',
      extras_require={
          'tf': ['tensorflow'],
          'tf_gpu': ['tensorflow-gpu'],
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
