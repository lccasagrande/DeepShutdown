from setuptools import setup

setup(name='DRLGrid',
      author='lccasagrande',
      version='0.0.1',
      python_requires='>=3.6',
      install_requires=[
          'gym',
          'tqdm',
          'numpy',
          'pandas',
          'joblib',
          'tensorflow-gpu',
          'seaborn',
          'matplotlib',
          'cloudpickle'
      ])
