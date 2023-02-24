from setuptools import find_packages, setup

setup(
    name='roaf',
    packages=['roaf'],
    version='0.1.0',
    description='data science project road accidents in France',
    author='Kay Langhammer',
    install_requires=[
        'pandas',
        'numpy>=1.14.5',
        'matplotlib>=2.2.0',
        'jupyter',
        'bokeh',
        'pyproj',
        ],
    license='',
    package_data={'roafr': ['data/schema.json']}
)
