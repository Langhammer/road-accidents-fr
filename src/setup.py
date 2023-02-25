from setuptools import setup

setup(
    name='roaf',
    packages=['roaf'],
    version='0.1.0',
    description='data science project road accidents in France',
    author='Kay Langhammer',
    author_email='info@kaylanghammer.de',
    url='https://github.com/Langhammer/road-accidents-fr',
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
