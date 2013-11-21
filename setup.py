from distutils.core import setup

setup(
    name='calliope',
    version='0.1.0',
    author='Stefan Pfenninger',
    author_email='stefan@pfenninger.org',
    description='Calliope: a multi-scale energy systems (MUSES) model',
    packages=['calliope'],
    install_requires=[
        "coopr >= 3.4.7842",
        "numpy >= 1.7.1",
        "pandas >= 0.12.0",
        "pyyaml >= 3.10"
    ],
)