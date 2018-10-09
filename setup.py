from setuptools import setup

setup(
    name='asist-nsf-2018',
    version='0.1.0',
    description='Processing data from the ASIST-NSF-2018 experiment',
    author='Milan Curcic',
    author_email='caomaco@gmail.com',
    url='https://github.com/sustain-lab/asist-nsf-2018',
    packages=['asist_nsf_2018'],
    install_requires=['asist-python'],
    dependency_links=['git+https://github.com/sustain-lab/asist-python.git@master#egg=asist-0.1.0'],
    test_suite='asist_nsf_2018.tests',
    license='MIT'
)
