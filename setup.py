from setuptools import setup

setup(
    name='RTPSpy',
    packages=['rtpspy'],
    package_data={'rtpspy': ['librtp.so']},
    version='1.0.1',
    author='Masaya Misaki',
    author_email='mamisaki@gmail.com',
    description='fMRI Real-Time Processing System in python',
    url='https://github.com/mamisaki/RTPSpy',
    classifiers=['Programming Language :: Python :: 3',
                 'License :: OSI Approved :: GNU GPLv3',
                 'Operating System :: Linux'],
    python_requires='>=3.10',
    scripts=['rtpspy/ants_run.py']
)
