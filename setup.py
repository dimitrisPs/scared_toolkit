#!/usr/bin/env python3

from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(name='scaredtk',
    version='0.0.1',
    description='collection of tools to manipulate MICCAI EndoVis 2019 SCARED dataset',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dimitrisps/scared-toolkit',
    author='Dimitris Psychogyios',
    author_email='d.psychogyios@gmail.com',
    scripts=[
        'scripts/aggregate_keyframes.py',
        'scripts/aggregate_keyframes.py',
        'scripts/disparity_to_original_depthmap.py',
        'scripts/evaluate.py',
        'scripts/extract_sequence_dataset.py',
        'scripts/generate_flow_sequence.py',
        'scripts/generate_keyframe_dataset.py',
        'scripts/generate_sequence_dataset.py'
    ],
    packages=['scaredtk'],
    install_requires=[
        'opencv-python>=4',
        'tifffile',
        'tqdm',
        'numpy',
        'plyfile',
        'pandas'
    ],
    python_requires='>=3.6, <4',
    classifiers=[
        'Environment :: Console',
        
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities',

        'License :: OSI Approved :: MIT License',

        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='SCARED, toolkit',
    project_urls={
        'Bug Reports': 'https://github.com/dimitrisps/scared-toolkit',
        'Source': 'https://github.com/dimitrisps/scared-toolkit',
    },
)