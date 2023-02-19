from setuptools import find_packages, setup


def requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def version():
    version_file = 'autotabular/version.py'
    with open(version_file, encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


setup(
    name='autotabular',
    version=version(),
    packages=find_packages(exclude=(
        'tests',
        'docs',
        'examples',
        'requirements',
        '*.egg-info',
    )),
    description='AutoTabular is an automl framework for tabular datasets',
    long_description=readme(),
    long_description_content_type='text/markdown',
    license='Apache Software License 2.0',
    install_requires=requirements('requirements/requirements.txt'),
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Environment :: GPU :: NVIDIA CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence :: AutoML',
    ],
    entry_points={
        'console_scripts': [
            'autotabular-classification=autotabular.classification.train:main',
            'autotabular-regression=autotabular.regression.train:main',
        ],
    }
)
