from setuptools import setup, find_packages

setup(
    name='mlektic',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.3',
        'pandas>=2.2.2',
        'polars>=1.4.1',
        'scikit_learn>=1.3.0',
        'tensorflow>=2.17.0',
        'tensorflow_intel>=2.17.0',
        'matplotlib>=3.9.0',
        'ipython>=8.20.0'],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'setuptools>=72.1.0'
        ],
    },
    author='Daniel Antonio García Escobar',
    author_email='contacto@dialektico.com',
    description='Machine learning library',
    url='https://github.com/DanielDialektico/mlektic',
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: Apache 20.0',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)