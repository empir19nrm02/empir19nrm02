from setuptools import setup, find_packages

setup(
    name='empir19nrm02',
    version='0.1.7',
    packages= find_packages(),
    url='https://github.com/empir19nrm02/empir19nrm02',
    license='',
    author='Udo Krüger',
    author_email='udo.krueger@technoteam.de',
    description='Meausrement Uncertainty for EMPIR 19nrm02',
    keywords = ['color', 'colorimetry','photometry','CIE','spectral data', 'measurement uncertainty'],

	install_requires=[
        'numpy',
		'scipy',
		'matplotlib',
		'pyxll',
        'luxpy',
        'pandas',
        'dataclasses',
        'seaborn',
        'sigfig',
      ],
	package_data={'': [	'data/SPD/*.csv', 'data/SPD/*.txt',
						'data/RES/*.csv', 'data/RES/*.txt',
						'data/CORR/*.csv', 'data/CORR/*.xlsx']},
    include_package_data = True,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3',
        ],
    # normally >=3.10
    python_requires='>=3.8',
)