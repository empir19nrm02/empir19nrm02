from setuptools import setup, find_packages

setup(
    name='empir19nrm02',
    version='0.0.2',
    packages= find_packages(),
    url='https://github.com/UdoKrueger/empir19nrm02',
    license='',
    author='Udo Krüger',
    author_email='udo.krueger@technoteam.de',
    description='Meausrement Uncertainty for EMPIR 19nrm02',
    keywords = ['color', 'colorimetry','photometry','CIE','spectral data', 'measurement uncertainty'],

  install_requires=[
        'numpy',
		'scipy',
		'matplotlib',
        'luxpy'
      ],
  package_data={'empir19nrm02': ['empir19nrm02/data/*.dat',
						  'empir19nrm02/data/*.txt',
						  'empir19nrm02/data/*.csv',
						  'empir19nrm02/data/spd/*.txt',
						  'empir19nrm02/data/spd/*.csv',
						  'empir19nrm02/data/res/*.txt',
						  'empir19nrm02/data/res/*.csv',
                          ]},
  include_package_data = True,
  classifiers=[
    'Development Status :: 1 - Planning',
    'Programming Language :: Python :: 3',
    ],
  python_requires='>=3.5',
)