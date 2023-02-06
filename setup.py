from setuptools import setup

with open("README.md", "r") as fh:
   long_description = fh.read()

setup(
	name='pysoplot',
	version='0.0.3b1',
	description='a Python geochronology library',
	license='MIT',
	packages=['pysoplot'],
	package_dir={'': 'src'},
	url="https://github.com/timpol/pysoplot",
	author="Timothy Pollard",
	author_email="pollard@student.unimelb.edu.au",
	platforms="all",
	classifiers=[
		"Development Status :: 4 - Beta",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Operating System :: MacOS :: MacOS X",
		"Operating System :: Microsoft :: Windows",
		"Operating System :: POSIX :: Linux",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3 :: Only",
		"Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
	],
	long_description=long_description,
	long_description_content_type="text/markdown",
	install_requires=[
		"numpy",
		"scipy",
		"matplotlib"
	],
	extras_require={
		"dev": []
	}
)
