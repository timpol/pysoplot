# Pysoplot

[![license](https://img.shields.io/github/license/timpol/pysoplot.svg)](https://github.com/timpol/pysoplot/blob/master/LICENSE.txt)

[//]: # ([![DOI]&#40;https://zenodo.org/badge/DOI/....svg&#41;]&#40;https://doi.org/...&#41;)

**Pysoplot** is a Python library that provides basic functions and routines for geochronology. It implements some of the core functionality of the popular (and now defunct) Isoplot/Ex. software, but also includes new algorithms and routines. Pysoplot is intended to be used to build custom geochronology scripts and data processing routines in order to meet individual requirements.

Pysoplot includes functions for:
* performing linear regression on 2-D data using the model 1, 2 and 3 algorithms popularised by Isoplot/Ex.
* performing linear regression on 2-D data using the robust spine algorithm of Powell et al. (2020) and a new "robust model 2" algorithm
* computing weighted averages using algorithms based on classical and robust statistics
* computing classical isochron and U-Pb concordia-intercept ages
* plotting isochron diagrams
* computing disequilibrium U-Pb ages
* plotting equilibrium and disequilibriam concordia curves, age ellipses, and uncertainty envelopes

* computing age uncertainties using Monte Carlo methods

For more info, see the online [documentation](https://timpol.github.io/pysoplot/).

## Installation

Run the following to install:

```python
pip install pysoplot
```

## Example usage
```python
import pysoplot as pp

# get Tera-Wasserburg test dataset 
dp = pp.data.LA0708
# transform data point errors from 2 sigma to 1 sigma absolute
dp = pp.transform.dp_errors(dp, 'abs2s')

# regress data
fit = pp.regression.robust_fit(*dp, plot=True, diagram='tw')
pp.misc.print_result(fit, 'Regression results')
fit['fig'].show()

# compute Tera-Wasserburg concordia-intercept age
result = pp.upb.concint_age(fit, method='Powell')
print(f"age: {result['age']:.2f} +/- {result['age_95pm']:.2f}")
```

## Acknowledgements

Acknowledgement of all third-party algorithms implemented in Pysoplot with links to publications will be added here soon... 

## License

Pysoplot is distributed under the MIT license.

## Contact

[Timothy Pollard](mailto:pollard@student.unimelb.edu.au)

