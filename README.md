# Pysoplot

**Pysoplot** is a Python library that provides basic functions and routines for geochronology. It implements some of the core functionality of the popular (and now defunct) Isoplot/Ex. software, but also includes new algorithms and routines. Pysoplot is intended to be used by geochronologists to build custom scripts and data processing routines in order to meet individual requirements.

Pysoplot includes functions for:
* regressing 2-D data using the model 1, 2 and 3 algorithms popularised by Isoplot/Ex.
* regressing 2-D data using the robust spine algorithm of Powell et al. (2020) and a new "robust model 2" algorithm
* computing weighted averages using algorithms based on classical and robust statistics
* computing classical isochron and U-Pb concordia-intercept ages
* plotting isochron diagrams
* computing disequilibrium U-Pb ages
* plotting equilibrium and disequilibriam concordia curves, age ellipses, and uncertainty envelopes

* computing age uncertainties using Monte Carlo methods

Full code documentation is coming soon.

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

Timothy Pollard - pollard@student.unimelb.edu.au

