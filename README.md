# Code for the paper *Accelerating delayed-acceptance Markov Chain Monte Carlo algorithms*

This repository contains all code the draft paper *Accelerating delayed-acceptance Markov Chain Monte Carlo algorithms using Gaussian processes*.

The files (in the master branch) are structured as following

/DWPSDE model
- source files for the implementations of the algorithms for the DWP-SDE model
- scripts to run the algorithms
- datasets and simulated data
- training data for DA and ADA

/DWPSDE model/Results
- all numerical results in .csv files

/Ricker model
- source files for the implementations of the algorithms for the Ricker model
- scripts to run the algorithms
- simulated data
- training data for DA and ADA

/Ricker model/Results
- all numerical results in .csv files

/adaptive update algorithms
- source files for the adaptive tuning algorithms

/gpmodel
- source files for the GP model

/select cases
- source files for the different methods to select which case to assume in the ADA algorithm

/utilities

- source files for various help functions


The `lunarc` branch contains the code used to run the algorithms on AURORA@LUNARC  http://www.lunarc.lu.se/resources/hardware/aurora/. The source code in the `lunarc` branch is similar to the source code on the `master` branch. The `lunarc` branch also contains scripts to run the algorithms on AURORA@LUNARC, and all numerical results.
