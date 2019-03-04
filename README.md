# Algorithm implementations and case studies for the paper *Accelerating delayed-acceptance Markov chain Monte Carlo algorithms*

This repository contains all code for the pre-print paper *Accelerating delayed-acceptance Markov chain Monte Carlo algorithms* https://arxiv.org/abs/1806.05982.

**N.B.:** The results in the pre-print at arXiv v1 are computed using the version of the code at tag *preprint_v1_master* and *preprint_v1_lunarc* for the `master` and the `lunarc` branch respectively.  

All computations are carried out on the AURORA@LUNARC http://www.lunarc.lu.se/resources/hardware/aurora/ cluster.

**The lunarc branch is depreciated, only the master branch is used now**

## File structure

The files are structured as following:

/DWPSDE model
- source files for the implementations of the algorithms for the DWP-SDE model
- scripts to run the algorithms
- run-scripts for LUNARC
- datasets and simulated data
- training data for DA and ADA

/DWPSDE model/Results
- all numerical results in .csv files

/DWPSDE model/lunarc_output
- output files from LUNARC

/DWPSDE model/analyses
- scripts for analyzing the results  

/Ricker model
- source files for the implementations of the algorithms for the Ricker model
- scripts to run the algorithms
- simulated data
- training data for DA and ADA
- run-scripts for LUNARC
- scripts for analyzing the results  

/Ricker model/Results
- all numerical results in .csv files

/Ricker model/lunarc_output
- output files from LUNARC

/adaptive update algorithms
- source files for the adaptive tuning algorithms

/gpmodel
- source files for the GP model

/select cases
- source files for the different methods to select which case to assume in the ADA algorithm

/utilities

- source files for various help functions

## Software  

#### Julia version

```julia
julia> versioninfo()
Julia Version 0.5.2
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz
  WORD_SIZE: 64
  BLAS: libmkl_rt
  LAPACK: libmkl_rt
  LIBM: libimf
  LLVM: libLLVM-3.7.1 (ORCJIT, haswell)
```

#### Packages
```julia
julia> include("utilities/print_used_packages.jl")

 - Distributions                 0.15.0
 - DataFrames                    0.10.1
 - StatsBase                     0.19.2
 - Optim                         0.7.8
 - Lasso                         0.1.0
 - StatsFuns                     0.5.0
 - JLD                           0.8.3
 - HDF5                          0.8.8
 - PyPlot                        2.3.2+             master
 - KernelDensity                 0.4.0
```
## Data

To get access to the reaction coordinate data sets, please contact the authors.
