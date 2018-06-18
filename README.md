# Algorithm implementations and case studies for the paper *Accelerating delayed-acceptance Markov Chain Monte Carlo algorithms*

This repository contains all code for the pre-print paper *Accelerating delayed-acceptance Markov Chain Monte Carlo algorithms* https://arxiv.org/abs/1806.05982.

**N.B.:** The results in the pre-print at arXiv v1 are computed using the version of the code at tag *preprint_v1* for both the `master` and the `lunarc` branch.  


## File structure

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

## Software  

#### Julia version

```julia
julia> versioninfo()
Julia Version 0.6.0
Commit 903644385b* (2017-06-19 13:05 UTC)
Platform Info:
OS: Windows (x86_64-w64-mingw32)
CPU: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
WORD_SIZE: 64
BLAS: libopenblas (USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell)
LAPACK: libopenblas64_
LIBM: libopenlibm
LLVM: libLLVM-3.9.1 (ORCJIT, skylake)
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
