Code used to run the algorithms on Lunarc (http://www.lunarc.lu.se/).

## Software

##### Julia version  

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

##### Packages

```julia 
julia>include("utilities/print_used_packages.jl")
 - Distributions                 0.13.0
 - DataFrames                    0.9.1
 - StatsBase                     0.15.0
 - Optim                         0.7.8
 - Lasso                         0.1.0
 - StatsFuns                     0.5.0
 - JLD                           0.6.10
 - HDF5                          0.8.1

```

