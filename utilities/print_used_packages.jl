# simple script to print the status (i.e. version) for
# all packages that we use

used_packages =  ["Distributions";
                  "DataFrames";
                  "StatsBase";
                  "Optim";
                  "StatsFuns";
                  "JLD";
                  "HDF5";
                  "PyPlot";
                  "KernelDensity"
                  "GLM"]

for pack in used_packages
  Pkg.status(pack)
end
