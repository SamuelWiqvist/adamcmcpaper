# simple script to print the status (i.e. version) for
# all packages that we use 

used_packages =  ["Distributions";
                  "DataFrames";
                  "StatsBase";
                  "Optim";
                  "Lasso";
                  "StatsFuns";
                  "JLD";
                  "HDF5";
                  "PyPlot"]

for pack in used_packages
  Pkg.status(pack)
end
