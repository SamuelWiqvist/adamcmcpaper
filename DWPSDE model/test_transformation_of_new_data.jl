
# set correct path
try
  cd("DWPSDE model")
catch
  warn("Already in the DWP-SDE folder.")
end

# load case models
cd("..")
include(pwd()*"\\select case\\selectcase.jl")
cd("DWPSDE model")

# load files and functions
include("set_up.jl")


using PyPlot
using StatsBase


## Old data

Z = load_data()

Z = Z[:]

summarystats(Z)

PyPlot.figure()
PyPlot.plot(Z)


PyPlot.figure()
PyPlot.plt[:hist](Z,50)


## New data

file = open("new_data_set.txt")
data = readlines(file)
close(file)

Z_data = zeros(length(data)-1)

idx = 2:length(data)

for i in idx
    try
        Z_data[i-1] = readdlm(IOBuffer(data[i]), Float64)[2]
    catch
        Z_data[i-1] = parse(data[i][end-1-4:end-1])
    end
end


summarystats(Z_data)

PyPlot.figure()
PyPlot.plot(Z_data)


PyPlot.figure()
PyPlot.plt[:hist](Z_data,50)


# thinned data
thinning = 100
idx_thinned = 1:thinning:length(Z_data)
Z_data = Z_data[idx_thinned]



# linear transformation of data to obtain a scaling where it is easier to
# construct the dwp model
Z_data_trans1 = 50*Z_data

summarystats(Z_data_trans1)

PyPlot.figure()
PyPlot.plot(Z_data_trans1)


PyPlot.figure()
PyPlot.plt[:hist](Z_data_trans1,50)

# linear transformation of data to obtain a scaling where it is easier to
# construct the dwp model
Z_data_trans1 = 50*Z_data
Z_data_trans1 = (Z_data-mean(Z_data_trans1) ) / (maximum(Z_data_trans1) - minimum(Z_data_trans1))

summarystats(Z_data_trans1)

PyPlot.figure()
PyPlot.plot(Z_data_trans1)

PyPlot.figure()
PyPlot.plt[:hist](Z_data_trans1,50)
