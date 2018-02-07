# This file contains the normplot function

# load packages
using PyPlot
using Distributions


# normplot

doc"""
    normplot(x)

Normal probability plot.
"""
function normplot(x::Vector)

  x = sort(x)
  p_i = zeros(length(x))
  z_i = zeros(length(x))
  prob_i = zeros(length(x))


  n = length(x)

  a = 1/3

  for i = 1:n
    p_i[i] = (i-a)/(n-a+1)
    z_i[i] = quantile(Normal(0,1), p_i[i])
    prob_i[i] = cdf(Normal(0,1), z_i[i])
  end

  PyPlot.figure(figsize=(7,5))
  PyPlot.plot(x, z_i, "b*")
  PyPlot.xlabel("Data")
  PyPlot.ylabel("Quantile")

  #=
  PyPlot.figure()
  PyPlot.plot(x, prob_i, "b*")
  PyPlot.xlabel("Data")
  PyPlot.ylabel("Probability")
  PyPlot.yscale("log")
  =#

end
