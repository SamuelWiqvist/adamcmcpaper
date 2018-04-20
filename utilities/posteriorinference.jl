# This file contains functions to compute posterior inference results

using StatsBase

# posterior quantile intervals

doc"""
    calcquantileint(data::Matrix, lower::Real=2.5,upper::Real=97.5)

Computes posterior quantile intervals, 2.5th and 97.5th quantiles as default.
"""
function calcquantileint(data::Matrix, lower::Real=2.5,upper::Real=97.5)

    # find dim. for data
    dim = minimum(size(data))

    # set nbr of intervals
    intervals = zeros(dim, 2)

    # transform data to column-major order if necessary
    if size(data)[1] > size(data)[2]
        data = data'
    end

    # calc intervals over all dimensions
    for i = 1:dim
        intervals[i,:] = quantile(data[i,:], [lower/100 upper/100])
    end

    # return intervals
    return intervals

end



doc"""
    calcquantileint(data::Matrix, lower::Real=2.5,upper::Real=97.5)

Computes posterior quantile intervals, 2.5th and 97.5th quantiles as default.
"""
function calcquantileint(data::Vector, lower::Real=2.5,upper::Real=97.5)


    # set nbr of intervals
    intervals = zeros(1, 2)

    # calc intervals over all dimensions
    intervals[1,:] = quantile(data[1,:], [lower/100 upper/100])

    # return intervals
    return intervals

end

# calc loss

doc"""
    loss(theta_true::Vector, theta_est::Matrix)
Computes the root mean square error for the parameter estimations.
"""
function loss(theta_true::Vector, Theta_est::Matrix)
  rmse = copy(theta_true)
  for i = 1:length(rmse)
    rmse[i] = sqrt(mean((theta_true[i]-Theta_est[i,:]).^2))
  end
  return rmse
end
