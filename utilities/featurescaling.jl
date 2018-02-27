# functions for scaling features

"""
    standardization!(data::Matrix)

In-place standardization of the features in the matrix `data`.
"""
function standardization!(data::Matrix)

  # find dim. for data
  dim = minimum(size(data))

  # standardize data
  if size(data)[1] > size(data)[2]
    for i = 1:dim
        data[:,i] = (data[:,i] - mean(data[:,i]))/std(data[:,i])
    end
  else
    for i = 1:dim
        data[i,:] = (data[i,:] - mean(data[i,:]))/std(data[i,:])
    end
  end

end


"""
    standardization(data::Matrix, return_column_major_order::Bool=true)

Standardization of the features in the matrix `data`.
"""
function standardization(data::Matrix, return_column_major_order::Bool=true)

  # find dim. for data
  dim = minimum(size(data))

  # transform data to column-major order if necessary
  if size(data)[1] > size(data)[2]
      data = data'
  end
  data_standardized = copy(data)

  # standadize data
  for i = 1:dim
      data_standardized[i,:] = (data[i,:] - mean(data[i,:]))/std(data[i,:])
  end

  # return data in column major order
  if !return_column_major_order
      data_standardized = data_standardized'
  end

  return data_standardized

end

"""
    standardization!(data::Vector)

Standardization of the vector `data`.
"""
function standardization(data::Vector)
  data_standardized = (data - mean(data))/std(data)
  return data_standardized
end


"""
    standardization!(data::Vector)

In-place standardization of the vector `data`.
"""
function standardization!(data::Vector)
  data[:] = (data - mean(data))/std(data)
end
