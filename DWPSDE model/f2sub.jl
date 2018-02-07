function f2sub(xs::SharedArray, index::UnitRange)
  for j in index
    xs[j] = xs[j]^2
    sleep(.001)
  end
end
