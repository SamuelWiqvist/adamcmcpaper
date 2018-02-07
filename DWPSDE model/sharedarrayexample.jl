# add processors

addprocs(3)

# set functions

function f(x::Vector, n::Int64)
  for j = 1:n
    for i = 1:length(x)
      x[i]  = x[i]^2
      sleep(.001)
    end
  end
end

function fs(xs::SharedArray, n::Int64)
  @sync for j = 1:n
    @parallel for i = 1:length(xs)
      xs[i]  = xs[i]^2
      sleep(.001)
    end
  end
end


function fs2(xs::SharedArray, index::SharedArray, n::Int64)
  @sync for j = 1:n
    @parallel for i = 1:size(index,1)
      f2sub(xs,index[i,1]:index[i,2])
      end
    end
end


@everywhere include("f2sub.jl")


n = 500
x = collect(1:n)
#xs = SharedArray(Int, (10,1), init = S -> S[Base.localindexes(S)] = myid())
xs = SharedArray(Int, (n ,1))

xs2 = SharedArray(Int, (n ,1))

# this is not correct!!!
nbr_per_proc = Int64(length(xs2)/nprocs())
index = SharedArray(Int,(nprocs(),2))
index[:,1] = collect(1:nbr_per_proc:length(xs))
index[:,2] = collect(nbr_per_proc:nbr_per_proc:length(xs))

xs[:] = collect(1:n)
xs2[:] = collect(1:n)

f(x,1)
fs(xs,1)
fs2(xs2, index, 1)

x = collect(1:n )
xs[:] = collect(1:n )
xs2[:] = collect(1:n )

tic()
f(x,1)
x = x + 1
toc()


tic()
fs(xs,1)
xs[:] = xs[:] +1
toc()

tic()
fs2(xs2, index, 1)
xs2[:] = xs2[:] +1
toc()


println(x)
println(xs)
println(nprocs())
println(sum(x-xs))
