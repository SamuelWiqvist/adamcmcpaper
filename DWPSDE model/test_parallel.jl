addprocs(4)

@everywhere function f(i, j, k)
#  sleep(0.001)
  i^2*j - 10*k
end

N = 1000

c = 1:N
k = 5

b = zeros(N)
tic()
for i = 1:N
  b[i] = f(i, c[i], k)
end
toc()



tic()
a = SharedArray{Float64}(N)
@parallel for i = 1:N
  a[i] = f(i, c[i], k)
end
toc()
