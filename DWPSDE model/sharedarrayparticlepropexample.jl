
addprocs(3)


@fastmath @inbounds function propagate_x_v2!(xpred::Array{Float64}, nbr::Int64,nbr_calc::Float64,dt::Float64,subsample_interval::Int64,subsample_interval_calc::Float64, N::Int64,N_calc::Float64, dB::Array{Float64},A::Float64,B::Float64,c::Float64,d::Float64,f::Float64,g::Float64,power1::Float64,power2::Float64,b_const::Float64)
  for k = 1:nbr
     @simd for j = 1:N #@simd
      #xpred[j] = xpred[j] - (A*xpred[j] - f + (power2*abs(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])^(power2 - 1)*sign(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])*(g - (B*power1*abs(c - B*xpred[j])^(power1 - 1)*sign(c - B*xpred[j]))/2))/2)*dt*subsample_interval_calc/nbr_calc + b_const*dB[j,k]
      xpred[j] = xpred[j] - (A*xpred[j] - f + (power2*abs(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])^(power2 - 1)*sign(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])*(g - (B*power1*abs(c - B*xpred[j])^(power1 - 1)*sign(c - B*xpred[j]))/2))/2)*dt/nbr_calc + b_const*dB[j,k]
    end
  end
end

function propagate_x_v2_s!(xpred::SharedArray, nbr::Int64,nbr_calc::Float64,dt::Float64,subsample_interval::Int64,subsample_interval_calc::Float64, N::Int64,N_calc::Float64, dB::SharedArray,A::Float64,B::Float64,c::Float64,d::Float64,f::Float64,g::Float64,power1::Float64,power2::Float64,b_const::Float64)
  @sync for k = 1:nbr
     @parallel for j = 1:N #@simd
      #xpred[j] = xpred[j] - (A*xpred[j] - f + (power2*abs(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])^(power2 - 1)*sign(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])*(g - (B*power1*abs(c - B*xpred[j])^(power1 - 1)*sign(c - B*xpred[j]))/2))/2)*dt*subsample_interval_calc/nbr_calc + b_const*dB[j,k]
      xpred[j] = xpred[j] - (A*xpred[j] - f + (power2*abs(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])^(power2 - 1)*sign(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])*(g - (B*power1*abs(c - B*xpred[j])^(power1 - 1)*sign(c - B*xpred[j]))/2))/2)*dt/nbr_calc + b_const*dB[j,k]
    end
  end
end

function propagate_x_v2_s2!(xpred::SharedArray, index::SharedArray, nbr::Int64,nbr_calc::Float64,dt::Float64,subsample_interval::Int64,subsample_interval_calc::Float64, N::Int64,N_calc::Float64, dB::SharedArray,A::Float64,B::Float64,c::Float64,d::Float64,f::Float64,g::Float64,power1::Float64,power2::Float64,b_const::Float64)
  @sync for k = 1:nbr
    @parallel for i = 1:size(index,1)
      propagate_x_v2_s2sub!(xpred,index[i,1]:index[i,2],k,nbr,nbr_calc,dt,subsample_interval,subsample_interval_calc, N,N_calc, dB,A,B,c,d,f,g,power1,power2,b_const)
      end
    end
end



@everywhere include("propagate_x_v2_s2sub.jl")





Κ = 0.3
Γ = 0.9
B = 1.
c = 28.5
d =  4.
A = 0.01
f = 0.
g = 0.03
power1 = 1.5
power2 = 1.8
sigma =  1.9
b_const = sqrt(2.*sigma^2 / 2.)
dt = 0.01



subsample_interval = 1
N = 1000
nbr = 1


(subsample_interval_calc, nbr_calc,N_calc) = map(Float64, (subsample_interval, nbr,N))

dB = randn(N,nbr)*sqrt(dt/nbr_calc)

xpred = 20 + (28-20)*rand(N)

xpreds = SharedArray(Float64, (N,1))
dBs = SharedArray(Float64, (N,1))

xpreds[:] = xpred
dBs[:] = dB


xs2 = SharedArray(Int, (n ,1))
xs2 = SharedArray(Int, (n ,1))

nbr_per_proc = Int64(length(xs2)/nprocs())
index = SharedArray(Int,(nprocs(),2))
index[:,1] = collect(1:nbr_per_proc:length(xs))
index[:,2] = collect(nbr_per_proc:nbr_per_proc:length(xs))




@time propagate_x_v2!(xpred, nbr,nbr_calc,dt,subsample_interval,subsample_interval_calc, N,N_calc, dB,A,B,c,d,f,g,power1,power2,b_const)

@time propagate_x_v2_s!(xpreds, nbr,nbr_calc,dt,subsample_interval,subsample_interval_calc, N,N_calc, dBs,A,B,c,d,f,g,power1,power2,b_const)
