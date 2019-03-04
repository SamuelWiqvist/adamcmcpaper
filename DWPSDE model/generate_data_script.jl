# load files and functions
include("set_up.jl")
using PyPlot
using KernelDensity
using Distributions
# scritp

# true parameter valuse
#Κ = 0.3
#Γ = 0.9
#B = 1.
#c = 28.5
#d =  4.
#A = 0.01
#f = 0.
#g = 0.03
#power1 = 1.5
#power2 = 1.8
#sigma =  1.9

################################################################################
###        old data set                                                      ###
################################################################################

# load new data 
# load data
Z = convert(Array,readtable("DWPSDE model/Results/sim data small problem/data_usedgp_training_7_par_lunarc_simdata_4_coressimdata.csv"))
Z_data = Z[:,1]



#Z_data = load_data()

PyPlot.figure()
PyPlot.plot(Z_data)

bins = 50
PyPlot.figure()
h1 = PyPlot.plt[:hist](Z_data,bins)

# load data

Z_data = load_data()

PyPlot.figure()
PyPlot.plot(Z_data')

bins = 100
PyPlot.figure()
h1 = PyPlot.plt[:hist](Z_data',bins)



# parameter values to test
Κ = 0.3
Γ = 0.9
B = 1.
c = 28.5
d =  4.
A = 0.01
A_sign = 1
f = 0.
g = 0.03
power1 = 1.5
power2 = 1.8
sigma =  1.9

# new parameter values
#=
B = 1.;
c = 28.5;
d =  4.2;
A = 0.0052;
f = 0.;
g = 0.0610;
power1 = 1.5;
power2 = 1.8;
sigma =  1.9;
=#




# two unknown parameters
theta = log([c,d]) # true values for the unknown parameters
theta_known = [Κ, Γ, A, A_sign, B, f, g, power1, power2, sigma] # set vector with known parameters

# 7 parameters
#theta = log([0.2415 1.0075 28.4738 3.8861 1.5036 1.5500 1.6472]) # true values for the unknown parameters
#theta_known = [A,B,f,g] # set vector with known parameters


(Z_sim, dt, diff_dt, X_thinned) = generate_data(theta, theta_known,1)

PyPlot.figure()
PyPlot.plot(X_thinned)

PyPlot.figure()
PyPlot.plot(Z_sim)

bins = 100
PyPlot.figure()
h1 = PyPlot.plt[:hist](Z_sim,bins)

# compare kernel densities for real data and simulated data
Z_data_m = zeros(2,max(length(Z_data), length(Z_sim)))

for i = 1:length(Z_data)
  Z_data_m[1,i] = Z_data[i]
end

for i = 1:length(Z_sim)
  Z_data_m[2,i] = Z_sim[i]
end


k_real_data = kde(Z_data_m[1,:])
k_sim_data = kde(Z_data_m[2,:])

PyPlot.figure()
PyPlot.plot(k_real_data.x,k_real_data.density, "b")
PyPlot.hold(true)
PyPlot.plot(k_sim_data.x,k_sim_data.density, "b--")


# save data to file
Z_out = zeros(length(Z_sim),1)

for i = 1:length(Z_out)
  Z_out[i] = Z_sim[i]
end

writetable("data_old_new_dt.csv", convert(DataFrame, Z_out))



################################################################################
###          new data set                                                    ###
################################################################################

# load data
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


# linear transformation of data to obtain a scaling where it is easier to
# construct the dwp model
Z_data = 50*Z_data

# plot data

PyPlot.figure()
PyPlot.plot(1:length(Z_data), Z_data)

bins = 100
PyPlot.figure()
h1 = PyPlot.plt[:hist](Z_data,bins)

# thinned data

thinning = 100

idx_thinned = 1:thinning:length(Z_data)
Z_data_thinned = zeros(Float64, idx_thinned)

j = 0
for i in idx_thinned
  j = j + 1
  Z_data_thinned[j] = Z_data[i]
end


# plot thinned data
PyPlot.figure()
PyPlot.plot(1:length(Z_data_thinned), Z_data_thinned)

bins = 100
PyPlot.figure()
h1 = PyPlot.plt[:hist](Z_data_thinned,bins)



# simulated data


# parameter values to test
Κ = 0.5;
Γ = 0.9;
B = 1.;
c = 22.5;
d =  13;
A = -0.0025;
f = 0.;
g = 0.;
power1 = 1.3;
power2 = 1.3;
sigma =  2.6;


# two unknown parameters
theta = log([c,d]) # true values for the unknown parameters
theta_known = [Κ, Γ, A, B, f, g, power1, power2, sigma] # set vector with known parameters
#=
thinning_sim = 1

N = Int64(3.5e6/thinning_sim)
dt = 20*1e-9 # 0.005*thinning_sim#20*1e-9
dt_U = 20*1e-9 # 0.005*thinning_sim#20*1e-9
diff_dt = 1

X_sim = zeros(N+1)
U_sim = zeros(N+1)

X_sim[1] = 35
U_sim[1] = 0

dB = zeros(N+1)


for i = 1:length(dB)
  dB[i] = sqrt(dt)*randn()
end

j = 1 # index for U_sim
b_const = sqrt(2*sigma^2 / 2)

#  should also consider the cache memory!
for i = 1:N # numerical integration of the X process
  X_sim[i+1] = X_sim[i] - (A*X_sim[i] - f + (power2*abs(abs(c - B*X_sim[i])^power1/2 - d + g*X_sim[i])^(power2 - 1)*sign(abs(c - B*X_sim[i])^power1/2 - d + g*X_sim[i])*(g - (B*power1*abs(c - B*X_sim[i])^(power1 - 1)*sign(c - B*X_sim[i]))/2))/2)*dt + b_const*dB[i]
  if  mod(i, diff_dt) == 0
    U_sim[j+1] = rand(Normal( U_sim[j]*exp(-Κ*dt_U), sqrt( Γ^2*( 1 - exp( -2*Κ*dt_U ) ) ) ),1)[1]
    j = j + 1
  end
end

X_thinned = X_sim[1:diff_dt:end]
Z_sim = X_thinned + U_sim
=#

Z_sim, X_thinned = generate_data(theta, theta_known)

PyPlot.figure()
PyPlot.plot(X_thinned)

PyPlot.figure()
PyPlot.plot(Z_sim)

bins = 100
PyPlot.figure()
h1 = PyPlot.plt[:hist](Z_sim,bins)


# compare kernel densities for real data and simulated data
Z_data_m = zeros(2,max(length(Z_data_thinned), length(Z_sim)))

for i = 1:length(Z_data_thinned)
  Z_data_m[1,i] = Z_data_thinned[i]
end

for i = 1:length(Z_sim)
  Z_data_m[2,i] = Z_sim[i]
end


k_real_data = kde(Z_data_m[1,:])
k_sim_data = kde(Z_data_m[2,:])

PyPlot.figure()
PyPlot.plot(k_real_data.x,k_real_data.density, "b")
PyPlot.hold(true)
PyPlot.plot(k_sim_data.x,k_sim_data.density, "b--")


# save data to file
Z_out = zeros(length(Z_sim),1)

for i = 1:length(Z_out)
  Z_out[i] = Z_sim[i]
end

writetable("data_new.csv", convert(DataFrame, Z_out))
