# this file is only used to plot the results computed at luanrc

using PyPlot
using DataFrames

# scaled up problem
alg_prop_da = Matrix(readtable("DWPSDE model/analyses/alg prop/alg_prop_da_simdatadt.csv"))
alg_prop_ada = Matrix(readtable("DWPSDE model/analyses/alg prop/alg_prop_ada_simdatadt.csv"))

# small problem
alg_prop_da = Matrix(readtable("DWPSDE model/analyses/alg prop/alg_prop_da_smallsimdatadt.csv"))
alg_prop_ada = Matrix(readtable("DWPSDE model/analyses/alg prop/alg_prop_ada_smallsimdatadt.csv"))


function print_stats(x::Vector)
  @printf "--------------------------\n"
  @printf "Mean:           %.4f\n"  mean(x)
  @printf "Std:            %.4f\n"  std(x)
  describe(x)
end

# text and lable size
text_size = 25
label_size = 20

# run time

println("Runtime:")
print_stats(alg_prop_da[:,1])
print_stats(alg_prop_ada[:,1])


# box plot
run_times = zeros(size(alg_prop_da,1), 2)
run_times[:,1] = alg_prop_da[:,1]
run_times[:,2] = alg_prop_ada[:,1]

PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.boxplot(run_times)
PyPlot.xticks([1, 2], ["DA", "ADA"])
ax[:tick_params]("both",labelsize = label_size)

# histogram
PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.plt[:hist](alg_prop_da[:,1],10, alpha = 0.6)
PyPlot.plt[:hist](alg_prop_ada[:,1],10, alpha = 0.6)
ax[:tick_params]("both",labelsize = label_size)

speed_up = alg_prop_da[:,1]./alg_prop_ada[:,1]
print_stats(speed_up)

PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.plt[:hist](speed_up,20, alpha = 0.6)
ax[:tick_params]("both",labelsize = label_size)


# nbr pf eval
println("Nbr pf eval:")
print_stats(alg_prop_da[:,2])
print_stats(alg_prop_ada[:,2])

# box plot
pf_eval = zeros(size(alg_prop_da,1), 2)
pf_eval[:,1] = alg_prop_da[:,2]
pf_eval[:,2] = alg_prop_ada[:,2]

PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.boxplot(pf_eval)
PyPlot.xticks([1, 2], ["DA", "ADA"])
ax[:tick_params]("both",labelsize = label_size)

# histogram
PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.plt[:hist](alg_prop_da[:,2],10, alpha = 0.6)
PyPlot.plt[:hist](alg_prop_ada[:,2],10, alpha = 0.6)
ax[:tick_params]("both",labelsize = label_size)

diff_nbr_pf = alg_prop_da[:,2] - alg_prop_ada[:,2]

print_stats(diff_nbr_pf)

PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.plt[:hist](diff_nbr_pf,10, alpha = 0.6)
ax[:tick_params]("both",labelsize = label_size)

# nbr pf eval in secound stage

println("Nbr pf eval in secound stage:")

print_stats(alg_prop_da[:,3])
print_stats(sum(alg_prop_ada[:,end-3:end],2)[:])

pf_eval = zeros(size(alg_prop_da,1), 2)
pf_eval[:,1] = alg_prop_da[:,3]
pf_eval[:,2] = sum(alg_prop_ada[:,end-3:end],2)

PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.boxplot(pf_eval)
ax[:tick_params]("both",labelsize = label_size)


PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.plt[:hist](alg_prop_da[:,3],10, alpha = 0.6)
PyPlot.plt[:hist](sum(alg_prop_ada[:,end-3:end],2)[:],10, alpha = 0.6)
ax[:tick_params]("both",labelsize = label_size)

diff_nbr_pf_secound_stage = alg_prop_da[:,3]./sum(alg_prop_ada[:,end-3:end],2)[:]

print_stats(diff_nbr_pf_secound_stage)

PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.plt[:hist](diff_nbr_pf_secound_stage,20, alpha = 0.6)
ax[:tick_params]("both",labelsize = label_size)


# nbr ord. mh.

print_stats(alg_prop_da[:,end])
print_stats(alg_prop_ada[:,3])

PyPlot.figure()
ax = axes()
PyPlot.plt[:hist](alg_prop_da[:,end],5, alpha = 0.6)
PyPlot.plt[:hist](alg_prop_ada[:,3],5, alpha = 0.6)
ax[:tick_params]("both",labelsize = label_size)

# nbr times in secound stage


print_stats(alg_prop_da[:,3])
print_stats(sum(alg_prop_ada[:,4:5],2)[:])

PyPlot.figure()
ax = axes()
PyPlot.plt[:hist](alg_prop_da[:,3],5, alpha = 0.6)
PyPlot.plt[:hist](sum(alg_prop_ada[:,4:5],2),5, alpha = 0.6)
ax[:tick_params]("both",labelsize = label_size)



# nbr cases1_3 and cases2_4

print_stats(alg_prop_ada[:,4])
print_stats(alg_prop_ada[:,5])

PyPlot.figure()
PyPlot.plt[:hist](alg_prop_ada[:,4],5, alpha = 0.6)

PyPlot.figure()
PyPlot.plt[:hist](alg_prop_ada[:,5],5, alpha = 0.6)


# analyses of cases

# nbr cases
print_stats(alg_prop_ada[:,6])
print_stats(alg_prop_ada[:,7])
print_stats(alg_prop_ada[:,8])
print_stats(alg_prop_ada[:,9])

PyPlot.figure()
PyPlot.plt[:hist](alg_prop_ada[:,6],5, alpha = 0.6)
PyPlot.figure()
PyPlot.plt[:hist](alg_prop_ada[:,7],5, alpha = 0.6)
PyPlot.figure()
PyPlot.plt[:hist](alg_prop_ada[:,8],5, alpha = 0.6)
PyPlot.figure()
PyPlot.plt[:hist](alg_prop_ada[:,9],5, alpha = 0.6)

# pf eval per case
print_stats(alg_prop_ada[:,10])
print_stats(alg_prop_ada[:,11])
print_stats(alg_prop_ada[:,12])
print_stats(alg_prop_ada[:,13])

PyPlot.figure()
PyPlot.plt[:hist](alg_prop_ada[:,10],5, alpha = 0.6)
PyPlot.figure()
PyPlot.plt[:hist](alg_prop_ada[:,11],5, alpha = 0.6)
PyPlot.figure()
PyPlot.plt[:hist](alg_prop_ada[:,12],5, alpha = 0.6)
PyPlot.figure()
PyPlot.plt[:hist](alg_prop_ada[:,13],5, alpha = 0.6)

# % pf eval per case

percentage_pf_eval = alg_prop_ada[:,10:13]./alg_prop_ada[:,6:9]

print_stats(percentage_pf_eval[:,1])
print_stats(percentage_pf_eval[:,2])
print_stats(percentage_pf_eval[:,3])
print_stats(percentage_pf_eval[:,4])

PyPlot.figure()
PyPlot.plt[:hist](percentage_pf_eval[:,1],5, alpha = 0.6)
PyPlot.figure()
PyPlot.plt[:hist](percentage_pf_eval[:,2],5, alpha = 0.6)
PyPlot.figure()
PyPlot.plt[:hist](percentage_pf_eval[:,3],5, alpha = 0.6)
PyPlot.figure()
PyPlot.plt[:hist](percentage_pf_eval[:,4],5, alpha = 0.6)
