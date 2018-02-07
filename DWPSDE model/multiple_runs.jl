# set correct folder

# load files and functions
include("set_up.jl")

# set parameters
nbr_iterations = 10000
nbr_particels = 25
burn_in = 5000
nbr_of_cores = 4

################################################################################
##                         set model parameters                               ##
################################################################################

problem_normal_prior_est_2_1_noadapt = set_up_problem(nbr_of_unknown_parameters=2)
problem_normal_prior_est_2_1_noadapt.alg_param.R = nbr_iterations
problem_normal_prior_est_2_1_noadapt.alg_param.N = nbr_particels
problem_normal_prior_est_2_1_noadapt.alg_param.burn_in = burn_in
problem_normal_prior_est_2_1_noadapt.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_2_1_noadapt.adaptive_update = noAdaptation(1e-2, eye(2)])

# use non identy matrix
problem_normal_prior_est_2_1_noadapt = set_up_problem(nbr_of_unknown_parameters=2)
problem_normal_prior_est_2_1_noadapt.alg_param.R = nbr_iterations
problem_normal_prior_est_2_1_noadapt.alg_param.N = nbr_particels
problem_normal_prior_est_2_1_noadapt.alg_param.burn_in = burn_in
problem_normal_prior_est_2_1_noadapt.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_2_1_noadapt.adaptive_update = noAdaptation(1e-2, 2*[1 0; 0 25])


# AM_w_eps

problem_normal_prior_est_2_1_AM_w_eps = set_up_problem(nbr_of_unknown_parameters=2)
problem_normal_prior_est_2_1_AM_w_eps.alg_param.R = nbr_iterations
problem_normal_prior_est_2_1_AM_w_eps.alg_param.N = nbr_particels
problem_normal_prior_est_2_1_AM_w_eps.alg_param.burn_in = burn_in
problem_normal_prior_est_2_1_AM_w_eps.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_2_1_AM_w_eps.adaptive_update = AMUpdate_w_eps(1e-2*eye(2), 1e-2, 200,1,1e-10)


# parameter settings ok!
problem_normal_prior_est_2_1_AM = set_up_problem(nbr_of_unknown_parameters=2)
problem_normal_prior_est_2_1_AM.alg_param.R = nbr_iterations
problem_normal_prior_est_2_1_AM.alg_param.N = nbr_particels
problem_normal_prior_est_2_1_AM.alg_param.burn_in = burn_in
problem_normal_prior_est_2_1_AM.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_2_1_AM.adaptive_update =  AMUpdate(eye(2), 2.38/sqrt(2), 1, 0.6, 25) # 2.38/sqrt(2)

# parameter settings ok!
problem_normal_prior_est_2_1_AM_gen = set_up_problem(nbr_of_unknown_parameters=2)
problem_normal_prior_est_2_1_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_2_1_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_2_1_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_2_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores

# old settings:
problem_normal_prior_est_2_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(2), 2.38/sqrt(2), 0.1, 1, 0.6, 25)
# new settings:
problem_normal_prior_est_2_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(2), 2.38/sqrt(2), 0.24, 1, 0.7, 25)

#problem_normal_prior_est_2_1_AM_gen.adaptive_update =  AMUpdate_gen(0.1*eye(2), 2/sqrt(2), 0.238, 0.5, 0.5, 25)

# parameters do not work for this case
problem_normal_prior_est_2_1_AM_comp_w = set_up_problem(nbr_of_unknown_parameters=2)
problem_normal_prior_est_2_1_AM_comp_w.alg_param.R = nbr_iterations
problem_normal_prior_est_2_1_AM_comp_w.alg_param.N = nbr_particels
problem_normal_prior_est_2_1_AM_comp_w.alg_param.burn_in = burn_in
problem_normal_prior_est_2_1_AM_comp_w.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_2_1_AM_comp_w.adaptive_update =  AMUpdate_comp_w(eye(2), 2.4/sqrt(2), 0.238, 1, 0.7, 100)


# parameters do not work for this case
problem_normal_prior_est_2_1_AM_gen_comp_w = set_up_problem(nbr_of_unknown_parameters=2)
problem_normal_prior_est_2_1_AM_gen_comp_w.alg_param.R = nbr_iterations
problem_normal_prior_est_2_1_AM_gen_comp_w.alg_param.N = nbr_particels
problem_normal_prior_est_2_1_AM_gen_comp_w.alg_param.burn_in = burn_in
problem_normal_prior_est_2_1_AM_gen_comp_w.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_2_1_AM_gen_comp_w.adaptive_update =  AMUpdate_gen_comp_w(eye(2), [1.8/sqrt(2) 1.8/sqrt(2)], 0.238, 1, 0.8, 50)

# set new data

theta_true = problem_normal_prior_est_2_1_AM.model_param.theta_true
theta_known = problem_normal_prior_est_2_1_AM.model_param.theta_known
(Z, dt, diff_dt) = generate_data(theta_true, theta_known)
Gadfly.plot(x = 1:length(Z), y= Z,  Geom.line)
problem_normal_prior_est_2_1_AM.data.Z = Z
problem_normal_prior_est_2_1_AM_gen.data.Z = Z

# set to thinned data
problem_normal_prior_est_2_1_AM_gen.alg_param.nbr_x0 = 2
problem_normal_prior_est_2_1_AM_gen.alg_param.nbr_x = 8

problem_normal_prior_est_2_1_AM_gen.alg_param.subsample_int = 4
problem_normal_prior_est_2_1_AM_gen.data.Z = Z_thinned


# priors: normal dist on non-log-scale

problem_normal_nonlog_est_2_1_noadapt = set_up_problem(nbr_of_unknown_parameters=2, prior_dist="nonlog")
problem_normal_nonlog_est_2_1_noadapt.alg_param.R = nbr_iterations
problem_normal_nonlog_est_2_1_noadapt.alg_param.N = nbr_particels
problem_normal_nonlog_est_2_1_noadapt.alg_param.burn_in = burn_in
problem_normal_nonlog_est_2_1_noadapt.alg_param.nbr_of_cores = nbr_of_cores

problem_normal_nonlog_est_2_1_noadapt.adaptive_update = noAdaptation(2.38/sqrt(2), eye(2))



problem_nonlog_prior_est_2_1_AP = set_up_problem(nbr_of_unknown_parameters=2, prior_dist="nonlog")
problem_nonlog_prior_est_2_1_AP.alg_param.R = nbr_iterations
problem_nonlog_prior_est_2_1_AP.alg_param.N = nbr_particels
problem_nonlog_prior_est_2_1_AP.alg_param.burn_in = burn_in
problem_normal_prior_est_2_1_AM.alg_param.nbr_of_cores = nbr_of_cores

problem_nonlog_prior_est_2_1_AP.adaptive_update = APUpdate(100,100,0.1,2.38/sqrt(2))

# works well
problem_nonlog_prior_est_2_1_AM = set_up_problem(nbr_of_unknown_parameters=2, prior_dist="nonlog")
problem_nonlog_prior_est_2_1_AM.alg_param.R = nbr_iterations
problem_nonlog_prior_est_2_1_AM.alg_param.N = nbr_particels
problem_nonlog_prior_est_2_1_AM.alg_param.burn_in = burn_in
problem_normal_prior_est_2_1_AM.alg_param.nbr_of_cores = nbr_of_cores

problem_nonlog_prior_est_2_1_AM.adaptive_update =  AMUpdate(eye(2), 2.4/sqrt(2), 1, 0.7, 50)

# workes somewhat well
problem_nonlog_prior_est_2_1_AM_gen = set_up_problem(nbr_of_unknown_parameters=2, prior_dist="nonlog")
problem_nonlog_prior_est_2_1_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_2_1_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_2_1_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_2_1_AM.alg_param.nbr_of_cores = nbr_of_cores

# old settings problem_nonlog_prior_est_2_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(2), 2/sqrt(2), 0.238, 1, 0.4, 50)
problem_nonlog_prior_est_2_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(2), 2/sqrt(2), 0.238, 0.7, 0.5, 25)

# problem with parameters...
problem_nonlog_prior_est_2_1_AM_gen_cw = set_up_problem(nbr_of_unknown_parameters=2, prior_dist="nonlog")
problem_nonlog_prior_est_2_1_AM_gen_cw.alg_param.R = nbr_iterations
problem_nonlog_prior_est_2_1_AM_gen_cw.alg_param.N = nbr_particels
problem_nonlog_prior_est_2_1_AM_gen_cw.alg_param.burn_in = burn_in
problem_nonlog_prior_est_2_1_AM_gen_cw.alg_param.nbr_of_cores = nbr_of_cores

problem_nonlog_prior_est_2_1_AM_gen_cw.adaptive_update =  AMUpdate_gen_comp_w(eye(2), [2/sqrt(2) 2/sqrt(2)], 0.238, 1, 0.4, 50)

problem_nonlog_prior_est_2_1_AM.data.Z = Z
problem_nonlog_prior_est_2_1_AM_gen.data.Z = Z

# using thinned data
problem_nonlog_prior_est_2_1_AM_gen.alg_param.nbr_x0 = 1
problem_nonlog_prior_est_2_1_AM_gen.alg_param.nbr_x = 4
problem_nonlog_prior_est_2_1_AM_gen.alg_param.subsample_int = 4
problem_nonlog_prior_est_2_1_AM_gen.data.Z = Z_thinned



# est 3 param


problem_normal_prior_est_3_1_noadapt = set_up_problem(nbr_of_unknown_parameters=3)
problem_normal_prior_est_3_1_noadapt.alg_param.R = nbr_iterations
problem_normal_prior_est_3_1_noadapt.alg_param.N = nbr_particels
problem_normal_prior_est_3_1_noadapt.alg_param.burn_in = burn_in
problem_normal_prior_est_3_1_noadapt.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_3_1_noadapt.adaptive_update = noAdaptation(1e-2, eye(3))



# use non identy matrix
problem_normal_prior_est_3_1_noadapt = set_up_problem(nbr_of_unknown_parameters=3)
problem_normal_prior_est_3_1_noadapt.alg_param.R = nbr_iterations
problem_normal_prior_est_3_1_noadapt.alg_param.N = nbr_particels
problem_normal_prior_est_3_1_noadapt.alg_param.burn_in = burn_in
problem_normal_prior_est_3_1_noadapt.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_3_1_noadapt.adaptive_update = noAdaptation(1e-2, 2*[50 0 0; 0 1 0; 0 0 25])




# normal prior
problem_normal_prior_est_3_1_AP = set_up_problem(nbr_of_unknown_parameters=3)
problem_normal_prior_est_3_1_AP.alg_param.R = nbr_iterations
problem_normal_prior_est_3_1_AP.alg_param.N = nbr_particels
problem_normal_prior_est_3_1_AP.alg_param.burn_in = burn_in
problem_normal_prior_est_3_1_AP.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_3_1_AP.adaptive_update = APUpdate(100,100,0.1,2.4/sqrt(3))

problem_normal_prior_est_3_1_AM = set_up_problem(nbr_of_unknown_parameters=3)
problem_normal_prior_est_3_1_AM.alg_param.R = nbr_iterations
problem_normal_prior_est_3_1_AM.alg_param.N = nbr_particels
problem_normal_prior_est_3_1_AM.alg_param.burn_in = burn_in
problem_normal_prior_est_3_1_AM.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_3_1_AM.adaptive_update =  AMUpdate([0.1 0 0; 0 0.2 0; 0 0 0.2], 2/sqrt(3), 1, 0.5, 25)

problem_normal_prior_est_3_1_AM_gen = set_up_problem(nbr_of_unknown_parameters=3)
problem_normal_prior_est_3_1_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_3_1_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_3_1_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_3_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
# old settings:
#problem_normal_prior_est_3_1_AM_gen.adaptive_update =  AMUpdate_gen([0.05 0 0; 0 0.1 0; 0 0 0.1], 2/sqrt(3), 0.238, 0.5, 0.5, 25)
# new settings:
problem_normal_prior_est_3_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.24, 1, 0.7, 25)

problem_normal_prior_est_3_1_AM_gen_cw = set_up_problem(nbr_of_unknown_parameters=3)
problem_normal_prior_est_3_1_AM_gen_cw.alg_param.R = nbr_iterations
problem_normal_prior_est_3_1_AM_gen_cw.alg_param.N = nbr_particels
problem_normal_prior_est_3_1_AM_gen_cw.alg_param.burn_in = burn_in
problem_normal_prior_est_3_1_AM_gen_cw.alg_param.nbr_of_cores = nbr_of_cores

problem_normal_prior_est_3_1_AM_gen_cw.adaptive_update =  AMUpdate_gen_comp_w(eye(3), [2/sqrt(2) 2/sqrt(2)  2/sqrt(2)], 0.238, 1, 0.4, 50)


problem_normal_prior_est_3_1_AM.data.Z = Z
problem_normal_prior_est_3_1_AM_gen.data.Z = Z_thinned


# non-log-scale prior

problem_nonlog_prior_est_3_1_AP = set_up_problem(nbr_of_unknown_parameters=3, prior_dist="nonlog")
problem_nonlog_prior_est_3_1_AP.alg_param.R = nbr_iterations
problem_nonlog_prior_est_3_1_AP.alg_param.N = nbr_particels
problem_nonlog_prior_est_3_1_AP.alg_param.burn_in = burn_in
problem_nonlog_prior_est_3_1_AP.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_3_1_AP.adaptive_update = APUpdate(100,100,0.1,2.4/sqrt(3))

# works ok
problem_nonlog_prior_est_3_1_AM = set_up_problem(nbr_of_unknown_parameters=3, prior_dist="nonlog")
problem_nonlog_prior_est_3_1_AM.alg_param.R = nbr_iterations
problem_nonlog_prior_est_3_1_AM.alg_param.N = nbr_particels
problem_nonlog_prior_est_3_1_AM.alg_param.burn_in = burn_in
problem_nonlog_prior_est_3_1_AM.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_3_1_AM.adaptive_update =  AMUpdate([0.1 0 0; 0 0.2 0; 0 0 0.2], 2/sqrt(3), 1, 0.5, 25)

# quite bad param estimations
problem_nonlog_prior_est_3_1_AM_gen = set_up_problem(nbr_of_unknown_parameters=3, prior_dist="nonlog")
problem_nonlog_prior_est_3_1_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_3_1_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_3_1_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_3_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores

# old settings:
#problem_nonlog_prior_est_3_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(3), 2/sqrt(3), 0.238, 1, 0.4, 50)
# new settings:
problem_nonlog_prior_est_3_1_AM_gen.adaptive_update =  AMUpdate_gen([0.05 0 0; 0 0.1 0; 0 0 0.1], 2/sqrt(3), 0.238, 0.5, 0.5, 25)
problem_nonlog_prior_est_3_2_AM_gen = set_up_problem(nbr_of_unknown_parameters=3, prior_dist="nonlog")
problem_nonlog_prior_est_3_2_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_3_2_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_3_2_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_3_2_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_3_2_AM_gen.adaptive_update =  AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.238, 1, 0.7, 50)


# quite bad param estimations
problem_nonlog_prior_est_3_1_AM_cw = set_up_problem(nbr_of_unknown_parameters=3, prior_dist="nonlog")
problem_nonlog_prior_est_3_1_AM_cw.alg_param.R = nbr_iterations
problem_nonlog_prior_est_3_1_AM_cw.alg_param.N = nbr_particels
problem_nonlog_prior_est_3_1_AM_cw.alg_param.burn_in = burn_in
problem_nonlog_prior_est_3_1_AM_cw.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_3_1_AM_cw.adaptive_update =  AMUpdate_comp_w(eye(3), [2/sqrt(3) 2/sqrt(3)  2/sqrt(3)], 0.238, 1, 0.4, 50)


# quite bad param estimations
problem_nonlog_prior_est_3_1_AM_gen_cw = set_up_problem(nbr_of_unknown_parameters=3, prior_dist="nonlog")
problem_nonlog_prior_est_3_1_AM_gen_cw.alg_param.R = nbr_iterations
problem_nonlog_prior_est_3_1_AM_gen_cw.alg_param.N = nbr_particels
problem_nonlog_prior_est_3_1_AM_gen_cw.alg_param.burn_in = burn_in
problem_nonlog_prior_est_3_1_AM_gen_cw.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_3_1_AM_gen_cw.adaptive_update =  AMUpdate_gen_comp_w(eye(3), [2/sqrt(3) 2/sqrt(3)  2/sqrt(3)], 0.238, 1, 0.4, 50)

problem_nonlog_prior_est_3_1_AM.alg_param.nbr_x0 = 1
problem_nonlog_prior_est_3_1_AM.alg_param.nbr_x = 4
problem_nonlog_prior_est_3_1_AM.alg_param.subsample_int = 4
problem_nonlog_prior_est_3_1_AM.data.Z = Z_thinned

problem_nonlog_prior_est_3_1_AM_gen.alg_param.nbr_x0 = 1
problem_nonlog_prior_est_3_1_AM_gen.alg_param.nbr_x = 4
problem_nonlog_prior_est_3_1_AM_gen.alg_param.subsample_int = 4
problem_nonlog_prior_est_3_1_AM_gen.data.Z = Z_thinned


#=
problem_normal_prior_est_3_2 = set_up_problem(nbr_of_unknown_parameters=3)
problem_normal_prior_est_3_2.alg_param.R = 10000
problem_normal_prior_est_3_2.alg_param.N = 25
problem_normal_prior_est_3_2.alg_param.burn_in = 5000
problem_normal_prior_est_3_2.model_param.theta_0 = [log(0.00001) log(2) log(1.5)]
=#

# est 4 parameters


problem_normal_prior_est_4_1_noadapt = set_up_problem(nbr_of_unknown_parameters=4)
problem_normal_prior_est_4_1_noadapt.alg_param.R = nbr_iterations
problem_normal_prior_est_4_1_noadapt.alg_param.N = nbr_particels
problem_normal_prior_est_4_1_noadapt.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_1_noadapt.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_4_1_noadapt.adaptive_update = noAdaptation(1e-2, eye(4))



# use non identy matrix
problem_normal_prior_est_4_1_noadapt = set_up_problem(nbr_of_unknown_parameters=4)
problem_normal_prior_est_4_1_noadapt.alg_param.R = nbr_iterations
problem_normal_prior_est_4_1_noadapt.alg_param.N = nbr_particels
problem_normal_prior_est_4_1_noadapt.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_1_noadapt.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_4_1_noadapt.adaptive_update = noAdaptation(1e-2, 2*diagm([10,10,1,25]))


# normal prior
problem_normal_prior_est_4_1_AP = set_up_problem(nbr_of_unknown_parameters=4)
problem_normal_prior_est_4_1_AP.alg_para4m.R = nbr_iterations
problem_normal_prior_est_4_1_AP.alg_param.N = nbr_particels
problem_normal_prior_est_4_1_AP.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_1_AP.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_4_1_AP.adaptive_update = APUpdate(200,200,0.1,2.4/sqrt(4))

problem_normal_prior_est_4_1_AM = set_up_problem(nbr_of_unknown_parameters=4)
problem_normal_prior_est_4_1_AM.alg_param.R = nbr_iterations
problem_normal_prior_est_4_1_AM.alg_param.N = nbr_particels
problem_normal_prior_est_4_1_AM.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_1_AM.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_4_1_AM.adaptive_update =  AMUpdate(eye(4), 2.38/sqrt(4), 1, 0.5, 50)

problem_normal_prior_est_4_1_AM_gen = set_up_problem(nbr_of_unknown_parameters=4)
problem_normal_prior_est_4_1_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_4_1_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_4_1_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
# old settings:
problem_normal_prior_est_4_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(4), 2.4/sqrt(4), 0.238, 1, 0.7, 25)
# new settings:
#problem_normal_prior_est_4_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(4), 2.1/sqrt(4), 0.238, 0.5, 0.4, 50)
#problem_normal_prior_est_4_1_AM_gen.adaptive_update =  AMUpdate_gen(diagm([0.05, 0.05, 0.1, 0.1]), 2.38/sqrt(4), 0.238, 1, 0.5, 25)

problem_normal_prior_est_4_2_AM_gen = set_up_problem(nbr_of_unknown_parameters=4)
problem_normal_prior_est_4_2_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_4_2_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_4_2_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
# old settings:
#problem_normal_prior_est_4_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(4), 2.38/sqrt(4), 0.238, 1, 0.5, 50)
# new settings:
problem_normal_prior_est_4_2_AM_gen.adaptive_update =  AMUpdate_gen(eye(4), 2.1/sqrt(4), 0.238, 0.5, 0.4, 50)


problem_normal_prior_est_4_1_AM_gen_cw = set_up_problem(nbr_of_unknown_parameters=4)
problem_normal_prior_est_4_1_AM_gen_cw.alg_param.R = nbr_iterations
problem_normal_prior_est_4_1_AM_gen_cw.alg_param.N = nbr_particels
problem_normal_prior_est_4_1_AM_gen_cw.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_1_AM_gen_cw.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_4_1_AM_gen_cw.adaptive_update =  AMUpdate_gen_comp_w(eye(4), 2.38/sqrt(4)*ones(1,4), 0.238, 1, 0.5, 50)


problem_normal_prior_est_4_1_AM.data.Z = Z
problem_normal_prior_est_4_1_AM_gen.data.Z = Z
problem_normal_prior_est_4_1_AM_gen_cw.data.Z = Z


# non-log-scale prior
problem_nonlog_prior_est_4_1_AP = set_up_problem(nbr_of_unknown_parameters=4, prior_dist="nonlog")
problem_nonlog_prior_est_4_1_AP.alg_param.R = nbr_iterations
problem_nonlog_prior_est_4_1_AP.alg_param.N = nbr_particels
problem_nonlog_prior_est_4_1_AP.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_1_AP.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_4_1_AP.adaptive_update = APUpdate(200,200,0.1,2.4/sqrt(4))

problem_nonlog_prior_est_4_1_AM = set_up_problem(nbr_of_unknown_parameters=4, prior_dist="nonlog")
problem_nonlog_prior_est_4_1_AM.alg_param.R = nbr_iterations
problem_nonlog_prior_est_4_1_AM.alg_param.N = nbr_particels
problem_nonlog_prior_est_4_1_AM.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_1_AM.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_4_1_AM.adaptive_update =  AMUpdate(eye(4), 2.38/sqrt(4), 1, 0.5, 50)

problem_nonlog_prior_est_4_1_AM_gen = set_up_problem(nbr_of_unknown_parameters=4, prior_dist="nonlog")
problem_nonlog_prior_est_4_1_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_4_1_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_4_1_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_4_1_AM_gen.adaptive_update =  AMUpdate_gen(diagm([0.1, 0.1, 0.2, 0.2]), 2.38/sqrt(4), 0.25, 1, 0.6, 25) #old settings AMUpdate_gen(eye(4), 2.38/sqrt(4), 0.238, 1, 0.5, 50)

problem_nonlog_prior_est_4_2_AM_gen = set_up_problem(nbr_of_unknown_parameters=4, prior_dist="nonlog")
problem_nonlog_prior_est_4_2_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_4_2_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_4_2_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_2_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_4_2_AM_gen.adaptive_update =  AMUpdate_gen(eye(4), 2/sqrt(4), 0.238, 1, 0.7, 50)


problem_nonlog_prior_est_4_1_AM_cw = set_up_problem(nbr_of_unknown_parameters=4, prior_dist="nonlog")
problem_nonlog_prior_est_4_1_AM_cw.alg_param.R = nbr_iterations
problem_nonlog_prior_est_4_1_AM_cw.alg_param.N = nbr_particels
problem_nonlog_prior_est_4_1_AM_cw.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_1_AM_cw.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_4_1_AM_cw.adaptive_update =  AMUpdate_comp_w(eye(4), 2.38/sqrt(4)*ones(1,4), 0.238, 1, 0.5, 50)


problem_nonlog_prior_est_4_1_AM_gen_cw = set_up_problem(nbr_of_unknown_parameters=4, prior_dist="nonlog")
problem_nonlog_prior_est_4_1_AM_gen_cw.alg_param.R = nbr_iterations
problem_nonlog_prior_est_4_1_AM_gen_cw.alg_param.N = nbr_particels
problem_nonlog_prior_est_4_1_AM_gen_cw.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_1_AM_gen_cw.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_4_1_AM_gen_cw.adaptive_update =  AMUpdate_gen_comp_w(eye(4), 2.38/sqrt(4)*ones(1,4), 0.238, 1, 0.5, 50)

problem_nonlog_prior_est_4_1_AM.data.Z = Z
problem_nonlog_prior_est_4_1_AM_gen.data.Z = Z
problem_nonlog_prior_est_4_1_AM_gen_cw.data.Z = Z


# est 5 parameters

# normal prior
problem_normal_prior_est_5_1_AP = set_up_problem(nbr_of_unknown_parameters=5)
problem_normal_prior_est_5_1_AP.alg_para4m.R = nbr_iterations
problem_normal_prior_est_5_1_AP.alg_param.N = nbr_particels
problem_normal_prior_est_5_1_AP.alg_param.burn_in = burn_in
problem_normal_prior_est_5_1_AP.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_5_1_AP.adaptive_update = APUpdate(200,200,0.1,2.4/sqrt(5))

problem_normal_prior_est_5_1_AM = set_up_problem(nbr_of_unknown_parameters=5)
problem_normal_prior_est_5_1_AM.alg_param.R = nbr_iterations
problem_normal_prior_est_5_1_AM.alg_param.N = nbr_particels
problem_normal_prior_est_5_1_AM.alg_param.burn_in = burn_in
problem_normal_prior_est_5_1_AM.alg_param.nbr_of_cores = nbr_of_cores

problem_normal_prior_est_5_1_AM.adaptive_update =  AMUpdate(eye(5), 2/sqrt(5), 1, 0.7, 50)

problem_normal_prior_est_5_1_AM_gen = set_up_problem(nbr_of_unknown_parameters=5)
problem_normal_prior_est_5_1_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_5_1_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_5_1_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_5_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores

#problem_normal_prior_est_5_1_AM_gen.adaptive_update =  AMUpdate_gen(diagm([0.05, 0.05, 0.05, 0.1, 0.1]), 2.38/sqrt(5), 0.238, 1, 0.5, 25)
problem_normal_prior_est_5_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(5), 1/sqrt(5), 0.238, 1, 0.5, 25)
problem_normal_prior_est_5_1_AM_gen.alg_param.nbr_of_cores = 4

problem_normal_prior_est_5_1_AM_gen_cw = set_up_problem(nbr_of_unknown_parameters=5)
problem_normal_prior_est_5_1_AM_gen_cw.alg_param.R = nbr_iterations
problem_normal_prior_est_5_1_AM_gen_cw.alg_param.N = nbr_particels
problem_normal_prior_est_5_1_AM_gen_cw.alg_param.burn_in = burn_in
problem_normal_prior_est_5_1_AM_gen_cw.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_5_1_AM_gen_cw.adaptive_update =  AMUpdate_gen_comp_w(eye(5), 2.38/sqrt(5)*ones(1,5), 0.238, 1, 0.5, 50)


problem_normal_prior_est_5_1_AM.data.Z = Z
problem_normal_prior_est_5_1_AM_gen.data.Z = Z
problem_normal_prior_est_5_1_AM_gen_cw.data.Z = Z


# non-log-scale prior
problem_nonlog_prior_est_5_1_AP = set_up_problem(nbr_of_unknown_parameters=5, prior_dist="nonlog")
problem_nonlog_prior_est_5_1_AP.alg_param.R = nbr_iterations
problem_nonlog_prior_est_5_1_AP.alg_param.N = nbr_particels
problem_nonlog_prior_est_5_1_AP.alg_param.burn_in = burn_in
problem_nonlog_prior_est_5_1_AP.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_5_1_AP.adaptive_update = APUpdate(200,200,0.1,2.4/sqrt(5))

problem_nonlog_prior_est_5_1_AM = set_up_problem(nbr_of_unknown_parameters=5, prior_dist="nonlog")
problem_nonlog_prior_est_5_1_AM.alg_param.R = nbr_iterations
problem_nonlog_prior_est_5_1_AM.alg_param.N = nbr_particels
problem_nonlog_prior_est_5_1_AM.alg_param.burn_in = burn_in
#problem_nonlog_prior_est_5_1_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_5_1_AM.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_5_1_AM.adaptive_update =  AMUpdate(eye(5), 2/sqrt(5), 1, 0.5, 50)


problem_nonlog_prior_est_5_1_AM_gen = set_up_problem(nbr_of_unknown_parameters=5, prior_dist="nonlog")
problem_nonlog_prior_est_5_1_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_5_1_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_5_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
#problem_nonlog_prior_est_5_1_AM_gen.adaptive_update =  AMUpdate_gen(diagm([0.05, 0.05, 0.05, 0.1, 0.1]), 2.38/sqrt(5), 0.238, 1, 0.5, 25)
problem_nonlog_prior_est_5_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(5), 1/sqrt(5), 0.238, 1, 0.5, 25)

problem_nonlog_prior_est_5_1_AM_gen_cw = set_up_problem(nbr_of_unknown_parameters=5, prior_dist="nonlog")
problem_nonlog_prior_est_5_1_AM_gen_cw.alg_param.R = nbr_iterations
problem_nonlog_prior_est_5_1_AM_gen_cw.alg_param.N = nbr_particels
problem_nonlog_prior_est_5_1_AM_gen_cw.alg_param.burn_in = burn_in
problem_nonlog_prior_est_5_1_AM_gen_cw.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_5_1_AM_gen_cw.adaptive_update =  AMUpdate_gen_comp_w(eye(5), 2.38/sqrt(5)*ones(1,5), 0.238, 1, 0.5, 50)

problem_nonlog_prior_est_5_1_AM.data.Z = Z
problem_nonlog_prior_est_5_1_AM_gen.data.Z = Z
problem_nonlog_prior_est_5_1_AM_gen_cw.data.Z = Z

# est 6 parameters

# normal prior
problem_normal_prior_est_6_1_AP = set_up_problem(nbr_of_unknown_parameters=6)
problem_normal_prior_est_6_1_AP.alg_param.R = nbr_iterations
problem_normal_prior_est_6_1_AP.alg_param.N = nbr_particels
problem_normal_prior_est_6_1_AP.alg_param.burn_in = burn_in
problem_nonlog_prior_est_6_1_AP.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_6_1_AP.adaptive_update = APUpdate(200,200,0.1,2.4/sqrt(3))

problem_normal_prior_est_6_1_AM = set_up_problem(nbr_of_unknown_parameters=6)
problem_normal_prior_est_6_1_AM.alg_param.R = nbr_iterations
problem_normal_prior_est_6_1_AM.alg_param.N = nbr_particels
problem_normal_prior_est_6_1_AM.alg_param.burn_in = burn_in
problem_nonlog_prior_est_6_1_AM.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_6_1_AM.adaptive_update =  AMUpdate(eye(6), 2.38/sqrt(6), 1, 0.5, 50)

problem_normal_prior_est_6_1_AM_gen = set_up_problem(nbr_of_unknown_parameters=6)
problem_normal_prior_est_6_1_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_6_1_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_6_1_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_6_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_6_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(6), 2.38/sqrt(6), 0.238, 1, 0.7, 50)
#problem_normal_prior_est_6_1_AM_gen.alg_param.nbr_of_cores = 4


problem_normal_prior_est_6_1_AM_gen_cw = set_up_problem(nbr_of_unknown_parameters=6)
problem_normal_prior_est_6_1_AM_gen_cw.alg_param.R = nbr_iterations
problem_normal_prior_est_6_1_AM_gen_cw.alg_param.N = nbr_particels
problem_normal_prior_est_6_1_AM_gen_cw.alg_param.burn_in = burn_in
problem_nonlog_prior_est_6_1_AM_gen_cw.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_6_1_AM_gen_cw.adaptive_update =  AMUpdate_gen_comp_w(eye(6), 2.38/sqrt(6)*ones(1,6), 0.238, 1, 0.5, 50)


problem_normal_prior_est_6_1_AM.data.Z = Z
problem_normal_prior_est_6_1_AM_gen.data.Z = Z
problem_normal_prior_est_6_1_AM_gen_cw.data.Z = Z


# non-log-scale prior
problem_nonlog_prior_est_6_1_AP = set_up_problem(nbr_of_unknown_parameters=6, prior_dist="nonlog")
problem_nonlog_prior_est_6_1_AP.alg_param.R = nbr_iterations
problem_nonlog_prior_est_6_1_AP.alg_param.N = nbr_particels
problem_nonlog_prior_est_6_1_AP.alg_param.burn_in = burn_in
problem_nonlog_prior_est_6_1_AP.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_6_1_AP.adaptive_update = APUpdate(200,200,0.1,2.4/sqrt(3))

problem_nonlog_prior_est_6_1_AM = set_up_problem(nbr_of_unknown_parameters=6, prior_dist="nonlog")
problem_nonlog_prior_est_6_1_AM.alg_param.R = nbr_iterations
problem_nonlog_prior_est_6_1_AM.alg_param.N = nbr_particels
problem_nonlog_prior_est_6_1_AM.alg_param.burn_in = burn_in
problem_nonlog_prior_est_6_1_AM.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_6_1_AM.adaptive_update =  AMUpdate(eye(6), 2.38/sqrt(6), 1, 0.4, 50)

problem_nonlog_prior_est_6_1_AM_gen = set_up_problem(nbr_of_unknown_parameters=6, prior_dist="nonlog")
problem_nonlog_prior_est_6_1_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_6_1_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_6_1_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_6_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_6_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(6), 2.38/sqrt(6), 0.238, 1, 0.7, 50)


problem_nonlog_prior_est_6_1_AM_gen_cw = set_up_problem(nbr_of_unknown_parameters=6, prior_dist="nonlog")
problem_nonlog_prior_est_6_1_AM_gen_cw.alg_param.R = nbr_iterations
problem_nonlog_prior_est_6_1_AM_gen_cw.alg_param.N = nbr_particels
problem_nonlog_prior_est_6_1_AM_gen_cw.alg_param.burn_in = burn_in
problem_nonlog_prior_est_6_1_AM_gen_cw.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_6_1_AM_gen_cw.adaptive_update =  AMUpdate_gen_comp_w(eye(6), 2.38/sqrt(6)*ones(1,6), 0.238, 1, 0.4, 50)

problem_nonlog_prior_est_6_1_AM.data.Z = Z
problem_nonlog_prior_est_6_1_AM_gen.data.Z = Z
problem_nonlog_prior_est_6_1_AM_gen_cw.data.Z = Z

# est 7 parameters


problem_normal_prior_est_7_1_AM_gen = set_up_problem(nbr_of_unknown_parameters=7)
problem_normal_prior_est_7_1_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_7_1_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_7_1_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_7_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_7_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(7), 2.38/sqrt(7), 0.238, 1, 0.7, 25)


problem_nonlog_prior_est_7_1_AM_gen = set_up_problem(nbr_of_unknown_parameters=7, prior_dist="nonlog")
problem_nonlog_prior_est_7_1_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_7_1_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_7_1_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_7_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_7_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(7), 2.38/sqrt(7), 0.238, 1, 0.7, 25)



# est 9 parameters
problem_normal_prior_est_9_1_AP = set_up_problem(nbr_of_unknown_parameters=6)
problem_normal_prior_est_9_1_AP.alg_param.R = nbr_iterations
problem_normal_prior_est_9_1_AP.alg_param.N = nbr_particels
problem_normal_prior_est_9_1_AP.alg_param.burn_in = burn_in
problem_nonlog_prior_est_9_1_AP.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_9_1_AP.adaptive_update = APUpdate(200,200,0.1,2.4/sqrt(9))

problem_normal_prior_est_9_1_AM = set_up_problem(nbr_of_unknown_parameters=9)
problem_normal_prior_est_9_1_AM.alg_param.R = nbr_iterations
problem_normal_prior_est_9_1_AM.alg_param.N = nbr_particels
problem_normal_prior_est_9_1_AM.alg_param.burn_in = burn_in
problem_nonlog_prior_est_9_1_AM.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_9_1_AM.adaptive_update =  AMUpdate(eye(9), 2.38/sqrt(9), 1, 0.8, 50)

problem_normal_prior_est_9_1_AM_gen = set_up_problem(nbr_of_unknown_parameters=9)
problem_normal_prior_est_9_1_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_9_1_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_9_1_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_9_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_9_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(9), 2.38/sqrt(9), 0.238, 1, 0.7, 50)

problem_normal_prior_est_9_1_AM_gen_cw = set_up_problem(nbr_of_unknown_parameters=9)
problem_normal_prior_est_9_1_AM_gen_cw.alg_param.R = nbr_iterations
problem_normal_prior_est_9_1_AM_gen_cw.alg_param.N = nbr_particels
problem_normal_prior_est_9_1_AM_gen_cw.alg_param.burn_in = burn_in
problem_nonlog_prior_est_9_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_9_1_AM_gen_cw.adaptive_update =  AMUpdate_gen_comp_w(eye(9), 2.38/sqrt(9)*ones(1,9), 0.238, 1, 0.8, 50)


problem_normal_prior_est_9_1_AM.data.Z = Z
problem_normal_prior_est_9_1_AM_gen.data.Z = Z
problem_normal_prior_est_9_1_AM_gen_cw.data.Z = Z


problem_nonlog_prior_est_9_1_AP = set_up_problem(nbr_of_unknown_parameters=9,prior_dist="nonlog")
problem_nonlog_prior_est_9_1_AP.alg_param.R = nbr_iterations
problem_nonlog_prior_est_9_1_AP.alg_param.N = nbr_particels
problem_nonlog_prior_est_9_1_AP.alg_param.burn_in = burn_in
problem_nonlog_prior_est_9_1_AP.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_9_1_AP.adaptive_update = APUpdate(200,200,0.1,2.4/sqrt(9))

problem_nonlog_prior_est_9_1_AM = set_up_problem(nbr_of_unknown_parameters=9,prior_dist="nonlog")
problem_nonlog_prior_est_9_1_AM.alg_param.R = nbr_iterations
problem_nonlog_prior_est_9_1_AM.alg_param.N = nbr_particels
problem_nonlog_prior_est_9_1_AM.alg_param.burn_in = burn_in
problem_nonlog_prior_est_9_1_AM.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_9_1_AM.adaptive_update =  AMUpdate(eye(9), 2.38/sqrt(9), 1, 0.4, 50)

problem_nonlog_prior_est_9_1_AM_gen = set_up_problem(nbr_of_unknown_parameters=9,prior_dist="nonlog")
problem_nonlog_prior_est_9_1_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_9_1_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_9_1_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_9_1_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_9_1_AM_gen.adaptive_update =  AMUpdate_gen(eye(9), 2.38/sqrt(9), 0.238, 1, 0.6, 50)

problem_nonlog_prior_est_9_1_AM_gen_cw = set_up_problem(nbr_of_unknown_parameters=9,prior_dist="nonlog")
problem_nonlog_prior_est_9_1_AM_gen_cw.alg_param.R = nbr_iterations
problem_nonlog_prior_est_9_1_AM_gen_cw.alg_param.N = nbr_particels
problem_nonlog_prior_est_9_1_AM_gen_cw.alg_param.burn_in = burn_in
problem_nonlog_prior_est_9_1_AM_gen_cw.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_9_1_AM_gen_cw.adaptive_update =  AMUpdate_gen_comp_w(eye(9), 2.38/sqrt(9)*ones(1,9), 0.238, 1, 0.6, 50)


problem_nonlog_prior_est_9_1_AM.data.Z = Z
problem_nonlog_prior_est_9_1_AM_gen.data.Z = Z
problem_nonlog_prior_est_9_1_AM_gen_cw.data.Z = Z


################################################################################
##                            run MCMC algorithms                             ##
################################################################################

# est 2 parameters

res_problem_normal_prior_est_2_1_noadapt  = @time MCWM(problem_normal_prior_est_2_1_noadapt)
#res_problem_nonlog_prior_est_2_1_noadapt  = @time MCWM(problem_nonlog_prior_est_2_1_noadapt)

#res_problem_normal_prior_est_2_1_AM_w_eps  = @time MCWM(problem_normal_prior_est_2_1_AM_w_eps)
res_problem_normal_prior_est_2_1_AM = @time MCWM(problem_normal_prior_est_2_1_AM)
res_problem_normal_prior_est_2_1_AM_gen  = @time MCWM(problem_normal_prior_est_2_1_AM_gen)
#res_problem_normal_prior_est_2_1_AM_comp_w  = @time MCWM(problem_normal_prior_est_2_1_AM_comp_w)
#res_problem_normal_prior_est_2_1_AM_gen_comp_w  = @time MCWM(problem_normal_prior_est_2_1_AM_gen_comp_w)


#res_problem_nonlog_prior_est_2_1_AM = @time MCWM(problem_nonlog_prior_est_2_1_AM)
#res_problem_nonlog_prior_est_2_1_AM_gen  = @time MCWM(problem_nonlog_prior_est_2_1_AM_gen)
#res_problem_nonlog_prior_est_2_1_AM_gen_cw  = @time MCWM(problem_nonlog_prior_est_2_1_AM_gen_cw)
#res_problem_nonlog_prior_est_2_1_AM_gen_cw  = @time MCWM(problem_nonlog_prior_est_2_1_AM_gen_cw)

# est 3 parameters

res_problem_normal_prior_est_3_1_noadapt  = @time MCWM(problem_normal_prior_est_3_1_noadapt)
#res_problem_nonlog_prior_est_2_1_noadapt  = @time MCWM(problem_nonlog_prior_est_2_1_noadapt)


#res_problem_normal_prior_est_3_1_AP = @time MCWM(problem_normal_prior_est_3_1_AP)
#res_problem_normal_prior_est_3_1_AM = @time MCWM(problem_normal_prior_est_3_1_AM)
res_problem_normal_prior_est_3_1_AM_gen = @time MCWM(problem_normal_prior_est_3_1_AM_gen)
#res_problem_normal_prior_est_3_1_AM_gen_cw  = @time MCWM(problem_normal_prior_est_3_1_AM_gen_cw)


#res_problem_nonlog_prior_est_3_1_AM = @time MCWM(problem_nonlog_prior_est_3_1_AM)
#res_problem_nonlog_prior_est_3_1_AM_gen = @time MCWM(problem_nonlog_prior_est_3_1_AM_gen)
#res_problem_nonlog_prior_est_3_1_AM_cw  = @time MCWM(problem_nonlog_prior_est_3_1_AM_cw)
#res_problem_nonlog_prior_est_3_1_AM_gen_cw  = @time MCWM(problem_nonlog_prior_est_3_1_AM_gen_cw)

#res_problem_nonlog_prior_est_3_2_AM_gen = @time MCWM(problem_nonlog_prior_est_3_2_AM_gen)

# est 4 parameters

res_problem_normal_prior_est_4_1_noadapt  = @time MCWM(problem_normal_prior_est_4_1_noadapt)
#res_problem_nonlog_prior_est_2_1_noadapt  = @time MCWM(problem_nonlog_prior_est_2_1_noadapt)



#res_problem_normal_prior_est_4_1_AP = @time MCWM(problem_normal_prior_est_4_1_AP)
#res_problem_normal_prior_est_4_1_AM = @time MCWM(problem_normal_prior_est_4_1_AM)
res_problem_normal_prior_est_4_1_AM_gen = @time MCWM(problem_normal_prior_est_4_1_AM_gen)
#res_problem_normal_prior_est_4_1_AM_gen_cw = @time MCWM(problem_normal_prior_est_4_1_AM_gen_cw)

#res_problem_nonlog_prior_est_4_1_AM = @time MCWM(problem_nonlog_prior_est_4_1_AM)
#res_problem_nonlog_prior_est_4_1_AM_gen = @time MCWM(problem_nonlog_prior_est_4_1_AM_gen)
#res_problem_nonlog_prior_est_4_1_AM_cw = @time MCWM(problem_nonlog_prior_est_4_1_AM_cw)
#res_problem_nonlog_prior_est_4_1_AM_gen_cw = @time MCWM(problem_nonlog_prior_est_4_1_AM_gen_cw)

#res_problem_normal_prior_est_4_2_AM_gen = @time MCWM(problem_normal_prior_est_4_2_AM_gen)
#res_problem_nonlog_prior_est_4_2_AM_gen = @time MCWM(problem_nonlog_prior_est_4_2_AM_gen)


#res_problem_normal_prior_est_4_1_AP = @time MCWM(problem_normal_prior_est_4_1_AP)
#res_problem_normal_prior_est_4_1_AM = @time MCWM(problem_normal_prior_est_4_1_AM)
#res_problem_normal_prior_est_4_1_AM_gen = @time MCWM(problem_normal_prior_est_4_1_AM_gen)
#res_problem_normal_prior_est_4_1_AM_gen_cw = @time MCWM(problem_normal_prior_est_4_1_AM_gen_cw)

#res_problem_nonlog_prior_est_4_1_AM = @time MCWM(problem_nonlog_prior_est_4_1_AM)
#res_problem_nonlog_prior_est_4_1_AM_gen = @time MCWM(problem_nonlog_prior_est_4_1_AM_gen)
#res_problem_nonlog_prior_est_4_1_AM_gen_cw = @time MCWM(problem_nonlog_prior_est_4_1_AM_gen_cw)

# est 5 parameters
#res_problem_normal_prior_est_5_1_AP = @time MCWM(problem_normal_prior_est_5_1_AP)
#res_problem_normal_prior_est_5_1_AM = @time MCWM(problem_normal_prior_est_5_1_AM)
res_problem_normal_prior_est_5_1_AM_gen = @time MCWM(problem_normal_prior_est_5_1_AM_gen)
#res_problem_normal_prior_est_5_1_AM_gen_cw  = @time MCWM(problem_normal_prior_est_5_1_AM_gen_cw)


#res_problem_nonlog_prior_est_5_1_AM = @time MCWM(problem_nonlog_prior_est_5_1_AM)
res_problem_nonlog_prior_est_5_1_AM_gen = @time MCWM(problem_nonlog_prior_est_5_1_AM_gen)
#res_problem_nonlog_prior_est_5_1_AM_cw  = @time MCWM(problem_nonlog_prior_est_5_1_AM_cw)
#res_problem_nonlog_prior_est_5_1_AM_gen_cw  = @time MCWM(problem_nonlog_prior_est_5_1_AM_gen_cw)


# est 6 parameters
#res_problem_normal_prior_est_6_1_AP = @time MCWM(problem_normal_prior_est_6_1_AP)
#res_problem_normal_prior_est_6_1_AM = @time MCWM(problem_normal_prior_est_6_1_AM)
res_problem_normal_prior_est_6_1_AM_gen = @time MCWM(problem_normal_prior_est_6_1_AM_gen)
#res_problem_normal_prior_est_6_1_AM_gen_cw  = @time MCWM(problem_normal_prior_est_6_1_AM_gen_cw)


#res_problem_nonlog_prior_est_6_1_AM = @time MCWM(problem_nonlog_prior_est_6_1_AM)
res_problem_nonlog_prior_est_6_1_AM_gen = @time MCWM(problem_nonlog_prior_est_6_1_AM_gen)
#res_problem_nonlog_prior_est_6_1_AM_cw  = @time MCWM(problem_nonlog_prior_est_6_1_AM_cw)
#res_problem_nonlog_prior_est_6_1_AM_gen_cw  = @time MCWM(problem_nonlog_prior_est_6_1_AM_gen_cw)


# est 7 parameters
res_problem_normal_prior_est_7_1_AM_gen = @time MCWM(problem_normal_prior_est_7_1_AM_gen)
res_problem_nonlog_prior_est_7_1_AM_gen = @time MCWM(problem_nonlog_prior_est_7_1_AM_gen)


# est 9 (all) parameters
res_problem_normal_prior_est_9_1_AM = @time MCWM(problem_normal_prior_est_9_1_AM)
res_problem_normal_prior_est_9_1_AM_gen = @time MCWM(problem_normal_prior_est_9_1_AM_gen)
res_problem_normal_prior_est_9_1_AM_gen_cw = @time MCWM(problem_normal_prior_est_9_1_AM_gen_cw)

res_problem_nonlog_prior_est_9_1_AM = @time MCWM(problem_nonlog_prior_est_9_1_AM)
res_problem_nonlog_prior_est_9_1_AM_gen = @time MCWM(problem_nonlog_prior_est_9_1_AM_gen)
res_problem_nonlog_prior_est_9_1_AM_gen_cw = @time MCWM(problem_nonlog_prior_est_9_1_AM_gen_cw)

################################################################################
##                            export data                                     ##
################################################################################

# 2 params
export_data(problem_normal_prior_est_2_1_noadapt, res_problem_normal_prior_est_2_1_noadapt)
export_data(problem_normal_prior_est_2_1_AM_w_eps,res_problem_normal_prior_est_2_1_AM_w_eps)
export_data(problem_normal_prior_est_2_1_AM, res_problem_normal_prior_est_2_1_AM)
export_data(problem_normal_prior_est_2_1_AM_gen, res_problem_normal_prior_est_2_1_AM_gen[1])
export_data(problem_normal_prior_est_2_1_AM_comp_w, res_problem_normal_prior_est_2_1_AM_comp_w)
export_data(problem_normal_prior_est_2_1_AM_gen_comp_w, res_problem_normal_prior_est_2_1_AM_gen_comp_w[1])

export_parameters(res_problem_normal_prior_est_2_1_AM_gen[2])
export_parameters(res_problem_normal_prior_est_2_1_AM_gen_comp_w[2])


export_data(problem_nonlog_prior_est_2_1_nonlog, res_problem_nonlog_prior_est_2_1_nonlog)
export_data(problem_nonlog_prior_est_2_1_AM, res_problem_nonlog_prior_est_2_1_AM)
export_data(problem_nonlog_prior_est_2_1_AM_gen, res_problem_nonlog_prior_est_2_1_AM_gen[1])
export_data(problem_nonlog_prior_est_2_1_AM_gen_cw, res_problem_nonlog_prior_est_2_1_AM_gen_cw[1])

export_parameters(res_problem_nonlog_prior_est_2_1_AM_gen[2])
export_parameters(res_problem_nonlog_prior_est_2_1_AM_gen_cw[2])

# 3 params
export_data(problem_normal_prior_est_3_1_noadapt, res_problem_normal_prior_est_3_1_noadapt)
export_data(problem_normal_prior_est_3_1_AM, res_problem_normal_prior_est_3_1_AM)
export_data(problem_normal_prior_est_3_1_AM_gen, res_problem_normal_prior_est_3_1_AM_gen[1])
export_data(problem_normal_prior_est_3_1_AM_gen_cw, res_problem_normal_prior_est_3_1_AM_gen_cw[1])

export_parameters(res_problem_normal_prior_est_3_1_AM_gen[2])
export_parameters(res_problem_normal_prior_est_3_1_AM_gen_cw[2])

export_data(problem_nonlog_prior_est_3_1_AM, res_problem_nonlog_prior_est_3_1_AM)
export_data(problem_nonlog_prior_est_3_1_AM_gen, res_problem_nonlog_prior_est_3_1_AM_gen[1])
export_data(problem_nonlog_prior_est_3_1_AM_gen_cw, res_problem_nonlog_prior_est_3_1_AM_gen_cw[1])

export_parameters(res_problem_nonlog_prior_est_3_1_AM_gen[2])
export_parameters(res_problem_nonlog_prior_est_3_2_AM_gen[2])
export_parameters(res_problem_nonlog_prior_est_3_1_AM_gen_cw[2])


# 4 params
export_data(problem_normal_prior_est_4_1_noadapt, res_problem_normal_prior_est_4_1_noadapt)

export_data(problem_normal_prior_est_4_1_AM, res_problem_normal_prior_est_4_1_AM)
export_data(problem_normal_prior_est_4_1_AM_gen, res_problem_normal_prior_est_4_1_AM_gen[1])
#export_data(problem_normal_prior_est_4_2_AM_gen, res_problem_normal_prior_est_4_2_AM_gen[1])
export_data(problem_normal_prior_est_4_1_AM_gen_cw, res_problem_normal_prior_est_4_1_AM_gen_cw[1])

export_parameters(res_problem_normal_prior_est_4_1_AM_gen[2])
export_parameters(res_problem_normal_prior_est_4_1_AM_gen_cw[2])

export_data(problem_nonlog_prior_est_4_1_AM, res_problem_nonlog_prior_est_4_1_AM)
export_data(problem_nonlog_prior_est_4_1_AM_gen, res_problem_nonlog_prior_est_4_1_AM_gen[1])
export_data(problem_nonlog_prior_est_4_2_AM_gen, res_problem_nonlog_prior_est_4_2_AM_gen[1])
export_data(problem_nonlog_prior_est_4_1_AM_cw, res_problem_nonlog_prior_est_4_1_AM_cw[1])
export_data(problem_nonlog_prior_est_4_1_AM_gen_cw, res_problem_nonlog_prior_est_4_1_AM_gen_cw[1])

export_parameters(res_problem_nonlog_prior_est_4_1_AM_gen[2])
export_parameters(res_problem_nonlog_prior_est_4_2_AM_gen[2])
export_parameters(res_problem_nonlog_prior_est_4_1_AM_cw[2])
export_parameters(res_problem_nonlog_prior_est_4_1_AM_gen_cw[2])

# 5 params
export_data(problem_normal_prior_est_5_1_AM, res_problem_normal_prior_est_5_1_AM)
export_data(problem_normal_prior_est_5_1_AM_gen, res_problem_normal_prior_est_5_1_AM_gen[1])
export_data(problem_normal_prior_est_5_1_AM_gen_cw, res_problem_normal_prior_est_5_1_AM_gen_cw[1])

export_parameters(res_problem_normal_prior_est_5_1_AM_gen[2])
export_parameters(res_problem_normal_prior_est_5_1_AM_gen_cw[2])

export_data(problem_nonlog_prior_est_5_1_AM, res_problem_nonlog_prior_est_5_1_AM)
export_data(problem_nonlog_prior_est_5_1_AM_gen, res_problem_nonlog_prior_est_5_1_AM_gen[1])
export_data(problem_nonlog_prior_est_5_1_AM_cw, res_problem_nonlog_prior_est_5_1_AM_cw[1])
export_data(problem_nonlog_prior_est_5_1_AM_gen_cw, res_problem_nonlog_prior_est_5_1_AM_gen_cw[1])

export_parameters(res_problem_nonlog_prior_est_5_1_AM_gen[2])
export_parameters(res_problem_nonlog_prior_est_5_1_AM_cw[2])
export_parameters(res_problem_nonlog_prior_est_5_1_AM_gen_cw[2])

# 6 params
export_data(problem_normal_prior_est_6_1_AM, res_problem_normal_prior_est_6_1_AM)
export_data(problem_normal_prior_est_6_1_AM_gen, res_problem_normal_prior_est_6_1_AM_gen[1])
export_data(problem_normal_prior_est_6_1_AM_gen_cw, res_problem_normal_prior_est_6_1_AM_gen_cw[1])

export_data(problem_nonlog_prior_est_6_1_AM, res_problem_nonlog_prior_est_6_1_AM)
export_data(problem_nonlog_prior_est_6_1_AM_gen, res_problem_nonlog_prior_est_6_1_AM_gen[1])
export_data(problem_nonlog_prior_est_6_1_AM_gen_cw, res_problem_nonlog_prior_est_6_1_AM_gen_cw[1])

export_parameters(res_problem_nonlog_prior_est_6_1_AM_gen_cw[2])


# 7 params
export_data(problem_normal_prior_est_7_1_AM_gen, res_problem_normal_prior_est_7_1_AM_gen[1])
export_data(problem_nonlog_prior_est_7_1_AM_gen, res_problem_nonlog_prior_est_7_1_AM_gen[1])



export_data(problem_normal_prior_est_9_1_AM, res_problem_normal_prior_est_9_1_AM)
export_data(problem_normal_prior_est_9_1_AM_gen, res_problem_normal_prior_est_9_1_AM_gen[1])
export_data(problem_normal_prior_est_9_1_AM_gen_cw, res_problem_normal_prior_est_9_1_AM_gen_cw[1])



# old code
######################################################################
######################################################################
######################################################################

# set parameters for subsampling
#subsample_int = 30

#Z_data_subsample = Z_data[1:subsample_int:end]

# est 2 param
# priors: normal dist on log-scale



# test functions

p1 = set_adaptive_alg_params(problem_normal_prior_est_2_1_AP.adaptive_update,2, [1,1], 100)
p2 = set_adaptive_alg_params(problem_normal_prior_est_2_1_AM.adaptive_update,2, [1,1], 100)
p3 = set_adaptive_alg_params(problem_normal_prior_est_2_1_AM_gen.adaptive_update,2, [1,1], 100)
p4 = set_adaptive_alg_params(problem_normal_prior_est_2_1_AM_gen_cw.adaptive_update,2, [1,1], 100)
p5 = set_adaptive_alg_params(problem_normal_prior_est_2_1_AM_gen_comp_w.adaptive_update,2, [1,1], 100)


print_covariance(problem_normal_prior_est_2_1_AP.adaptive_update,p1, 10)
print_covariance(problem_normal_prior_est_2_1_AM.adaptive_update,p2, 10)
print_covariance(problem_normal_prior_est_2_1_AM_gen.adaptive_update,p3, 10)
print_covariance(problem_normal_prior_est_2_1_AM_gen_cw.adaptive_update,p4, 10)
print_covariance(problem_normal_prior_est_2_1_AM_gen_comp_w.adaptive_update,p5, 10)

Theta_star = zeros(1,2)

gaussian_random_walk(problem_normal_prior_est_2_1_AP.adaptive_update, p1, [1, 1], 2)
gaussian_random_walk(problem_normal_prior_est_2_1_AM.adaptive_update, p2, [1, 1], 2)
gaussian_random_walk(problem_normal_prior_est_2_1_AM_gen.adaptive_update, p3, [1, 1], 2)
gaussian_random_walk(problem_normal_prior_est_2_1_AM_gen_cw.adaptive_update, p4, [1, 1], 2)
(X_star,Z) = gaussian_random_walk(problem_normal_prior_est_2_1_AM_gen_comp_w.adaptive_update, p5, [1, 1], 2)


Theta = rand(100,2)
adaptation(problem_normal_prior_est_2_1_AM.adaptive_update, p2, Theta, 51,log(0.1))
adaptation(problem_normal_prior_est_2_1_AM_gen.adaptive_update, p3, Theta, 51, log(0.1))
adaptation(problem_normal_prior_est_2_1_AM_gen_cw.adaptive_update, p4, Theta, 51,log(0.1))
adaptation(problem_normal_prior_est_2_1_AM_gen_comp_w.adaptive_update, p5, Theta, 51,log(0.1))



#=
problem_normal_prior_est_2_2 = set_up_problem(nbr_of_unknown_parameters=2)
problem_normal_prior_est_2_2.alg_param.R = 10000
problem_normal_prior_est_2_2.alg_param.N = 25
problem_normal_prior_est_2_2.alg_param.burn_in = 5000
problem_normal_prior_est_2_2.model_param.theta_0 = [log(2) log(1.5)]
=#

# est 2 param
res_problem_normal_prior_est_2_1_AP = @time MCWM(problem_normal_prior_est_2_1_AP)
res_problem_normal_prior_est_2_1_AM = @time MCWM(problem_normal_prior_est_2_1_AM)
res_problem_normal_prior_est_2_1_AM_gen  = @time MCWM(problem_normal_prior_est_2_1_AM_gen)

res_problem_normal_prior_est_2_2 = @time MCWM(problem_normal_prior_est_2_2)

res_PMCMC_problem_normal_prior_est_2_1_AP = @time PMCMC(problem_normal_prior_est_2_1_AP)
res_PMCMC_problem_normal_prior_est_2_1_AM = @time PMCMC(problem_normal_prior_est_2_1_AM)


export_data(problem_normal_prior_est_2_1_AP, res_problem_normal_prior_est_2_1_AP)
export_data(problem_normal_prior_est_2_1_AM, res_problem_normal_prior_est_2_1_AM)
export_data(problem_normal_prior_est_2_1_AM_gen, res_problem_normal_prior_est_2_1_AM_gen[1])
export_data(problem_normal_prior_est_2_1_AM_gen_cw, res_problem_normal_prior_est_2_1_AM_gen_cw[1])

export_data(problem_normal_prior_est_2_1_AP, res_PMCMC_problem_normal_prior_est_2_1_AP)
export_data(problem_normal_prior_est_2_1_AM, res_PMCMC_problem_normal_prior_est_2_1_AM)


# est 3 param
res_problem_normal_prior_est_3_1_AP = @time MCWM(problem_normal_prior_est_3_1_AP)
res_problem_normal_prior_est_3_1_AM = @time MCWM(problem_normal_prior_est_3_1_AM)
res_problem_normal_prior_est_3_1_AM_gen = @time MCWM(problem_normal_prior_est_3_1_AM_gen)


res_PMCMC_problem_normal_prior_est_2_1_AP = @time PMCMC(problem_normal_prior_est_2_1_AP)
res_PMCMC_problem_normal_prior_est_2_1_AM = @time PMCMC(problem_normal_prior_est_2_1_AM)

res_PMCMC_problem_normal_prior_est_3_1_AP = @time PMCMC(problem_normal_prior_est_3_1_AP)
res_PMCMC_problem_normal_prior_est_3_1_AM = @time PMCMC(problem_normal_prior_est_3_1_AM)


export_data(problem_normal_prior_est_3_1_AP, res_problem_normal_prior_est_3_1_AP)
export_data(problem_normal_prior_est_3_1_AM, res_problem_normal_prior_est_3_1_AM)
export_data(problem_normal_prior_est_3_1_AM_gen, res_problem_normal_prior_est_3_1_AM_gen[1])
export_data(problem_normal_prior_est_3_1_AM_gen_cw, res_problem_normal_prior_est_3_1_AM_gen_cw[1])

export_data(problem_normal_prior_est_3_1_AP, res_PMCMC_problem_normal_prior_est_3_1_AP)
export_data(problem_normal_prior_est_3_1_AM, res_PMCMC_problem_normal_prior_est_3_1_AM)

plot(x=1:length(res_problem_normal_prior_est_3_1_AM_gen[2]), y=res_problem_normal_prior_est_3_1_AM_gen[2],Geom.point, Geom.line)

# est 4 params
res_problem_normal_prior_est_4_1_AP = @time MCWM(problem_normal_prior_est_4_1_AP)
res_problem_normal_prior_est_4_2_AM = @time MCWM(problem_normal_prior_est_4_2_AM)

res_PMCMC_normal_prior_est_4_1_AP = @time PMCMC(problem_normal_prior_est_4_1_AP)
res_PMCMC_normal_prior_est_4_2_AM = @time PMCMC(problem_normal_prior_est_4_2_AM)


export_data(problem_normal_prior_est_4_1_AP, res_problem_normal_prior_est_4_1_AP)
export_data(problem_normal_prior_est_4_2_AM, res_problem_normal_prior_est_4_2_AM)

export_data(problem_normal_prior_est_4_1_AP, res_PMCMC_normal_prior_est_4_1_AP)
export_data(problem_normal_prior_est_2_1_AM, res_PMCMC_normal_prior_est_4_2_AM)


# est 6 params
res_problem_normal_prior_est_6_1_AP = @time MCWM(problem_normal_prior_est_6_1_AP)
res_problem_normal_prior_est_6_1_AM = @time MCWM(problem_normal_prior_est_6_1_AM)
res_problem_normal_prior_est_6_1_gen = @time MCWM(problem_normal_prior_est_6_1_AM_gen)

export_data(problem_normal_prior_est_6_1_AP, res_problem_normal_prior_est_6_1_AP)
export_data(problem_normal_prior_est_6_1_AM, res_problem_normal_prior_est_6_1_AM)
export_data(problem_normal_prior_est_6_1_AM_gen, problem_normal_prior_est_6_1_AM_gen[1])
export_data(problem_normal_prior_est_6_1_AM_gen_cw, problem_normal_prior_est_6_1_AM_gen_cw[1])



################################################################################
################################################################################
# no subsampling
R = 5000 # nbr of iterations for MH
N = 50 # nbr of particles for SMC
burn_in = 500

nbr_x0 = 1
nbr_x = 1
subsample_int = 1


(Theta_est_p50, loglik_est_p50, accept_vec_est_p50, prior_vec_est_p50) = @time MCWM(Z_data,R,N,theta_0,Theta_parameters,dt,nbr_x0,nbr_x,
                                                                subsample_int, prior_dist, theta_known, gain_start,
                                                                U, H, cd_par, false , true)

# subsampling
R = 5000 # nbr of iterations for MH
N = 100 # nbr of particles for SMC
burn_in = 500

nbr_x0 = 1
nbr_x = 10
subsample_int = 30

(Theta_ss1, loglik_ss1, accept_vec_ss1, prior_vec_ss1) = @time MCWH(Z_data_subsample,R,N,theta_0,Theta_parameters,dt,nbr_x0,nbr_x,
                                                                              subsample_int, prior_dist, theta_known, gain_start,
                                                                              U, H, cd_par, false , false)

# subsampling
R = 2000 # nbr of iterations for MH
N = 250 # nbr of particles for SMC
burn_in = 500

nbr_x0 = 1
nbr_x = 10
subsample_int = 30

(Theta_ss2, loglik_ss2, accept_vec_ss2, prior_vec_ss2) = @time MCWH(Z_data_subsample,R,N,theta_0,Theta_parameters,dt,nbr_x0,nbr_x,
                                                                                subsample_int, prior_dist, theta_known, gain_start,
                                                                                U, H, cd_par, false )




# subsampling
R = 2000 # nbr of iterations for MH
N = 500 # nbr of particles for SMC
burn_in = 500

nbr_x0 = 1
nbr_x = 10
subsample_int = 30

(Theta_ss3, loglik_ss3, accept_vec_ss3, prior_vec_ss3) = @time MCWH(Z_data_subsample,R,N,theta_0,Theta_parameters,dt,nbr_x0,nbr_x,
                                                                                      subsample_int, prior_dist, theta_known, gain_start,
                                                                                      U, H, cd_par, false )
# subsampling
R = 100 # nbr of iterations for MH
N = 1000 # nbr of particles for SMC
burn_in = 500

nbr_x0 = 1
nbr_x = 10
subsample_int = 30

(Theta_ss4, loglik_ss4, accept_vec_ss4, prior_vec_ss4) = @time MCWH(Z_data_subsample,R,N,theta_0,Theta_parameters,dt,nbr_x0,nbr_x,
                                                                                                                  subsample_int, prior_dist, theta_known, gain_start,
                                                                                                                  U, H, cd_par, false )


# run PCMCM
R = 1000 # nbr of iterations for MH
N = 25 # nbr of particles for SMC
burn_in = 500

nbr_x0 = 1
nbr_x = 1
subsample_int = 1

(Theta_PMCMC, loglik_PMCMC, accept_vec_PMCMC, prior_vec_PMCMC) = @time PMCMC_adaptive(Z_data,R,N,theta_0,Theta_parameters,dt,nbr_x0,nbr_x,
                                                                                     subsample_int, prior_dist, theta_known, gain_start,
                                                                                      U, H, cd_par, false, false)

# run PCMCM with parallel pf
R = 1000 # nbr of iterations for MH
N = 200 # nbr of particles for SMC
burn_in = 500

nbr_x0 = 1
nbr_x = 1
subsample_int = 1

(Theta_PMCMC_p, loglik_PMCMC_p, accept_vec_PMCMC_p, prior_vec_PMCMC_p) = @time PMCMC_adaptive(Z_data,R,N,theta_0,Theta_parameters,dt,nbr_x0,nbr_x,
                                                                                    subsample_int, prior_dist, theta_known, gain_start,
                                                                                    U, H, cd_par, false, true)


# export results

# output data from no subsample
export_data(theta_true, burn_in, Theta_est,loglik_est,accept_vec_est,Theta_parameters)

export_data(theta_true, burn_in, Theta_est_p25,loglik_est_p25,accept_vec_est_p25,Theta_parameters,prior_dist)

export_data(theta_true, burn_in, Theta_est_p50,loglik_est_p50,accept_vec_est_p50,Theta_parameters)

# output data from  subsample N = 100
export_data(theta_true, burn_in, Theta_ss1,loglik_ss1,accept_vec_ss1,Theta_parameters)

# output data from  subsample N = 250
export_data(theta_true, burn_in, Theta_ss2,loglik_ss2,accept_vec_ss2,Theta_parameters)

# output data from  subsample N = 500
export_data(theta_true, burn_in, Theta_ss3,loglik_ss3,accept_vec_ss3,Theta_parameters)


# output data from  PMCMC
export_data(theta_true, burn_in, Theta_PMCMC,loglik_PMCMC,accept_vec_PMCMC,Theta_parameters)


# output data from  PMCMC
export_data(theta_true, burn_in, Theta_PMCMC_p,loglik_PMCMC_p,accept_vec_PMCMC_p,Theta_parameters)
