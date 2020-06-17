# This .R code file consists of:
# Problem 4: Randomized matrix multiplication
# Problem 5: Power method
# Problem 6: Sketching for Least-squares
# Arthurs: Chenghan Sun, Ninghui Li
# NOTE: please run this code in the directory (folder) of HW1


############ Problem 4 Randomized matrix multiplication ############
# set working directory
setwd("./")  # set relative path to the current directory

# load resource data as matrices
path_src_file_A = "./STA243_homework_1_matrix_A.csv"
path_src_file_B = "./STA243_homework_1_matrix_B.csv"
src_A = read.csv(path_src_file_A, header = F)
src_B = read.csv(path_src_file_B, header = F)
matA = as.matrix(src_A)
matB = as.matrix(src_B)

# Part (a)
# implementation of Algorithm 2: Randomized Matrix Multiplication Algorithm
# function 4.1: compute prob
gen_prob = function(matA, matB, gen_flag){
  # calculate probabilities pk as the preliminary step for random matmul algorithm
  # Param:
    # matA: the first matrix noted as A
    # matB: the second matrix noted as B
    # gen_flag: two sampling methods implemented as:
      # 1. uniformly randomly select the outer products: "uniform"
      # 2. non-uniform sampling: "nonuniform"
  # Return:
    # prob: n-dimensional probabilities distribution
  n = ncol(matA)
  if (gen_flag == "uniform"){
    pk = 1/ n
    prob = replicate(n, pk)
  }
  else if (gen_flag == "nonuniform"){
    temp_prob = c()  # initialize vector prob
    for (k in 1:n) {
      norm_A = norm(matA[, k], type="2")
      norm_B = norm(matB[k, ], type="2")
      pk = norm_A * norm_B
      temp_prob = c(temp_prob, pk)  # vector append
    }
    prob = temp_prob / sum(temp_prob)
  }
  else {
    print("Please enter the argument: gen_flag (sampling method) !")
  }
  return(prob)
}

# function 4.2: randomized matrix multiplication
randmatmul_alg2 = function(matA, matB, prob_dist, r){
  # algorithm: randomized matrix multiplication
  # Param:
    # matA: the first matrix noted as A
    # matB: the second matrix noted as B
    # prob_dist: two sampling methods implemented as:
      # 1. uniformly randomly select the outer products: "uniform"
      # 2. non-uniform sampling: "nonuniform"
    # r: r rank-one components
  # Return:
    # matM: output random matrix
  probs = gen_prob(matA, matB, prob_dist)  # instance of function: gen_prob
  n = ncol(matA)
  samp_list = sample(1:n, r, replace=T, prob=probs)  # perform random selection with probilities = probs
  sel_probs = probs[samp_list]  # selected probilities
  matM = matA[, samp_list] %*% diag(1 / (r * sel_probs)) %*% matB[samp_list,]
  return(matM)
}

# Part (b)
# Apply the algorithm in Part (a) to matA and matB with r = 20, 50, 100, 200.
# direct apply
matM_r20 = randmatmul_alg2(matA, matB, "nonuniform", 20)
matM_r50 = randmatmul_alg2(matA, matB, "nonuniform", 50)
matM_r100 = randmatmul_alg2(matA, matB, "nonuniform", 100)
matM_r200 = randmatmul_alg2(matA, matB, "nonuniform", 200)

# Part (c)
# Calculate the relative approximation error  for each of the estimates found in Part (b)
matC = matA %*% matB  # exact matmul
matM_list = list(matM_r20, matM_r50, matM_r100, matM_r200)  # list contains list of matrix M
r_vec = c(20, 50, 100, 200)

# loop through r's
error_MC_list = c()  # initialize error list
for (i in 1:length(matM_list)){
  # calculate the relative approximation error
  diff = as.matrix(matM_list[[i]]) - matC
  error_MC = norm(diff, type="F") / (norm(matA, type="F") * norm(matB, type="F"))
  error_MC_list = c(error_MC_list, error_MC)
}
# show as a table
table_error = data.frame("Sample size r" = r_vec, "Errors" = error_MC_list)
print(table_error)

# output:
# > print(table_error)
# Sample.size.r     Errors
# 1            20 0.18514221
# 2            50 0.12469411
# 3           100 0.07876438
# 4           200 0.05016654

# Part (d)
# Visualize the estimates from (b) using the image() function in R
par(mfrow=c(2,2))
# loop through r's
for (i in 1:length(matM_list)) {
  image(as.matrix(matM_list[[i]]), main=substitute(paste("Sample size r = ", i), 
                                                         list(i=r_vec[i])))
}


############ Problem 5 ############
# Complete the code power_sim.R and run the test routine to produce the plot.
power_iteration = function(A, v0, eps = 1e-6, maxiter=100) {
  # Algorithm 3 Power Method
  # Param:
    # A: matrix to be took in 
    # v0: initial vector, denoted as c0 in the notes
  # Return:
    # v_final: output of approximate eigenvector
  c_in = v0
  iter = 0
  for (i in 1:maxiter) {
    c_out = A %*% c_in  # calculate new c
    c_out = c_out / sqrt(sum(c_out^2))  # with normalization 
    error_eigen = 1 - sum(c_in * c_out)^2

    if (error_eigen < eps) {
      print("Algorithm finished")
      break
    } else if (iter == maxiter) {
      print("Algorithm unfinished and stopped with the default max iterations")
      print(paste("Current error = ", error_eigen))
    } else {
      c_in = c_out
      iter = iter + 1
    }
  }
  v_final = c_out
  return(v_final)
}

# testing part (provided)
set.seed(5)
E = matrix(rnorm(100), 10, 10)
v = c(1, rep(0, 9))
lams = 1:10
prods = c()
for (lambda in lams) {
  X = lambda*outer(v, v) + E  # 10x10
  v0 = rep(1, nrow(E))
  v0 = v0/sqrt(sum(v0^2))
  vv = power_iteration(X, v0)
  prods = c(prods, abs(v %*% vv))
}
par(mfrow=c(1,1))  # reset figure size
plot(lams, prods, "b")


############ Problem 6 ############
# Sketching for Least-squares
# load package for fhm()
library("phangorn")

# Part (a)
# Implement the Sketched-OLS algorithm: following P5 of SketchingLS.pdf, Algorithm 1 The Sketched-OLS algorithm

# Here defines some helper functions for the main Sketched-OLS function
helper_r = function(dim_d, dim_n, epsi) {
  # helper function 6.1: initialize r
  # Param:
    # dim_d: number of column of matrix 
    # dim_n: number of rows of matrix 
    # epsi: error parameter
  # Return:
    # r: an integer as upper trails 
  r = as.integer(dim_d * log(dim_n) / epsi)
  return (r)
}

helper_sampleS = function(HDX, HDy, dim_n, r) {
  # helper function 6.2: generate sub-sampling matrix
  # Param:
    # HDX: matrix multiplication result of HDX
    # HDy: matrix multiplication result of HDy
    # dim_n: number of rows of matrix 
    # r: an integer as upper trails 
  # Return:
    # list(sampleX, sampley): list of matrix multiplication result of SHDX and SHDy
  flag = sqrt(dim_n / r)
  sampleS = sample(1:dim_n, size=r, replace=T)
  sampleX = flag*HDX[sampleS, ]
  sampley = flag*HDy[sampleS]
  return (list(sampleX, sampley))
}

helper_diagD = function(X, y, dim_n) {
  # helper function 6.3: generate D ∈ Rn×n be a diagonal matrix
  # Param:
    # X: the design matrix 
    # y: the response vector 
    # dim_n: number of rows of matrix 
  # Return:
    # list(DX, Dy): list of matrix multiplication result of DX and Dy
  selectDii = sample(c(1,-1), size=dim_n, replace=T, prob=c(1/2, 1/2))
  DX = apply(X, 2, function(i) selectDii*i)
  Dy = selectDii * y
  return(list(DX, Dy))
}

# main function: Sketched-OLS
main_SketchedOLS = function (X, y, epsi) {
  # Param:
    # X: design matrix (nxd)
    # y: response y (n)
    # epsi: error parameter
  # Return:
    # S_list: list of matrix multiplication result of SHDX and SHDy
  dim_n = dim(X)[1]
  dim_d = dim(X)[2]
  r = helper_r(dim_d, dim_n, epsi)  # instance of r
  D_list = helper_diagD(X, y, dim_n)
  DX = D_list[[1]]  # instance D*X
  Dy = D_list[[2]]  # instance D*y
  HDX = apply(DX, 2, fhm)
  HDy = fhm(Dy)
  S_list = helper_sampleS(HDX, HDy, dim_n, r)
  #SHDX = S_list[1]  # no need to unpack 
  #SHDy = S_list[2]  # no need to unpack 
  return (S_list)
}

# Part (b)
# Generate design matrix X and response y with elements drawn iid from a Uniform(0, 1) distribution
design_X = matrix(nrow=1048576, ncol=20)
for (i in 1:20) {
  design_X[, i] = runif(1048576, 0, 1)
}

# generate design matrix X
respon_Y = as.matrix(runif(1048576, 0, 1))  # generate response Y

# Part (c)
# Compare the calculation time for the full least squares problem and the sketched OLS.
# Note: first calculate X∗ = ΦX and y∗ = Φy, and apply system.time() function on (XT∗ X∗)^(-1) XT∗ y∗
# and compare to the calculation time of (XT X)^(-1) XT y
# Repeat these steps for  = .1, .05, .01, .001 and present your results in a table
epsi_vector = c(0.1, 0.05, 0.01, 0.001)
perform_LS = function(X, y) {
  # perform OLS
  beta_s = solve(t(X) %*% X) %*% (t(X) %*% y)
  return (beta_s)
}

# In the following, considering the randomness of time measurements, we performed the calculation for 200 
# time to get stable conclusions
# Time measurements list 
usr_time_vec = c()
sys_time_vec = c()
elapsed_time_vec = c()

# Time for OLS
for (i in 1:200) {
  t_ols = system.time(perform_LS(design_X, as.matrix(respon_Y)))
  usr_time_vec = c(usr_time_vec, t_ols[[1]])
  sys_time_vec = c(sys_time_vec, t_ols[[2]])
  elapsed_time_vec = c(elapsed_time_vec, t_ols[[3]])
}
# printout average OLS time stats
print(paste("Average user time for OLS = ", mean(usr_time_vec)))
print(paste("Average system time for OLS = ", mean(sys_time_vec)))
print(paste("Average elapsed time for OLS = ", mean(elapsed_time_vec)))

# output:
# [1] "Average user time for OLS =  0.512559999999985"
# [1] "Average system time for OLS =  0.0879299999999978"
# [1] "Average elapsed time for OLS =  0.665754999998317"

# For Sketched OLS: 
# initialize time measurements vectors
usr_time_vec_epsi1 = c()  # 0.1
sys_time_vec_epsi1 = c()
elapsed_time_vec_epsi1 = c()
usr_time_vec_epsi2 = c()  # 0.05
sys_time_vec_epsi2 = c()
elapsed_time_vec_epsi2 = c()
usr_time_vec_epsi3 = c()  # 0.01
sys_time_vec_epsi3 = c()
elapsed_time_vec_epsi3 = c()
usr_time_vec_epsi4 = c()  # 0.001
sys_time_vec_epsi4 = c()
elapsed_time_vec_epsi4 = c()

# Time for Sketched OLS with epsilons = 0.1
for (i in 1:200) {
  SHDXy_list1 = main_SketchedOLS(design_X, respon_Y, epsi_vector[1])
  SHDX1 = SHDXy_list1[[1]]  # unpack
  SHDy1 = SHDXy_list1[[2]]  # unpack
  time_sket1 = system.time(perform_LS(SHDX1, SHDy1))
  usr_time_vec_epsi1 = c(usr_time_vec_epsi1, time_sket1[[1]])
  sys_time_vec_epsi1 = c(sys_time_vec_epsi1, time_sket1[[2]])
  elapsed_time_vec_epsi1 = c(elapsed_time_vec_epsi1, time_sket1[[3]])
}
# time stats
print(paste("Average user time for Sketched OLS with epsi = 0.1: ", mean(usr_time_vec_epsi1)))
print(paste("Average system time for Sketched OLS with epsi = 0.1: ", mean(sys_time_vec_epsi1)))
print(paste("Average elapsed time for Sketched OLS with epsi = 0.1: ", mean(elapsed_time_vec_epsi1)))

# output:
# [1] "Average user time for Sketched OLS with epsi = 0.1:  0.00142500000001064"
# [1] "Average system time for Sketched OLS with epsi = 0.1:  0.000289999999997974"
# [1] "Average elapsed time for Sketched OLS with epsi = 0.1:  0.00180500000104075"

# Time for Sketched OLS with epsilons = 0.05
for (i in 1:200) {
  SHDXy_list2 = main_SketchedOLS(design_X, respon_Y, epsi_vector[2])
  SHDX2 = SHDXy_list2[[1]]  # unpack
  SHDy2 = SHDXy_list2[[2]]  # unpack
  time_sket2 = system.time(perform_LS(SHDX2, SHDy2))
  usr_time_vec_epsi2 = c(usr_time_vec_epsi2, time_sket2[[1]])
  sys_time_vec_epsi2 = c(sys_time_vec_epsi2, time_sket2[[2]])
  elapsed_time_vec_epsi2 = c(elapsed_time_vec_epsi2, time_sket2[[3]])
}
# time stats
print(paste("Average user time for Sketched OLS with epsi = 0.05: ", mean(usr_time_vec_epsi2)))
print(paste("Average system time for Sketched OLS with epsi = 0.05: ", mean(sys_time_vec_epsi2)))
print(paste("Average elapsed time for Sketched OLS with epsi = 0.05: ", mean(elapsed_time_vec_epsi2)))

# output:
# [1] "Average user time for Sketched OLS with epsi = 0.05:  0.00258000000000266"
# [1] "Average system time for Sketched OLS with epsi = 0.05:  0.000475000000000705"
# [1] "Average elapsed time for Sketched OLS with epsi = 0.05:  0.00320000000065193"

# Time for Sketched OLS with epsilons = 0.01
for (i in 1:200) {
  SHDXy_list3 = main_SketchedOLS(design_X, respon_Y, epsi_vector[3])
  SHDX3 = SHDXy_list3[[1]]  # unpack
  SHDy3 = SHDXy_list3[[2]]  # unpack
  time_sket3 = system.time(perform_LS(SHDX3, SHDy3))
  usr_time_vec_epsi3 = c(usr_time_vec_epsi3, time_sket3[[1]])
  sys_time_vec_epsi3 = c(sys_time_vec_epsi3, time_sket3[[2]])
  elapsed_time_vec_epsi3 = c(elapsed_time_vec_epsi3, time_sket3[[3]])
}
# time stats
print(paste("Average user time for Sketched OLS with epsi = 0.01: ", mean(usr_time_vec_epsi3)))
print(paste("Average system time for Sketched OLS with epsi = 0.01: ", mean(sys_time_vec_epsi3)))
print(paste("Average elapsed time for Sketched OLS with epsi = 0.01: ", mean(elapsed_time_vec_epsi3)))

# output:
# [1] "Average user time for Sketched OLS with epsi = 0.01:  0.0113200000000097"
# [1] "Average system time for Sketched OLS with epsi = 0.01:  0.000120000000000573"
# [1] "Average elapsed time for Sketched OLS with epsi = 0.01:  0.0117449999996461"

# Time for Sketched OLS with epsilons = 0.001
for (i in 1:200) {
  SHDXy_list4 = main_SketchedOLS(design_X, respon_Y, epsi_vector[4])
  SHDX4 = SHDXy_list4[[1]]  # unpack
  SHDy4 = SHDXy_list4[[2]]  # unpack
  time_sket4 = system.time(perform_LS(SHDX4, SHDy4))
  usr_time_vec_epsi4 = c(usr_time_vec_epsi4, time_sket4[[1]])
  sys_time_vec_epsi4 = c(sys_time_vec_epsi4, time_sket4[[2]])
  elapsed_time_vec_epsi4 = c(elapsed_time_vec_epsi4, time_sket4[[3]])
}
# time stats
print(paste("Average user time for Sketched OLS with epsi = 0.001: ", mean(usr_time_vec_epsi4)))
print(paste("Average system time for Sketched OLS with epsi = 0.001: ", mean(sys_time_vec_epsi4)))
print(paste("Average elapsed time for Sketched OLS with epsi = 0.001: ", mean(elapsed_time_vec_epsi4)))

# output:
# [1] "Average user time for Sketched OLS with epsi = 0.001:  0.118554999999974"
# [1] "Average system time for Sketched OLS with epsi = 0.001:  0.00997000000000298"
# [1] "Average elapsed time for Sketched OLS with epsi = 0.001:  0.131505000000179"

