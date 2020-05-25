# This .R code file consists of:
# Problem 1: the MNIST handwritten digit dataset pre-processing 
# Problem 3: implement the EM algorithm
# Arthurs: Chenghan Sun, Ran Sun 
# NOTE: please run this code in the directory (folder) of HW3

############ Problem 1 ############
library(readmnist)
# the MNIST handwritten digit dataset pre-processing
# set path to the current directory
setwd("/Users/furinkazan/Box/STA_243/Comp_Stats/Project-3") 

# data loading for training / test set images
train_file = file("train-images-idx3-ubyte", "rb")
test_file = file("t10k-images-idx3-ubyte", "rb")
src_train = readBin(train_file, integer(), n=4, endian="big")
src_test = readBin(test_file, integer(), n=4, endian="big")

# load as matrix 
conv_train = readBin(train_file, what='raw', n=prod(src_train[2:4]), endian="big")
train_data = matrix(as.integer(conv_train), src_train[2], prod(src_train[3:4]), byrow=TRUE)
conv_test = readBin(test_file, what='raw', n=prod(src_test[2:4]), endian="big")
test_data = matrix(as.integer(conv_test), src_test[2], prod(src_test[3:4]), byrow=TRUE)
close(train_file)
close(test_file)

# data load for training / test set labels
src_train_labels = Read.mnist('train-labels-idx1-ubyte')
src_test_labels = Read.mnist('t10k-labels-idx1-ubyte')
train_labels = src_train_labels$labels
test_labels = src_test_labels$labels

# Image compression
img_compress = function(mat, ori_dim=28, new_dim=14){
  # reduce each 28×28 image into new 14×14 image by deviding 2×2 non-overlapping blocks
  img_comp = c()  # initialize vector
  for (i in 1:new_dim){
    for (j in 1:new_dim){
      block_1 = (29-2*j)*ori_dim + (2*i-1)
      block_2 = (28-2*j)*ori_dim + (2*i-1)
      block_3 = (29-2*j)*ori_dim + 2*i
      block_4 = (28-2*j)*ori_dim + 2*i
      img_comp = cbind(img_comp, (mat[, block_1] + mat[, block_2] + mat[, block_3] + mat[, block_4]) / 4)
    }
  }
  return(img_comp)
}

# Apply image compression
train_data_comp = img_compress(train_data)
test_data_comp = img_compress(test_data)

# view compressed images
par(mfrow=c(5,5))
par(mar=c(0,0,0,0))
for(i in 1:25){
  m = matrix(train_data_comp[i,],14,14,byrow = TRUE)
  image(m[,14:1])
}

# only clustering the digits {0, 1, 2, 3, 4} 
digits = c(0,1,2,3,4)
train_data_comp_cluster = train_data_comp[train_labels %in% digits, ]
train_labels_cluster = train_labels[train_labels %in% digits]

test_data_comp_cluster = test_data_comp[test_labels %in% digits,]
test_labels_cluster = test_labels[test_labels %in% digits]

# Note: use the above four processed datasets for Part 3 EM algorithm 

# view clustering images
par(mfrow=c(5,5))
par(mar=c(0,0,0,0))
for(i in 1:25){
  m = matrix(train_data_comp_cluster[i,],14,14,byrow = TRUE)
  image(m[,14:1])
}

############ Problem 3 ############
#（i): Program the EM algorithm you derived for mixture of spherical Gaussians. Assume 5
# clusters. Terminate the algorithm when the fractional change of the log-likelihood goes under
# 0.0001. (Try 3 random initializations and present the best one in terms of maximizing 
# the likelihood function).

# First of all, define some helper functions before the EM - spherical Gaussians algorithm
# Functions to initialize group of parameters for the gaussian kernel

# @Helper Function 1 (apply for both algorithms)
init_para_pi = function(num_cluster) {
  # randomly initialize distribution of parameter pi
  # Params:
    # num_cluster: number of clusters
  # Return:
    # init_pi: initialized parameter pi with length of num_cluster
  para_pi = runif(num_cluster)
  init_pi = para_pi / sum(para_pi)  # weight percents 
  return(init_pi)
}

# @Helper Function 2 (apply for both algorithms)
init_para_mu = function(data_X, num_cluster, para_pi) {
  # initialize parameter mu (normal distribution) base on parameter pi
  # Params:
    # data_X: data set
    # num_cluster: number of clusters
    # para_pi: initialized parameter pi
  # Return:
    # init_mu: list of initialized parameter mu
  init_mu = c()
  data_idx = sample(1:num_cluster, nrow(data_X), replace=T, prob=para_pi)
  for(c in 1:num_cluster) {
    mu = apply(data_X[data_idx==c, ], 2, mean)
    init_mu = cbind(init_mu, mu)
  }
  return(t(init_mu))
}

# @Helper Function 3 (apply for both algorithms)
init_para_sigma = function(data_X, num_cluster, para_pi, gaussians) {
  # initialize parameter sigma (normal distribution) base on hint 3 of HW3
  # Params:
    # data_X: data set
    # num_cluster: number of clusters
    # para_pi: initialized parameter pi
    # gaussians: type of gaussians
  # Return:
    # init_sigma: list of initialized parameter sigma
  init_sigma = c()
  adjust = 0.05  # based on hint #2 in HW3
  data_idx = sample(1:num_cluster, nrow(data_X), replace=T, prob=para_pi)
  for(c in 1:num_cluster) {
    if (gaussians == 'sphe'){
      sigma = sd(data_X[c, ])^2
      init_sigma = c(init_sigma, sigma)  # 1 param
    }
    else if (gaussians == 'diag') {
      sigma = apply(data_X[data_idx==c, ], 2, sd)^2 + adjust  # based on hint #2 in HW3
      init_sigma = cbind(init_sigma, sigma)  # d params
    }
  }
  return(t(init_sigma))
}

# Based on the hints, construct a helper function to compute log-likelihood by matirx Fij
# @Helper Function 4 (only apply for spherical case)
sphe_LL_constructor = function(data_X, num_cluster, para_pi, para_mu, para_sigma) {
  # compute log-likelihood by matirx Fij
  # Params:
    # data_X: dataset 
    # num_cluster: number of clusters
    # para_pi: parameter pi
    # para_mu: parameter mu
    # para_sigma: parameter sigma
  # Return:
    # list of two vars:
      # 1. Matrix Fij
      # 2. log-likelihood
  # implement Matrix Fij as result of Problem 2 Part B
  log_kernel_F = c()
  d = ncol(data_X)
  # contrcut loop for log-kernel of normal density 
  for(c in 1:num_cluster){
    kernel = -(t(data_X) - para_mu[c, ])^2 / 2
    log_term_1 = kernel / para_sigma[c]
    sum_log_term_1 = apply(log_term_1, 2, sum)
    log_term_2 = -d/2*log(para_sigma[c])
    log_term_3 = log(para_pi[c])
    log_term_4 = -d/2*log(2*pi)
    log_term_tot = sum_log_term_1 + log_term_2 + log_term_3 + log_term_4
    log_kernel_F = cbind(log_kernel_F, log_term_tot)
  }
  A = apply(log_kernel_F, 1, max)  # based on hint #1 in HW3
  kernel_F = exp(log_kernel_F - A)  # based on hint #1 in HW3
  sum_kernel_F = apply(kernel_F, 1, sum)
  matF = kernel_F / sum_kernel_F  # final form of matrix Fij
  LL = sum(log(sum_kernel_F)) + sum(A)  # final form of log-likelihood
  return(list(matF, LL))
}

# Based on the results of Problem 2 Part e M-step for mixture of spherical Gaussians,
# construct three update functions for parameters

# @Helper Function 5 (apply for both algorithms)
update_para_pi = function(matF) {
  # update parameter pi
  # Params:
    # matF: matrix Fij
  # Return:
    # updated_pi: updated parameter pi
  updated_pi = apply(matF, 2, mean)
  return(updated_pi)
}

# @Helper Function 6 (apply for both algorithms)
update_para_mu = function(data_X, matF) {
  # update parameter mu 
  # Params:
    # data_X: dataset 
    # matF: matrix Fij
  # Return:
    # updated_mu: updated parameter mu
  matF_weights = t(matF) / apply(matF, 2, sum)
  updated_mu = matF_weights %*% data_X
  return(updated_mu)
}

# @Helper Function 7 (apply for both algorithms)
update_para_sigma = function(data_X, num_cluster, matF, updated_mu, gaussians) {
  # update parameter sigma
  # Params:
    # data_X: dataset 
    # num_cluster: number of clusters
    # matF: matrix Fij
    # gaussians: type of gaussians
    # updated_mu: updated parameter mu
  # Return:
    # updated_sigma: updated parameter sigma
  updated_sigma = c()
  adjust = 0.05  # based on hint #2 in HW3
  matF_weights = t(matF) / apply(matF, 2, sum)
  
  for(c in 1:num_cluster) {
    kernel = (t(data_X) - updated_mu[c, ])^2
    kernel_weight = t(matF_weights)[ ,c]
    proto_sigma = kernel %*% kernel_weight
    if (gaussians == 'sphe'){
      sigma = mean(proto_sigma)
      updated_sigma = c(updated_sigma, sigma)
    }
    else if (gaussians == 'diag') {
      updated_sigma = cbind(updated_sigma, proto_sigma)
    }
  }
  
  # based on hint #2 in HW3
  if (gaussians == 'diag') {
    updated_sigma = t(updated_sigma) + adjust
  }
  return(updated_sigma)
}

# implement EM - mixture of spherical Gaussians algorithm
# @main EM function (only apply for spherical case)
EM_sphe_gaus = function(data_X, num_cluster, tol=0.0001, maxiters=500) {
  # implement EM - mixture of spherical Gaussians
  # Params:
    # data_X: dataset 
    # num_cluster: number of clusters
    # tol: tolerance 
    # maxiters: maximum number ofiterations
  # Return:
    # list of vars: 1-3: updated parameters; 4: matF; 5: log-likelihood
  # initialize all parameters
  init_pi = init_para_pi(num_cluster)
  init_mu = init_para_mu(data_X, num_cluster, init_pi)
  init_sigma = init_para_sigma(data_X, num_cluster, init_pi, 'sphe')
  
  # compute log-likelihood
  sphe_results = sphe_LL_constructor(data_X, num_cluster, init_pi, init_mu, init_sigma)
  
  # set counter
  iter = 10
  diff = 100
  
  # while loop for updating rules, ensemble helper functions 
  for(iter in 1:maxiters) {
    process_matF = sphe_results[[1]]
    process_LL = sphe_results[[2]]
    new_pi = update_para_pi(process_matF)
    new_mu = update_para_mu(data_X, process_matF)
    new_sigma = update_para_sigma(data_X, num_cluster, process_matF, new_mu, 'sphe')
    
    new_sphe_results = sphe_LL_constructor(data_X, num_cluster, new_pi, new_mu, new_sigma)
    new_matF = new_sphe_results[[1]]
    new_LL = new_sphe_results[[2]]
    diff = abs(new_LL - process_LL) / abs(process_LL)
    
    # stopping criteria
    if(diff < tol) {
      print(paste("Algorithm finished by reaching the tolerance."))
      print(paste('Total number of iterations for EM-spherical Gaussians = ', iter))
      break
    }
    
    if(iter >= maxiters) {
      warning('Algorithm unfinished by reaching the maximum iterations.')
      break
    }
    
    # update 
    sphe_results = new_sphe_results
    iter = iter + 1
  }
  print(paste('Final log-likelihood for EM-spherical Gaussians = ', sphe_results[[2]]))
  return(list(new_pi, new_mu, new_sigma, sphe_results[[1]], sphe_results[[2]]))
}

# now perform the EM - mixture of spherical Gaussians algorithm
# As required, try 3 random initializations

# @Experiment 1
EM_sphe_results_1 = EM_sphe_gaus(train_data_comp_cluster, 5)

# @Outputs
  #[1] "Algorithm finished by reaching the tolerance."
  #[1] "Total number of iterations for EM-spherical Gaussians =  18"
  #[1] "Final log-likelihood for EM-spherical Gaussians =  -31831226.1846782"

# @Experiment 2
EM_sphe_results_2 = EM_sphe_gaus(train_data_comp_cluster, 5)

# @Outputs

# @Experiment 3
EM_sphe_results_3 = EM_sphe_gaus(train_data_comp_cluster, 5)

# @Outputs



# Now make use of the true labels and calculate the error of the algorithm
# define a function for this task
# @(only apply for spherical case)
EM_sphe_labels_pred = function(data_X, train_labels, num_cluster, EM_results) {
  # make use of the true labels and calculate the error of the EM algorithm
  # Workflow: 
    # data (n X d) -> matF (n X 5) --> cluster (n X 1) --> map labels (n X 1) 
    # --> predicted lables (n X 1)
  # Params:
    # data_X: dataset 
    # train_labels: training data labels 
    # num_cluster: number of clusters
    # EM_results: list of vars: 1-3: updated parameters; 4: matF; 5: log-likelihood
  # Return:
    # pred_labels: prediction labels 
  
  # mapping labels to clusters 
  matF = EM_results[[4]]  # get back output of matrix Fij
  map_labels = c()  # cluster labels list 
  for (c in 1:num_cluster) {
    max_prob_idx = apply(matF, 1, max)
    true_labels = train_labels[(matF == max_prob_idx)[ ,c]]
    tab_true_labels = table(true_labels)
    most_label = names(tab_true_labels)[tab_true_labels == max(tab_true_labels)]
    map_labels = c(map_labels, most_label)
  }
  
  # predict labels 
  new_sphe_results = sphe_LL_constructor(data_X, num_cluster, EM_results[[1]], EM_results[[2]], EM_results[[3]])
  new_matF = new_sphe_results[[1]]
  pred_labels = rep(0, nrow(data_X))
  for(c in 1:num_cluster) {
    max_prob_idx = apply(new_matF, 1, max)
    pred_labels[(new_matF == max_prob_idx)[ ,c]] = map_labels[c]
  }
  return(pred_labels)
}

#@ (apply for both algorithms)
EM_pred_error =  function(train_data, train_labels, test_labels, pred_labels, verb_dataset) {
  # Calculate the error rate
  # Params:
    # train_data: training set 
    # train_labels: training data labels 
    # test_labels: test data labels 
    # pred_labels: prediction labels 
    # verb_dataset: verbose training set or test set
  # Return:
    # pred_error: prediction error rate
  n1 = nrow(train_data)
  n2 = length(test_labels)
  if (verb_dataset == 'train') {
    pred_error = sum(pred_labels != train_labels) / n1
  }
  else if (verb_dataset == 'test') {
    pred_error = sum(pred_labels != test_labels) / n2
  }
  print(paste('The predictive error rate = ', pred_error))
  return(pred_error)
}

# Calculate the error of the EM - mixture of spherical Gaussians algorithm

# @prediction error rate for training set
trainset_sphe_pred_labels = EM_sphe_labels_pred(train_data_comp_cluster, 
                                          train_labels_cluster, 5, EM_sphe_results_1)
trainset_sphe_pred_error = EM_pred_error(train_data_comp_cluster, train_labels_cluster, 
                                         test_labels_cluster, trainset_sphe_pred_labels, 'train')
# @Output:
  # 0.127533

# @prediction error rate for test set
testset_sphe_pred_labels = EM_sphe_labels_pred(test_data_comp_cluster, 
                                               train_labels_cluster, 5, EM_sphe_results_1)
testset_sphe_pred_error = EM_pred_error(train_data_comp_cluster, train_labels_cluster, 
                                         test_labels_cluster, testset_sphe_pred_labels, 'test')
# @Output:
  # 0.1196731

# last component, plot section 
# @(apply for both algorithms)
EM_cluster_plot = function(num_cluster, final_para_mu) {
  # visualize clustering result
  # Params:
    # num_cluster: number of cluster
    # final_para_mu: final parameter mu
  par(mfrow = c(2,3))
  for (c in 1:num_cluster) {
    m = matrix(final_para_mu[c, ], 14, 14, byrow=TRUE)
    image(m, main=paste("cluster = ", c))
  }
}

# @Output
EM_cluster_plot(5, EM_sphe_results_1[[2]])

# end of (i)


# (ii) Program the EM algorithm you derived for mixture of diagonal Gaussians. Assume 5
# clusters. Terminate the algorithm when the fractional change in the log-likelihood goes under
# 0.0001. (Try 3 random initializations and present the best one in terms of maximizing the
# likelihood function).

# For this EM - diagonal Gaussians algorithm
# We update three new functions specifically for this section, other helper functions could be applied 
# for both algorithms

# @new function 1 (only apply for diagonal case)
diag_LL_constructor = function(data_X, num_cluster, para_pi, para_mu, para_sigma) {
  # compute log-likelihood by matirx Fij
  # Params:
    # data_X: dataset 
    # num_cluster: number of clusters
    # para_pi: parameter pi
    # para_mu: parameter mu
    # para_sigma: parameter sigma
  # Return:
    # list of two vars:
      # 1. Matrix Fij
      # 2. log-likelihood
  # implement Matrix Fij as result of Problem 2 Part B
  log_kernel_F = c()
  d = ncol(data_X)
  # contrcut loop for log-kernel of normal density 
  for(c in 1:num_cluster){
    kernel = -(t(data_X) - para_mu[c, ])^2 / 2
    log_term_1 = kernel / para_sigma[c, ]
    sum_log_term_1 = apply(log_term_1, 2, sum)
    log_term_2 = -1/2*sum(log(para_sigma[c,]))
    log_term_3 = log(para_pi[c])
    log_term_4 = -d/2*log(2*pi)
    log_term_tot = sum_log_term_1 + log_term_2 + log_term_3 + log_term_4
    log_kernel_F = cbind(log_kernel_F, log_term_tot)
  }
  A = apply(log_kernel_F, 1, max)  # based on hint #1 in HW3
  kernel_F = exp(log_kernel_F - A)  # based on hint #1 in HW3
  sum_kernel_F = apply(kernel_F, 1, sum)
  matF = kernel_F / sum_kernel_F  # final form of matrix Fij
  LL = sum(log(sum_kernel_F)) + sum(A)  # final form of log-likelihood
  return(list(matF, LL))
}

# implement EM - mixture of diagonal Gaussians algorithm
# @main EM function (only apply for diagonal case)
EM_diag_gaus = function(data_X, num_cluster, tol=0.0001, maxiters=500) {
  # implement EM - mixture of diagonal Gaussians
  # Params:
    # data_X: dataset 
    # num_cluster: number of clusters
    # tol: tolerance 
    # maxiters: maximum number ofiterations
  # Return:
    # list of vars: 1-3: updated parameters; 4: matF; 5: log-likelihood
  # initialize all parameters
  init_pi = init_para_pi(num_cluster)
  init_mu = init_para_mu(data_X, num_cluster, init_pi)
  init_sigma = init_para_sigma(data_X, num_cluster, init_pi, 'diag')
  
  # compute log-likelihood
  diag_results = diag_LL_constructor(data_X, num_cluster, init_pi, init_mu, init_sigma)
  
  # set counter
  iter = 10
  diff = 100
  
  # while loop for updating rules, ensemble helper functions 
  for(iter in 1:maxiters) {
    process_matF = diag_results[[1]]
    process_LL = diag_results[[2]]
    new_pi = update_para_pi(process_matF)
    new_mu = update_para_mu(data_X, process_matF)
    new_sigma = update_para_sigma(data_X, num_cluster, process_matF, new_mu, 'diag')
    
    new_diag_results = diag_LL_constructor(data_X, num_cluster, new_pi, new_mu, new_sigma)
    new_matF = new_diag_results[[1]]
    new_LL = new_diag_results[[2]]
    diff = abs(new_LL - process_LL) / abs(process_LL)
    
    # stopping criteria
    if(diff < tol) {
      print(paste("Algorithm finished by reaching the tolerance."))
      print(paste('Total number of iterations for EM-diagonol Gaussians = ', iter))
      break
    }
    
    if(iter >= maxiters) {
      warning('Algorithm unfinished by reaching the maximum iterations.')
      break
    }
    
    # update 
    diag_results = new_diag_results
    iter = iter + 1
  }
  print(paste('Final log-likelihood for EM-diagonol Gaussians = ', diag_results[[2]]))
  return(list(new_pi, new_mu, new_sigma, diag_results[[1]], diag_results[[2]]))
}









