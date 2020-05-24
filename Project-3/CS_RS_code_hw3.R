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
# （i): Program the EM algorithm you derived for mixture of spherical Gaussians. Assume 5
# clusters. Terminate the algorithm when the fractional change of the log-likelihood goes under
# 0.0001. (Try 3 random initializations and present the best one in terms of maximizing 
# the likelihood function).

# First of all, define some helper functions before the EM - spherical Gaussians algorithm
# Functions to initialize group of parameters for the gaussian kernel
# @Helper Function 1
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

# @Helper Function 2
init_para_mu = function(data_X, num_cluster, para_pi) {
  # initialize parameter mu (normal distribution) base on parameter pi
  # Params:
    # data_X: data set
    # num_cluster: number of clusters
    # para_pi: initialized parameter pi
  # Return:
    # para_mu: list of initialized parameter mu
  para_mu = c()
  data_idx = sample(1:num_cluster, nrow(data_X), replace=T, prob=para_pi)
  for(c in 1:num_cluster) {
    mu = apply(data_X[c, ], 2, mean)
    para_mu = cbind(para_mu, mu)
  }
  return(t(para_mu))
}

# @Helper Function 3
init_para_sigma = function(data_X, num_cluster, para_pi, gaussians) {
  # initialize parameter sigma (normal distribution) base on parameter pi
  # Params:
    # data_X: data set
    # num_cluster: number of clusters
    # para_pi: initialized parameter pi
    # gaussians: type of gaussians
  # Return:
    # para_sigma: list of initialized parameter sigma
  para_sigma = c()
  adjust = 0.05
  for(c in 1:num_cluster) {
    if (gaussians == 'sphe'){
      sigma = sd(data_X[c, ])^2
    }
    else if (gaussians == 'diag') {
      sigma = apply(data_X[c, ], 2, sd)^2 + adjust
    }
    para_sigma = cbind(para_sigma, sigma)
  }
  return(t(para_sigma))
}

# Based on the hints, construct a helper function to compute log-likelihood by matirx Fij
sphe_matF_constructor = function(X, para_pi, para_mu, para_sigma, ) {
  # assign a sample xi to a cluster j, implement Matrix Fij
  # Params:
  # Return: 
  
  # 
  
}












