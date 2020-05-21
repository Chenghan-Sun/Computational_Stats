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

# view clustering images
par(mfrow=c(5,5))
par(mar=c(0,0,0,0))
for(i in 1:25){
  m = matrix(train_data_comp_cluster[i,],14,14,byrow = TRUE)
  image(m[,14:1])
}

############ Problem 3 ############















