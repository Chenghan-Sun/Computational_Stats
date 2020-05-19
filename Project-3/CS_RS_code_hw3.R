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
conv_train = readBin(train_file, integer(), n=prod(src_train[2:4]), endian="big")
train_data = matrix(conv_train, src_train[2], prod(src_train[3:4]), byrow=TRUE)
conv_test = readBin(test_file, integer(), n=prod(src_test[2:4]), endian="big")
test_data = matrix(conv_test, src_test[2], prod(src_test[3:4]), byrow=TRUE)
close(train_file)
close(test_file)

# load training / test set labels
train_labels = Read.mnist('train-labels-idx1-ubyte')
test_labels = Read.mnist('t10k-labels-idx1-ubyte')

# Image compression: create 14×14 images
img_compress = function(matrix, ori_dim=28, new_dim=14){
  img_comp = c()  # initialize vector
  for (i in 1:new_dim){
    for (j in 1:new_dim){
      block_1 = 
      block_2 = 
      img_comp = cbind(img_comp, )
    }
  }
}


#image(matrix(train_labels$labels[], 14,14,byrow = TRUE))



############ Problem 3 ############
