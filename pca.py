#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ====================================================================
# Created By: Chun-Wei Chiang - https://chunwei.org/
# Created Date: 9/7/18, 10:21 PM
# Goal: Get the eigen vector and draw the projection using pca

from mnist import MNIST
from numpy import *
from pprint import pprint
import matplotlib.pyplot as plt

def getEigen(mat_data): 
	vec_mean = mean(mat_data, axis = 0) #axis(0: column, 1: row)
	diff_data = mat_data - vec_mean #remove mean
	mat_cov = cov(diff_data, rowvar = 0) #get covariance matrix. For mnist, it should be a (28*28)^2 matrix
	set_printoptions(threshold=nan)
	vec_eig_vals,mat_eig_vects = linalg.eig(mat(mat_cov)) #Compute the eigenvalues(28*1) and right eigenvectors(28*28) of a square array.
	eig_val_index = argsort(-vec_eig_vals) #sort, sort goes largest to smallest

	return vec_mean, diff_data, mat_eig_vects, eig_val_index

def getTopEigen(d_eig_vects, d_sort_index, top_n):
	top_eig_index = d_sort_index[:top_n]
	top_eigen = d_eig_vects[:,top_eig_index] #get the top n eigen vector. Acttually it is the tranport of eigen
	return top_eigen


def getProjections(d_diff, top_eigen):
	#The differnce between the each data (5000*784) and the mean multiple the eigen vector (784*20) to get lower dimensional data.
	proj = d_diff * top_eigen 
	return proj

def getRevProjections(low_d_proj, top_eigen, d_mean):
	#reverse the projection to the original coordinate axis
	rev_proj = (low_d_proj * top_eigen.T) +  d_mean
	return rev_proj.real

def getReconstructN(sample_img,d_mean, d_eig_vects, d_sort_index, top_n):
	eigen = getTopEigen(d_eig_vects, d_sort_index, top_n)
	d_diff = sample_img - d_mean
	low_d_proj = getProjections(d_diff, eigen)
	reconstruct = getRevProjections(low_d_proj, eigen, d_mean)
	
	return reconstruct

def drawEigenXY(eigen, y, x, digit):
	dimension_num = eigen.shape[1]
	for i in range (dimension_num):
		# pprint(eigen.T[i])
		v = eigen.T[i]
		plt.subplot(x,y, i+1)
		m = array(v.real).reshape(28,28)
		plt.imshow(m, cmap='gray')
		plt.axis('off')

	# plt.show()
	file_name = "img/seperate/eigen_" + str(digit)+ ".png"
	plt.savefig(file_name)

def drawEigen (eigen):
	dimension_num = eigen.shape[1]
	plot_h = floor(sqrt(dimension_num))
	plot_w = ceil(dimension_num/plot_h)
	for i in range (dimension_num):
		# pprint(eigen.T[i])
		v = eigen.T[i]
		plt.subplot(plot_w,plot_h, i+1)
		m = array(v.real).reshape(28,28)
		plt.imshow(m, cmap='gray')
		plt.axis('off')

	# plt.show()
	file_name = "img/all/eigen.png"
	plt.savefig(file_name)

	



def getDataSet(data_amount):
	# For global PCA
	mndata = MNIST('sample') #sample is the directory of the MNIST
	images, labels = mndata.load_training() #image is 28*28 image, labels is the number which the image presents
	mat_training = zeros((data_amount,28*28)) #initialize the matrix size
	lbl_training = [] 

	for i in range(data_amount) : #get the data_amount of the data
		mat_training[i] = images[i]
		lbl_training.append(labels[i])

	return mat_training, lbl_training


def getDigitData(digit, data_amount):
	mndata = MNIST('sample') #sample is the directory of the MNIST
	images, labels = mndata.load_training() #image is 28*28 image, labels is the number which the image presents
	target = []

	for i in range(data_amount) : #get the data_amount of the data
		if labels[i] == digit:
			target.append(i);

	mat_training = zeros((len(target),28*28)) #initialize the matrix size
	for i in range(len(target)) :
		mat_training[i] = images[target[i]]

	return mat_training
