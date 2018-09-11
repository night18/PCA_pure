#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ====================================================================
# Created By: Chun-Wei Chiang - https://chunwei.org/
# Created Date: 9/9/18, 2:52 PM
# Goal: Using PCA for 10 digits in MNIST (all class). 

from pca import * 

def drawXY(img, x, y, i):
	plt.subplot(x,y,i)
	plt.imshow(img.reshape(28,28), cmap='gray')
	plt.axis('off')

mat_training, lbl_training = getDataSet(5000)
train_mean, train_diff, vec_eigen, top_eigen_idx = getEigen(mat_training)
top_eigen = getTopEigen(vec_eigen, top_eigen_idx, 20)
drawEigen(top_eigen)
digits = {}

for i in range(len(lbl_training)):
	if not digits.has_key(lbl_training[i]):
		digits[lbl_training[i]] = i


for digit in range(10):
	org = array(mat_training[digits[digit]])
	drawXY(org, 1, 4, 1)
	top2 = getReconstructN(org, train_mean, vec_eigen, top_eigen_idx, 2)
	drawXY(top2, 1, 4, 2)
	top5 = getReconstructN(org, train_mean, vec_eigen, top_eigen_idx, 5)
	drawXY(top5, 1, 4, 3)
	top10 = getReconstructN(org, train_mean, vec_eigen, top_eigen_idx, 10)
	drawXY(top10, 1, 4, 4)
	# plt.show() 
	file_name = "img/all/recon_" + str(digit)+ ".png"
	plt.savefig(file_name)