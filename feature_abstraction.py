#!/usr/bin/env python3
import numpy as np
from glob import glob
import cv2
import h5py
import os
'''
	This script is a demo of feature abstraction using optical flow operations.

	Follow the nuScenes's tutorial (https://www.nuscenes.org/) to extract the mini set. 
	Only images from the CAM_FRONT channel are sued.

	There are 10 video scenes in the v1.0-mini version. 
	All frames (static PNG images) extracted from one video scene shall be placed in one scene folder. 
	Place all scene folders in one folder, e.g. data/nuscenes-v1.0-mini/ 

	Each scene will be splitted into a test (1-48 frames) and a train (49-last frames) feature file in hdf5 format

'''

class FeatureAbstraction:
	def __init__(self, sourcepath):

		# Specify output root
		self.dstroot = sourcepath.split("/")[0] + "/"				
		# List file for train and test loaders
		self.list_train, self.list_test = [], []
		# Resize inputs
		self.newdim = (320, 240)

		for scenefolder in glob(sourcepath + "*"):
			frames = []
			# It is time series. Frame order matters !!!
			for imagefile in sorted(glob(scenefolder + "/*.png")):
				frames.append(imagefile)

			# Fetch the 1st frame
			im1 = cv2.cvtColor(cv2.resize(cv2.imread(frames[0]), self.newdim), cv2.COLOR_BGR2GRAY)
			features_x, features_y = [], []
			im2, flow = None, None
			for i in range(1, len(frames)):
				if im2 is None:
					im2 = cv2.cvtColor(cv2.resize(cv2.imread(frames[i]), self.newdim), cv2.COLOR_BGR2GRAY)
				else:
					im1 = im2[:]
					im2 = cv2.cvtColor(cv2.resize(cv2.imread(frames[i]), self.newdim), cv2.COLOR_BGR2GRAY)

				flow = cv2.calcOpticalFlowFarneback(im1, im2, flow, 
						pyr_scale = 0.5, levels = 1, iterations = 1, 
						winsize = 11, poly_n = 5, poly_sigma = 1.1,  
						flags = 0 if flow is None else cv2.OPTFLOW_USE_INITIAL_FLOW )
				features_x.append(cv2.resize(flow[..., 0], None, fx=0.5, fy=0.5))
				features_y.append(cv2.resize(flow[..., 1], None, fx=0.5, fy=0.5))

			# Write optic flow fields of one video episode to a h5 file
			# First 48 frames for test, rest for train
			trainfile = self.dstroot + "train." + scenefolder.split("/")[-1] + ".h5"
			with h5py.File(trainfile, "w") as f:
				f.create_dataset("x", data=features_x[48:])
				f.create_dataset("y", data=features_y[48:])
			self.list_train.append(trainfile)

			testfile = self.dstroot + "test." + scenefolder.split("/")[-1] + ".h5"
			with h5py.File(testfile, "w") as f:
				f.create_dataset("x", data=features_x[:48])
				f.create_dataset("y", data=features_y[:48])
			self.list_test.append(testfile)

		h5fillist_train = self.dstroot + sourcepath.split("/")[-2]+".train"
		with open(h5fillist_train, "w") as f:
			for scene in self.list_train: 
				f.write(scene+"\n")

		h5filelist_test = self.dstroot + sourcepath.split("/")[-2]+".test"
		with open(h5filelist_test, "w") as f:
			for scene in self.list_test: 
				f.write(scene+"\n")

		print("See feature extraction results in {} and {}".format(h5fillist_train, h5filelist_test))


if __name__ == "__main__":
	sourceroot = 'data/nuscenes-v1.0-mini/'
	if os.path.isdir(sourceroot):
		FeatureAbstraction(sourceroot)