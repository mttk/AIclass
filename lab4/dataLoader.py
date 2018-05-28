import numpy as np 
import os, sys 

def loadFrom(dataSource):
	"""
		Load a space separated list of float values from a file. The
		method assumes that the target variable will be the last one
		in the sequence.
	"""

	with open(dataSource, 'r') as dataFile: 
		Xs, ys = [], []
		for line in dataFile: 
			# remove trailing whitespace and split over spaces
			parts = [float(el) for el in line.strip().split()]

			Xs.append(parts[:-1])
			ys.append(parts[-1])

	Xs, ys = np.array(Xs), np.array(ys)

	print "Loaded data"
	print "Shape of input variable: ", Xs.shape, "Shape of output variable: ", ys.shape 
	return Xs, ys 
