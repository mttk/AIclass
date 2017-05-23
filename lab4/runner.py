from networkLayers import *
from transferFunctions import * 
from neuralNet import * 
from geneticAlgorithm import * 

import plotter
import matplotlib.pyplot as plt
import numpy as np
import dataLoader

###
#   Global constants, I/O paths
###

SIN_TRAIN = './data/sine_train.txt'
SIN_TEST = './data/sine_test.txt'

RASTRIGIN_TRAIN = './data/rastrigin_train.txt'
RASTRIGIN_TEST = './data/rastrigin_test.txt'

ROSENBROCK_TRAIN = './data/rosenbrock_train.txt'
ROSENBROCK_TEST = './data/rosenbrock_test.txt'

if __name__ == '__main__':
	# set the random seed for reproducibility of results
	# setting the random seed forces the same results of randoming each
	# time you start the program - that way you can demonstrate your results
	np.random.seed(11071998)


	# Load the train / test data
	# X is the input matrix, y is the target vector
	# X can be a vector (and will be, in the first assignment) as well 

	"""
		To change the function being approximated, just change the paths 
		to the dataset in the arguments of the data loader.s
	"""
	X_train, y_train = dataLoader.loadFrom(SIN_TRAIN)
	X_test, y_test = dataLoader.loadFrom(SIN_TEST)

	# for check, print out the shapes of the input variables
	# the first dimension is the number of input samples, the second dimension
	# is the number of variables 

	print "Train data shapes: ", X_train.shape, y_train.shape 
	print "Test data shapes: ", X_test.shape, y_test.shape 

	# The dimensionality of the input layer of the network is the second
	# dimension of the shape 

	if len(X_train.shape) > 1:
		input_size = X_train.shape[1]
	else: 
		input_size = 1

	# the size of the output layer
	output_size = 1

	NN = NeuralNetwork()

	#  Define the layers of your
	#        neural networks
	#############################
	#       YOUR CODE HERE      #
	#############################



	####################
	#  YOUR CODE ENDS  #
	####################

	def errorClosure(w):
		"""
			A closure is a variable that stores a function along with the environment.
			The environment, in this case are the variables x, y as well as the NN
			object representing a neural net. We store them by defining a method inside
			a method where those values have been initialized. This is a "hacky" way of 
			enforcing the genetic algorithm to work in a generalized manner. This way,
			the genetic algorithm can be applied to any problem that optimizes an error 
			(in this case, this function) by updating a vector of values (in this case,
			defined only by the initial size of the vector). 

			In plain - the genetic algorithm doesn't know that the neural network exists,
			and the neural network doesn't know that the genetic algorithm exists. 
		"""
		# Set the weights to the pre-defined network
		NN.setWeights(w)
		# Do a forward pass of the etwork and evaluate the error according to the
		# oracle (y)
		return NN.forwardStep(X_train, y_train)

	# Check the constructor (__init__) of the GeneticAlgorithm for further instructions
	# on what the parameters are. Feel free to change / adapt any parameters. The defaults
	# are as follows 


	#######################################
	#    MODIFY CODE AT WILL FROM HERE    #
	#######################################

	elitism = 1 # Keep this many of top units in each iteration
	populationSize = 6 # The number of chromosomes
	mutationProbability  = .8 # Probability of mutation
	mutationScale = 10. # Standard deviation of the gaussian noise
	numIterations = 10000 # Number of iterations to run the genetic algorithm for
	errorTreshold = 1e-6 # Lower threshold for the error while optimizing

	GA = GeneticAlgorithm(NN.size(), errorClosure,
		elitism = elitism,
		populationSize = populationSize,
		mutationProbability = mutationProbability,
		mutationScale = mutationScale, 
		numIterations = numIterations, 
		errorTreshold = errorTreshold)


	print_every = 100 # Print the output every this many iterations
	plot_every = 100 # Plot the actual vs estimated functions every this many iterations

	# emulated do-while loop
	done = False
	while not done: 
		done, iteration, best = GA.step()

		if iteration % print_every == 0: 
			print "Error at iteration %d = %f" % (iteration, errorClosure(best))

		if iteration % plot_every == 0: 
			NN.setWeights(best)
			plotter.plot(X_train, y_train, NN.output(X_train)) 
			plotter.plot_surface(X_train, y_train, NN)

	print "Training done, running on test set"
	NN.setWeights(best)

	print "Error on test set: ", NN.forwardStep(X_test, y_test)
	plotter.plot(X_test, y_test, NN.output(X_test))
	plotter.plot_surface(X_test, y_test, NN)
