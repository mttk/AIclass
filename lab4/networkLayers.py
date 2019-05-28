import numpy as np

class NetworkLayer(object):
	"""
		Base class for representing different possible types of neural network 
		layers.
		Defines the base methods and documentations of the class and its methods.
	"""

	def output(self, X):
		"""
			Pass the input values (X) through the layer of the network
		"""
		pass 

	def size(self):
		"""
			Return the number of weights in the layer 
		""" 
		return 0  

	def getWeightsFlat(self): 
		"""
			Return the array of weigts in the layer 	
		"""
		return np.array([]) 

	def setWeights(self, weights):
		"""
			Reset the weight vector/matrix 
		"""

		pass

	def __repr__(self):
		"""
			The string __repr__esentation of the object
		"""
		return self.name


class IdentityLayer(NetworkLayer):
	"""
		A dummy layer that simply passes the input through.
	"""
	i = 1 # static counter

	def __init__(self, name='Identity layer'):
		"""
			Dummy initialization - no weight vector.
		"""
		self.name = name + " " + str(IdentityLayer.i+1)
		# static referencing
		IdentityLayer.i+=1

	def output(self, X):
		return X


class LinearLayer(NetworkLayer): 
	"""
		A layer that outputs a dot product between the input (X)
		and the weight (W) vector. 
	""" 
	i = 1 # static counter


	def __init__(self, inputShape, outputShape, name='Linear Layer'):
		"""
			The linear layer also takes a weight vector as an input
		"""
		self.name = name + " " + str(LinearLayer.i+1)
		# static referencing
		LinearLayer.i+=1

		self.weights = np.random.randn(inputShape, outputShape) * 0.1
		self.biases = np.zeros(outputShape)

	def output(self, X): 	
		"""
			Calculate the dot product of weights and inputs + biases 
		"""	
		out = X.dot(self.weights) + self.biases
		return out

	def size(self):
		"""
			Return the total number of weights in the layer
		"""
		return self.weights.size + self.biases.size

	def getWeightsFlat(self):
		"""
			Returns a one-dimensional representation of the weights in the layer 
		"""
		return np.append(self.weights.flatten(), self.biases.flatten())		

	def setWeights(self, flat_vector):
		"""
			Assumes that the weights will be stored in the same order as the flattened ones 
		""" 

		# separated for readability
		# first split the bias and vector parts
		bias_part = flat_vector[-self.biases.size:]
		weight_part = flat_vector[:-self.biases.size]

		# reshape to fit - biases are always 1-dimensional (1 bias per node in layer)
		self.biases = bias_part
		self.weights = np.reshape(weight_part, self.weights.shape)




class SigmoidLayer(NetworkLayer):
	"""
		A layer that outputs the sigmoid function for each element 
		in the input (X). 
	""" 
	i = 1 # static counter

	def __init__(self, name='Sigmoid Layer'):
		"""
			The linear layer also takes a weight vector as an input
		"""
		self.name = name + " " + str(SigmoidLayer.i+1)
		# static referencing
		SigmoidLayer.i+=1


	def sigmoid(self, x): 
		return 1. / (1. + np.exp(-x))

	def output(self, X): 
		out = self.sigmoid(X)
		return out 


class FunctionLayer(NetworkLayer):
	"""
		A generalization of the sigmoid layer for any provided transfer
		function - the outpu of the layer is the function applied to 
		each element in the input. 
	"""

	i = 1 # static counter

	def __init__(self, f):
		"""
			The linear layer also takes a weight vector as an input
		"""
		self.name = f.__name__ + " Function Layer " + str(FunctionLayer.i+1)
		# static FunctionLayer
		FunctionLayer.i+=1

		self.f = f

	def output(self, X): 
		"""
			Appies the given function (Note: the function has to be vectorized for 
			the layer to work in the general case)
		"""
		out = self.f(X)
		return out 

