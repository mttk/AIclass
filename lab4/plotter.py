import numpy as np 

# package import checking
MATPLOTLIB_AVAILABLE = True 
try: 
	import matplotlib.pyplot as plt 
	from mpl_toolkits.mplot3d import axes3d, Axes3D, art3d
	from matplotlib import cm
except ImportError: 
	print "Matplotlib library is not available - please install it in order to plot."
	MATPLOTLIB_AVAILABLE = False 


def plot(X, y_actual, y_predicted=None):
	# exit if library doesn't exist
	if not MATPLOTLIB_AVAILABLE: 
		print "Matplotlib not available - exiting plot"
		return 

	# separate cases
	input_dim = X.shape[1]

	if input_dim == 1: 
		plot_2d(X, y_actual, y_predicted)
	elif input_dim == 2: 
		plot_3d(X, y_actual, y_predicted)
	else:
		print "Invalid dimensions of input %d - can only plot 2- and 3-d functions" % input_dim 

def plot_2d(X, y_actual, y_predicted=None):
	"""
		Plot a two-dimensional function
	"""
	if not y_predicted is None:
		plt.title("Predicted vs actual function values")
	else: 
		plt.title("Approximated function samples")

	plt.plot(X, y_actual, 'g.', label = "Actual values")
	if not y_predicted is None:
		plt.plot(X, y_predicted, 'b.', label = "Predicted values")
	plt.show()

def plot_3d(X, y_actual, y_predicted=None):
	fig = plt.figure()

	if y_predicted is None:
		plt.title("Predicted vs actual function values")
	else: 
		plt.title("Approximated function samples")

	ax = Axes3D(fig)

	ax.view_init(elev=30, azim=70)

	scatter_actual = ax.scatter(X[:,0], X[:,1], y_actual, c='g', depthshade=False)
	if not y_predicted is None:
		scatter_predicted = ax.scatter(X[:,0], X[:,1], y_predicted, c='b', depthshade=False)
	
	if y_predicted is None:
		plt.legend((scatter_actual, scatter_predicted),
				('Actual values', 'Predicted values'),
				scatterpoints = 1)

	plt.grid()
	plt.show()


def plot_surface(X, y_actual, NN):
	# exit if library doesn't exist
	if not MATPLOTLIB_AVAILABLE: 
		print "Matplotlib not available - exiting plot"
		return 

	# separate cases
	input_dim = X.shape[1]

	if input_dim == 1: 
		plot_surface_2d(X, y_actual, NN)
	elif input_dim == 2: 
		plot_surface_3d(X, y_actual, NN)
	else:
		print "Invalid dimensions of input %d - can only plot 2- and 3-d functions" % input_dim 

def plot_surface_2d(X, y_actual, NN):
	plt.title("Predicted function with marked training samples")

	plt.plot(X, y_actual, 'go', label = "Actual values")
	
	size = 100

	# estimate the actual surface
	xmin = X.min()
	xmax = X.max()

	xs = np.linspace(xmin, xmax, size)
	xs = xs.reshape(xs.shape[0], -1)

	plt.plot(xs, NN.output(xs), 'b', label = "Predicted surface")

	plt.show()

def plot_surface_3d(X, y_actual, NN):
	fig = plt.figure()
	plt.title("Predicted function with marked training samples")
	ax = Axes3D(fig)

	size = X.shape[0]

	ax.view_init(elev=30, azim=70)
	scatter_actual = ax.scatter(X[:,0], X[:,1], y_actual, c='g', depthshade=False)

	x0s = sorted(X[:,0])
	x1s = sorted(X[:,1])

	x0s, x1s = np.meshgrid(x0s, x1s)
	predicted_surface = np.zeros((size, size))

	for i in range(size):
		for j in range(size):
			predicted_surface[i,j] = NN.output(np.array([x0s[i,j], x1s[i,j]]))

	surf = ax.plot_surface(x0s, x1s, predicted_surface, rstride=2, cstride=2, linewidth=0, cmap=cm.coolwarm, alpha=0.5)

	plt.grid()
	plt.show()
