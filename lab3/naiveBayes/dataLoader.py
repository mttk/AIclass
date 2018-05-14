import cPickle, os, sys, pacman
import util
from game import Directions

# for feature extraction
POSSIBLE_MOVES = set([Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP])
STAY = Directions.STOP

class FeatureExtractor(object):
	"""
	The common class for every feature extractor you might implement.

	The feature extactor accepts a GameState and returns a dictionary
	of features where the keys are the names of the features, and the
	values are the features
	"""

	def __init__(self, name='baseExtractor'):
		"""
		Any required initialisation goes here
		"""
		self.name = name

	def extract(state):
		"""
		The logic for extracting feature(s) from a given GameState object
		"""
		abstract

class FoodEaten(FeatureExtractor):
	"""
	A feature extractor checks if the food count reduced
	"""
	def __init__(self, name='foodEaten'):
		self.name = name

	def extract(self, state):
		"""
		input: GameState object
		return: if the food count reduced
		"""

		if len(state.getGhostPositions()) < 1:
			print("Error: game has zero ghosts on the map.")
			sys.exit(0)

		legalMoves = state.getLegalActions()
		illegalMoves = POSSIBLE_MOVES - set(legalMoves) # we will create placeholder features for the illegal moves

		featureName = 'foodEaten_{}' # the {} is a placeholder for later string formatting
									 # we will compute the same feature for each possible next state
									 # and for each of nGhosts ghosts
		features = {}
		curFood = state.getNumFood()
		for action in legalMoves:
			successorState = state.generateSuccessor(0, action)
			
			features[featureName.format(action)] = (curFood - successorState.getNumFood()) > 0

		for action in illegalMoves: 
			successorState = state.generateSuccessor(0, STAY)
			
			features[featureName.format(action)] = (curFood - successorState.getNumFood()) > 0

		return features

class GhostCloser(FeatureExtractor):
	"""
	A feature extractor checks if the minimum distance towards a ghost reduced
	"""
	def __init__(self, name='ghostCloser'):
		self.name = name

	def extract(self, state):
		"""
		input: GameState object
		return: did we move closer towards the closest ghost for each action
		"""

		if len(state.getGhostPositions()) < 1:
			print("Error: game has zero ghosts on the map.")
			sys.exit(0)

		legalMoves = state.getLegalActions()
		illegalMoves = POSSIBLE_MOVES - set(legalMoves) # we will create placeholder features for the illegal moves

		featureName = 'ghostCloser_{}' # the {} is a placeholder for later string formatting
									 # we will compute the same feature for each possible next state
									 # and for each of nGhosts ghosts
		features = {}
		closestGhost = self.computeDistance(state)
		for action in legalMoves:
			successorState = state.generateSuccessor(0, action)
			
			features[featureName.format(action)] = (closestGhost - self.computeDistance(successorState)) > 0

		for action in illegalMoves: 
			successorState = state.generateSuccessor(0, STAY)
			
			features[featureName.format(action)] = (closestGhost - self.computeDistance(successorState)) > 0

		return features

	def computeDistance(self, state):
		"""
		Computes the distance between pacman and the closest ghost
		"""

		ghosts = state.getGhostPositions()
		pacmanPosition = state.getPacmanPosition()

		closestGhostDistance = min([util.manhattanDistance(ghost, pacmanPosition) for ghost in ghosts])
		return int(closestGhostDistance)


class FoodCounter(FeatureExtractor):
	"""
	A simple feature extractor which returns the number of food dots 
	on the map.
	"""

	def __init__(self, name='foodCounter'):
		self.name = name

	def extract(self, state):
		"""
		input: GameState object
		return: the number of food dots in every subsequent state
		"""

		legalMoves = state.getLegalActions()
		illegalMoves = POSSIBLE_MOVES - set(legalMoves) # we will create placeholder features for the illegal moves

		featureName = 'foodCount_{}' # the {} is a placeholder for later string formatting
									 # we will compute the same feature for each possible next state
		features = {}

		# features for all possible transitions
		for action in legalMoves:
			successorState = state.generateSuccessor(0, action)
			features[featureName.format(action)] = successorState.getNumFood()

		# for impossible transitions, pretend that we stay at the same spot
		for action in illegalMoves:
			successorState = state.generateSuccessor(0, STAY)

			features[featureName.format(action)] = successorState.getNumFood()

		return features

class ClosestGhost(FeatureExtractor):
	"""
	A feature extractor that computes the distance of the closest ghost
	"""
	def __init__(self, name='closestGhost'):
		self.name = name

	def extract(self, state):
		"""
		input: GameState object
		return: the distance of the closest ghost in every subsequent state
		"""

		if len(state.getGhostPositions()) < 1:
			print("Error: game has zero ghosts on the map.")
			sys.exit(0)

		legalMoves = state.getLegalActions()
		illegalMoves = POSSIBLE_MOVES - set(legalMoves) # we will create placeholder features for the illegal moves

		featureName = 'closestGhost_{}' # the {} is a placeholder for later string formatting
									 # we will compute the same feature for each possible next state
									 # and for each of nGhosts ghosts
		features = {}
		for action in legalMoves:
			successorState = state.generateSuccessor(0, action)
			
			features[featureName.format(action)] = self.computeDistance(successorState)

		for action in illegalMoves: 
			successorState = state.generateSuccessor(0, STAY)

			features[featureName.format(action)] = self.computeDistance(successorState)

		return features

	def computeDistance(self, state):
		"""
		Computes the distance between pacman and the closest ghost
		"""

		ghosts = state.getGhostPositions()
		pacmanPosition = state.getPacmanPosition()

		closestGhostDistance = min([util.manhattanDistance(ghost, pacmanPosition) for ghost in ghosts])
		return int(closestGhostDistance)


class GhostDistances(FeatureExtractor):
	"""
	A feature extractor which returns the distances from nGhosts ghosts
	"""

	def __init__(self, name='ghostDistance', nGhosts=2):
		"""
		We add an additional paramter "nGhosts" since we need 
		to keep the number of features fixed, and the games might
		have different numbers of ghosts. 
		Therefore, if there are less than nGhosts ghosts, we will fill the 
		remaining positions with placeholders, and otherwise we will
		use just the first nGhosts.

		"""
		self.name = name
		self.nGhosts = nGhosts

	def extract(self, state):
		"""
		input: GameState object
		return: distances of nGhosts ghosts in every subsequent state
		"""

		if len(state.getGhostPositions()) < 1:
			print("Error: game has zero ghosts on the map.")
			sys.exit(0)


		legalMoves = state.getLegalActions()
		illegalMoves = POSSIBLE_MOVES - set(legalMoves) # we will create placeholder features for the illegal moves

		featureName = 'ghostDistance_{}_{}' # the {} is a placeholder for later string formatting
									 # we will compute the same feature for each possible next state
									 # and for each of nGhosts ghosts
		features = {}

		# features for all possible transitions
		for action in legalMoves:
			successorState = state.generateSuccessor(0, action)
			self.computeFeatures(features, action, successorState, featureName)	

		# for impossible transitions, pretend that we stay at the same spot
		for action in illegalMoves:
			successorState = state.generateSuccessor(0, STAY)
			self.computeFeatures(features, action, successorState, featureName)

		return features

	def computeFeatures(self, features, action, state, featureName):
		"""
		Method that eliminates code duplication
		"""
		ghosts = state.getGhostPositions()
		pacmanPosition = state.getPacmanPosition()

		for ghostIdx, ghostPosition in enumerate(ghosts):
				
			if ghostIdx >= self.nGhosts: 
				# take only the first nGhosts ghosts
				break	
				
			dist = int(util.manhattanDistance(ghostPosition, pacmanPosition))
			features[featureName.format(ghostIdx, action)] = dist

		#print ghostIdx
			
		# in case there are less than nGhosts ghosts, fill the feature
		# values with placeholders
		if ghostIdx < self.nGhosts:
			for placeholder in range(ghostIdx, self.nGhosts):
				# in case there is at least one ghost, dist will
				# have stored distance from that ghost

				features[featureName.format(placeholder, action)] = dist 


def loadPacmanStatesFile(filename):
	"""
	read and return the compressed data file
	"""
	f = open(filename, 'r')
	result = cPickle.load(f)
	f.close()
	return result 


def loadPacmanData(filename, n=0):
	"""
	Return game states from specified recorded games as data, and actions taken as labels
	"""
	components = loadPacmanStatesFile(filename)
	return components['states'], components['actions']

def extractFeatures(featureExtractors, pacmanData):
	"""
	Run all of the feature extractors on the game state training data
	"""
	features = {}

	# pacmanData is a list of GameState instances with the snapshots of the pacman game
	for gameState in pacmanData:
		# for each game state, apply all feature extractors and combine the features
		for extractor in featureExtractors:
			extracted_features = extractor.extract(gameState)

			# combining features
			for feature in extracted_features:
				# only on first outer iteration
				if feature not in features:
					features[feature] = []
				features[feature].append(extracted_features[feature])
	return features

def loadDataset(source, sep='\t'):
	"""
	Read the data from a tsv file into the format:
	list(dict) -> features, list(string) -> labels
	"""
	# training features & target labels
	instances = []
	labels = []

	# all the possible values of features in the dataset, necessary for smoothing
	featureValues = util.Counter() 
	

	with open(source, 'r') as infile:
		
		header = infile.readline().strip().split(sep) # first line is always the header
		target = header[-1] # value to predict
		features = header[:-1] # feature names 

		# initialise the unique feature values for each feature
		for feature in features:
			featureValues[feature] = set()

		# read and parse the instances
		for line in infile:
			instance = {}
			parts = line.strip().split(sep)
			labels.append(parts[-1])
			for feat, value in zip(features, parts[:-1]):
				instance[feat] = value
				featureValues[feat].add(value) # for smoothing

			instances.append(instance)
	return instances, labels, featureValues

def convertDataToFeatures(agentType='contest', infolder='pacmandata', outfolder='classifier_data'):
	print infolder, outfolder
	if not os.path.exists(infolder):
		print "Directory does not exist: {}".format(infolder)
		sys.exit(0)

	if not os.path.exists(outfolder):
		print "Directory does not exist: {}".format(outfolder)
		sys.exit(0)

	print ("Extracting features from: '{}' agent train and test files in the folder: '{}', \
			 writing output to folder: '{}'".format(agentType, infolder, outfolder))

	# create all the wanted feature extractors --> there are multiple other at your disposal
	pacFeatures = []
	pacFeatures.append(FoodEaten())
	pacFeatures.append(GhostCloser())

	for file in os.listdir(infolder):
		if not file.startswith(agentType):
			# not the save file of the agent we want
			continue 
		else: 
			stateData, actionData = loadPacmanData(os.path.join(infolder, file), 'r')

			features = extractFeatures(pacFeatures, stateData)
			featureNames = sorted(features.keys())

			header = '\t'.join(list(featureNames) + ["target"])

			with open(os.path.join(outfolder, file.replace('.pkl','.tsv')), 'w') as outfile:
				print "\tExtracting from: {}".format(file)
				outfile.write(header + "\n")

				for idx, action in enumerate(actionData):
					for feature in featureNames:
						outfile.write(str(features[feature][idx]) + "\t")
					outfile.write(action+"\n")



if __name__ == '__main__':
	# command line arguments
	args = sys.argv[1:]

	# ugly command line parsing script
	if len(args) < 1: 
		convertDataToFeatures()
	elif len(args) < 2: 
		convertDataToFeatures(agentType=args[0])
	elif len(args) < 3: 
		convertDataToFeatures(agentType=args[0], infolder=args[1])
	else:
		convertDataToFeatures(agentType=args[0], infolder=args[1], outfolder=args[2])
