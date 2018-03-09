from game import Directions
from game import Agent
from game import Actions
import util
import time
import logic
import search
import pacard

class PacardAgent(Agent):
    """
    This very general search agent finds a path to the goal of a problem 
    by using a search algorithm. The search algorithm you will implement for 
    this excercise is based on refutation resolution.


    Note: You should NOT change any code in PacardAgent
    """

    def __init__(self, fn='logicBasedSearch', prob='LogicSearchProblem'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems
        # Get the search function from the name and heuristic
        if fn not in dir(pacard):
            raise AttributeError, fn + ' is not a search function in pacard.py.'
        func = getattr(pacard, fn)
        self.searchFunction = func
        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in logicAgents.py.'
        self.searchType = globals()[prob]
        print('[PacardAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for PacardAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


class LogicSearchProblem(pacard.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.capsules = gameState.getCapsules()
        
        # hacky
        self.gameState = gameState

        ghosts = gameState.getGhostPositions()
        if len(ghosts) != 1:
            print 'Warning: this does not look a Wumpus maze'
        self.wumpus = ghosts[0]

        if start != None: self.startState = start


        if warn and (gameState.getNumFood() != 1):
            print 'Warning: this does not look a Wumpus maze'
        food = gameState.getFood() 
        self.goal = food.asList()[0]

        self.costFn = costFn
        self.visualize = visualize

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

    def isWumpus(self, state):
        """
        Check if wumpus is present at the given state.
        You should NOT use this function in your implementation - it is simply 
        for the purposes of checking if the game is over - you could (and will)
        sometimes end up in the spots where Wumpus is.
        """
        return state == self.wumpus

    def isTeleporter(self, state):
        """
        Check if wumpus is present at the given state.
        You should NOT use this function in your implementation - it is simply 
        for the purposes of checking if the game is over.
        """
        return state == self.goal

    def isPoisonCapsule(self, state):
        """
        Check if wumpus is present at the given state.
        You should NOT use this function in your implementation - it is simply 
        for the purposes of checking if the game is over - you could (and will)
        sometimes end up in the spots where the poisoned capsules are.
        """
        return state in self.capsules

    def isWumpusClose(self, state):
        """
        You SHOULD use this function in your implementation. This function checks
        if the stench of the wumpus can be sensed from the square you are at.
        """
        return util.manhattanDistance(self.wumpus, state) == 1

    def isPoisonCapsuleClose(self, state):
        """
        You SHOULD use this function in your implementation. This function checks
        if the poison from the pills can be sensed from the square you are at.
        """
        return any([util.manhattanDistance(capsule, state) == 1 for capsule in self.capsules])

    def isTeleporterClose(self, state):
        """
        You SHOULD use this function in your implementation. This function checks
        if the glow of the teleporter can be seen from the square you are at.
        """
        return util.manhattanDistance(self.goal, state) == 1


    def reconstructPath(self, visitedStates):
        """
        Returns the fastest path taken while visiting a sequence of states 
        in the given order. The path is not necessarily the optimal solution 
        through the given states. 

        You should use this method to return the transitions as a result of your 
        search problem.
        """
        path = []
        # trivial example, the start is the finish
        if len(visitedStates) <= 1: 
            return path 
        # call bfs on each pair of points
        current = visitedStates[0]

        for future in visitedStates[1:]: 
            path += search.pathBetween(current, future, visitedStates, self.gameState)
            current = future

        return path
