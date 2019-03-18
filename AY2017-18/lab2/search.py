import util
import copy

from util import Queue
from game import Directions
from game import Agent
from game import Actions

class SearchNode:
    """
    This class represents a node in the graph which represents the search problem.
    """

    def __init__(self, position, parent=None, transition=None, cost=0, heuristic=0):
        """
        Basic constructor which copies the values. Remember, you can access all the 
        values of a python object simply by referencing them - there is no need for 
        a getter method. 
        """
        self.position = position
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic
        self.transition = transition

    def isRootNode(self):
        """
        Check if the node has a parent.
        returns True in case it does, False otherwise
        """
        return self.parent == None 

    def unpack(self):
        """
        Return all relevant values for the current node.
        Returns position, parent node, cost, heuristic value
        """
        return self.position, self.parent, self.cost

    def backtrack(self):
        """
        Reconstruct a path to the initial state from the current node.
        """
        moves = []
        
        node = copy.deepcopy(self)
        while node.parent is not None: 
            moves.insert(0, node.transition)
            node = node.parent

        return moves


def constrainedBreadthFirstSearch(problem, legalStates):
    """
    A breadth-first search that finds a shortest path to a 
    state going only through given states.
    """
    # we need a set to remember all visited states
    visitedStates = set()

    # DFS works in LIFO fashion
    searchQueue = Queue()

    # add an initial state so the stack is not empty
    startState = problem.getStartState()
    startNode = SearchNode(startState)
    searchQueue.push(startNode)

    # iterate until completion
    while not searchQueue.isEmpty():
        currentNode = searchQueue.pop()
        currentState = currentNode.position

        # check for end
        if problem.isGoalState(currentState):
            return currentNode.backtrack()

        if currentState in visitedStates: 
            continue

        visitedStates.add(currentState)

        for futureState, move, _ in problem.getSuccessors(currentState):
            if futureState not in visitedStates and futureState in legalStates: 
                futureNode = SearchNode(futureState, parent=currentNode, transition=move)
                searchQueue.push(futureNode)

    print "Search finished, final state not found!"
    return



class PositionSearchProblem():
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

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
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

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

def pathBetween(point1, point2, legalStates, gameState):
    """
    Returns a possible shortest path through visited states 
    between any two points, using constrained BFS 
    The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: pathBetween( (2,4), (5,6), visitedStates, gameState)

    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return constrainedBreadthFirstSearch(prob, legalStates)
