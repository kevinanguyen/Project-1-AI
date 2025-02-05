# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack  # Use the provided Stack class

    stack = Stack()  # Initialize stack for DFS
    visited = set()  # Keep track of visited nodes

    start_state = problem.getStartState()
    stack.push((start_state, []))  # Push (state, path) to the stack

    while not stack.isEmpty():
        state, path = stack.pop()  # Get the last added node

        if problem.isGoalState(state):
            return path  # Return the path if we reached the goal

        if state not in visited:
            visited.add(state)  # Mark as visited

            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    stack.push((successor, path + [action]))  # Push new state with updated path

    return []  # Return an empty path if no solution found
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue  # Use the provided Queue class

    queue = Queue()  # Initialize queue for BFS
    visited = set()  # Track visited states

    start_state = problem.getStartState()
    queue.push((start_state, []))  # Push (state, path) to the queue

    while not queue.isEmpty():
        state, path = queue.pop()  # Get the oldest added node

        if problem.isGoalState(state):
            return path  # Return the path if we reached the goal

        if state not in visited:
            visited.add(state)  # Mark as visited

            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    queue.push((successor, path + [action]))  # Push new state with updated path

    return []  # Return an empty path if no solution found
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue  # Use the provided PriorityQueue class

    pq = PriorityQueue()  # Initialize priority queue for UCS
    visited = set()  # Track visited states
    start_state = problem.getStartState()

    pq.push((start_state, [], 0), 0)  # Push (state, path, cost) with priority = cost

    while not pq.isEmpty():
        state, path, cost = pq.pop()  # Get the lowest-cost node

        if problem.isGoalState(state):
            return path  # Return the path if we reached the goal

        if state not in visited:
            visited.add(state)  # Mark as visited

            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    new_cost = cost + stepCost  # Compute new total cost
                    pq.push((successor, path + [action], new_cost), new_cost)  # Push with updated cost

    return []  # Return an empty path if no solution found
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue  # Use provided PriorityQueue
    
    fringe = PriorityQueue()  # Initialize priority queue
    visited = set()  # Track visited states

    start_state = problem.getStartState()
    fringe.push((start_state, [], 0), heuristic(start_state, problem))  # (state, path, cost), priority

    while not fringe.isEmpty():
        state, path, cost = fringe.pop()  # Get state with lowest f(n) = g(n) + h(n)

        if problem.isGoalState(state):
            return path  # Return the found path

        if state not in visited:
            visited.add(state)  # Mark as visited

            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    new_cost = cost + stepCost  # g(n): Total path cost
                    priority = new_cost + heuristic(successor, problem)  # f(n) = g(n) + h(n)
                    fringe.push((successor, path + [action], new_cost), priority)

    return []  # If no solution found
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
