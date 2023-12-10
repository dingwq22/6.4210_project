from pydrake.all import RigidTransform
from typing import (Callable, Iterable, List, Sequence, Tuple, Dict, Optional,
                    Any, Union)

from abc import abstractmethod
import collections
import itertools
import functools
import random

import heapq as hq  # Can use this as a priority queue

import numpy as np

########## Graph-Search-Related Utilities and Class Definitions ##########

State = Any
Action = Any

StateSeq = List[State]
ActionSeq = List[State]
CostSeq = RewardSeq = List[float]


class Problem(object):
  """The abstract base class for either a path cost problem or a reward problem."""

  def __init__(self, initial: State):
    self.initial = initial

  @abstractmethod
  def actions(self, state: State) -> Iterable[Action]:
    """Returns the allowed actions in a given state.

    The result would typically be a list. But if there are many actions,
    consider yielding them one at a time in an iterator,
    rather than building them all at once.
    """
    ...

  @abstractmethod
  def step(self, state: State, action: Action) -> State:
    """Returns the next state when executing a given action in a given state.

    The action must be one of self.actions(state).
    """
    ...


class PathCostProblem(Problem):
  """An abstract class for a path cost problem, based on AIMA.

  To formalize a path cost problem, you should subclass from this and implement
  the abstract methods.
  Then you will create instances of your subclass and solve them with the
  various search functions.
  """

  @abstractmethod
  def goal_test(self, state: State) -> bool:
    """Checks if the state is a goal."""
    ...

  @abstractmethod
  def step_cost(self, state1: State, action: Action, state2: State) -> float:
    """Returns the cost incurred at state2 from state1 via action."""
    ...

  def h(self, state: State) -> float:
    """Returns the heuristic value, a lower bound on the distance to goal."""
    return 0


class RewardProblem(Problem):
  """An abstract class for a finite-horizon reward problem, based on AIMA.

  To formalize a reward problem, you should subclass from this and implement
  the abstract methods.
  Then you will create instances of your subclass and solve them with the
  various search functions.
  """

  def __init__(self, initial: State, horizon: int):
    self.initial = initial
    self.horizon = horizon

  @abstractmethod
  def reward(self, state1: State, action: Action, state2: State) -> float:
    """Returns the reward given at state2 from state1 via action.

    A reward at each step must be no greater than `self.rmax`.
    """
    ...

  @property
  @abstractmethod
  def rmax(self) -> float:
    """Returns the maximum reward per step."""
    ...



def parse_map(map):
  
    '''
    map: map represented as np array
    return initial=(r, c, direction), goal=(r, c), size = (h, w)
    e.g. initial=(1, 1, 0), goal=(3, 3), size = (3,3)
    '''
    h, w = map.shape
    wall = True
    initial, goal = None, []
    door_open = True
    for i in range(h):
        for j in range(w):
            if map[i,j]:
                obj = map[i,j][0]

                if obj == 'L':
                   door_open = False

                if obj in ['0', '1', '2', '3']: #agent
                    initial = (i, j, int(obj), ())

                if obj=='G': #goal
                    goal = (i,j)
    if not initial or not goal:
       raise ValueError(f'{initial, goal} one of initial/goal is None')
    return tuple(list(initial)+[door_open]), goal

def parse_map2(map):
    '''
    map: map represented as np array
    return initial=(r, c, direction), goal=(r, c), size = (h, w)
    e.g. initial=(1, 1, 0), goal=(3, 3), size = (3,3)
    '''
    h, w = map.shape
    wall = True
    initial, goal = None, []
    door_open = True
    for i in range(h):
        for j in range(w):
            if map[i,j]:
                obj = map[i,j][0]

                if obj == 'L':
                   door_open = False

                if obj in ['0', '1', '2', '3']: #agent
                    initial = (i, j, int(obj))

                if obj=='G': #goal
                    goal.append((i,j))
    if not initial or not goal:
       raise ValueError(f'{initial, goal} one of initial/goal is None')
    return tuple(list(initial)+[tuple(False for _ in range(len(goal)))]+[door_open]), goal

class GridProblem(PathCostProblem):
    """A grid problem."""

    def __init__(self, map):
        self.map = map
        initial, goal = parse_map(map)
        print('initial state', initial)
        super().__init__(initial)
        self.goal = goal
        self.all_grid_actions = ["forward", "left", "right"]
        self.grid_dir_to_delta = {
            0: (0, 1),
            1: (1, 0),
            2: (0, -1),
            3: (-1, 0)
        }
        (self.height, self.width) = map.shape

    # def __init__(self, map, initial=(1, 1, 0), goal=(3, 3), size = (3,3)):
    #     self.map = map
    #     initial, goal = parse_map(map)
    #     super().__init__(initial)
    #     self.goal = goal
    #     self.all_grid_actions = ["forward", "left", "right"]
    #     self.grid_dir_to_delta = {
    #         0: (0, 1),
    #         1: (1, 0),
    #         2: (0, -1),
    #         3: (-1, 0)
    #     }
    #     (self.height, self.width) = size
    

    def actions(self, state):
        (r, c, d, carrying, dooropen) = state
        actions = ["left", "right"]
        dr, dc = self.grid_dir_to_delta[d]
        new_r, new_c = r + dr, c + dc
        next_grid = self.map[new_r, new_c][0]
        # Check if in bounds
        if  (0 < new_r < self.height and
            0 < new_c < self.width) and\
            next_grid not in ['W'] and\
            (next_grid != 'K' or 'key' in state[3]) and\
            (next_grid not in ['L'] or state[4]): #door not locked, or do locked door in map
            
            actions.append("forward")
        if next_grid == 'K':
           actions.append("pickup_key")
        if next_grid == 'L' and 'key' in state[3] and not state[4]: #carrying key, door not open
           actions.append("open_door")
        return actions

    def step(self, state, action):
        (r, c, d, carrying, dooropen) = state
        if action == 'left':
            return (r, c, (d - 1) % 4, carrying, dooropen)
        elif action == 'right':
            return (r, c, (d + 1) % 4, carrying, dooropen)
        elif action == 'forward':
            dr, dc = self.grid_dir_to_delta[d]
            new_r, new_c = r + dr, c + dc
            return (new_r, new_c, d, carrying, dooropen)
        elif action == "pickup_key":
            return (r, c, d, tuple(list(carrying)+['key']), dooropen)
        elif action == "open_door":
            return (r, c, d, carrying, True)

        else:
           raise ValueError(f'invalid action: {action}')

    def goal_test(self, state):
        return state[:2] == self.goal

    def step_cost(self, state1, action, state2):
        return 1

    def h(self, state):
        """Manhattan distance."""
        return abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])

#unit test
# grid = GridProblem()
# print(grid.actions((1, 1, 1)))
# print(grid.step((1, 1, 1), 'left'))



# A useful data structure for best-first search
Node = collections.namedtuple("Node",
                              ["state", "parent", "action", "cost", "g"])

def run_best_first_search(
    problem: PathCostProblem,
    get_priority: Callable[[Node], float],
    step_budget: int = 1000) -> Tuple[StateSeq, ActionSeq, CostSeq, int]:
  """A generic heuristic search implementation.

  Depending on `get_priority`, can implement A*, GBFS, or UCS.

  The `get_priority` function here should determine the order
  in which nodes are expanded. For example, if you want to
  use path cost as part of this determination, then the
  path cost (node.g) should appear inside of get_priority,
  rather than in this implementation of `run_best_first_search`.

  Important: for determinism (and to make sure our tests pass),
  please break ties using the state itself. For example,
  if you would've otherwise sorted by `get_priority(node)`, you
  should now sort by `(get_priority(node), node.state)`.

  Args:
    problem: a path cost problem.
    get_priority: a callable taking in a search Node and returns the priority
    step_budget: maximum number of `problem.step` before giving up.

  Returns:
    state_sequence: A list of states.
    action_sequence: A list of actions.
    cost_sequence: A list of costs.
    num_steps: number of taken `problem.step`s. Must be less than or equal to `step_budget`.

  Raises:
    error: SearchFailed, if no plan is found.
  """
  #"state", "parent", "action", "cost", "g"
  s0 = problem.initial

  n = Node(s0, None, None, 0, 0)
  frontier = [(0, s0, n)]
  hq.heapify(frontier)
  reached = {s0:n}
  count = 0
  while len(frontier)!=0 and count<step_budget:
    n = hq.heappop(frontier)[2]
    s = n.state

    if problem.goal_test(s):
      # print(reached, s)
      # return reached, s
      curr_node = reached[s]
      state_sequence, action_sequence, cost_sequence = [], [], []
      while True:
        curr_state = curr_node.state
        # print(curr_state)
        state_sequence.append(curr_state)
        action_sequence.append(curr_node.action)
        cost_sequence.append(curr_node.cost)
        curr_node = curr_node.parent
        if not curr_node:
          break
      # print(state_sequence)
      # print(cost_sequence[:-1][::-1])
      return state_sequence[::-1], action_sequence[:-1][::-1], cost_sequence[:-1][::-1], count

    for a in problem.actions(s):
      count+=1
      s1 = problem.step(s, a)
      step_cost = problem.step_cost(s, a, s1)
      path_cost = n.g + step_cost
      if not s1 in reached:

        n1 = Node(s1, n, a, step_cost, path_cost)
        reached[s1] = n1
        hq.heappush(frontier, (get_priority(n1), s1, n1))
    # if s==(3,2):
    #   print(frontier)
  print('no path found')
  return None

def run_uniform_cost_search(problem: PathCostProblem, step_budget: int = 1000):
    return run_best_first_search(problem, lambda node: node.g, step_budget)

def get_astar_plan(grid_init = None, step_budget = 1000):
    grid_problem = GridProblem(grid_init)
    get_priority_fn = lambda node: node.g + grid_problem.h(node.state)
    # result = run_best_first_search(grid_problem, get_priority_fn)
    # print(result)
    result = run_best_first_search(grid_problem, get_priority = get_priority_fn, step_budget = step_budget)
    if not result:
       return None
    # print([i[:3] for i in result[0]])
    return result

direction_map = {
    1: 0,
    0: np.pi/2,
    3: np.pi,
    2: 3/2*np.pi
}

def get_trajectory_from_states(states):
    traj = []
    for (r, c, d, x, y) in states:
        traj.append([2*c-6, -2*r+6, 0])
    return traj

# map_default = np.array([['WG', 'WG', 'WG', 'WG', 'WG', 'WG', 'WG'],
#                 ['WG', '3^', '  ', '  ', 'WG', 'WG', 'WG'],
#                 ['WG', '  ', 'WG', 'WG', '  ', '  ', 'WG'],
#                 ['WG', '  ', '  ', '  ', '  ', '  ', 'WG'],
#                 ['WG', 'WG', 'WG', '  ', 'WG', '  ', 'WG'],
#                 ['WG', 'WG', 'WG', '  ', '  ', '  ', 'WG'],
#                 ['WG', 'WG', 'WG', 'WG', 'WG', 'WG', 'WG']])
map_default = np.array([['WG', 'WG', 'WG', 'WG', 'WG', 'WG', 'WG'],
                ['WG', '3^', '  ', '  ', '  ', '  ', 'WG'],
                ['WG', '  ', '  ', '  ', '  ', '  ', 'WG'],
                ['WG', '  ', '  ', '  ', '  ', '  ', 'WG'],
                ['WG', '  ', '  ', '  ', '  ', '  ', 'WG'],
                ['WG', '  ', '  ', '  ', '  ', '  ', 'WG'],
                ['WG', 'WG', 'WG', 'WG', 'WG', 'WG', 'WG']])

map_empty = np.array([['WG', 'WG', 'WG', 'WG', 'WG', 'WG', 'WG'],
                ['WG', '3^', '  ', '  ', '  ', '  ', 'WG'],
                ['WG', '  ', '  ', '  ', '  ', '  ', 'WG'],
                ['WG', '  ', '  ', '  ', '  ', '  ', 'WG'],
                ['WG', '  ', '  ', '  ', '  ', '  ', 'WG'],
                ['WG', '  ', '  ', '  ', '  ', '  ', 'WG'],
                ['WG', 'WG', 'WG', 'WG', 'WG', 'WG', 'WG']])

def trajectory_plan(goal=(5,-5), obstacles = [(1, 4), (1, 5), (2, 2), (2, 3), (4, 1), (4, 2), (4, 4), (5, 1), (5, 2)], gripper_pose=None):
  x, y = goal 
  print('map goal (x, y):', goal)
  goal = (round(-0.5*y + 3), round(0.5*x + 3)) #r, c
  print('grid goal (r, c)', goal)
  curr_map = map_empty.copy()
  curr_map[goal[0], goal[1]] = 'GG'
  for obs in obstacles:
    curr_map[obs] = 'WG'
  result = get_astar_plan(curr_map,step_budget =1000)
  print('traj in grid coord:', result[0])
  total_traj = get_trajectory_from_states(result[0]) + [[x, y, 0]]
  print('traj in xy coord:', total_traj)
  return total_traj 




maps = np.array([['WG', 'WG', 'WG', 'WG', 'WG'],
                ['WG', '  ', 'LY', '  ', 'WG'],
                ['WG', 'KY', 'WG', '  ', 'WG'],
                ['WG', '3^', 'WG', 'GG', 'WG'],
                ['WG', 'WG', 'WG', 'WG', 'WG']])
maps2 = np.array([['WG', 'WG', 'WG', 'WG', 'WG', 'WG', 'WG'],
                ['WG', '3^', '  ', '  ', 'WG', 'WG', 'WG'],
                ['WG', '  ', 'WG', 'WG', '  ', '  ', 'WG'],
                ['WG', '  ', '  ', '  ', '  ', '  ', 'WG'],
                ['WG', 'WG', 'WG', '  ', 'WG', '  ', 'WG'],
                ['WG', 'WG', 'WG', '  ', '  ', 'GG', 'WG'],
                ['WG', 'WG', 'WG', 'WG', 'WG', 'WG', 'WG']])

obstacles = [(1, 4), (1, 5), (2, 2), (2, 3), (4, 1), (4, 2), (4, 4), (5, 1), (5, 2)]



# add inverse dynamics controller
# actuation import port -> choose inverse dynamics

# g = GridProblem(maps2)
# # print(parse_map(maps))
# # print(g.actions((1, 1, 0, (), False)))
# result = get_astar_plan(maps2,step_budget =1000)

# traj = trajectory_plan([0.95, 0.93])

map3 = np.array([['WG', 'WG', 'WG', 'WG', 'WG', 'WG', 'WG'],
                ['WG', '3^', '  ', '  ', 'WG', 'WG', 'WG'],
                ['WG', '  ', 'WG', 'WG', '  ', '  ', 'WG'],
                ['WG', '  ', '  ', 'GG', '  ', '  ', 'WG'],
                ['WG', 'WG', 'WG', '  ', 'WG', '  ', 'WG'],
                ['WG', 'WG', 'WG', '  ', '  ', 'GG', 'WG'],
                ['WG', 'WG', 'WG', 'WG', 'WG', 'WG', 'WG']])

print(parse_map(map3))