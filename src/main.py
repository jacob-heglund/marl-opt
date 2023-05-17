import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from utils.subtask_controller import SubtaskController
from utils.high_level_mdp import HLMDP
import networkx as nx
import pdb

import argparse


# TODO create experiment running framework (specify in bash files)

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-s', '--success_rate', default=0.9, type=float, help="Success rate requirement")
parser.add_argument('--low_comms_success_prob_1', default=0.9, type=float, help="Success rate for low comms")
parser.add_argument('--high_comms_success_prob_1', default=0.9, type=float, help="Success rate for high comms")

parser.add_argument('--low_comms_success_prob_2', default=0.9, type=float, help="Success rate for low comms")
parser.add_argument('--high_comms_success_prob_2', default=0.9, type=float, help="Success rate for high comms")

args = parser.parse_args()
# initialize subtask controllers
controller_dict = {}


"""
#TODO implement this
scenario 5 - the multi-agent labyrinth as we will actually implement it
includes "helper goals" to get the team between rooms easier and reduce the complexity of learning the policy
"""

success_prob_requirement = args.success_rate
#TODO why define init and goal states in a list like this?
init_states = [0, 0]
goal_states = [11, 0]

#TODO I shouldn't have to manually specify all this stuff. Just the start state, end state, and transition condition
state_action_transition_dict = {}

state_action_transition_dict[0] = {
    "avail_actions": [0, 7],
    "final_states": [1, 7],
    }

# left path
state_action_transition_dict[1] = {
    "avail_actions": [1],
    "final_states": [2]
    }

state_action_transition_dict[2] = {
    "avail_actions": [2],
    "final_states": [3]
    }

state_action_transition_dict[3] = {
    "avail_actions": [3],
    "final_states": [4]
    }

state_action_transition_dict[4] = {
    "avail_actions": [4],
    "final_states": [5]
    }

state_action_transition_dict[5] = {
    "avail_actions": [5],
    "final_states": [6]
    }

state_action_transition_dict[6] = {
    "avail_actions": [6],
    "final_states": [11]
    }

state_action_transition_dict[7] = {
    "avail_actions": [8],
    "final_states": [8]
    }

state_action_transition_dict[8] = {
    "avail_actions": [9],
    "final_states": [9]
    }

state_action_transition_dict[9] = {
    "avail_actions": [10],
    "final_states": [10]
    }

state_action_transition_dict[10] = {
    "avail_actions": [11],
    "final_states": [11]
    }


# populate controller dictionary
for start_state in state_action_transition_dict:
    for i in range(len(state_action_transition_dict[start_state]["avail_actions"])):
        action = state_action_transition_dict[start_state]["avail_actions"][i]
        final_state = state_action_transition_dict[start_state]["final_states"][i]
        #TODO this isn't right, should be "if action in state_action_transition_dict[start_state]["avail_actions"]"

        # navigation tasks always succeed regardless of comms level
        if action in [0, 2, 4, 6, 7, 9, 11]:
            success_function_idx = 6
            controller_dict[action] = SubtaskController(action, init_states=[start_state, 0], final_states=[final_state, 0], success_function_idx=success_function_idx)

        # button-toggling tasks are mixed in comms level success
        elif action in [1, 3, 5, 8, 10]:
            success_function_idx = 7

            if action in [1, 5, 8]:
                controller_dict[action] = SubtaskController(action, init_states=[start_state, 0], final_states=[final_state, 0], success_function_idx=success_function_idx, low_comms_success_prob=args.low_comms_success_prob_1, high_comms_success_prob=args.high_comms_success_prob_1)

            elif action in [3, 10]:
                # interesting case: the values are relatively close, so you don't know a priori how the algorithm will trade off
                low_comms_success_prob = args.low_comms_success_prob_2
                high_comms_success_prob = args.low_comms_success_prob_2
                controller_dict[action] = SubtaskController(action, init_states=[start_state, 0], final_states=[final_state, 0], success_function_idx=success_function_idx, low_comms_success_prob=args.low_comms_success_prob_2, high_comms_success_prob=args.low_comms_success_prob_2)



# pdb.set_trace()
# # populate controller dictionary
# for start_state in state_action_transition_dict:
#     for i in range(len(state_action_transition_dict[start_state]["avail_actions"])):
#         action = state_action_transition_dict[start_state]["avail_actions"][i]
#         final_state = state_action_transition_dict[start_state]["final_states"][i]
#         controller_dict[action] = SubtaskController(action, init_states=[start_state, 0], final_states=[final_state, 0], success_function_idx=6)


# set up communication optimization problem

#TODO the hlmdp state and action indices don't match the states and actions I specify above
hlmdp = HLMDP(init_states=init_states, goal_states=goal_states, controller_dict=controller_dict)
policy, optimal_comms_vals, chosen_success_probs, goal_reach_prob, feasible_flag = hlmdp.solve_minimal_comms_vals(success_prob_requirement)


# init networkx graph for visualization
G = nx.DiGraph()
edge_labels = {}

# populate edge data from the high-level MDP solution
for start_state in hlmdp.S:
    if (start_state != hlmdp.s_fail):
        #TODO how to visualize the policy for each state?
        ## not a priority, it will take more effort than it is worth right now
        ### maybe color the outgoing edges of a state based on the probability of taking that action
        ### or as a weight that is "above" the edge
        print("If this is True, you fixed the bug:", hlmdp.s_g == 11)
        pdb.set_trace()

        G.add_node(start_state)

        if (start_state != hlmdp.s_g):
            for i in range(len(state_action_transition_dict[start_state]["avail_actions"])):
                action = state_action_transition_dict[start_state]["avail_actions"][i]
                final_state = state_action_transition_dict[start_state]["final_states"][i]

                # only show the action index on the edges
                edge_labels[(start_state, final_state)] = f"a: {action}"

                # show a bunch on information on the edges (gets cut off b/c some edges are too short)
                # edge_labels[(start_state, final_state)] = f"a: {action}, p: {chosen_success_probs[action]}, c: {optimal_comms_vals[action]}"

                G.add_edge(start_state, final_state, action_idx=action, optimal_comms_val=optimal_comms_vals[action], chosen_success_prob=chosen_success_probs[action])

pos = nx.spectral_layout(G)
plt.figure()
nx.draw(G, pos, labels={node: node for node in G.nodes()})
nx.draw_networkx_edge_labels(G, pos, label_pos=0.65, edge_labels=edge_labels, rotate=False)
plt.savefig("high_level_mdp.png")

print("\n\n")
print(f"Policy: \n{policy}\n")
print(f"Chosen success probabilities: {chosen_success_probs}")
print(f"Communication values: {optimal_comms_vals}")
print(f"Summed communication values: {sum(optimal_comms_vals.values())}")
print(f"Goal reach probability: {goal_reach_prob}")


#TODO once you have an optimal policy, it may make sense to "refine" the optimal policy so comms are minimized just for the states that can be visited under that policy

"""
notes on graph visualization
# visualize the state-transition graph
## would be nice to visualize the policy
## is there a nice way to visualize the learned communication values and chosen success probabilities (transition function) too?

# draw the graph with networkx and matplotlib
# this layout is pretty dope tbh
## I tried a bunch of the other layouts and they don't look great for our graph
## spectral is also kinda how I drew the graph in the first place, so I think it is a fairly intuitive way to display the graph
# would be nice to roate it so 0 is at the top to mirror the actual layout of the labyrinth
## but the final image in our paper will be made using tikz in latex anyways, so don't worry about it. This image is just for my own sanity check and to have an intermediate result.
## in the final image, it should have the "subtask" labels, as well as the possibility of entering a failure state like the image in "Verifiable"
"""

"""
####################
# policies going out of u0
start_state = 0
controller_dict[0] = SubtaskController(0, init_states=[start_state, 0], final_states=[1, 0], success_function_idx=6)
controller_dict[1] = SubtaskController(1, init_states=[start_state, 0], final_states=[2, 0], success_function_idx=6)
####################
# policies going out of u1
start_state = 1
controller_dict[2] = SubtaskController(2, init_states=[start_state, 0], final_states=[3, 0], success_function_idx=6)
####################
# policies going out of u2
start_state = 2
controller_dict[3] = SubtaskController(3, init_states=[start_state, 0], final_states=[4, 0], success_function_idx=6)
####################
# policies going out of u3
start_state = 3
controller_dict[4] = SubtaskController(4, init_states=[start_state, 0], final_states=[5, 0], success_function_idx=6)
controller_dict[5] = SubtaskController(5, init_states=[start_state, 0], final_states=[6, 0], success_function_idx=6)
####################
# policies going out of u4
start_state = 4
controller_dict[7] = SubtaskController(7, init_states=[start_state, 0], final_states=[8, 0], success_function_idx=6)
controller_dict[9] = SubtaskController(9, init_states=[start_state, 0], final_states=[10, 0], success_function_idx=6)
####################
# policies going out of u5
start_state = 5
controller_dict[10] = SubtaskController(10, init_states=[start_state, 0], final_states=[10, 0], success_function_idx=6)
####################
# policies going out of u6
start_state = 6
controller_dict[6] = SubtaskController(6, init_states=[start_state, 0], final_states=[7, 0], success_function_idx=6)
####################
# policies going out of u7
start_state = 7
controller_dict[14] = SubtaskController(14, init_states=[start_state, 0], final_states=[10, 0], success_function_idx=6)
####################
# policies going out of u8
start_state = 8
controller_dict[8] = SubtaskController(8, init_states=[start_state, 0], final_states=[9, 0], success_function_idx=6)
####################
# policies going out of u9
start_state = 9
controller_dict[15] = SubtaskController(15, init_states=[start_state, 0], final_states=[5, 0], success_function_idx=6)
####################
# policies going out of u10
start_state = 10
controller_dict[11] = SubtaskController(15, init_states=[start_state, 0], final_states=[5, 0], success_function_idx=6)
"""


"""
# scenario 1 - many nonlinear success functions
# pushing the optimization to its limits (at least for our purposes anyways)
## Apparently there is a limit on the size of models you can solve with a free Gurobi license.
### For 10 sampled communication values, that limit is 17 tasks, which it can solve no problem.
### I originally wanted it to solve with 50 tasks, but it wouldn't even run. It just output a "limited model size for free license" message

# for this particular setup (mixture of many nonlinear success functions + 17 tasks + success_prob_requirement of 0.9), it actually takes a little bit (around 20 seconds on a fast computer) but still produces an optimal solution. I think the optimization can handle anything we throw at it within the scope of this project.
success_prob_requirement = 0.90
init_states = [0, 0]
final_states = [15, 0]

controller_dict[0] = SubtaskController(0, init_states=[0, 0], final_states=[1, 0], success_function_idx=0)
controller_dict[1] = SubtaskController(1, init_states=[1, 0], final_states=[2, 0], success_function_idx=0)
controller_dict[2] = SubtaskController(2, init_states=[2, 0], final_states=[3, 0], success_function_idx=0)
controller_dict[3] = SubtaskController(3, init_states=[3, 0], final_states=[4, 0], success_function_idx=0)
controller_dict[4] = SubtaskController(4, init_states=[4, 0], final_states=[5, 0], success_function_idx=1)
controller_dict[5] = SubtaskController(5, init_states=[5, 0], final_states=[6, 0], success_function_idx=1)
controller_dict[6] = SubtaskController(6, init_states=[6, 0], final_states=[7, 0], success_function_idx=1)
controller_dict[7] = SubtaskController(7, init_states=[7, 0], final_states=[8, 0], success_function_idx=1)
controller_dict[8] = SubtaskController(8, init_states=[8, 0], final_states=[9, 0], success_function_idx=2)
controller_dict[9] = SubtaskController(9, init_states=[9, 0], final_states=[10, 0], success_function_idx=2)
controller_dict[10] = SubtaskController(10, init_states=[10, 0], final_states=[11, 0], success_function_idx=2)
controller_dict[11] = SubtaskController(11, init_states=[11, 0], final_states=[12, 0], success_function_idx=2)
controller_dict[12] = SubtaskController(12, init_states=[12, 0], final_states=[13, 0], success_function_idx=3)
controller_dict[13] = SubtaskController(13, init_states=[13, 0], final_states=[14, 0], success_function_idx=3)
controller_dict[14] = SubtaskController(14, init_states=[14, 0], final_states=[15, 0], success_function_idx=3)
"""

"""
'''
scenario 2 - smaller task with two possible paths, one easy and one hard.
- expected result: optimal policy allocates enough to the path u_0 -> u_1 -> u_3 -> u_5 (u_f). I want to see the resulting policy reflects the lower cost of communications along that path.
-- Yep, it accomplishes this.


6 = 100% successful subtask completion rate
5 = 0% successful subtask completion rate

'''

success_prob_requirement = 0.95
init_states = [0, 0]
final_states = [5, 0]

####################
# policies going out of u_0
controller_dict[0] = SubtaskController(0, init_states=[0, 0], final_states=[1, 0], success_function_idx=6)
controller_dict[1] = SubtaskController(1, init_states=[0, 0], final_states=[2, 0], success_function_idx=5)
####################
# policies going out of u_1
controller_dict[2] = SubtaskController(2, init_states=[1, 0], final_states=[3, 0], success_function_idx=6)
####################
# policies going out of u_2
controller_dict[3] = SubtaskController(3, init_states=[2, 0], final_states=[4, 0], success_function_idx=5)
####################
# policies going out of u_3
controller_dict[4] = SubtaskController(4, init_states=[3, 0], final_states=[5, 0], success_function_idx=6)
####################
# policies going out of u_4
controller_dict[5] = SubtaskController(5, init_states=[4, 0], final_states=[5, 0], success_function_idx=5)
####################
"""

"""
# scenario 3 - smaller task with non-trivial higher-level structure
# Like the labryinth I have planned. There should be an obviously "better" path, and I want to see the resulting policy reflects the lower cost of communications along that path
success_prob_requirement = 0.95
init_states = [0, 0]
final_states = [5, 0]

'''
# 6 = 100% successful subtask completion rate
# 5 = 0% successful subtask completion rate
'''
####################
# policies going out of u_0
controller_dict[0] = SubtaskController(0, init_states=[0, 0], final_states=[1, 0], success_function_idx=6)
controller_dict[1] = SubtaskController(1, init_states=[0, 0], final_states=[2, 0], success_function_idx=5)
####################
# policies going out of u_1
controller_dict[2] = SubtaskController(2, init_states=[1, 0], final_states=[3, 0], success_function_idx=6)
controller_dict[3] = SubtaskController(3, init_states=[1, 0], final_states=[4, 0], success_function_idx=5)
####################
# policies going out of u_2
controller_dict[4] = SubtaskController(4, init_states=[2, 0], final_states=[3, 0], success_function_idx=5)
controller_dict[5] = SubtaskController(5, init_states=[2, 0], final_states=[4, 0], success_function_idx=5)
####################
# policies going out of u_3
controller_dict[6] = SubtaskController(6, init_states=[3, 0], final_states=[5, 0], success_function_idx=6)
####################
# policies going out of u_4
controller_dict[7] = SubtaskController(7, init_states=[4, 0], final_states=[5, 0], success_function_idx=5)
####################
"""


"""
# scenario 4 - 2-state task with several actions. Use for super-simple sanity check to debug the optimization
success_prob_requirement = 0.95
init_states = [0, 0]
final_states = [1, 0]

'''
test cases to check it is working as expected
#################################################
- case 1: set to 1, 1, 1
- expected result: should give 100% completion rate with comms vals = 0 for all tasks
-- yep
Policy:
[[ 1.  0.  0.]
 [ 1.  0.  0.]
 [-1. -1. -1.]]

Communication values: {0: 0.0, 1: 0.0, 2: 0.0}
Chosen success probabilities: {0: 1.0, 1: 1.0, 2: 1.0}
Goal reach probability: 1.0

controller_dict[0] = SubtaskController(0, init_states=[0, 0], final_states=[1, 0], success_function_idx=1)
controller_dict[1] = SubtaskController(1, init_states=[0, 0], final_states=[1, 0], success_function_idx=1)
controller_dict[2] = SubtaskController(2, init_states=[0, 0], final_states=[1, 0], success_function_idx=1)
#################################################
- case 2: set to 0, 0, 0
- expected result: either balances some comms between all tasks, or puts a bunch in 1 task (and 0 in others) to get the desired completion probability
-- yep
Policy:
[[ 1.  0.  0.]
 [ 1.  0.  0.]
 [-1. -1. -1.]]

Communication values: {0: 1.0, 1: 0.0, 2: 0.0}
Chosen success probabilities: {0: 1.0, 1: 0.0, 2: 0.0}
Goal reach probability: 1.0

controller_dict[0] = SubtaskController(0, init_states=[0, 0], final_states=[1, 0], success_function_idx=0)
controller_dict[1] = SubtaskController(1, init_states=[0, 0], final_states=[1, 0], success_function_idx=0)
controller_dict[2] = SubtaskController(2, init_states=[0, 0], final_states=[1, 0], success_function_idx=0)
#################################################
- case 3: various permutations of (6, 5, 5)
controller_dict[0] = SubtaskController(0, init_states=[0, 0], final_states=[1, 0], success_function_idx=6)
controller_dict[1] = SubtaskController(1, init_states=[0, 0], final_states=[1, 0], success_function_idx=5)
controller_dict[2] = SubtaskController(2, init_states=[0, 0], final_states=[1, 0], success_function_idx=5)
- Expected result: to make sure it sends the maximum probability along the action with the highest probability of success
-- Yes, has the desired result
(6, 6, 6) (and all other values I tried) make sense
Policy:
[[ 1.  0.  0.]
 [ 1.  0.  0.]
 [-1. -1. -1.]]

Communication values: {0: 0.0, 1: 0.0, 2: 0.0}
Chosen success probabilities: {0: 0.99, 1: 0.99, 2: 0.99}
Goal reach probability: 0.99
#################################################
- case 4: various permutations of (1, 0, 0)
0 - f(x) = x, so highest success probability when comms val = 1
1 - f(x) = 1-x, so highest success probability when comms val = 0
- expected result: it can achieve 100% with comms = 0, so it should do that and the policy should be take the action with f(x) = 1-x 95% of the time or more
-- Yep, it achieves this result

controller_dict[0] = SubtaskController(0, init_states=[0, 0], final_states=[1, 0], success_function_idx=1)
controller_dict[1] = SubtaskController(1, init_states=[0, 0], final_states=[1, 0], success_function_idx=0)
controller_dict[2] = SubtaskController(2, init_states=[0, 0], final_states=[1, 0], success_function_idx=0)
"""
