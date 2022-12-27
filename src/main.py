import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from utils.subtask_controller import SubtaskController
from utils.high_level_mdp import HLMDP
import networkx as nx
import pdb


# initialize subtask controllers
controller_dict = {}

"""
# scenario 1 - many nonlinear success functions
# pushing the optimization to its limits (at least for our purposes anyways)
## Apparently there is a limit on the size of models you can solve with a free Gurobi license.
### For 10 sampled communication values, that limit is 17 tasks, which it can solve no problem.
### I originally wanted it to solve with 50 tasks, but it wouldn't even run. It just output a "limited model size for free license" message

# for this particular setup (mixture of many nonlinear success functions + 17 tasks + prob_threshold of 0.9), it actually takes a little bit (around 20 seconds on a fast computer) but still produces an optimal solution. I think the optimization can handle anything we throw at it within the scope of this project.
prob_threshold = 0.90
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

prob_threshold = 0.95
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
prob_threshold = 0.95
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
prob_threshold = 0.95
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

"""
#TODO implmenent this and make sure it works for some reasonable assumptions on the success function
#TODO the transitions don't have the right init_states and final_states, but that is only something I will know after I actually implement the full labyrinth environment
## for now, the init_states and final_states are shorthand, with the number in the first entry as the indices of the high-level states the edge is associated with

scenario 5 - the multi-agent labyrinth as we will actually implement it
includes "helper goals" to get the team between rooms easier and reduce the complexity of learning the policy
"""
prob_threshold = 0.95
init_states = [0, 0]
final_states = [13, 0]

state_action_transition_dict = {}

state_action_transition_dict[0] = {
    "avail_actions": [0, 1],
    "final_states": [1, 2]
    }

state_action_transition_dict[1] = {
    "avail_actions": [2],
    "final_states": [3]
    }

state_action_transition_dict[2] = {
    "avail_actions": [3],
    "final_states": [4]
    }

state_action_transition_dict[3] = {
    "avail_actions": [4, 5],
    "final_states": [5, 6]
    }

state_action_transition_dict[4] = {
    "avail_actions": [7, 9],
    "final_states": [8, 10]
    }

state_action_transition_dict[5] = {
    "avail_actions": [10],
    "final_states": [11]
    }

state_action_transition_dict[6] = {
    "avail_actions": [6],
    "final_states": [7]
    }

state_action_transition_dict[7] = {
    "avail_actions": [14],
    "final_states": [10]
    }

state_action_transition_dict[8] = {
    "avail_actions": [8],
    "final_states": [9]
    }

state_action_transition_dict[9] = {
    "avail_actions": [15],
    "final_states": [5]
    }

state_action_transition_dict[10] = {
    "avail_actions": [11],
    "final_states": [12]
    }

state_action_transition_dict[11] = {
    "avail_actions": [12],
    "final_states": [13]
    }

state_action_transition_dict[12] = {
    "avail_actions": [13],
    "final_states": [13]
    }

G = nx.DiGraph()
edge_labels = {}

for start_state in state_action_transition_dict:
    G.add_node(start_state)

    for i in range(len(state_action_transition_dict[start_state]["avail_actions"])):
        action = state_action_transition_dict[start_state]["avail_actions"][i]
        final_state = state_action_transition_dict[start_state]["final_states"][i]
        controller_dict[action] = SubtaskController(action, init_states=[start_state, 0], final_states=[final_state, 0], success_function_idx=6)

        edge_labels[(start_state, final_state)] = action


        G.add_edge(start_state, final_state, action_idx=action)

# draw the graph with networkx and matplotlib

# this layout is pretty dope tbh
## I tried a bunch of the other layouts and they don't look great for our graph
## spectral is also kinda how I drew the graph in the first place, so I think it is a fairly intuitive way to display the graph
# would be nice to roate it so 0 is at the top to mirror the actual layout of the labyrinth
## but the final image in our paper will be made using tikz in latex anyways, so don't worry about it. This image is just for my own sanity check and to have an intermediate result.
## in the final image, it should have the "subtask" labels, as well as the possibility of entering a failure state like the image in "Verifiable"

pos = nx.spectral_layout(G)
plt.figure()
nx.draw(G, pos, labels={node: node for node in G.nodes()})
nx.draw_networkx_edge_labels(G, pos, label_pos=0.65, edge_labels=edge_labels)
plt.savefig("high_level_mdp.png")





# ####################
# # policies going out of u0
# start_state = 0
# controller_dict[0] = SubtaskController(0, init_states=[start_state, 0], final_states=[1, 0], success_function_idx=6)
# controller_dict[1] = SubtaskController(1, init_states=[start_state, 0], final_states=[2, 0], success_function_idx=6)
# ####################
# # policies going out of u1
# start_state = 1
# controller_dict[2] = SubtaskController(2, init_states=[start_state, 0], final_states=[3, 0], success_function_idx=6)
# ####################
# # policies going out of u2
# start_state = 2
# controller_dict[3] = SubtaskController(3, init_states=[start_state, 0], final_states=[4, 0], success_function_idx=6)
# ####################
# # policies going out of u3
# start_state = 3
# controller_dict[4] = SubtaskController(4, init_states=[start_state, 0], final_states=[5, 0], success_function_idx=6)
# controller_dict[5] = SubtaskController(5, init_states=[start_state, 0], final_states=[6, 0], success_function_idx=6)
# ####################
# # policies going out of u4
# start_state = 4
# controller_dict[7] = SubtaskController(7, init_states=[start_state, 0], final_states=[8, 0], success_function_idx=6)
# controller_dict[9] = SubtaskController(9, init_states=[start_state, 0], final_states=[10, 0], success_function_idx=6)
# ####################
# # policies going out of u5
# start_state = 5
# controller_dict[10] = SubtaskController(10, init_states=[start_state, 0], final_states=[10, 0], success_function_idx=6)
# ####################
# # policies going out of u6
# start_state = 6
# controller_dict[6] = SubtaskController(6, init_states=[start_state, 0], final_states=[7, 0], success_function_idx=6)
# ####################
# # policies going out of u7
# start_state = 7
# controller_dict[14] = SubtaskController(14, init_states=[start_state, 0], final_states=[10, 0], success_function_idx=6)
# ####################
# # policies going out of u8
# start_state = 8
# controller_dict[8] = SubtaskController(8, init_states=[start_state, 0], final_states=[9, 0], success_function_idx=6)
# ####################
# # policies going out of u9
# start_state = 9
# controller_dict[15] = SubtaskController(15, init_states=[start_state, 0], final_states=[5, 0], success_function_idx=6)
# ####################
# # policies going out of u10
# start_state = 10
# controller_dict[11] = SubtaskController(15, init_states=[start_state, 0], final_states=[5, 0], success_function_idx=6)






#TODO once you have an optimal policy, it may make sense to "refine" the optimal policy so comms are minimized just for the states that can be visited under that policy



# set up communication optimization problem
hlmdp = HLMDP(init_states=init_states, goal_states=final_states, controller_dict=controller_dict)

# this is just to show that the basic optimization works given the classes I have implemented
# policy, reach_prob, feasible_flag = hlmdp.solve_max_reach_prob_policy()

policy, optimal_comms_vals, chosen_success_probs, goal_reach_prob, feasible_flag = hlmdp.solve_minimal_comms_vals(prob_threshold)


"""
print("\n\n")
print(f"Policy: \n{policy}\n")


print(f"Communication values: {optimal_comms_vals}")
print(f"Chosen success probabilities: {chosen_success_probs}")
print(f"Goal reach probability: {goal_reach_prob}")
pdb.set_trace()
"""

# plot success probability vs communication threshold
# for controller_idx, controller in enumerate(controller_list):
#     plt.plot(controller.communication_thresholds, controller.success_prob_list, label=f"Controller {controller_idx}")
#     plt.xlabel("Communication threshold $\lambda_c$")
#     plt.ylabel("Empirical subtask success probability $\hat{\sigma}_c$ ")
#     plt.legend()
#     plt.savefig("success_prob_list.png")

# initialize high-level MDP



# solve high-level MDP for minimal communication while meeting prob_threshold

