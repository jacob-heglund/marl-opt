import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from utils.subtask_controller import SubtaskController
from utils.high_level_mdp import HLMDP
import pdb


# initialize subtask controllers
controller_dict = {}

#TODO
# "Sanity check" scenarios - make sure the optimization is actually doing what we want

#TODO develop visualizations for the solutions

"""
# scenario 3 - many nonlinear success functions
TODO has not been updated for change from controller_list to controller_dict
# pushing the optimization to its limits (at least for our purposes anyways)
## Apparently there is a limit on the size of models you can solve with a free Gurobi license.
### For 10 sampled communication values, that limit is 17 tasks, which it can solve no problem.
### I originally wanted it to solve with 50 tasks, but it wouldn't even run. It just output a "limited model size for free license" message

# for this particular setup (mixture of many nonlinear success functions + 17 tasks + prob_threshold of 0.9), it actually takes a little bit (around 20 seconds on a fast computer) but still produces an optimal solution. I think the optimization can handle anything we throw at it within the scope of this project.
prob_threshold = 0.90
init_states = [0, 0]
final_states = [17, 0]

controller_list.append(SubtaskController(0, init_states=[0, 0], final_states=[1, 0], success_function_idx=0))
controller_list.append(SubtaskController(1, init_states=[1, 0], final_states=[2, 0], success_function_idx=0))
controller_list.append(SubtaskController(2, init_states=[2, 0], final_states=[3, 0], success_function_idx=0))
controller_list.append(SubtaskController(3, init_states=[3, 0], final_states=[4, 0], success_function_idx=0))
controller_list.append(SubtaskController(4, init_states=[4, 0], final_states=[5, 0], success_function_idx=1))
controller_list.append(SubtaskController(5, init_states=[5, 0], final_states=[6, 0], success_function_idx=1))
controller_list.append(SubtaskController(6, init_states=[6, 0], final_states=[7, 0], success_function_idx=1))
controller_list.append(SubtaskController(7, init_states=[7, 0], final_states=[8, 0], success_function_idx=1))
controller_list.append(SubtaskController(8, init_states=[8, 0], final_states=[9, 0], success_function_idx=2))
controller_list.append(SubtaskController(9, init_states=[9, 0], final_states=[10, 0], success_function_idx=2))
controller_list.append(SubtaskController(10, init_states=[10, 0], final_states=[11, 0], success_function_idx=2))
controller_list.append(SubtaskController(11, init_states=[11, 0], final_states=[12, 0], success_function_idx=2))
controller_list.append(SubtaskController(12, init_states=[12, 0], final_states=[13, 0], success_function_idx=3))
controller_list.append(SubtaskController(13, init_states=[13, 0], final_states=[14, 0], success_function_idx=3))
controller_list.append(SubtaskController(14, init_states=[14, 0], final_states=[15, 0], success_function_idx=3))
controller_list.append(SubtaskController(15, init_states=[15, 0], final_states=[16, 0], success_function_idx=3))
controller_list.append(SubtaskController(16, init_states=[16, 0], final_states=[17, 0], success_function_idx=4))
controller_list.append(SubtaskController(17, init_states=[17, 0], final_states=[18, 0], success_function_idx=4))
"""

"""
'''
scenario 5 - smaller task with two possible paths, one easy and one hard.
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

# scenario 6 - smaller task with non-trivial higher-level structure
# Like the labryinth I have planned. There should be an obviously "better" path, and I want to see the resulting policy reflects the lower cost of communications along that path
prob_threshold = 0.95
init_states = [0, 0]
final_states = [5, 0]

'''
# 6 = 100% successful subtask completion rate
# 5 = 0% successful subtask completion rate
'''
#TODO this doesn't make any sense
## if I just have the MDP u_0 -> u_1 -> u_3 -> u_f, each with success function 6, it finds a solution no problem
## but when I add on other states where they have success functions 5 (leaving the only feasible path as u_0 -> u_1 -> u_3 -> u_f), it doesn't find a solution
### I need to understand this issue before moving on to RL stuff, becauase that will only add another layer of complication and training times that will impede my process of understanding what is going on here
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
# scenario 7 - 2-state task with several actions. Use to debug issues with optimization
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






#TODO once you have an optimal policy, it may make sense to "refine" the optimal policy so comms are minimized just for the states that can be visited under that policy



# set up communication optimization problem
hlmdp = HLMDP(init_states=init_states, goal_states=final_states, controller_dict=controller_dict)

# this is just to show that the basic optimization works given the classes I have implemented
# policy, reach_prob, feasible_flag = hlmdp.solve_max_reach_prob_policy()

#TODO would be nice to visualize the policy for the high-level MDP
## this isn't necessary at all, and will take some time. You need to prioritize the low-level RL stuff first.

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

