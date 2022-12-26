import numpy as np
from gurobipy import Model, GRB

import pdb

class HLMDP(object):
    """
    Class representing the MDP model of the high-level decision making process.
    """

    def __init__(self, init_states, goal_states, controller_dict):
        """
        Inputs
        ------
        init_states : list
            List of tuples representing the possible initial states of the system.
        goal_states : list
            List of tuples representing the target goal states of the system.
        controller_dict : dict
            Dict of MinigridController objects (the sub-systems being used as components of the overall RL system).
        """

        self.init_states = init_states
        self.goal_states = goal_states
        self.controller_dict = controller_dict

        self.state_list = []
        self.S = None
        self.s_i = None
        self.s_g = None
        self.s_fail = None
        self._construct_state_space()

        self.avail_actions = {}
        self.avail_states = {}
        self.A = self.controller_dict.keys()
        self._construct_action_space()

        self._construct_avail_actions()


        self.N_S = len(self.S) # Number of states in the high-level MDP
        self.N_A = len(self.A) # Number of actions in the high-level MDP

        self.P = np.zeros((self.N_S, self.N_A, self.N_S), dtype=np.float)
        self._construct_transition_function()

        self.successor = {}
        self._construct_successor_map()

        self.predecessors = {}
        self._construct_predecessor_map()

    def update_transition_function(self):
        """
        Re-construct the transition function to reflect any changes in the empirical
        measurements of how likely each controller is to succeed.
        """
        self.P = np.zeros((self.N_S, self.N_A, self.N_S), dtype=np.float)
        self._construct_transition_function()

    def _construct_state_space(self):
        for controller_ind in self.controller_dict.keys():
            controller = self.controller_dict[controller_ind]
            controller_init_states = controller.get_init_states()
            controller_final_states = controller.get_final_states()
            if controller_init_states not in self.state_list:
                self.state_list.append(controller_init_states)
            if controller_final_states not in self.state_list:
                self.state_list.append(controller_final_states)

        self.state_list.append(-1) # Append another state representing the absorbing "task failed" state
        self.S = np.arange(len(self.state_list))

        self.s_i = self.state_list.index(self.init_states)
        self.s_g = self.state_list.index(self.goal_states)
        self.s_fail = self.state_list.index(-1)

    def _construct_action_space(self):
        for s in self.S:
            self.avail_actions[s] = []

        for controller_ind in self.controller_dict.keys():
            controller = self.controller_dict[controller_ind]
            controller_init_states = controller.get_init_states()
            init_s = self.state_list.index(controller_init_states)
            self.avail_actions[init_s].append(controller_ind)

    def _construct_avail_actions(self):
        for a in self.A:
            self.avail_states[a]=[]

        for s in self.S:
            avail_actions = self.avail_actions[s]
            for action in avail_actions:
                self.avail_states[action].append(s)

    def _construct_transition_function(self):
        for s in self.S:
            avail_actions = self.avail_actions[s]
            for action in avail_actions:
                success_prob = self.controller_dict[action].get_success_prob()
                controller_next_states = self.controller_dict[action].get_final_states()
                next_s = self.state_list.index(controller_next_states)

                self.P[s, action, next_s] = success_prob
                self.P[s, action, self.s_fail] = 1 - success_prob

    def _construct_successor_map(self):
        for s in self.S:
            avail_actions = self.avail_actions[s]
            for action in avail_actions:
                controller_next_states = self.controller_dict[action].get_final_states()
                next_s = self.state_list.index(controller_next_states)

                self.successor[(s, action)] = next_s

    def _construct_predecessor_map(self):
        for s in self.S:
            self.predecessors[s] = []
            for sp in self.S:
                avail_actions = self.avail_actions[sp]
                for action in avail_actions:
                    if self.successor[(sp, action)] == s:
                        self.predecessors[s].append((sp, action))

    def solve_minimal_comms_vals(self, prob_threshold):
        """computes the minimum communication for a team of agents required to reach a goal state of an MDP given sampled success probabilities and a task completion probability constraint that must be satisficed

        Returns:
            policy (array): stochastic policy for each state of the MDP
            optimal_comms_vals (dict): minimal communication values for the team of agents to use for each subtask
            goal_reach_prob (float): probability of reaching the goal state from an initial state under the communication minimization problem
        """

        # initialize gurobi model
        model = Model("my_model")

        # specifiy non-convex optimization
        model.params.NonConvex = 2

        # load sampled communication threshold and associated empirical success probabilities for each subtask
        comms_vals = {}
        success_prob_vals = {}

        for a in self.A:
            comms_vals[a] = self.controller_dict[a].get_comm_threshold_list()
            success_prob_vals[a] = self.controller_dict[a].get_success_prob_list()

        # define state-action occupancy measure decision variable, x(s, a)
        state_act_vars = {}

        # dummy action for goal state
        self.avail_actions[self.s_g] = [0]

        for s in self.S:
            for a in self.avail_actions[s]:
                state_act_vars[s, a] = model.addVar(lb=0, name=f"state_act_{str(s)}_{str(a)}")
        model.update()


        # define binary decision variable to select the communication threshold and associated success probability
        ## assume we sample the same number of communication values for each subtask
        n_sampled_vals = len(comms_vals[0])
        binary_decision_vars = {}
        for a in self.A:
            constraint = 0
            for val_idx in range(n_sampled_vals):
                binary_decision_vars[a, val_idx] = model.addVar(vtype=GRB.BINARY, name=f"binary_decision_var_{str(a)}_{str(val_idx)}")

                # add a constraint so only one decision variable can be equal to 1 and the rest are 0
                ## binary_decision_var == 1 indicates we have chosen that index for the communcation and empirical completion probability
                ## implement by specifying the binary_decision_vars sum to 1 for each subtask
                constraint += binary_decision_vars[a, val_idx]

            model.addConstr(constraint == 1)

        # update model so I can reference the binary decision variables in defining other constraints and the objective
        model.update()


        # define the objective function to minimize
        ## J = \sum_a \sum_i d_{i, a} * \lambda_{i, a}
        objective = 0
        for a in self.A:
            for val_idx in range(n_sampled_vals):
                # define objective in terms of the binary decision vars already defined in the model
                binary_decision_var = model.getVarByName(f"binary_decision_var_{str(a)}_{str(val_idx)}")

                comms_val = comms_vals[a][val_idx]
                objective += binary_decision_var * comms_val

        model.setObjective(objective, GRB.MINIMIZE)
        model.update()


        # define Bellman flow constraint for each state-action occupancy measure
        """
        Bellman flow: the occupancy measure of the current state-action is the sum of the occupancy measures of all state-actions that enter the current state-action multiplied by the probability they will succeed in reaching the current state from the predecessor state

        Occupancy measure: the expected number of visitations (i.e., number of times times a particular action "a" is executed in state "s") scaled so the occupancy measure is a probability function (occupancy_s_a / (\sum_s \sum_a occupancy_s_a)). You can use this to get a stochastic policy.

        The initial state has no predecessor states, so we set the summed occupancies of thet initial state equal to 1 if in the initial state (which is how the \delta_{u_I} function is defined)

        Constraint definition: \sum_a x(s, a) - \sum_{s', a'} x(s', a') p_{a'} - \delta_{s_I}(s) = 0
        where s' and a' are predecessor state-actions of s
        """
        for s in self.S:
            constraint = 0

            # add occupancy for available actions
            for a in self.avail_actions[s]:
                constraint += state_act_vars[s, a]

            # add incoming occupancy for predecessor state-actions
            for s_pred, a_pred in self.predecessors[s]:

                # the "!=" statements are there so we don't double count reaching the goal and failure states
                if (s_pred != self.s_g) and (s_pred != self.s_fail):
                    for val_idx in range(n_sampled_vals):
                        # define constraint in terms of the binary decision vars already defined in the model
                        # only one of the completion_probs will be non-zero b/c of how the binary_decision_vars are constrained
                        binary_decision_var = model.getVarByName(f"binary_decision_var_{str(a_pred)}_{str(val_idx)}")
                        completion_prob = binary_decision_var * success_prob_vals[a_pred][val_idx]

                        constraint -= state_act_vars[s_pred, a_pred] * completion_prob
                # pdb.set_trace()
                # pred_constraint *= state_act_vars[s_pred, a_pred]
                # constraint -= pred_constraint
            # define initial state occupancy
            if s == self.s_i:
                constraint -= 1

            #TODO remove once the model works as expected
            # # add constraint to the model
            # print(s)
            # print(constraint)
            # pdb.set_trace()

            model.addConstr(constraint == 0)
        model.update()


        # define constraint on global task completion probability
        ## state_act_vars[s_g, 0] is the expected probability of reaching the final state starting from the initial state, so setting state_act_vars[s_g, 0] >= prob_threshold is exactly what we want here
        ## this is because state_act_vars[s_g, 0] has the state-action occupancy information from all predecessor states, which itself has information about completion probability
        ## this is equivalent to putting a constraint on the values that the Bellman Flow condition can take for a particular state, and could be defined for other "key states" that have some constraint on the visitation probability if that makes sense based on the environment
        for s in self.S:
            if s == self.s_g:
                # since there is only 1 "dummy action" in the final state, we can write the constraint like this (instead of as a sum over actions)
                model.addConstr(state_act_vars[s, 0] >= prob_threshold)
        model.update()

        # solve the optimization problem
        model.optimize()

        # print variable values
        # for var in model.getVars():
        #     print(f"Variable: {var.VarName} --- Value: {var.x}")

        # translate model output (binary decision variables) into chosen communication values and success probabilites
        chosen_idx = {}
        optimal_comms_vals = {}
        chosen_success_probs = {}

        for a in self.A:
            for val_idx in range(n_sampled_vals):
                if binary_decision_vars[a, val_idx].x == 1:
                    chosen_idx[a] = val_idx
                    optimal_comms_vals[a] = comms_vals[a][val_idx]
                    chosen_success_probs[a] = success_prob_vals[a][val_idx]

        goal_reach_prob = state_act_vars[self.s_g, 0].x

        # round goal_reach_prob to 1 if it is supposed to be 1
        ## Gurobi likes to output numbers like 1.0000000000000004 b/c it uses floating point numbers
        if goal_reach_prob >= 1:
            if (round(goal_reach_prob, 5) == 1.0):
                goal_reach_prob = 1.0

        # compute the policy from the model output
        if model.SolCount == 0:
            feasible_flag = False
        else:
            feasible_flag = True

        if feasible_flag:
            # construct the policy from occupancy variables
            policy = np.zeros((self.N_S, self.N_A), dtype=np.float)
            for s in self.S:
                if len(self.avail_actions[s]) == 0:
                    # return filler value if no actions available
                    policy[s, :] = -1
                else:
                    occupancy_state = np.sum([state_act_vars[s, a].x for a in self.avail_actions[s]])
                    # set policy to uniform random over available actions if the optimization does not output an occupancy measure for this state
                    #TODO this doesn't make sense. Why wouldn't the optimization output a policy for a given state?
                    ## this breaks my stuff in the case of a simple 2-state transition with 3 actions, and success_prob given by f(x) = x for all tasks. Assigns equal probability to each action, which is WRONG

                    # pdb.set_trace()
                    if occupancy_state == 0.0:
                        for a in self.avail_actions[s]:
                            policy[s, a] = 1 / len(self.avail_actions[s])
                    if occupancy_state > 0.0:
                        for a in self.avail_actions[s]:
                            policy[s, a] = state_act_vars[s, a].x / occupancy_state

        else:
            policy = -1 * np.ones((self.N_S, self.N_A), dtype=np.float)
            goal_reach_prob = -1

        print("\n\n")
        print(f"Policy: \n{np.round(policy, 2)}\n")


        print(f"Communication values for subtask policies: {optimal_comms_vals}")
        print(f"Chosen success probabilities for subtask policies: {chosen_success_probs}")
        print(f"Goal reach probability: {goal_reach_prob}")
        pdb.set_trace()

        return policy, optimal_comms_vals, chosen_success_probs, goal_reach_prob, feasible_flag

