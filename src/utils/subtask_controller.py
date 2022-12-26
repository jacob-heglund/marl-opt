import numpy as np
import pdb

class SubtaskController():
    def __init__(self, controller_idx, init_states, final_states, success_function_idx=0):
        self.controller_idx = controller_idx
        self.init_states = init_states
        self.final_states = final_states

        self.success_function_idx = success_function_idx

        # values of communication threshold to be sampled
        #TODO change back to 11 when done debuggin
        self.communication_thresholds = np.linspace(0.0, 1.0, num=11)

        # choose these functions y = f(x) such that y \in [0, 1] for x \in [0, 1]
        self.success_functions = {}

        self.success_functions[0] = self.f_0
        self.success_functions[1] = self.f_1
        self.success_functions[2] = self.f_2
        self.success_functions[3] = self.f_3
        self.success_functions[4] = self.f_4
        self.success_functions[5] = self.f_5
        self.success_functions[6] = self.f_6

        # list of success probs at sampled points of f_c
        self.success_prob_list = self._sample_success_probability_function()

        # temporary success prob to get HLMDP working in the first place
        self.success_prob = 0.99


    def _sample_success_probability_function(self):
        """generates samples from a simulated success probability function at different values of communication between agents
        """
        success_vals = self.success_functions[self.success_function_idx](self.communication_thresholds)
        return success_vals

    def f_0(self, x):
        return x

    def f_1(self, x):
        return 1 - x

    def f_2(self, x):
        return x**0.5

    def f_3(self, x):
        return 6*np.abs(x**3 - x**2)

    def f_4(self, x):
        return np.abs(np.cos(10*x + 0.7))

    def f_5(self, x):
        vals = [0.0 for i in range(len(x))]
        return vals

    def f_6(self, x):
        vals = [1.0 for i in range(len(x))]
        return vals

    def get_init_states(self):
        return self.init_states

    def get_final_states(self):
        return self.final_states

    def get_success_prob_list(self):
        return self.success_prob_list

    def get_comm_threshold_list(self):
        return self.communication_thresholds

    def get_success_prob(self):
        return self.success_prob

    '''
    TODO put back in high_leve_mdp.py if needed
    def solve_max_reach_prob_policy(self):
        """
        Find the meta-policy that maximizes probability of reaching the goal state.

        Outputs
        -------
        policy : numpy (N_S, N_A) array
            Array representing the solution policy. If there is no feasible solution, an array of
            -1 is returned.
        reach_prob : float
            The probability of reaching the goal state under the policy.
        feasible_flag : bool
            Flag indicating whether or not a feasible solution was found.
        """
        self.update_transition_function()

        #initialize gurobi model
        linear_model = Model("abs_mdp_linear")

        #dictionary for state action occupancy
        state_act_vars={}

        avail_actions = self.avail_actions.copy()

        #dummy action for goal state
        avail_actions[self.s_g]=[0]

        #create occupancy measures, probability variables and reward variables
        for s in self.S:
            for a in avail_actions[s]:
                state_act_vars[s,a]=linear_model.addVar(lb=0,name="state_act_"+str(s)+"_"+str(a))

        #gurobi updates model
        linear_model.update()

        #MDP bellman or occupancy constraints for each state
        for s in self.S:
            cons=0
            #add outgoing occupancy for available actions
            for a in avail_actions[s]:
                cons+=state_act_vars[s,a]

            #add ingoing occupancy for predecessor state actions
            for s_bar, a_bar in self.predecessors[s]:
                #this if clause ensures that you dont double count reaching goal and failure
                if not s_bar == self.s_g and not s_bar == self.s_fail:
                    cons -= state_act_vars[s_bar, a_bar] * self.P[s_bar, a_bar, s]
            #initial state occupancy
            if s == self.s_i:
                cons=cons-1

            #sets occupancy constraints
            linear_model.addConstr(cons==0)

        # set up the objective
        obj = 0
        obj+= state_act_vars[self.s_g, 0] # Probability of reaching goal state

        #set the objective, solve the problem
        linear_model.setObjective(obj, GRB.MAXIMIZE)
        linear_model.optimize()

        if linear_model.SolCount == 0:
            feasible_flag = False
        else:
            feasible_flag = True

        if feasible_flag:
            # Construct the policy from the occupancy variables
            policy = np.zeros((self.N_S, self.N_A), dtype=np.float)
            for s in self.S:
                if len(self.avail_actions[s]) == 0:
                    policy[s, :] = -1 # If no actions are available, return garbage value
                else:
                    occupancy_state = np.sum([state_act_vars[s,a].x for a in self.avail_actions[s]])
                    # If the state has no occupancy measure under the solution, set the policy to
                    # be uniform over available actions
                    if occupancy_state == 0.0:
                        for a in self.avail_actions[s]:
                            policy[s,a] = 1 / len(self.avail_actions[s])
                    if occupancy_state > 0.0:
                        for a in self.avail_actions[s]:
                            policy[s, a] = state_act_vars[s,a].x / occupancy_state
        else:
            policy = -1 * np.ones((self.N_S, self.N_A), dtype=np.float)

        reach_prob = state_act_vars[self.s_g, 0].x

        return policy, reach_prob, feasible_flag

        '''