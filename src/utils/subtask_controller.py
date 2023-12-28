import numpy as np
import json
import os
import pdb

class SubtaskController():
    def __init__(self, controller_idx, init_states, final_states, success_function_idx=0, low_comms_success_prob=1.0, high_comms_success_prob=1.0):
        self.controller_idx = controller_idx
        self.init_states = init_states
        self.final_states = final_states

        self.success_function_idx = success_function_idx
        self.low_comms_success_prob = low_comms_success_prob
        self.high_comms_success_prob = high_comms_success_prob

        # values of communication threshold to be sampled
        self.communication_thresholds = np.linspace(0.0, 1.0, num=4)

        # choose these functions y = f(x) such that y \in [0, 1] for x \in [0, 1]
        self.success_functions = {}

        self.load_success_functions()
        # self.success_functions[0] = self.f_0
        # self.success_functions[1] = self.f_1
        # self.success_functions[2] = self.f_2
        # self.success_functions[3] = self.f_3
        # self.success_functions[4] = self.f_4
        # self.success_functions[5] = self.f_5
        # self.success_functions[6] = self.f_6
        # self.success_functions[7] = self.f_7
        # # list of success probs at sampled points of f_c
        # self.success_prob_list = self._sample_success_probability_function()

        self.comms_vals = self.success_functions[success_function_idx]["comms_vals"]
        self.success_prob_list = self.success_functions[success_function_idx]["success_probs"]

        # temporary success prob to get HLMDP working in the first place
        self.success_prob = 0.99

    def load_success_functions(self):
        for i in range(1, 5):
            with open(f"./results/success_probs_box_{i}_vdn_low_casec_high.json") as f:
                data = json.load(f)
                data_tmp = data["success_probs"]["3"]

            self.success_functions[i] = {
                "comms_vals": list(data_tmp.keys()),
                "success_probs": list(data_tmp.values())
            }


    def _sample_success_probability_function(self):
        """generates samples from a simulated success probability function at different values of communication between agents
        """
        #TODO this is a really dumb way to implement this
        if self.success_function_idx == 7:
            success_vals = self.success_functions[self.success_function_idx](self.communication_thresholds, self.low_comms_success_prob, self.high_comms_success_prob)
        else:
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

    def f_7(self, x, low_val, high_val):
        # easier for 1 agent to complete
        high_vals = [0.97 for i in range(int(len(x)/2))]
        low_vals = [0.85 for i in range(int(len(x)/2))]

        vals = high_vals + low_vals
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

