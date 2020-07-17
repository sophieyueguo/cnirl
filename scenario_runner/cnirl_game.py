import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from copy import deepcopy
import time
import torch
from operator import itemgetter
import pandas as pd
from scipy.special import logsumexp


GAMMA = 0.9
NUM_LANES = 3
ROAD_LENGTH = 40
CLEAR_RANGE = 1
MIN_CP_X = 0
MAX_CP_X = 0
MIN_CP_Y = 9
MAX_CP_Y = 37

VISION = 16
SAFE_DIST = 0.5*VISION
CRITICAL_POS = [[0,9], [2,16], [0,23], [2,30], [0,37]]
STEP_THRES = 70

MIN_TIME_TO_HELP = 1



NUM_TRAIN_DEMO = 500
NUM_TEST_DEMO = 500

LENGTH_PHI = 8
LENGTH_PHI_2 = 9

############### utility #######
itertation_num = 50   # iteration # for these iterations
MAX_A = 709
MIN_A = -20
MAX_EXP = np.exp(MAX_A)
MIN_EXP = np.exp(MIN_A)

def limit_exp(x):
    if x <= MIN_A:
        return MIN_EXP
    return limit_inf(np.exp(x))

def limit_inf(x):
    if np.isinf(x):
        if x > 0:
            return MAX_EXP
        elif x < 0:
            return -MAX_EXP
    return x

def limit_inf_list(l):
    return [limit_inf(x) for x in l]

def limit_zero(e):
    if abs(e) < MIN_EXP:
        e = np.sign(e)*MIN_EXP
    return e

def limit_nan_lists(gradient):
    new_gradient = [np.zeros(LENGTH_PHI) for _ in range(len(gradient))]
    for i in range(len(gradient)):
        for j in range(LENGTH_PHI):
            if np.isnan(gradient[i][j]):
                print ('there is nan number!!!!!!!!!!')
                new_gradient[i][j] = 0.0
            else:
                new_gradient[i][j] = gradient[i][j]
    return new_gradient




def neighbors(s):
    x ,y = s[0], s[1]
    return [[x+1,y],[x-1,y],[x,y+1],[x,y]]

def InBound(s, num_lanes, road_length):
    return (0 <= s[0] <= num_lanes-1 and 0 <= s[1] <= road_length-1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class TrafficGame:
    def __init__(self, num_lanes, road_length, gamma, beta,
                 critical_pos, vision, clear_range, min_cp_x, max_cp_x, min_cp_y, max_cp_y, safe_dist=1.0):

        self.beta = beta                       # exploration temprature
        self.num_lanes = num_lanes             # grid world number of driving lanes
        self.road_length = road_length         # lenth of the road
        self.gamma = gamma                     # discount factor

        self.num_norms = 2                     # number of norms
        self.norm_grs = [[-3., -1.], [1., 1.]] # ground truth norm rewards
        self.num_non_crit_feat = 3             # number of non-critical features
        self.num_actions = 4                   # number of actions

        self.critical_pos = critical_pos       # critical positions
        self.vision = vision
        self.clear_range = clear_range
        self.min_cp_x = min_cp_x
        self.max_cp_x = max_cp_x
        self.min_cp_y = min_cp_y
        self.max_cp_y = max_cp_y
        self.safe_dist = safe_dist

        self.length_phi = LENGTH_PHI                    # should be consistent with self.phi,self.phi_2
        self.length_phi_2 = LENGTH_PHI_2

        # The following is the ground truth rewards
        domain_R_coefficients = np.array([3., 0., 0., 0., 0., 0., 0., 0.]).reshape(-1,1)
        normative_R_coefficients_0 = np.array([0., 3., 0., 1., 1., 1., 1., 1.]).reshape(-1,1)
        normative_R_coefficients_1 = np.array([.1, -1., 0., -.5, -.5, -.5, -.5, -.5]).reshape(-1,1)

        normative_R_coefficients_0_0 = np.array([.8, 0., 0., 2., 0., 0., 0., 0.]).reshape(-1,1)
        normative_R_coefficients_0_1 = np.array([.8, 0., 0., 0., 2., 0., 0., 0.]).reshape(-1,1)
        normative_R_coefficients_0_2 = np.array([.8, 0., 0., 0., 0., 2., 0., 0.]).reshape(-1,1)
        normative_R_coefficients_0_3 = np.array([.8, 0., 0., 0., 0., 0., 2., 0.]).reshape(-1,1)
        normative_R_coefficients_0_4 = np.array([.8, 0., 0., 0., 0., 0., 0., 2.]).reshape(-1,1)

        normative_R_coefficients_1_0 = np.array([1., 0., 0., -10., 0., 0., 0., 0.]).reshape(-1,1)
        normative_R_coefficients_1_1 = np.array([1., 0., 0., 0., -10., 0., 0., 0.]).reshape(-1,1)
        normative_R_coefficients_1_2 = np.array([1., 0., 0., 0., 0., -10., 0., 0.]).reshape(-1,1)
        normative_R_coefficients_1_3 = np.array([1., 0., 0., 0., 0., 0., -10., 0.]).reshape(-1,1)
        normative_R_coefficients_1_4 = np.array([1., 0., 0., 0., 0., 0., 0., -10.]).reshape(-1,1)


        self.R_coefficients_list = [domain_R_coefficients, normative_R_coefficients_0, normative_R_coefficients_1,
                                   normative_R_coefficients_0_0, normative_R_coefficients_0_1,normative_R_coefficients_0_2,
                                   normative_R_coefficients_0_3, normative_R_coefficients_0_4,
                                   normative_R_coefficients_1_0, normative_R_coefficients_1_1,normative_R_coefficients_1_2,
                                   normative_R_coefficients_1_3, normative_R_coefficients_1_4]

        self.help_car_R_coefficients_list = [domain_R_coefficients,
                                              normative_R_coefficients_0_0, normative_R_coefficients_0_1,normative_R_coefficients_0_2,
                                              normative_R_coefficients_0_3, normative_R_coefficients_0_4]

        self.avoid_car_R_coefficients_list = [domain_R_coefficients,
                                              normative_R_coefficients_1_0, normative_R_coefficients_1_1,normative_R_coefficients_1_2,
                                              normative_R_coefficients_1_3, normative_R_coefficients_1_4]

        self.norm_R_coefficients_list = [domain_R_coefficients,
                                   normative_R_coefficients_0_0, normative_R_coefficients_0_1,normative_R_coefficients_0_2,
                                   normative_R_coefficients_0_3, normative_R_coefficients_0_4,
                                   normative_R_coefficients_1_0, normative_R_coefficients_1_1,normative_R_coefficients_1_2,
                                   normative_R_coefficients_1_3, normative_R_coefficients_1_4]
        R_unormalized_list = []
        num_R = len(self.R_coefficients_list)
        for r_ind in range(num_R):
            R_unormalized_list.append(np.zeros([num_lanes,road_length]))
            for i in range(num_lanes):
                for j in range(road_length):
                    state = [i,j]
                    state_repre = self.phi(state)
                    R_unormalized_list[r_ind][i,j] = np.dot(self.R_coefficients_list[r_ind].T, state_repre)

#         min_r_value = min([np.amin(r) for r in R_unormalized_list])
#         max_r_value = max([np.amax(r) for r in R_unormalized_list])
#         self.R_list = [(r-min_r_value)/max_r_value for r in R_unormalized_list]
        self.R_list = R_unormalized_list



        self.V_list = [self.ValueIterationWithR(iteration_num=200, R=r) for r in self.R_list]

        


    # feature transform of a state, for reward functions' usage
    # state features are
    # index 0: 1/0(whether has reached the goal),
    # index 1: 1/0(whether is in the critical location)
    # index 2: action - temporarily it is zero for now
    # index 3-7: exact position in critical locations if in them
    def phi(self, state):
        representation = [0,0,0,0,0,0,0,0]
        if state[1] == self.road_length-1:
            representation[0] = 1
        if [state[0],state[1]] in self.critical_pos:
            representation[1] = 1
            representation[3+self.critical_pos.index([state[0],state[1]])] = 1
        for cp_ind in range(len(self.critical_pos)):
            cp = self.critical_pos[cp_ind]
            if state != cp:
                if cp[1] >= state[1] > cp[1]-SAFE_DIST and abs(state[0] - cp[0]) <= self.safe_dist:
                    representation[cp_ind+3] = 0.5
        # feature = np.zeros(LENGTH_PHI)
        # if state[1] == ROAD_LENGTH - 1:
        #     feature[0] = 1
        # for i in range(len(CRITICAL_POS)):
        #     cp = CRITICAL_POS[i]
        #     if state[0] == cp[0] and state[1] == cp[1]:
        #         feature[1+i] = 0.5
        #
        # return np.array(feature, dtype=np.float64).reshape(-1,1)
        return np.array(representation, dtype=np.float64).reshape(-1,1)

    # feature transform of a state-context pair, for reward activation
    def phi_2(self,s,c):
        f = np.append(np.array(c), 1)
        return np.array(f, dtype=np.float64).reshape(-1,1)

    # return all valid actions of current state: a subset of {0,1,2,3}
    def ValidActions(self,state):
        validactions = []
        next_Pos_list = neighbors(state)
        for i in range(len(next_Pos_list)):
            if InBound(next_Pos_list[i], self.num_lanes, self.road_length):
                validactions.append(i)
        return validactions

    # move at state with action, return the next state
    def move(self,state,action):
        return neighbors(state)[action]

    # value iteration using reward map R, return the value map V
    def ValueIterationWithR(self, iteration_num, R):
        num_lanes, road_length = self.num_lanes, self.road_length
        V = np.zeros([num_lanes, road_length])
        for _ in range(iteration_num):
            V_new = np.zeros([num_lanes, road_length])
            for i in range(num_lanes):
                for j in range(road_length):
                    state = [i,j]
                    validactions = self.ValidActions(state)

                    def EvaluateQ(action):
                        ns = self.move(state,action) # deterministic transitions
                        return R[ns[0],ns[1]] + self.gamma*V[ns[0],ns[1]]

                    Q_list = list(map(EvaluateQ, validactions))
                    V_new[i,j] = np.max(Q_list)
            V = V_new
        return V


    # the ground truth reward activation function, used for generating expert demos
    def true_reward_activation(self, state_context_features):
        if state_context_features[0][0]==1 and state_context_features[1][0]==0 and state_context_features[2][0]==0:
            act = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
            positions_with_car = np.where(np.array([state_context_features[c][0] for c in range(3, 8)]) == 1)[0]
            positions_with_car += 3
            if len(positions_with_car) == 1:
                act[positions_with_car] += 1
            else:
                act = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0])
            return act
        elif state_context_features[0][0]==1 and state_context_features[1][0]==1 and state_context_features[2][0]==0:
            act = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
            positions_with_car = np.where(np.array([state_context_features[c][0] for c in range(3, 8)]) == 1)[0]
            positions_with_car += 8
            if len(positions_with_car) == 1:
                act[positions_with_car] += 1
            else:
                act = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0])
            return act
        else:
            return np.array([1,0,0,0,0,0,0,0,0,0,0,0,0])

    ### CORE function: Modularized NMDP trajectory
    ### for each state-context, compute the reward probabilities via RewardActivFunc, then act according to corresponding reward map
    ### RewardActivFunc outputs probability distribution P(R|s,c) with input phi_2(s,c), according to the R's order in R_list
    def play_MNMDP_task(self, AgentInitialState, InputContextFunc, R_list, V_list,
                        RewardActivFunc, conditions, display=False, explore=True, step_thres=STEP_THRES):

        moving_car_positions = generate_moving_car_positions(conditions)
        AgentState = deepcopy(AgentInitialState)
        AgentTrajectory = []
        valuesMaps = []
        rewardIndex_list = []
        context_list = []
        action_list = []

        time_in_help = 0

        t = -1
        while True:
            if display:
                print(AgentState)
            t += 1
            if t >= step_thres:
                break

            currentContext = InputContextFunc(AgentState, t, moving_car_positions, conditions)
            context_list.append(currentContext)

            if AgentState[1]==self.road_length-1:
                break

            AgentTrajectory.append(AgentState)
            validactions = self.ValidActions(AgentState)

            def EvaluateQ(action, rewardInd):
                ns = self.move(AgentState,action)
                return R_list[rewardInd][ns[0],ns[1]] + self.gamma*V_list[rewardInd][ns[0],ns[1]]

            # compute which reward to follow for current state-context, using the given RewardActivFunc
            scores = RewardActivFunc(self.phi_2(AgentState,currentContext))
#             print ('t', t, 'scores', scores, 'AgentState', AgentState, 'Context', currentContext)
            rewardIndex = np.argmax(scores)
            rewardIndex_list.append(rewardIndex)
            valuesMaps.append(R_list[rewardIndex])

            if explore:
                # boltzmann exploration policy
                Q_values = np.array(list(map(lambda action: EvaluateQ(action, rewardIndex), validactions)))
                logprob = self.beta*Q_values - logsumexp(self.beta*Q_values)
                prob = [limit_exp(l) for l in logprob]
                prob /= sum(prob)
                rd_indices = np.random.choice(len(validactions), 1, p=prob)
                action = (np.array(validactions)[rd_indices])[0]
                action_list.append(action)
                AgentState = self.move(AgentState,action)
            else:
                # no exploration
                action = max(validactions, key = EvaluateQ_1)
                action_list.append(action)
                AgentState = self.move(AgentState,action)

        return np.array(AgentTrajectory),action_list,context_list,valuesMaps,rewardIndex_list,np.array(moving_car_positions)


def generate_moving_car_positions(conditions):

    # avoid the moving cars
    MOVINGCARSPEED = 1.2
    if sum(conditions) >= 4:
        MOVINGCARSPEED = 1.1
    #     np.random.uniform(low=1.0, high=1.1)

    positions = []

    prev_x = 1.
    prev_y = 1.
    t_in_cp = 0
    MIN_T_IN_CP = 5


    for t in range(STEP_THRES):
        proceed = True
        if [prev_x, int(prev_y)] in CRITICAL_POS and t_in_cp <= MIN_T_IN_CP:
            if conditions[CRITICAL_POS.index([prev_x, int(prev_y)])] == 0:
                x, y = prev_x, prev_y
                t_in_cp += 1
                proceed = False

        elif [prev_x, int(prev_y)] not in CRITICAL_POS and t_in_cp <= MIN_T_IN_CP         and [prev_x, int(prev_y+0.5*MOVINGCARSPEED)] in CRITICAL_POS and [prev_x, int(prev_y+MOVINGCARSPEED)] not in CRITICAL_POS:
            if conditions[ CRITICAL_POS.index([prev_x, int(prev_y+0.5*MOVINGCARSPEED)] )] == 0:
                x, y = prev_x, prev_y
                t_in_cp += 1
                proceed = False


        if proceed:
            t_in_cp = 0
            x_choices = [prev_x, 1.]
            x = x_choices[np.random.choice(2)]
            for cp_ind in range(len(CRITICAL_POS)):
                if abs(CRITICAL_POS[cp_ind][1] - prev_y) < 2:
                    x = CRITICAL_POS[cp_ind][0]

            if x != prev_x:
                y = prev_y
            else:
                y = prev_y + MOVINGCARSPEED

        positions.append([x, y])

        prev_x = x
        prev_y = y

    return positions



def CriticalConditionsContext(agent_pos, t, moving_car_positions, conditions):
    context = [0, 0, 0, 0, 0, 0, 0, 0]

    HELP_AHEAD = 2
    AVOID_AHEAD = 5
    ahead_time = [HELP_AHEAD, AVOID_AHEAD]

    if agent_pos[1] >= 0:
        context[0] = 1 # norm reward activated

        car_pos = [moving_car_positions[t][0], int(moving_car_positions[t][1])]
        if car_pos[1] >= MAX_CP_Y + CLEAR_RANGE:
            context[2] = 1
        else:
            for cp_ind in range(len(CRITICAL_POS)):
                cp = CRITICAL_POS[cp_ind]
                cond = conditions[cp_ind]

                if 0 <= cp[1] - car_pos[1] <= ahead_time[cond]:

                    context[cp_ind+3] = 1
                    context[1] = cond# see what norm it is to avoid or help
                    break

    return context

input_context_functions = [CriticalConditionsContext]

def is_close_to_cp(cp, agent_pos):
    if abs(cp[0] - agent_pos[0]) < 1:
        if 0 <= cp[1] - agent_pos[1] <= 1:
            return True
    return False

def performance_avoiding(D):
    num_should_avoid = 0.
    num_fail_avoid = 0.

    for (s_list,a_list,c_list,R_list) in D:
        cp_met = set([])
        cp_isfail = {}
        for t in range(len(s_list)):
            s,a,c = s_list[t],a_list[t],c_list[t]
            if c[0] == 1 and c[1] == 1:
                positions_with_car = np.where(np.array(c)[3: 8] == 1)[0]
                if len(positions_with_car) == 1:
                    cp_list = CRITICAL_POS[positions_with_car[0]]
                    cp = (cp_list[0], cp_list[1])
                    if cp not in cp_met:
                        cp_met.add(cp)
                        num_should_avoid += 1
                        cp_isfail[cp] = 0
                    if is_close_to_cp(cp, s):
                        cp_isfail[cp] = 1
        num_fail_avoid += sum([cp_isfail[cp] for cp in cp_isfail])

    avoid_fail_rate = num_fail_avoid / num_should_avoid
    print ('num_should_avoid', num_should_avoid)
    print ('num_fail_avoid', num_fail_avoid)
    return avoid_fail_rate



def performance_helping(D):
    num_should_help = 0.
    num_success_help = 0.

    for (s_list,a_list,c_list,R_list) in D:
        cp_met = set([])
        cp_issucc = {}
        for t in range(len(s_list)):
            s,a,c = s_list[t],a_list[t],c_list[t]
            if c[0] == 1 and c[1] == 0:
                positions_with_car = np.where(np.array(c)[3: 8] == 1)[0]
                if len(positions_with_car) == 1:
                    cp_list = CRITICAL_POS[positions_with_car[0]]
                    cp = (cp_list[0], cp_list[1])
                    if cp not in cp_met:
                        cp_met.add(cp)
                        num_should_help += 1
                        cp_issucc[cp] = 0
                    if is_close_to_cp(cp, s):
                        cp_issucc[cp] = 1
        num_success_help += sum([cp_issucc[cp] for cp in cp_issucc])

    help_sucess_rate = num_success_help / num_should_help
    print ('num_should_help', num_should_help)
    print ('num_success_help', num_success_help)
    return help_sucess_rate
