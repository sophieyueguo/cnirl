#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from copy import deepcopy
import time
import torch
from operator import itemgetter
import pandas as pd
from scipy.special import logsumexp

import small_cnirl_game as cnirl_game

GAMMA = cnirl_game.GAMMA
NUM_LANES = cnirl_game.NUM_LANES
ROAD_LENGTH = cnirl_game.ROAD_LENGTH
CLEAR_RANGE = cnirl_game.CLEAR_RANGE
MIN_CP_X = cnirl_game.MIN_CP_X
MAX_CP_X = cnirl_game.MAX_CP_X
MIN_CP_Y = cnirl_game.MIN_CP_Y
MAX_CP_Y = cnirl_game.MAX_CP_Y

VISION = cnirl_game.VISION
CRITICAL_POS = cnirl_game.CRITICAL_POS

NUM_TRAIN_DEMO = cnirl_game.NUM_TRAIN_DEMO
NUM_TEST_DEMO = cnirl_game.NUM_TEST_DEMO


itertation_num = cnirl_game.itertation_num
MAX_A = cnirl_game.MAX_A
MIN_A = cnirl_game.MIN_A
MAX_EXP = cnirl_game.MAX_EXP
MIN_EXP = cnirl_game.MIN_EXP

NUM_EPISODE_RUN_BETEWEEN_TESTS = 10

input_context_functions = cnirl_game.input_context_functions

from small_cnirl_game import limit_exp as limit_exp
from small_cnirl_game import limit_inf as limit_int
from small_cnirl_game import limit_inf_list as limit_inf_list
from small_cnirl_game import limit_zero as limit_zero
from small_cnirl_game import limit_nan_lists as limit_nan_lists


# MAX_THETA_CHANGE = 5e-4
# MAX_ALPHA_CHANGE = 1e-3

THETA_LEARNING_RATE = 0.00001
ALPHA_LEARNING_RATE = 0.01

NUM_RAND_START = 1

D = []
for i in range(NUM_TRAIN_DEMO):
    dir_name = 'expert_demo/cnirl_use/' + str(i)
    states = np.load(dir_name + '/states.npy')
    actions = np.load(dir_name + '/actions.npy')
    contexts = np.load(dir_name + '/contexts.npy')
    contexts = contexts[:, 0:cnirl_game.CONTEXT_END_IND]
    T = 0
    while states[T][1] < ROAD_LENGTH:
        T += 1
    D.append((states[0:T], actions[0:T], contexts[0:T]))

# for (s_list, a_list, c_list) in D:
#     print (s_list)
#     print (a_list)
#     print (c_list)
#     print ()

input_index = 0
input_conditions = np.load('test_demo/conditions' + str(input_index) + '.npy')
expert_avoid_fail_r = np.load('test_demo/expert_avoid_fail_r' + str(input_index) + '.npy')
expert_help_succ_r = np.load('test_demo/expert_help_succ_r' + str(input_index) + '.npy')




def V_Q_Pi_Z_iteration(theta):
    K = len(theta)
    Q = np.zeros([K,game.num_lanes,game.road_length,game.num_actions])
    V = np.zeros([K,game.num_lanes,game.road_length])
    for k in range(K):
        for x in range(game.num_lanes):
            for y in range(game.road_length):
                V[k,x,y] = np.dot(theta[k].T,game.phi([x,y]))[0][0]

    Z = np.zeros([K,game.num_lanes,game.road_length])
    Pi = np.zeros([K,game.num_lanes,game.road_length,game.num_actions])

    for _ in range(itertation_num):
        for k in range(K):
            for x in range(game.num_lanes):
                for y in range(game.road_length):
                    actions = game.ValidActions([x,y])
                    for a in actions:
                        s_prime = game.move([x,y],a)
                        Q[k,x,y,a] = np.dot(theta[k].T,game.phi([x,y]))[0][0] + game.gamma*V[k,s_prime[0],s_prime[1]]
                    logZ_kxy = logsumexp([game.beta*Q[k,x,y,a] for a in actions])
                    Z[k,x,y] = np.exp(logZ_kxy)


                    V_kxy = []
                    for a in game.ValidActions([x,y]):
                        logPi_kxya = game.beta*Q[k,x,y,a] - logZ_kxy
                        Pi[k,x,y,a] = np.exp(logPi_kxya)
                        V_kxy.append(Pi[k,x,y,a]*Q[k,x,y,a])
                    V[k,x,y] = sum(V_kxy)


    return V,Q,Pi,Z



# using the parameters theta, compute V,Q,dV,dQ,Pi,dPi,Z,dZ
def V_dV_Q_dQ_Pi_dPi_Z_dZ_iteration(theta):
    K = len(theta)
    Q = np.zeros([K,game.num_lanes,game.road_length,game.num_actions])  # Q[k,x,y,a] is the Q-value Q((x,y),a) for theta_k
    dQ = np.zeros([K,game.num_lanes,game.road_length,game.num_actions,game.length_phi])  # Q[k,x,y,a,:] is the derivative of Q(s,a) w.r.t theta_k

    V = np.zeros([K,game.num_lanes,game.road_length])    # V[k,x,y] is the V-value of V((x,y)) for theta_k
    for k in range(K):
        for x in range(game.num_lanes):
            for y in range(game.road_length):
                V[k,x,y] = np.dot(theta[k].T,game.phi([x,y]))[0][0]

    dV = np.zeros([K,game.num_lanes,game.road_length,game.length_phi]) # dV[k,x,y,:] is the derivative of V w.r.t. theta_k
    for k in range(K):
        for x in range(game.num_lanes):
            for y in range(game.road_length):
                dV[k,x,y,:] = game.phi([x,y]).reshape(-1)

    Z = np.zeros([K,game.num_lanes,game.road_length])
    dZ = np.zeros([K,game.num_lanes,game.road_length,game.length_phi])

    Pi = np.zeros([K,game.num_lanes,game.road_length,game.num_actions])
    dPi = np.zeros([K,game.num_lanes,game.road_length,game.num_actions,game.length_phi])

    for _ in range(itertation_num):
        for k in range(K):
            for x in range(game.num_lanes):
                for y in range(game.road_length):
                    Z[k,x,y] = 0
                    dZ[k,x,y,:] = np.zeros(game.length_phi)
                    # for a in game.ValidActions([x,y]):
                    for a in range(game.num_actions):
                        s_prime = game.move([x,y],a)
                        Q[k,x,y,a] = np.dot(theta[k].T,game.phi([x,y]))[0][0] + game.gamma*V[k,s_prime[0],s_prime[1]]
                        dQ[k,x,y,a,:] = game.phi([x,y]).reshape(-1) + game.gamma*dV[k,s_prime[0],s_prime[1],:]
                        Z[k,x,y] += np.exp(game.beta*Q[k,x,y,a])
                        dZ[k,x,y,:] += game.beta*np.exp(game.beta*Q[k,x,y,a])*dQ[k,x,y,a,:]

                    V[k,x,y] = 0
                    dV[k,x,y,:] = np.zeros(game.length_phi)
                    # for a in game.ValidActions([x,y]):
                    for a in range(game.num_actions):
                        Pi[k,x,y,a] = np.exp(game.beta*Q[k,x,y,a])/Z[k,x,y]
                        dPi[k,x,y,a,:] = (game.beta*Z[k,x,y]*np.exp(game.beta*Q[k,x,y,a])*dQ[k,x,y,a,:]-np.exp(game.beta*Q[k,x,y,a])*dZ[k,x,y,:])/(Z[k,x,y]**2)
                        V[k,x,y] += Pi[k,x,y,a]*Q[k,x,y,a]
                        dV[k,x,y,:] += (Q[k,x,y,a]*dPi[k,x,y,a,:] + Pi[k,x,y,a]*dQ[k,x,y,a,:])

    return V,dV,Q,dQ,Pi,dPi,Z,dZ


# this function is rolled out during computing gradient of alpha, needs to notice
# return a rank-1 np array consisting of distribution of R's
# here parameter x should be phi_2(s,c)
def rewardActivationwithAlpha(x,alpha):
    # currently a softmax regression, borrowing functions in pytorch as intermediate
    alpha_torch = torch.tensor(alpha)
    x_torch = torch.tensor(x)
    return torch.nn.functional.softmax(torch.mm(alpha_torch,x_torch).view(-1),dim=0).numpy()


# debugging
def tmprewardActivationwithAlpha(input):
    if input[0][0] == 0:
        return [1, 0]
    else:
        return [0, 1]


# With alpha fixed, optimize theta for one gradient step
def optimize_theta(theta_old, alpha_old):

    theta = theta_old.copy()
    alpha = alpha_old.copy()

    V,dV,Q,dQ,Pi,dPi,Z,dZ = V_dV_Q_dQ_Pi_dPi_Z_dZ_iteration(theta)

    # compute the log-likelihood
    L = 0
    for (s_list, a_list, c_list) in D:
        for t in range(len(s_list)):
            s,a,c = s_list[t],a_list[t],c_list[t]
            # print ('alpha', alpha)
            print ('tmp scores', tmprewardActivationwithAlpha(game.phi_2(s,c)))
            scores = rewardActivationwithAlpha(game.phi_2(s,c),alpha)
            print ('scores', scores)
            print ('game.phi_2(s,c)', game.phi_2(s,c))
            print ()

            lsum = sum([scores[i]*Pi[i,s[0],s[1],a] for i in range(len(scores))])
            if lsum == 0:
                print (Pi[i,s[0],s[1],a])
            L += np.log(lsum)

    # calculate the gradient
    gradient = [np.zeros(game.length_phi) for _ in range(len(theta))]

    for (s_list,a_list,c_list) in D:
        for t in range(len(s_list)):
            s,a,c = s_list[t],a_list[t],c_list[t]
            # print ('alpha', alpha)
            scores = rewardActivationwithAlpha(game.phi_2(s,c),alpha)
            print ('tmp scores', tmprewardActivationwithAlpha(game.phi_2(s,c)))
            scores = rewardActivationwithAlpha(game.phi_2(s,c),alpha)
            print ('scores', scores)
            print ('game.phi_2(s,c)', game.phi_2(s,c))
            print ()

            denom = sum(scores[i]*Pi[i,s[0],s[1],a] for i in range(len(theta)))
            for i in range(len(theta)):
                gradient[i] += scores[i]*dPi[i,s[0],s[1],a]/denom

    gradient = limit_nan_lists(gradient)
    # step_size = MAX_THETA_CHANGE/max([np.amax(abs(g)) for g in gradient])
    step_size = THETA_LEARNING_RATE
    if step_size < MIN_EXP:
        print ('should stop, theta step size too small')
        step_size = MIN_EXP

    for i in range(len(theta)):
        theta[i] += step_size*gradient[i].reshape(-1,1)

    return theta, Pi


### given theta, optimize alpha for one step
def optimize_alpha(theta_old, alpha_old):

    alpha = alpha_old.copy()
    theta = theta_old.copy()

    llh, gradient = calculate_llh_and_gradient(theta, alpha)
    # step_size = MAX_ALPHA_CHANGE/np.amax(np.abs(gradient))
    # print ('gradient', gradient)
    step_size = ALPHA_LEARNING_RATE
    if step_size < MIN_EXP:
        print ('should stop, alpha step size too small')
        step_size = MIN_EXP

    alpha += step_size * gradient
    return alpha, llh



def calculate_llh_and_gradient(theta, alpha):
    V,Q,Pi,Z = V_Q_Pi_Z_iteration(theta)
    # compute the log-likelihood, here roll out all functions and use pytorch
    alpha_torch = torch.tensor(alpha,requires_grad=True) # require gradient computation
    L = torch.tensor(0.,dtype=torch.double)
    for (s_list,a_list,c_list) in D:
        for t in range(len(s_list)):
            s,a,c = s_list[t],a_list[t],c_list[t]
            scores = torch.nn.functional.softmax(torch.mm(alpha_torch,torch.tensor(game.phi_2(s,c))).view(-1),dim=0)
            # print ('scores', scores)
            # print ('Pi[i,s[0],s[1],a]', Pi[:,s[0],s[1],a])
            # print (sum([scores[i]*Pi[i,s[0],s[1],a] for i in range(len(scores))]))
            # if np.isinf(torch.log()):
            #
            # else:
            if sum(Pi[:,s[0],s[1],a]) > 0:
                L += torch.log(sum([scores[i]*Pi[i,s[0],s[1],a] for i in range(len(scores))]))
    # compute gradient
    L.backward()
    gradient = alpha_torch.grad.data.numpy()
    llh = L.data.numpy()
    return llh, gradient


###############################################################################
# change the beta of the game
game = cnirl_game.TrafficGame(num_lanes=NUM_LANES, road_length=ROAD_LENGTH, gamma=GAMMA, beta=5.0,
                   critical_pos=CRITICAL_POS, vision=VISION, clear_range=CLEAR_RANGE,
                   min_cp_x=MIN_CP_X, max_cp_x=MAX_CP_X, min_cp_y=MIN_CP_Y, max_cp_y=MAX_CP_Y)

# # see the ground truth's log likelihood
# theta_truth = game.R_coefficients_list
# V,Q,Pi,Z = V_Q_Pi_Z_iteration(theta_truth)

# L = 0
# for (s_list,a_list,c_list) in D:
#     for t in range(len(s_list)):
#         s,a,c = s_list[t],a_list[t],c_list[t]
#         scores = game.true_reward_activation(game.phi_2(s,c))
#         L += np.log(sum([scores[k]*Pi[k,s[0],s[1],a] for k in range(len(scores))]))
#
# print("truth Log-Likelihood:",L)



def test_performance(theta, alpha, K):
    game.beta = 15.0
    R_theta_list = [np.zeros([game.num_lanes,game.road_length]) for _ in range(K)]
    for i in range(game.num_lanes):
        for j in range(game.road_length):
            state = [i,j]
            state_repre = game.phi(state)
            for k in range(K):
                R_theta_list[k][i,j] = np.dot(theta[k].T, state_repre)

    plt.close('all')
    for cp0 in range(2):
        reward_mat = np.zeros([game.num_lanes, game.road_length])
        for x in range(game.num_lanes):
            for y in range(game.road_length):
                reward_mat[x, y] = R_theta_list[cp0][x,y]
        # print (reward_mat)
        plt.clf()
        plt.imshow(reward_mat);
        plt.colorbar()
        # plt.show()
        plt.savefig('tmp_R' + str(cp0))
    print ()


    # pre-compute the value maps using the optimized rewards
    V_theta_list = [game.ValueIterationWithR(iteration_num=200, R=R_theta_list[k]) for k in range(K)]

    D_test = []


    for i in range(NUM_TEST_DEMO):
        single_conditions = input_conditions[i, :]
         # 0 means the human agent needs to be helped, 1 means the human agent needs to be avoid
        # single_conditions = np.random.choice(2, len(CRITICAL_POS))
        trajectory, actions, context_list, valuesMaps, rewardIndices, moving_car_positions = game.play_MNMDP_task(AgentInitialState=[1,0],
                                                                    InputContextFunc=input_context_functions[np.random.choice(len(input_context_functions))],
                                                                    R_list=R_theta_list,
                                                                    V_list=V_theta_list,
                                                                    RewardActivFunc=lambda x:rewardActivationwithAlpha(x,alpha),
                                                                    conditions = single_conditions,
                                                                    display = False,
                                                                    explore=True)
        # D_test.append((trajectory,actions,context_list,rewardIndices))
        D_test.append((trajectory, actions, moving_car_positions, single_conditions))


    print("generated",len(D_test),"testing trajectories")
    avoid_fail_r =  cnirl_game.performance_avoiding(D_test)
    help_succ_r = cnirl_game.performance_helping(D_test)
    print ('avoid fail rate', avoid_fail_r)
    print ('help rate', help_succ_r)

    game.beta=5.0

    return avoid_fail_r, help_succ_r




def print_avoid_fail(avoid_fail_rates, expert_avoid_fail_rates):
    # Data for plotting avoid failure
    t = np.arange(0.0, 10*len(avoid_fail_rates), 10)

    fig, ax = plt.subplots()
    ax.axis(ymin=0.0,ymax=1.0)
    line, = ax.plot(t, np.array(avoid_fail_rates), label='result after training')
    expert_line, = ax.plot(t, np.array(expert_avoid_fail_rates),  label='expert')

    ax.legend()

    ax.set(xlabel='episode', ylabel='avoid fail rates',
           title='The failure rates of avoiding the critical locations')
    ax.grid()

    fig.savefig("cnirl_save/current/avoid_fail.png")
    np.save('cnirl_save/current/avoid_fail_rates', np.array(avoid_fail_rates))
    # plt.show()



def print_help_succ(help_succ_rates, expert_help_succ_rates):
    # Data for plotting helping success
    t = np.arange(0.0, 10*len(help_succ_rates), 10)

    fig, ax = plt.subplots()
    ax.axis(ymin=0.0,ymax=1.0)
    line, = ax.plot(t, np.array(help_succ_rates), label='result after training')
    expert_line, = ax.plot(t, np.array(expert_help_succ_rates), label='expert')

    ax.legend()

    ax.set(xlabel='episode', ylabel='help success rates',
           title='The success rates of making help in the critical locations')
    ax.grid()

    fig.savefig("cnirl_save/current/help_succ.png")
    np.save('cnirl_save/current/help_succ_rates', np.array(help_succ_rates))
    # plt.show()





###############################################################################
K = 2*len(CRITICAL_POS)
num_tmp = NUM_RAND_START

# tmp_alpha = [np.random.rand(K,game.length_phi_2) for tm in range(num_tmp)]
# tmp_theta = [[np.random.rand(game.length_phi).reshape(-1,1) for _ in range(K)] for tm in range(num_tmp)]
#
#
# llhs = [calculate_llh_and_gradient(tmp_theta[tm], tmp_alpha[tm])[0] for tm in range(num_tmp)]
# selected_ind = llhs.index(max(llhs))
# alpha = tmp_alpha[selected_ind]
# theta = tmp_theta[selected_ind]
#
#
# llh = llhs[selected_ind]
# print("Log-Likelihood:", llh)
# print("initial_alpha:")
# print(alpha)
# print("initial_theta:")
# print(theta)
# print ("intial llh", llh)

alpha = np.random.rand(K, game.length_phi_2)
# alpha = np.zeros([K, game.length_phi_2])
theta = [np.zeros(game.length_phi).reshape(-1, 1) for _ in range(K)]
llh = calculate_llh_and_gradient(theta, alpha)[0]


MAX_ITER = 1000
DELTA_LLH_THRESHOLD = -5000  # threshold of llh for a bad descent
NUM_LLH = 10  # number of iterations to monitor termination
MAX_LLH_DELTA_SUM = 1.0 # if sum of the consecutive delta smaller than this value, iteration terminates
llh_deltas = []


best_theta = theta
best_alpha = alpha
best_llh = llh

avoid_fail_rates = []
help_succ_rates = []
expert_avoid_fail_rates = []
expert_help_succ_rates = []

print ('start theta alpha alternative gradient descent')
for iteration in range(MAX_ITER):
    print("------------------------------------------------------------------")
    print("iteration:",iteration)

    # fix alpha, optimize theta
    new_theta, new_Pi = optimize_theta(theta, alpha)

    # fix theta, optimize alpha
    new_alpha, new_llh = optimize_alpha(new_theta, alpha)


    print("Log-Likelihood:", new_llh)

    if np.isnan(new_llh):
        print ('nan, stop')
        break
    else:
        llh_deltas.append(abs(new_llh-llh))
        theta, alpha, llh = new_theta, new_alpha, new_llh

        if llh > best_llh:
            best_theta = theta
            best_alpha = alpha
            best_llh = llh
            print ('best theta, alpha, llh updated')
        print ('good')

        if (iteration) % NUM_EPISODE_RUN_BETEWEEN_TESTS == 0:
            avoid_fail_r, help_succ_r = test_performance(best_theta, best_alpha, K)
            avoid_fail_rates.append(avoid_fail_r)
            help_succ_rates.append(help_succ_r)
            expert_avoid_fail_rates.append(expert_avoid_fail_r)
            expert_help_succ_rates.append(expert_help_succ_r)
            print_avoid_fail(avoid_fail_rates, expert_avoid_fail_rates)
            print_help_succ(help_succ_rates, expert_help_succ_rates)

            # if (iteration)%100 == 0:
            #     print (avoid_fail_rates, expert_avoid_fail_rates)
            #     print (help_succ_rates, expert_help_succ_rates)


    # if len(llh_deltas) > NUM_LLH:
    #     if sum(llh_deltas[(len(llh_deltas) - NUM_LLH):]) < MAX_LLH_DELTA_SUM:
    #         print ('last 10 elements', llh_deltas[(len(llh_deltas) - NUM_LLH):])
    #         print ('sum', sum(llh_deltas[(len(llh_deltas) - NUM_LLH):]))
    #         break


    print("------------------------------------------------------------------")


theta = best_theta
alpha = best_alpha

# print("truth reward weights:")
# for r in game.R_coefficients_list:
#     print (r)
print("learned reward weights theta:")
print(theta)
print("learned activation parameters alpha:")
print(alpha)

print ('best llh', best_llh)
print (avoid_fail_rates, expert_avoid_fail_rates)
print (help_succ_rates, expert_help_succ_rates)
