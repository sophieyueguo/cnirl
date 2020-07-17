import numpy as np
import matplotlib.pyplot as plt
import time
import torch

import cnirl_game

RUNTIME_IND = 1
NUM_BABY = 2
IS_TESTING_PERFORMANCE = False

NUM_TRAIN = 200
NUM_TEST = 100
NUM_ITERATION = 300

ALPHA_STEPSIZE_COEFF = 0.01
THETA_STEPSIZE_COEFF = 0.01


cnirl_train_save_path = 'data/train_data_cnirluse/' + str(NUM_BABY) + 'obj/'
RUNTIME_SAVE_PATH = 'runtime_results/cnirl/' + str(NUM_BABY) + 'obj/' + str(RUNTIME_IND)

D = []
for i in range(NUM_TRAIN):
    states = np.load(cnirl_train_save_path + 'states' + str(i) + '.npy')
    actions = np.load(cnirl_train_save_path + 'actions' + str(i) + '.npy')
    contexts = np.load(cnirl_train_save_path + 'contexts' + str(i) + '.npy')
    D.append((states, actions, contexts))

game = cnirl_game.NavigationGame(num_baby=NUM_BABY)



itertation_num = 200   # iteration # for these iterations
# computes V,Q,Pi,Z for reward parameter theta
# this function is conditioned on the game setting(such as uses game.size), not isolate
# model-based method: requires domain transitions, which can be easily estimated
def V_Q_Pi_Z_iteration(theta):
    K = theta.shape[0]
    Q = np.zeros([K,game.size,game.size,5])  # Q[k,x,y,a] is the Q-value Q((x,y),a) for theta_k
    for k in range(K):
        for x in range(game.size):
            for y in range(game.size):
                state_repre = game.phi([x,y])
                for a in game.ValidActions([x,y]):
                    Q[k,x,y,a] = np.dot(theta[k],state_repre)

    V = np.zeros([K,game.size,game.size])    # V[k,x,y] is the V-value of V((x,y)) for theta_k
    for k in range(K):
        V[k,:,:] = np.dot(game.phi_map,theta[k])

    Z = np.zeros([K,game.size,game.size])
    Pi = np.zeros([K,game.size,game.size,5])

    for _ in range(itertation_num):
        for k in range(K):
            for x in range(game.size):
                for y in range(game.size):
                    Z[k,x,y] = 0
                    for a in game.ValidActions([x,y]):
                        s_prime = game.move([x,y],a)
                        Q[k,x,y,a] = np.dot(theta[k],game.phi([x,y])) + game.gamma*V[k,s_prime[0],s_prime[1]]
                        Z[k,x,y] += np.exp(game.beta*Q[k,x,y,a])

                    V[k,x,y] = 0
                    for a in game.ValidActions([x,y]):
                        Pi[k,x,y,a] = np.exp(game.beta*Q[k,x,y,a])/Z[k,x,y]
                        V[k,x,y] += Pi[k,x,y,a]*Q[k,x,y,a]

    return V,Q,Pi,Z

# using the parameters theta, compute V,Q,dV,dQ,Pi,dPi,Z,dZ
def V_dV_Q_dQ_Pi_dPi_Z_dZ_iteration(theta, V,dV,Q,dQ,Pi,dPi,Z,dZ):

    for iteration in range(itertation_num):
        dPi_before = dPi.copy()
        for k in range(K):
            for x in range(game.size):
                for y in range(game.size):
                    Z[k,x,y] = 0
                    dZ[k,x,y,:] = np.zeros(game.length_phi)
                    for a in game.ValidActions([x,y]):
                        s_prime = game.move([x,y],a)
                        Q[k,x,y,a] = np.dot(theta[k],game.phi([x,y])) + game.gamma*V[k,s_prime[0],s_prime[1]]
                        dQ[k,x,y,a,:] = game.phi([x,y]) + game.gamma*dV[k,s_prime[0],s_prime[1],:]
                        Z[k,x,y] += np.exp(game.beta*Q[k,x,y,a])
                        dZ[k,x,y,:] += game.beta*np.exp(game.beta*Q[k,x,y,a])*dQ[k,x,y,a,:]

                    V[k,x,y] = 0
                    dV[k,x,y,:] = np.zeros(game.length_phi)
                    for a in game.ValidActions([x,y]):
                        Pi[k,x,y,a] = np.exp(game.beta*Q[k,x,y,a])/Z[k,x,y]
                        dPi[k,x,y,a,:] = (game.beta*Z[k,x,y]*np.exp(game.beta*Q[k,x,y,a])*dQ[k,x,y,a,:]-np.exp(game.beta*Q[k,x,y,a])*dZ[k,x,y,:])/(Z[k,x,y]**2)
                        V[k,x,y] += Pi[k,x,y,a]*Q[k,x,y,a]
                        dV[k,x,y,:] += (Q[k,x,y,a]*dPi[k,x,y,a,:] + Pi[k,x,y,a]*dQ[k,x,y,a,:])
        dPi_after = dPi.copy()
        norm_diff = np.sum((dPi_after-dPi_before)**2)
        if norm_diff < 0.01:
            print("compute grad using", iteration, "iteration")
            break
    return V,dV,Q,dQ,Pi,dPi,Z,dZ



# see the ground truth's log likelihood
theta_truth = np.array(game.R_coeff_list)

tick = time.time()
V,Q,Pi,Z = V_Q_Pi_Z_iteration(theta_truth)
print(time.time()-tick)

L = 0
for (s_list,a_list,c_list) in D:
    for t in range(len(s_list)):
        s,a,c = s_list[t],a_list[t],c_list[t]
        scores = game.true_reward_activation(game.phi_2(s,c))
        E_Q_sac = sum([scores[i]*Q[i,s[0],s[1],a] for i in range(len(scores))])
        temp=np.array([sum([scores[i]*Q[i,s[0],s[1],a_prime]for i in range(len(scores))])for a_prime in game.ValidActions(s)])
        L += game.beta*E_Q_sac - np.log(np.sum(np.exp(game.beta*temp)))

print("truth Log-Likelihood:",L)



# this function is rolled out during computing gradient of alpha, needs to notice
# return a rank-1 np array consisting of distribution of R's
# here parameter x should be phi_2(s,c)
def rewardActivationwithAlpha(phi_2,alpha):
    # currently a softmax regression, borrowing functions in pytorch as intermediate
    alpha_torch = torch.tensor(alpha)
    phi_2_torch = torch.tensor(phi_2)
    out = torch.softmax(torch.mm(alpha_torch, phi_2_torch).view(-1),dim=0)
    return out.numpy()


# def optimize_alpha_fix_theta(alpha_old, theta_old, stepsize_alpha, regu):
#
#     theta = theta_old.copy()
#     alpha = alpha_old.copy()
#     alpha_torch = torch.tensor(alpha,requires_grad=True)
#     V,dV,Q,dQ,Pi,dPi,Z,dZ = V_dV_Q_dQ_Pi_dPi_Z_dZ_iteration(theta)
#     # compute the log-likelihood, here roll out all functions and use pytorch
#     L = torch.tensor(0.,dtype=torch.double)
#     for (s_list,a_list,c_list) in D:
#         for t in range(len(s_list)):
#             s,a,c = s_list[t],a_list[t],c_list[t]
#             scores = torch.softmax(torch.mm(alpha_torch,torch.tensor(game.phi_2(s,c))).view(-1),dim=0)
#             E_Q_sac = torch.sum(scores*torch.tensor(Q[:,s[0],s[1],a]))
#             temp=torch.tensor([torch.sum(scores*torch.tensor(Q[:,s[0],s[1],a_prime])) for a_prime in game.ValidActions(s)])
#             L += game.beta*E_Q_sac - torch.log(torch.sum(torch.exp(game.beta*temp)))
#     print("Log-Likelihood:",L.data.numpy())
#
#     print("optimizing alpha-------------")
#     L_regu = L + regu*torch.sum(alpha_torch**2)
#     L_regu.backward()
#     gradient_alpha = alpha_torch.grad.data.numpy()
#     print("gradient of alpha:")
#     print(gradient_alpha)
#     if stepsize_alpha is None:
#         stepsize_alpha = 0.08/max(np.max(np.abs(gradient_w1)),np.max(np.abs(gradient_w2)))
#
#     alpha += stepsize_alpha * gradient_alpha
#     return alpha, theta
#
# def optimize_theta_fix_alpha(alpha_old, theta_old, stepsize_theta):
#
#     theta = theta_old.copy()
#     alpha = alpha_old.copy()
#     alpha_torch = torch.tensor(alpha,requires_grad=True)
#     V,dV,Q,dQ,Pi,dPi,Z,dZ = V_dV_Q_dQ_Pi_dPi_Z_dZ_iteration(theta)
#     # compute the log-likelihood, here roll out all functions and use pytorch
#     L = torch.tensor(0.,dtype=torch.double)
#     for (s_list,a_list,c_list) in D:
#         for t in range(len(s_list)):
#             s,a,c = s_list[t],a_list[t],c_list[t]
#             scores = torch.softmax(torch.mm(alpha_torch,torch.tensor(game.phi_2(s,c))).view(-1),dim=0)
#             E_Q_sac = torch.sum(scores*torch.tensor(Q[:,s[0],s[1],a]))
#             temp=torch.tensor([torch.sum(scores*torch.tensor(Q[:,s[0],s[1],a_prime])) for a_prime in game.ValidActions(s)])
#             L += game.beta*E_Q_sac - torch.log(torch.sum(torch.exp(game.beta*temp)))
#     print("Log-Likelihood:",L.data.numpy())
#
#     print("optimizing theta-------------")
#     # calculate the gradient
#     gradient = np.zeros([len(theta),game.length_phi])
#     for (s_list,a_list,c_list) in D:
#         for t in range(len(s_list)):
#             s,a,c = s_list[t],a_list[t],c_list[t]
#             vActions = game.ValidActions(s)
#             scores = rewardActivationwithAlpha(game.phi_2(s,c),alpha)
#             for i in range(len(theta)):
#                 temp=np.array([sum([scores[i]*Q[i,s[0],s[1],a_prime] for i in range(len(scores))]) for a_prime in vActions])
#                 temp=np.exp(game.beta*temp)
#                 temp_2 = dQ[i,s[0],s[1],vActions,:]
#                 gradient[i]+=game.beta*scores[i]*(dQ[i,s[0],s[1],a,:]-sum([temp[i]*temp_2[i] for i in range(len(temp))])/(np.sum(temp)))
#
#     # print("gradient of theta:")
#     # print(gradient)
#
#     if stepsize_theta is None:
#         stepsize_theta = 0.06/np.max(np.abs(gradient))
#
#     theta += stepsize_theta * gradient
#
#     return alpha, theta

def optimize_alpha_and_theta(alpha_old, theta_old, stepsize_alpha, stepsize_theta, V,dV,Q,dQ,Pi,dPi,Z,dZ):

    theta = theta_old.copy()
    alpha = alpha_old.copy()
    alpha_torch = torch.tensor(alpha,requires_grad=True)
    # compute the log-likelihood, here roll out all functions and use pytorch
    L = torch.tensor(0.,dtype=torch.double)
    for (s_list,a_list,c_list) in D:
        for t in range(len(s_list)):
            s,a,c = s_list[t],a_list[t],c_list[t]
            scores = torch.softmax(torch.mm(alpha_torch,torch.tensor(game.phi_2(s,c))).view(-1),dim=0)
            E_Q_sac = torch.sum(scores*torch.tensor(Q[:,s[0],s[1],a]))
            temp=torch.tensor([torch.sum(scores*torch.tensor(Q[:,s[0],s[1],a_prime])) for a_prime in game.ValidActions(s)])
            L += game.beta*E_Q_sac - torch.log(torch.sum(torch.exp(game.beta*temp)))
    print("Log-Likelihood:",L.data.numpy())

    # print("optimizing alpha-------------")
    L.backward()
    gradient_alpha = alpha_torch.grad.data.numpy()
    # print("gradient of alpha:")
    # print(gradient_alpha)
    if stepsize_alpha is None:
        # stepsize_alpha = 0.1/np.max(np.abs(gradient_alpha))
        stepsize_alpha = ALPHA_STEPSIZE_COEFF/np.max(np.abs(gradient_alpha))

    alpha += stepsize_alpha * gradient_alpha

    # print("optimizing theta-------------")
    # calculate the gradient
    gradient = np.zeros([len(theta),game.length_phi])
    for (s_list,a_list,c_list) in D:
        for t in range(len(s_list)):
            s,a,c = s_list[t],a_list[t],c_list[t]
            vActions = game.ValidActions(s)
            scores = rewardActivationwithAlpha(game.phi_2(s,c),alpha)
            for i in range(len(theta)):
                temp=np.array([sum([scores[i]*Q[i,s[0],s[1],a_prime] for i in range(len(scores))]) for a_prime in vActions])
                temp=np.exp(game.beta*temp)
                temp_2 = dQ[i,s[0],s[1],vActions,:]
                gradient[i]+=game.beta*scores[i]*(dQ[i,s[0],s[1],a,:]-sum([temp[i]*temp_2[i] for i in range(len(temp))])/(np.sum(temp)))
    # print("gradient of theta:")
    # print(gradient)

    if stepsize_theta is None:
        # stepsize_theta = 0.1/np.max(np.abs(gradient))
        stepsize_theta = THETA_STEPSIZE_COEFF/np.max(np.abs(gradient))

    theta += stepsize_theta * gradient

    return alpha, theta



# initialize parameters randomly
K = 1 + len(game.babyPos_list)
current_alpha = np.random.randn(K,game.length_phi_2) / 10
current_theta = np.random.randn(K,game.length_phi) / 10

# initialize all maps
Q = np.zeros([K,game.size,game.size,5])  # Q[k,x,y,a] is the Q-value Q((x,y),a) for theta_k
for k in range(K):
    for x in range(game.size):
        for y in range(game.size):
            for a in game.ValidActions([x,y]):
                Q[k,x,y,a] = np.dot(current_theta[k],game.phi([x,y]))
dQ = np.zeros([K,game.size,game.size,5,game.length_phi])  # dQ[k,x,y,a,:] is the derivative of Q(s,a) w.r.t theta_k

V = np.zeros([K,game.size,game.size])    # V[k,x,y] is the V-value of V((x,y)) for theta_k
for k in range(K):
    V[k,:,:] = np.dot(game.phi_map,current_theta[k])

dV = np.zeros([K,game.size,game.size,game.length_phi]) # dV[k,x,y,:] is the derivative of V w.r.t. theta_k
for k in range(K):
    dV[k,:,:,:] = game.phi_map

Z = np.zeros([K,game.size,game.size])
dZ = np.zeros([K,game.size,game.size,game.length_phi])

Pi = np.zeros([K,game.size,game.size,5])
dPi = np.zeros([K,game.size,game.size,5,game.length_phi])



def simple_plot(save_path, title, l):
    t = np.arange(0.0, 10*len(l), 10)
    fig, ax = plt.subplots()
    ax.axis(ymin=0.0,ymax=1.0)
    line, = ax.plot(t, np.array(l), label='result after training')
    ax.legend()
    ax.set(xlabel='episode', ylabel='average',
           title=title)
    ax.grid()
    fig.savefig(save_path)


def evaluate(alpha,theta):
    print("start evaluating")
    size = game.size
    R_theta_list = [np.zeros([size,size]) for _ in range(K)]

    for ind in range(K):
        R_theta_list[ind] = np.dot(game.phi_map, theta[ind])

    for ind in range(K):
        plt.close('all')
        # print (reward_mat)
        plt.clf()
        plt.imshow(R_theta_list[ind]);
        plt.colorbar()
        # plt.show()
        plt.savefig('tmp_R' + str(ind))

    # pre-compute the value maps using the optimized rewards
    V_theta_list = [game.ValueIterationWithR(iteration_num=200, R=R_theta_list[i]) for i in range(K)]

    Reward_list = []
    help_rates = []
    crash_rates = []
    cnirl_test_save_path = 'data/test_data_cnirluse/' + str(NUM_BABY) + 'obj/'
    game.beta += 2.0
    for i in range(NUM_TEST):
        critime = np.load(cnirl_test_save_path + 'critime' + str(i) + '.npy')
        stoptime = np.load(cnirl_test_save_path + 'stoptime' + str(i) + '.npy')
        _,_,_,_,_,totalR,help_rate,crash_rate = game.play_MNMDP_task(R_list=R_theta_list,
                                                                V_list=V_theta_list,
                                                                RewardActivFunc=lambda x:rewardActivationwithAlpha(x,alpha),
                                                                Crytime_list = critime, Stoptime_list = stoptime)

        Reward_list.append(totalR)
        help_rates.append(help_rate)
        crash_rates.append(crash_rate)
    game.beta -= 2.0
    return np.mean(Reward_list), np.mean(help_rates), np.mean(crash_rates)


num_help_list = []
num_crash_list = []
start_time = time.time()
for iteration in range(NUM_ITERATION):
    if iteration == 1:
        time_spend = time.time() - start_time
        np.save(RUNTIME_SAVE_PATH, time_spend)
        print("--- %s seconds ---" % (time_spend))

    print("------------------------------------------------------------------")
    print("iteration:",iteration)
    if iteration % 10 == 0 and IS_TESTING_PERFORMANCE:
        Reward,num_help,num_crash = evaluate(current_alpha,current_theta)

        print(Reward, num_help, num_crash)
        num_help_list.append(num_help)
        num_crash_list.append(num_crash)
        np.save('results/cnirl/current/help_obj' + str(NUM_BABY), np.array(num_help_list))
        np.save('results/cnirl/current/crash_obj' + str(NUM_BABY), np.array(num_crash_list))
        simple_plot('results/cnirl/current/help_obj' + str(NUM_BABY), 'help', num_help_list)
        simple_plot('results/cnirl/current/crash_obj' + str(NUM_BABY), 'crash', num_crash_list)

    V,dV,Q,dQ,Pi,dPi,Z,dZ = V_dV_Q_dQ_Pi_dPi_Z_dZ_iteration(current_theta, V,dV,Q,dQ,Pi,dPi,Z,dZ)
    current_alpha, current_theta = optimize_alpha_and_theta(current_alpha, current_theta, None, None, V,dV,Q,dQ,Pi,dPi,Z,dZ)

print("truth reward weights:")
print(theta_truth)
print("learned reward weights:")
print(current_theta)
print("learned activation parameters")
print(current_alpha)

# size = game.size
# R_theta_list = [np.zeros([size,size]) for _ in range(K)]
#
# for ind in range(K):
#     R_theta_list[ind] = np.dot(game.phi_map, current_theta[ind])
#
# # pre-compute the value maps using the optimized rewards
# V_theta_list = [game.ValueIterationWithR(iteration_num=200, R=R_theta_list[i]) for i in range(K)]
#
# critime = []
# stoptime = []
# for i in range(len(game.babyPos_list)):
#     temp = np.random.randint(5,60)
#     critime.append(temp)
#     stoptime.append(temp+10)
# trajectory, actions, context_list, valuesMaps, scores_list = game.play_MNMDP_task(R_list=R_theta_list,
#                                                                     V_list=V_theta_list,
#                                                                     RewardActivFunc=lambda x:rewardActivationwithAlpha(x,current_alpha),
#                                                                     Crytime_list = critime, Stoptime_list = stoptime)
# print(context_list)
# for i in scores_list:
#     print(np.argmax(i))
#
#
# # align the value maps for visualization
# max_ = np.max(valuesMaps[0])
# for i in range(len(valuesMaps)):
#     valuesMap = valuesMaps[i]
#     local_max = np.max(valuesMap)
#     valuesMaps[i] *= (max_/local_max)
# plt.imshow(R_theta_list[3].T, cmap='viridis')
