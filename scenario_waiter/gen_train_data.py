import numpy as np

import cnirl_game
import augmented_game

NUM_TRAIN = 200
Ts = cnirl_game.Ts
CRY_DURATION = cnirl_game.CRY_DURATION


for NUM_BABY in range(1, 10):
    T = Ts[NUM_BABY-1]
    cgame = cnirl_game.NavigationGame(num_baby=NUM_BABY)
    agame = augmented_game.NavigationGame(num_baby=NUM_BABY)

    # dataset of trajectories, each entry is ([s_1,...,s_T],[a_1,...,a_T],[c_1,...,c_T]), s-a-c pairs
    cnirl_save_path = 'data/train_data_cnirluse/' + str(NUM_BABY) + 'obj/'
    other_save_path = 'data/train_data_otheruse/' + str(NUM_BABY) + 'obj/'

    Reward_list = []
    help_rate_list = []
    crash_rate_list = []

    for i in range(NUM_TRAIN):
        critime = []
        stoptime = []
        for _ in range(NUM_BABY):
            temp = np.random.randint(0,T)
            critime.append(temp)
            stoptime.append(temp+CRY_DURATION)
        trajectory, actions, context_list,_,_,totalR, help_rate, crash_rate = cgame.play_MNMDP_task(
                                                            R_list=cgame.R_list,
                                                            V_list=cgame.V_list,
                                                            RewardActivFunc = cgame.true_reward_activation,
                                                            Crytime_list = critime, Stoptime_list = stoptime)
        # print ('traj', trajectory)
        # print ('actions', actions)
        # print ('context_list', context_list)
        # print ('trajectory length', len(trajectory))
        np.save(cnirl_save_path + 'states' + str(i), trajectory)
        np.save(cnirl_save_path + 'actions' + str(i), np.array(actions).astype(int))
        np.save(cnirl_save_path + 'contexts' + str(i), np.array(context_list).astype(int))

        states = []
        for t in range(len(trajectory)):
            x, y = trajectory[t]
            context = context_list[t]
            c = int(''.join([str(cond) for cond in context]), 2)
            states.append(agame.state_to_int[str(np.array([x, y, c]))])
        np.save(other_save_path + 'states' + str(i), np.array(states).astype(int))
        np.save(other_save_path + 'actions' + str(i), np.array(actions).astype(int))

        Reward_list.append(totalR)
        help_rate_list.append(help_rate)
        crash_rate_list.append(crash_rate)

    print ('number of baby is', NUM_BABY)
    print ("generated", NUM_TRAIN, "trajectories")
    print ("expert average reward on train data:", np.mean(Reward_list))
    print ("expert average help rate on train data:", np.mean(help_rate_list))
    print ("expert average crash rate on train data:", np.mean(crash_rate_list))
    print ()

    np.save(cnirl_save_path + 'avg_reward', np.mean(Reward_list))
    np.save(cnirl_save_path + 'avg_help', np.mean(help_rate_list))
    np.save(cnirl_save_path + 'avg_crash', np.mean(crash_rate_list))

    np.save(other_save_path + 'avg_help', np.mean(help_rate_list))
    np.save(other_save_path + 'avg_crash', np.mean(crash_rate_list))
