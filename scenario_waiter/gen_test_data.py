import numpy as np

import cnirl_game

NUM_TEST = 200
Ts = cnirl_game.Ts
CRY_DURATION = cnirl_game.CRY_DURATION


for NUM_BABY in range(1, 10):
    T = Ts[NUM_BABY-1]
    game = cnirl_game.NavigationGame(num_baby=NUM_BABY)

    # dataset of trajectories, each entry is ([s_1,...,s_T],[a_1,...,a_T],[c_1,...,c_T]), s-a-c pairs
    cnirl_save_path = 'data/test_data_cnirluse/' + str(NUM_BABY) + 'obj/'
    other_save_path = 'data/test_data_otheruse/' + str(NUM_BABY) + 'obj/'

    Reward_list = []
    help_rate_list = []
    crash_rate_list = []

    for i in range(NUM_TEST):
        critime = []
        stoptime = []
        for _ in range(NUM_BABY):
            temp = np.random.randint(0,T)
            critime.append(temp)
            stoptime.append(temp+CRY_DURATION)
        _,_,_,_,_,totalR,help_rate,crash_rate = game.play_MNMDP_task(
                                                R_list=game.R_list,
                                                V_list=game.V_list,
                                                RewardActivFunc = game.true_reward_activation,
                                                Crytime_list = critime, Stoptime_list = stoptime)

        np.save(cnirl_save_path + 'critime' + str(i), np.array(critime).astype(int))
        np.save(cnirl_save_path + 'stoptime' + str(i), np.array(stoptime).astype(int))

        np.save(other_save_path + 'critime' + str(i), np.array(critime).astype(int))
        np.save(other_save_path + 'stoptime' + str(i), np.array(stoptime).astype(int))

        Reward_list.append(totalR)
        help_rate_list.append(help_rate)
        crash_rate_list.append(crash_rate)

    print ('number of baby is', NUM_BABY)
    print ("generated", NUM_TEST, "trajectories")
    print ("expert average reward on test data:", np.mean(Reward_list))
    print ("expert average help rate on test data:", np.mean(help_rate_list))
    print ("expert average crash rate on test data:", np.mean(crash_rate_list))
    print ()

    np.save(cnirl_save_path + 'avg_reward', np.mean(Reward_list))
    np.save(cnirl_save_path + 'avg_help', np.mean(help_rate_list))
    np.save(cnirl_save_path + 'avg_crash', np.mean(crash_rate_list))

    np.save(other_save_path + 'avg_help', np.mean(help_rate_list))
    np.save(other_save_path + 'avg_crash', np.mean(crash_rate_list))
