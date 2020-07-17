import numpy as np

import cnirl_game
NUM_TEST_DEMO = cnirl_game.NUM_TEST_DEMO

GAMMA = cnirl_game.GAMMA
NUM_LANES = cnirl_game.NUM_LANES
ROAD_LENGTH = cnirl_game.ROAD_LENGTH
CLEAR_RANGE = cnirl_game.CLEAR_RANGE
VISION = cnirl_game.VISION
CRITICAL_POS = cnirl_game.CRITICAL_POS
MIN_CP_X = cnirl_game.MIN_CP_X
MAX_CP_X = cnirl_game.MAX_CP_X
MIN_CP_Y = cnirl_game.MIN_CP_Y
MAX_CP_Y = cnirl_game.MAX_CP_Y


input_context_functions = cnirl_game.input_context_functions

def gen_test_conditions(save_index):
    save_path = 'test_demo/conditions' + str(save_index)
    conditions = np.zeros([NUM_TEST_DEMO, len(cnirl_game.CRITICAL_POS)])
    for i in range(NUM_TEST_DEMO):
        conditions[i, :] = np.random.choice(2, len(CRITICAL_POS))
    np.save(save_path, conditions.astype(int))

def test_test_conditions(input_index):
    input_path = 'test_demo/conditions' + str(input_index) + '.npy'
    D_expert = []
    game = cnirl_game.TrafficGame(num_lanes=NUM_LANES, road_length=ROAD_LENGTH, gamma=GAMMA, beta=15.0,
                       critical_pos=CRITICAL_POS, vision=VISION, clear_range=CLEAR_RANGE,
                       min_cp_x=MIN_CP_X, max_cp_x=MAX_CP_X, min_cp_y=MIN_CP_Y, max_cp_y=MAX_CP_Y)


    input_conditions = np.load(input_path)
    for i in range(NUM_TEST_DEMO):
        single_conditions = input_conditions[i, :]
        trajectory, actions, context_list, valuesMaps, rewardIndices,moving_car_positions = game.play_MNMDP_task(AgentInitialState=[1,0],
                                                                        InputContextFunc=input_context_functions[np.random.choice(len(input_context_functions))],
                                                                        R_list=game.R_list,
                                                                        V_list=game.V_list,
                                                                        RewardActivFunc = game.true_reward_activation,
                                                                        conditions = single_conditions)
        D_expert.append((trajectory,actions,context_list,rewardIndices))

    expert_avoid_fail_r = cnirl_game.performance_avoiding(D_expert)
    expert_help_succ_r = cnirl_game.performance_helping(D_expert)
    print ('expert_avoid_fail_r', expert_avoid_fail_r)
    print ('expert_help_succ_r', expert_help_succ_r)
    np.save('test_demo/expert_avoid_fail_r' + str(input_index), expert_avoid_fail_r)
    np.save('test_demo/expert_help_succ_r' + str(input_index), expert_help_succ_r)

# gen_test_conditions(1)
test_test_conditions(0)
