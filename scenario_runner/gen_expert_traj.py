import cnirl_game
import numpy as np
import os

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

game = cnirl_game.TrafficGame(num_lanes=NUM_LANES, road_length=ROAD_LENGTH, gamma=GAMMA, beta=20.0,
                   critical_pos=CRITICAL_POS, vision=VISION, clear_range=CLEAR_RANGE,
                   min_cp_x=MIN_CP_X, max_cp_x=MAX_CP_X, min_cp_y=MIN_CP_Y, max_cp_y=MAX_CP_Y)

# generate some trajectories
D = [] # dataset of trajectories, each entry is ([s_1,...,s_T],[a_1,...,a_T],[c_1,...,c_T], ground truth reward indices)
moving_car_positions_list = []
conditions = []
for _ in range(NUM_TRAIN_DEMO):
    # 0 means the human agent needs to be helped, 1 means the human agent needs to be avoid
    single_conditions = np.random.choice(2, len(CRITICAL_POS))
    input_context_functions = cnirl_game.input_context_functions
    trajectory, actions, context_list, valuesMaps, rewardIndices, moving_car_positions = game.play_MNMDP_task(AgentInitialState=[1,0],
                                                                    InputContextFunc=input_context_functions[np.random.choice(len(input_context_functions))],
                                                                    R_list=game.R_list,
                                                                    V_list=game.V_list,
                                                                    RewardActivFunc = game.true_reward_activation,
                                                                    conditions = single_conditions)
    D.append((trajectory,actions,context_list,rewardIndices))
    moving_car_positions_list.append(moving_car_positions)
    conditions.append(single_conditions)

for i in range(NUM_TRAIN_DEMO):
    states, actions, contexts, rewards = D[i]
    dir_name = 'expert_demo/cnirl_use/' + str(i)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    np.save(dir_name + '/states', np.array(states).astype(int))
    np.save(dir_name + '/actions', np.array(actions).astype(int))
    np.save(dir_name + '/contexts', np.array(contexts).astype(int))
    np.save(dir_name + '/rewards', np.array(rewards))


for i in range(NUM_TRAIN_DEMO):
    states, actions, contexts, rewards = D[i]
    moving_car_positions = moving_car_positions_list[i]
    single_conditions = conditions[i]
    new_states = []
    for t in range(len(states)):
        new_s = np.append(states[t], moving_car_positions[t].astype(int))
        new_s = np.append(new_s, single_conditions)
        new_states.append(new_s)
    dir_name = 'expert_demo/else_use/' + str(i)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    np.save(dir_name + '/states', np.array(new_states).astype(int))
    np.save(dir_name + '/actions', np.array(actions).astype(int))
    np.save(dir_name + '/rewards', np.array(rewards))
