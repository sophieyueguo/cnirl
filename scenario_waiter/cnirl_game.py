import numpy as np
from copy import deepcopy

potential_bPs = [[1,10],[12,1],[1,5],[8,10],[10,6],[4,12],[13,12],[6,2]]
Ts = np.arange(5, 50, 5).astype(int)
CRY_DURATION = 30

### Some simple functions

# for a point [x,y], return [[x+1,y],[x-1,y],[x,y+1],[x,y-1]], aka right,left,up,down neighbors
def neighbors(a):
    x, y = a[0], a[1]
    return [[x+1,y],[x-1,y],[x,y+1],[x,y-1]]

# return true if the point is in the map (0,0)-(size-1,size-1)
def InBound(a,size):
    return (0 <= a[0] <= size-1 and 0 <= a[1] <= size-1)

# return true if the distance (measured by the maximum coordinate difference) of a and b is <= size
def IsNear(a,b,size):
    if (abs(a[0]-b[0]) <= size) and (abs(a[1]-b[1]) <= size):
        return True
    return False

# computes the L2-Euclidean distance of two 2D input points
def EDist(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)



class NavigationGame:
    ### go to the goalPos while maybe need to avoid or help babies on the road
    ## deterministic transitions all the time currently

    # representation of a state:
    #      s = [x, y], the coordinate on the gridworld

    # features of a state (used for defining reward functions):
    #      phi(s)=[1/0(indicator for goalPos), indicators for each baby's position]; length should be #babies + 1

    # contexts at each timestep:
    #      [1/0 for each baby indicating whether crying or not]; length: #babies

    # features of state-context pair:
    #      phi_2(s,c) has length #babies + 1;
    #      the coordinate is 1 for the nearest crying baby; 0 for others; (since our agent always go towards the nearest crying baby)
    #      if no crying baby then the last coordinate is 1 and others are 0

    def __init__(self, num_baby, size=15, gamma=0.95, initialPos=[4,0], goalPos=[14,14], beta=5, goalR=1, helpR=3, avoidR=-3):
        self.beta = beta           # exploration temprature
        self.size = size           # grid world size*size, the map is (0,0)->(size-1,size-1)
        self.gamma = gamma         # discount factor for value iterations, etc
        self.initialPos = initialPos  # initial Position of the agent on the map
        self.goalPos = goalPos        # goal Position (milk position)
        self.babyPos_list = potential_bPs[0: num_baby]   # list consisting of baby positions
        # goalR: reward when step on goal pos
        # helpR: reward when help baby (should be bigger than goalR or else the agent will go directly to goal)
        # avoidR: reward when crash baby, should be negative

        # the length should be consistent with self.phi, self.phi_2 defined below
        self.length_phi = 1 + len(self.babyPos_list)
        self.length_phi_2 = 1 + len(self.babyPos_list)

        # phi_map, as its name, stores all the phi([x,y])'s in a map for convenience of retrieval
        # phi_map[x,y,:] would be phi([x,y])
        self.phi_map = np.zeros([size,size,self.length_phi])
        for i in range(self.size):
            for j in range(self.size):
                state = [i,j]
                self.phi_map[i,j] = self.phi(state)


        # now initialize the expert's reward coeffs and maps
        # NOTICE that it should have the same order as the the expert's reward activation function; see self.true_reward_activation
        # recall phi(s)=[1/0(indicator for goalPos), indicators for each baby's position]
        self.R_coeff_list = []  # list of reward coefficients; DOT PRODUCT with phi(s) is reward(s) since linearly model
        self.R_list = []        # list of Reward maps
        self.V_list = []        # list of value maps

        # the expert has a high reward for the nearest baby pos,
        for i in range(len(self.babyPos_list)):
            # now the reward map where the ith baby is the nearest crying baby: positive at goal and ith baby
            # note that since phi(s) has a first coordinate which is indicator of goalPos, the positive reward for ith baby
            #    should be at i+1 th position (so that they are aligned)

            # temp represents the reward coefficients to be filled;
            temp = avoidR * np.ones(1 + len(self.babyPos_list))
            temp[0] = goalR
            temp[i+1] = helpR
            self.R_coeff_list.append(temp)
            # dot product with phi gives the reward for each state
            self.R_list.append(np.dot(self.phi_map, temp))
            # compute the value map
            self.V_list.append(self.ValueIterationWithR(iteration_num=200, R=self.R_list[-1]))

        # for the last one (corresponding to the case where no babies is crying)
        # just add the goal reward and nagative for all babies
        temp = avoidR * np.ones(1 + len(self.babyPos_list))
        temp[0] = goalR
        self.R_coeff_list.append(temp)
        self.R_list.append(np.dot(self.phi_map, temp))
        self.V_list.append(self.ValueIterationWithR(iteration_num=200, R=self.R_list[-1]))

    # feature transform of a state, for reward functions' usage
    def phi(self, s):
        representation = []

        # 1st coordinate the indicator of goal pos
        if s == self.goalPos:
            representation.append(1)
        else:
            representation.append(0)

        # subsequent coordinates are indicators of each baby pos
        for i in range(len(self.babyPos_list)):
            babyPos = self.babyPos_list[i]
            if s == babyPos:
                representation.append(1)
            else:
                representation.append(0)

        # check the length
        assert len(representation) == 1 + len(self.babyPos_list)
        return np.array(representation,dtype=np.float64)

    # feature transform of a state-context pair, for reward activation
    # note the state s is not used here; but generally it can be used in our framework so it's added as a parameter
    def phi_2(self,s,c_):
        return_list = np.zeros(1 + len(self.babyPos_list))
        c = deepcopy(c_)
        # find all crying baby's indices
        crying_indices = []
        for i in range(len(c)):
            if c[i] == 1:
                crying_indices.append(i)
        # if no crying babies, set the last coordinate to be 1 and return
        if len(crying_indices) == 0:
            return_list[-1] = 1
            return return_list.reshape(-1,1)

        # find the index with nearest euclidean distance, set the cooresponding coordinate to 1
        index = min(crying_indices, key=lambda x: EDist(self.babyPos_list[x], s))
        return_list[index] = 1
        # return 2D numpy array (for later usage)
        return return_list.reshape(-1,1)

    # return all valid actions of current state: a subset of {0,1,2,3} right,left,up,down order
    def ValidActions(self,state):
        validactions = []
        next_Pos_list = neighbors(state)   # right,left,up,down order
        for i in range(len(next_Pos_list)):
            if InBound(next_Pos_list[i],self.size):
                validactions.append(i)
        validactions.append(4)   # represents stop
        return validactions

    # move at state with action, return the next state
    def move(self,state,action):
        x, y = state[0],state[1]
        if action < 4:
            return [[x+1,y],[x-1,y],[x,y+1],[x,y-1]][action]
        # else it's stop
        return [x, y]

    # value iteration using reward map R, return the value map V
    def ValueIterationWithR(self, iteration_num, R):
        size = self.size
        V = np.zeros([size,size])
        for _ in range(iteration_num):
            V_new = np.zeros([size,size])
            # do the iteration
            for i in range(size):
                for j in range(size):
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
    def true_reward_activation(self, phi_2):
        # return exactly phi_2 but reshaped to be a 1D list
        # the coordinate 1 indicates the cooreponding reward function
        return phi_2.reshape(-1)

    ### CORE function: MNMDP trajectory
    ### for each state-context, compute the reward scores via RewardActivFunc, then act according to corresponding reward map
    ### RewardActivFunc outputs scores with input phi_2(s,c), according to the Reward's order in R_list
    def play_MNMDP_task(self, R_list, V_list,
                        RewardActivFunc, Crytime_list, Stoptime_list, display=False, step_thres=200):

        # Crytime_list: time that each baby starts crying; lengths: #babies
        # Stoptime_list: time that each baby stops crying; lengths: #babies
        # display: debug usage; display relevant info when set to true; see all places with 'if display: blablabla'
        # step_thres: end the game when exceed - too long

        size = self.size
        AgentState = deepcopy(self.initialPos)  # recording the agent's current pos
        AgentTrajectory = []    # list of history agent states
        valuesMaps = []         # recording the value map of the agent states, for plotting the animating
        scores_list = []        # recording the agent's scores for each reward at each time step
        context_list = []
        action_list = []
        # always starts with all babies silent
        currentContext = [0 for _ in range(len(self.babyPos_list))]

        # three things we use to check how the agent "mimicks" the expert's policy
        totalR = 0.   # total rewards got from now
        num_help = 0.  # total number of helps during the task
        # num_crash = 0. # total number of crashes Modification: alternative way to count
        num_should_help = 0.
        crash_indices = np.zeros(len(self.babyPos_list))

        # Let's start the game
        t = -1
        isEnd = False   # whether game should end, updated when reaching goal pos
        while not isEnd:
            if display:
                print(AgentState)
            t += 1
            if t >= step_thres:
                break

            currentContext = deepcopy(currentContext)

            # update crying status of babies, a.k.a, updating contexts
            for i in range(len(self.babyPos_list)):
                if t == Crytime_list[i]:
                    currentContext[i] = 1
                    num_should_help += 1
            for i in range(len(self.babyPos_list)):
                babyPos = self.babyPos_list[i]
                if AgentState == babyPos:
                    if currentContext[i] == 0:
                        # the agent crashed into this baby
                        # num_crash += 1
                        crash_indices[i] = 1
                    if currentContext[i] == 1:
                        # the agent helped this baby
                        currentContext[i] = 0
                        num_help += 1

            # update stopping status
            for i in range(len(self.babyPos_list)):
                if t == Stoptime_list[i]:
                    currentContext[i] = 0

            # check if end game
            if AgentState==self.goalPos:
                isEnd = True

            # store relevant info
            context_list.append(currentContext)
            AgentTrajectory.append(AgentState)

            # compute the action to take according to the rewardActivFunc and reward functions
            validactions = self.ValidActions(AgentState)
            # compute which reward to follow for current state-context, using the given RewardActivFunc
            scores = RewardActivFunc(self.phi_2(AgentState,currentContext))
            scores_list.append(scores)
            # compute the weighted value maps
            weighted_map = sum([scores[i]*(R_list[i]+self.gamma*V_list[i]) for i in range(len(scores))])
            valuesMaps.append(weighted_map)

            def EvaluateQ(action):
                ns = self.move(AgentState,action)
                return weighted_map[ns[0],ns[1]]
            # standard boltzmann exploration policy with temperature beta
            Q_values = np.array(list(map(lambda action: EvaluateQ(action), validactions)))
            Exp_Q = np.exp(self.beta*Q_values)
            Exp_Q = Exp_Q/np.sum(Exp_Q)
            rd_indices = np.random.choice(len(validactions), 1, p=Exp_Q)
            action = (np.array(validactions)[rd_indices])[0]
            action_list.append(action)
            AgentState = self.move(AgentState,action)

            # compute the expert's reward function and add the expert's reward (which is the objective reward) for this action
            expert_scores = self.true_reward_activation(self.phi_2(AgentState,currentContext))
            expert_R_map = sum([expert_scores[i]*R_list[i] for i in range(len(expert_scores))])
            totalR += expert_R_map[AgentState[0],AgentState[1]]

        help_rate = num_help/num_should_help
        crash_rate = sum(crash_indices)/float(len(self.babyPos_list))
        return np.array(AgentTrajectory),action_list,context_list,valuesMaps,scores_list,totalR,help_rate,crash_rate
