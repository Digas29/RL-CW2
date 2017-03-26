import cv2
import numpy as np

from enduro.agent import Agent
from enduro.action import Action
from turn import Turn


class FunctionApproximationAgent(Agent):
    def __init__(self):
        super(FunctionApproximationAgent, self).__init__()
        # Add member variables to your class here
        self.num_features = 3
        self.weights = np.random.randn(self.num_features)
        self.total_reward = 0

        # Helper dictionaries that allow us to move from actions to
        # Q table indices and vice versa
        self.idx2act = {i: a for i, a in enumerate(self.getActionsSet())}
        self.act2idx = {a: i for i, a in enumerate(self.getActionsSet())}

        # Learning rate
        self.alpha = 1e-4
        # Discounting factor
        self.gamma = 0.9
        # Exploration rate
        self.epsilon = 0.01


        # Log the obtained reward during learning
        self.last_episode = 1
        self.episode_log = np.zeros(6510) - 1.
        self.log = []

    def initialise(self, road, cars, speed, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """

        self.current_features = self.buildFeaturesValues(road, cars, speed, grid)
        self.next_features = self.current_features

        # Reset the total reward for the episode
        self.total_reward = 0



    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        # You can get the set of possible actions and print them with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal

        self.current_features = self.next_features

        self.q_values = np.dot(self.weights.T,self.current_features)

        # If exploring
        if np.random.uniform(0., 1.) < self.epsilon:
            probs = np.exp(self.q_values) / np.sum(np.exp(self.q_values))
            idx = np.random.choice(4, p=probs)
            self.action = self.idx2act[idx]
        else:
            # Select the greedy action
            self.action = self.idx2act[np.argmax(self.q_values)]


        self.reward =  self.move(self.action)
        self.total_reward += self.reward

    def sense(self, road, cars, speed, grid):
        """ Constructs the next state from sensory signals.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """
        self.next_features = self.buildFeaturesValues(road, cars, speed, grid)

    def learn(self):
        """ Performs the learning procedure. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        next_q_values = self.weights.T.dot(self.next_features)
        new_weights = self.weights + self.alpha * (self.reward + self.gamma * np.max(next_q_values) - self.q_values[self.act2idx[self.action]]) * self.current_features[:,self.act2idx[self.action]]
        self.weights = new_weights

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """

        # Initialise the log for the next episode
        if episode != self.last_episode:
            iters = np.nonzero(self.episode_log >= 0)
            rewards = self.episode_log[iters]
            print "{0}: {1}".format(episode-1, rewards[-1])
            self.log.append((np.asarray(iters).flatten(), rewards, np.copy(self.weights)))
            self.last_episode = episode
            self.episode_log = np.zeros(6510) - 1.

        # Log the reward at the current iteration
        self.episode_log[iteration] = self.total_reward

    def buildFeaturesValues(self, road, cars, speed, grid):
        feature_values = np.zeros([self.num_features,len(self.getActionsSet())])

        for i in range(self.num_features):
            if i == 0:
                feature_values[i] = self.stayInCenter(road,grid)
            elif i == 1:
                feature_values[i] = self.avoidCars(cars)
            else:
                feature_values[i] = self.accelaration(speed)
        return feature_values

    def stayInCenter(self, road, grid):
        feature_values = np.zeros(len(self.getActionsSet()))
        turn = Turn.NOOP

        road_x = road[0][0][0]
        base_x = road[-1][0][0]

        diff = road_x - base_x
        if diff > 85:
            turn = Turn.RIGHT
        elif diff < -85:
            turn = Turn.LEFT

        pos = np.where(grid[0]==2)[0][0]

        dist_to_center = min(np.abs(pos-4),np.abs(pos-5))

        for i,a in enumerate(self.getActionsSet()):
            if turn == Turn.NOOP:
                if a == Action.ACCELERATE or a == Action.BRAKE:
                    if dist_to_center == 0:
                        feature_values[i] = 1
                    else:
                        feature_values[i] = 0
                elif a == Action.LEFT:
                    new_dist = min(np.abs(pos-1-4),np.abs(pos-1-5))
                    if new_dist < dist_to_center or new_dist == 0:
                        feature_values[i] = 1
                    else:
                        feature_values[i] = 0
                else:
                    new_dist = min(np.abs(pos+1-4),np.abs(pos+1-5))
                    if new_dist < dist_to_center or new_dist == 0:
                        feature_values[i] = 1
                    else:
                        feature_values[i] = 0
            elif turn == Turn.LEFT:
                if a == Action.ACCELERATE or a == Action.BRAKE:
                    new_dist = min(np.abs(pos+1-4),np.abs(pos+1-5))
                    if new_dist < dist_to_center or new_dist == 0:
                        feature_values[i] = 1
                    else:
                        feature_values[i] = 0
                elif a == Action.LEFT:
                    if pos > 3:
                        feature_values[i] = 1
                    else:
                        feature_values[i] = 0
                else:
                    new_dist = min(np.abs(pos+2-4),np.abs(pos+2-5))
                    if new_dist < dist_to_center or new_dist == 0:
                        feature_values[i] = 1
                    else:
                        feature_values[i] = 0
            else:
                if a == Action.ACCELERATE or a == Action.BRAKE:
                    new_dist = min(np.abs(pos-1-4),np.abs(pos-1-5))
                    if new_dist < dist_to_center or new_dist == 0:
                        feature_values[i] = 1
                    else:
                        feature_values[i] = 0
                elif a == Action.LEFT:
                    new_dist = min(np.abs(pos-2-4),np.abs(pos-2-5))
                    if new_dist < dist_to_center or new_dist == 0:
                        feature_values[i] = 1
                    else:
                        feature_values[i] = 0
                else:
                    if pos < 6:
                        feature_values[i] = 1
                    else:
                        feature_values[i] = 0

        return feature_values

    def avoidCars(self, cars):
        feature_values = np.ones(len(self.getActionsSet()))
        for i,a in enumerate(self.getActionsSet()):
            if len(cars['others']) != 0:
                if cars['self'][1] - cars['others'][0][1] + cars['others'][0][3] < 30:
                    collision = self.carCollision(cars['self'][0], cars['self'][2], cars['others'][0][0], cars['others'][0][2])
                    if collision == 'right':
                        feature_values[self.act2idx[Action.ACCELERATE]] = 0
                        feature_values[self.act2idx[Action.BRAKE]] = 0
                        feature_values[self.act2idx[Action.RIGHT]] = 0
                    elif collision == 'left':
                        feature_values[self.act2idx[Action.ACCELERATE]] = 0
                        feature_values[self.act2idx[Action.BRAKE]] = 0
                        feature_values[self.act2idx[Action.LEFT]] = 0
        return feature_values

    def carCollision (self, selfX, selfW, carX, carW):
        if carX >= selfX  and carX <= selfX + selfW:
            return 'right'
        elif carX + carW >= selfX  and carX + carW <= selfX + selfW:
            return 'left'
        else:
            return 'ok'

    def accelaration(self, speed):
        feature_values = np.zeros(len(self.getActionsSet()))
        feature_values[self.act2idx[Action.ACCELERATE]] = 1
        return feature_values

if __name__ == "__main__":
    a = FunctionApproximationAgent()
    a.run(True, episodes=500, draw=True)
    pickle.dump(a.log, open("log.p", "wb"))
