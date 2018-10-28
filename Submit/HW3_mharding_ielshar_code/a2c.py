import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import gym
import timeit

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce


np.random.seed(1234)
tf.set_random_seed(1234)


mean_reward_list=[]
std_dev_list=[]
def critic_network(env):
    model = keras.Sequential()
    model.add(keras.layers.Dense(20, input_dim=env.observation_space.shape[0], activation='relu'))
    model.add(keras.layers.Dense(20, activation='relu'))
    model.add(keras.layers.Dense(20, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))

    return model

class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, env, model, lr, critic_model, critic_lr, num_episodes, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.model = model
        self.critic_model = critic_model
        self.critic_lr = critic_lr
        self.critic_model.compile(optimizer=keras.optimizers.Adam(lr=self.critic_lr), loss='MSE')
        self.n = n
        self.lr = lr
        self.num_episodes = num_episodes
        self.numStates = env.observation_space.shape[0]
        self.numActions = env.action_space.n
        self.num_test_episodes = 100
        self.file_index = 1
        # TODO: Define any training operations and optimizers here, initialize
        # your variables, or alternately compile your model here.
        states = self.model.input
        softmax = self.model.output
        G_return = K.placeholder(shape=(None,))
        onehot_action = K.placeholder(shape=(None, self.numActions))
        prob_actions = K.sum(softmax * onehot_action, axis=1)
        log_prob_actions = K.log(prob_actions)
        loss = - log_prob_actions * G_return
        loss = K.mean(loss)
        Adam = keras.optimizers.Adam(lr=self.lr)
        updates = Adam.get_updates(self.model.trainable_weights,[],loss)
        self.fit = K.function(inputs=[states,onehot_action,G_return],
                              outputs=[],
                              updates=updates)
  

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        start_time = timeit.default_timer()
        loop_time = start_time
        mean_reward = []
        episodes_axes = []
        error = []
        for episode in range(1,self.num_episodes + 1):
            states, actions, rewards = self.generate_episode(env)
            onehot_actions = keras.utils.to_categorical(actions, num_classes=self.numActions)
            #Downscale rewards by 1e-2
            rewards = np.multiply(1e-2, np.array(rewards))
            T = len(rewards)
            R = np.zeros((T,1))
            # compute R 
            for t in range(T-1, -1, -1):
                if (t + self.n >= T):
                    Vend = 0.0
                else:
                    state = np.reshape(states[t+self.n], [1, self.numStates])
                    Vend = self.critic_model.predict(state)
                R[t,0] = (gamma**self.n)*Vend
                for k in range(self.n):
                        R[t,0] += (gamma**k)*rewards[t+k] if (t + k < T) else 0.0

            states_for_critic = np.reshape(states, [T, self.numStates])
            v = self.critic_model.predict(states_for_critic)
            G = np.subtract(R, v)
            G = G.ravel()
            states = np.squeeze(states)
            self.fit([states, onehot_actions, G])
            self.critic_model.train_on_batch(states_for_critic, R)  
            
            if  episode % 500 == 0:
                print("episode=", episode)
                total_returns = np.zeros((self.num_test_episodes))
                for ep in range(self.num_test_episodes):
                    states, actions, rewards = self.generate_episode(env)
                    total_returns[ep] = np.sum(rewards)
                mean_tot_return = np.mean(total_returns)
                std_dev = np.std(total_returns)
                mean_reward.append(mean_tot_return)
                episodes_axes.append(episode)
                print("mean_rewards="+str(mean_tot_return))
                mean_reward_list.append(mean_tot_return)
                print("std_dev_rewards="+str(std_dev))
                std_dev_list.append(std_dev)
                error.append(std_dev)
#                plt.figure()
                plt.errorbar(episodes_axes,mean_reward,error,capsize=3)
                plt.xlabel('Episodes')
                plt.ylabel('Mean & Std_dev Reward per Episode')
                plt.legend(loc='best')
                elapsed_time = timeit.default_timer() - loop_time
                print("Time="+str(elapsed_time))
                print("Elapsed_time="+str(timeit.default_timer() - start_time))
                loop_time = timeit.default_timer()

            if  episode % 1000 == 0:
                plt.savefig('A2C_LC_N='+str(self.n)+'_' + str(self.file_index) + '.png')
                plt.clf()
                self.model.save('A2C_Actor_model_N='+str(self.n)+'_'+ str(self.file_index) + '.h5')
                self.critic_model.save('A2C_Critic_model_N='+str(self.n)+'_'+ str(self.file_index) + '.h5')
                self.file_index += 1 

        

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.0008, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-3, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=100, help="The value of N in N-step A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    env.seed(1234)

    # Load the actor model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using A2C and plot the learning curves.
    critic_model = critic_network(env)
    agent = A2C(env, model, lr, critic_model, critic_lr, num_episodes, n)
    print("N=",agent.n)
    agent.train(env)


if __name__ == '__main__':
    main(sys.argv)
