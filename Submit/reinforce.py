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



np.random.seed(1234)
tf.set_random_seed(1234)



class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, env, model, lr, num_episodes):
        self.model = model
        self.lr = lr
        self.num_episodes = num_episodes
        self.numStates = env.observation_space.shape[0]
        self.numActions = env.action_space.n
        self.num_test_episodes = 100
        self.file_index = 1
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.
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

        
    def compute_G(self,R, gamma=1):
        G = np.zeros_like(R, dtype=np.float32)
        temp = 0
        for t in reversed(range(len(R))):
            temp = temp * gamma + R[t]
            G[t] = temp
        #normalize   
#        G -= G.mean() / G.std()
        return G

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        start_time = timeit.default_timer()
        loop_time = start_time
        mean_reward = []
        episodes_axes = []
        error = []
        for episode in range(1,self.num_episodes + 1):
            states, actions, rewards = self.generate_episode(env)
#            print("total return:",np.sum(rewards))#,"LR:", K.get_value(self.model.optimizer.lr) )
            states = np.squeeze(states)
#            print(states.shape)
            onehot_actions = keras.utils.to_categorical(actions, num_classes=self.numActions)
            #Downscale rewards by 1e-2
            rewards = np.multiply(1e-2, rewards)
            G = self.compute_G(rewards)
#            T = len(rewards)
#            print(T)
#            G = np.zeros(T)
#            for t in range(T-1, -1, -1):
#                for k in range(t, T):
#                    G[t] += gamma**(k-t)*rewards[k]
            self.fit([states, onehot_actions, G])
            
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
                print("std_dev_rewards="+str(std_dev))
                error.append(std_dev)
#                plt.figure()
                plt.errorbar(episodes_axes,mean_reward,error,capsize=5)
                plt.xlabel('Episodes')
                plt.ylabel('Mean & Std_dev Reward per Episode')
                plt.legend(loc='best')
                elapsed_time = timeit.default_timer() - loop_time
                print("Time="+str(elapsed_time))
                print("Elapsed_time="+str(timeit.default_timer() - start_time))
                loop_time = timeit.default_timer()

            if  episode % 1000 == 0:
                plt.savefig('Reinforce_LC_' + str(self.file_index) + '.png')
                plt.clf()
                self.model.save('Reinforce_model_' + str(self.file_index) + '.h5')
                self.file_index += 1 

        

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []
        
        state =  env.reset()
        done = False
        while not done:
            state = np.expand_dims(state,0)
            prob = self.model.predict(state, batch_size=1).flatten()
            action = np.random.choice(self.numActions, 1, p=prob)[0]
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        states = np.array(states)
#        print(states)
#        print(states.shape)
        actions = np.array(actions)
        rewards = np.array(rewards)
    
        return states, actions, rewards


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-3, help="The learning rate.")

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
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    env.seed(1234)
    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.
    agent = Reinforce(env, model, lr, num_episodes)
    agent.train(env)

if __name__ == '__main__':
    main(sys.argv)
