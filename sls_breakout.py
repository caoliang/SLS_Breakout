from PIL import Image
import numpy as np
import gym
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, LeakyReLU, Dense, Activation, Flatten
from keras.layers import Multiply, Lambda, Permute
from keras.optimizers import Adam
from keras.initializers import he_normal

import keras.backend as K
from keras.callbacks import Callback, LambdaCallback, CSVLogger
from tensorflow.keras.utils import plot_model

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class AtariProcessor(Processor):
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(self.input_shape).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == self.input_shape
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

class StepLogger(Callback):
    
    def __init__(self, filename, separator=','):
        self.sep = separator
        self.filename = filename
        self.init_file()
    
    def on_step_end(self, step_num, logs):
#        print(f"step: {step_num}")
#        for key, value in logs['info'].items():
#            print(f"{key}: {value}")
        
        with open(self.filename, "a+") as step_file:
            step_file.write(str(logs['episode']))
            step_file.write(self.sep)
            step_file.write(str(step_num))       
            step_file.write(self.sep)
            step_file.write(str(logs['reward']))
            step_file.write(self.sep)
            step_file.write(str(logs['info']['ale.lives']))
            step_file.write('\n')
            step_file.flush()

    def init_file(self):
        with open(self.filename, "w+") as step_file:
            step_file.write('episode')
            step_file.write(self.sep)
            step_file.write('step')        
            step_file.write(self.sep)
            step_file.write('reward') 
            step_file.write(self.sep)
            step_file.write('lives') 
            step_file.write('\n')
            step_file.flush()

class EpisodeLogger(Callback):
    
    def __init__(self, filename, separator=','):
        self.sep = separator
        self.filename = filename
        self.episode = 0
        self.init_file()
    
    def on_episode_end(self, episode_num, logs):
#        print("=====")
#        print(f"episode: {episode_num}")
        
        with open(self.filename, "a+") as step_file:
            step_file.write(str(episode_num))
            step_file.write(self.sep)
            step_file.write(str(logs['nb_episode_steps']))       
            step_file.write(self.sep)
            step_file.write(str(logs['nb_steps']))
            step_file.write(self.sep)
            step_file.write(str(logs['episode_reward']))
            step_file.write('\n')
            step_file.flush()

    def init_file(self):
        with open(self.filename, "w+") as step_file:
            step_file.write('episode')
            step_file.write(self.sep)
            step_file.write('step')        
            step_file.write(self.sep)
            step_file.write('total') 
            step_file.write(self.sep)
            step_file.write('reward') 
            step_file.write('\n')
            step_file.flush()

class AwardLogger(Callback):
    
    def __init__(self, filename, separator=','):
        self.sep = separator
        self.filename = filename
        self.step = 0
        self.award = 0
        self.init_file()
    
    def on_step_end(self, batch, logs):
        self.step = self.step + 1
        self.award = self.award + logs['reward']
        
        with open(self.filename, "a+") as award_file:
            award_file.write(str(logs['episode']))
            award_file.write(self.sep)
            award_file.write(str(self.step))       
            award_file.write(self.sep)
            award_file.write(str(self.award)) 
            award_file.write('\n')
            award_file.flush()

    def on_episode_end(self, batch, logs):
        self.step = 0
        self.award = 0

    def init_file(self):
        with open(self.filename, "w+") as award_file:
            award_file.write('episode')
            award_file.write(self.sep)
            award_file.write('step')        
            award_file.write(self.sep)
            award_file.write('reward') 
            award_file.write('\n')
            award_file.flush()

    # Show step-award diagram
    def plot_award(self):
        headers = ['episode', 'step', 'award']
        df = pd.read_csv(self.filename, sep=self.sep, names=headers, 
                         skiprows=1)
        #print(list(df.columns.values))
        
        fig, ax = plt.subplots()
        df.groupby('episode').plot(x='step', y='award', ax=ax, legend=False)


def main(args):
    # Supress waring message for CPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Breakout environment name
    default_env = 'BreakoutDeterministic-v0'
    #default_env = 'BreakoutDeterministic-v4'
    
    window_length = 4
    nb_steps = 1750000
    # learning reate, based on later DeepMind paper called 
    # "Rainbow: Combining Improvements in Deep Reinforcement Learning" 
    # by Hessel et al. 2017 RMSProp was substituted for Adam 
    # with a learning rate of 0.0000625
    lr_rate = 0.0000625
    
    # Application mode: train, test
    # train: Training breakout deep qlearning network
    # test: Test breakout deep qlearning network with pre-trained model
#    default_app_mode = 'train'
    default_app_mode = 'test'
    
    if len(args) == 0:
        app_mode = default_app_mode
    else:
        app_mode = args[0]
        
    if len(args) > 1:
        if args[1] == 'v4':
            ENV_NAME = 'BreakoutDeterministic-v4'
        else:
            ENV_NAME = 'BreakoutDeterministic-v0'
    else:
        ENV_NAME = default_env
    
    INPUT_SHAPE = (84, 84)
    
    try: 
        env = gym.make(ENV_NAME)

        np.random.seed(123)
        env.seed(123)
        nb_actions = env.action_space.n
        
        input_frame = Input(shape=(window_length,) + INPUT_SHAPE)
        dqn_out = Permute((2, 3, 1))(input_frame)
        # Set he initializer for relu activation function
        dqn_out = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', 
                         kernel_initializer=he_normal())(dqn_out)
        dqn_out = Conv2D(64, (4, 4), strides=(2, 2), activation='relu',
                         kernel_initializer=he_normal())(dqn_out)
        dqn_out = Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                         kernel_initializer=he_normal())(dqn_out)
        dqn_out = Flatten()(dqn_out)
        dqn_out = Dense(512)(dqn_out)
        dqn_out = LeakyReLU()(dqn_out)
        dqn_out = Dense(nb_actions)(dqn_out)
        dqn_out = Activation('linear')(dqn_out)
        model = Model(inputs=[input_frame], outputs=[dqn_out])
        
        print(model.summary())
        
        memory = SequentialMemory(limit=nb_steps, window_length=window_length)
        
        #policy = BoltzmannQPolicy()
        
#        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 
#                                      attr='eps', 
#                                      value_max=1., 
#                                      value_min=.1, 
#                                      value_test=.05,
#                                      nb_steps=1000000)
#        
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 
                                      attr='eps', 
                                      value_max=.001, 
                                      value_min=.0001, 
                                      value_test=.00005,
                                      nb_steps=1000000)
                
        processor = AtariProcessor(input_shape=INPUT_SHAPE)
        
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, 
                       # Whether to enable dueling network
                       enable_dueling_network = False,
                       # Training starts afert warm-up steps 
                       nb_steps_warmup=50000,
                       processor=processor, 
                       #controls how often the target network is updated
                       target_model_update=10000, 
                       policy=policy, 
                       gamma=.99,
                       train_interval=4, 
                       delta_clip=1.)
        
        dqn.compile(Adam(lr=lr_rate),
                    metrics=['mae'])
        
        weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
        log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
        
        if app_mode == 'train':
            # Load existing weights if exists
            if os.path.exists(weights_filename):
                dqn.load_weights(weights_filename)
                        
            checkpoint_weights_filename = 'dqn_' + ENV_NAME + \
                                          '_weights_{step}.h5f'
            
            # Create step_logger and espisode_loger to monitor training
#            step_filename = 'dqn_' + ENV_NAME + '_step.csv'
#            episode_filename = 'dqn_' + ENV_NAME + '_episode.csv'
#            step_logger = StepLogger(step_filename)
#            episode_logger = EpisodeLogger(episode_filename)
#            
            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, 
                                                 interval=250000)]
            callbacks += [FileLogger(log_filename, interval=100)]
            
            # Add step_logger and espisode_logger as callbacks
#            callbacks += [LambdaCallback(
#                            on_step_end=step_logger.on_step_end)]
#            callbacks += [LambdaCallback(
#                            on_episode_end=episode_logger.on_episode_end)]
#            
            dqn.fit(env, 
                    callbacks=callbacks, 
                    log_interval=10000,
                    nb_steps=nb_steps, 
                    visualize=False,
                    # To avoid hungs when all lifes are lost
                    nb_max_episode_steps=2000,
                    verbose=2,
                    #whether check life lost and start new episode
                    enable_life_lost_episode=False)
            
            # Save weights after training completed
            dqn.save_weights(weights_filename, overwrite=True)
            
            env.reset()
            
            # Evaluation 5 episodes to show training results
            dqn.test(env, nb_episodes=5, visualize=False)
            
        elif app_mode == 'test':
            awards_list = []
            #env.reset()
            csv_filename = 'dqn_' + ENV_NAME + '_test.csv'
            csv_logger = AwardLogger(csv_filename)
                        
            def print_test_logs(batch, logs):
                #print(batch)
                #print(logs)
                awards_list.append(logs['episode_reward'])
                
            callbacks = [LambdaCallback(on_episode_end=print_test_logs)]
            callbacks += [LambdaCallback(on_step_end=csv_logger.on_step_end)]
            callbacks += [LambdaCallback(
                            on_episode_end=csv_logger.on_episode_end)]
            dqn.load_weights(weights_filename)
            dqn.test(env, callbacks=callbacks, nb_episodes=5, 
                     visualize=True, nb_max_episode_steps=2000)
            
            mean_award = np.mean(awards_list)
            print('Average awards: {0:0.2f}'.format(mean_award))
            
            # show step-award diagram
            csv_logger.plot_award()
            
        elif app_mode == 'plot-model':
            model_filename = 'dqn_' + ENV_NAME + '_model.pdf'
            dql_model = dqn.model
            plot_model(dql_model, 
                       to_file=model_filename, 
                       show_shapes=True, 
                       show_layer_names=False,
                       rankdir='TB')
        elif app_mode == 'plot-train':
            
            json_log_file = 'dqn_' + ENV_NAME + '_log.json'
            records     = pd.read_json(json_log_file)
            fig, ax = plt.subplots(2)
#            plt.plot(records['episode'], records['loss'])
            fig.suptitle("Loss Value vs Espisode Reward")
            
            ax[0].plot(records['episode'], records['loss'], label='losss')
            ax[1].plot(records['episode'], records['episode_reward'], 
                          label='reward')
            
            ax[0].set_ylabel('Losss')
            ax[1].set_ylabel('Reward')
                        
            #plt.yticks([0, 0.005, 0.010, 0.050, 0.100])
            #plt.title('Loss Value / Mean Q',fontsize=12)
            #plt.legend(loc="upper left")
            ax[1].set_xlabel("Episode")
            #ax = plt.gca()
            #ax.set_xticklabels([])
            
            plt.show()
            
        else:
            print(f"Only support 'train', 'test', 'plot-model', " +
                  f"'plot-train' mode")
    finally:
        if env is not None:
            env.close()

        
if __name__ == "__main__":
    main(sys.argv[1:])
