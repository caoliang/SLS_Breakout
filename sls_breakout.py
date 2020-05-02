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
import keras.backend as K
from keras.callbacks import Callback, LambdaCallback, CSVLogger

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
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
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

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
            award_file.write(',')
            award_file.write(str(self.step))       
            award_file.write(',')
            award_file.write(str(self.award)) 
            award_file.write('\n')
            award_file.flush()

    def on_episode_end(self, batch, logs):
        self.step = 0
        self.award = 0

    def init_file(self):
        with open(self.filename, "w+") as award_file:
            award_file.write('episode')
            award_file.write(',')
            award_file.write('step')        
            award_file.write(',')
            award_file.write('reward') 
            award_file.write('\n')
            award_file.flush()

    def plot_award(self):
        headers = ['episode', 'step', 'award']
        df = pd.read_csv(self.filename, sep=',', names=headers, 
                         skiprows=1)
        #print(list(df.columns.values))
        
        fig, ax = plt.subplots()
        df.groupby('episode').plot(x='step', y='award', ax=ax, legend=False)


def main(args):
    # Supress waring message for CPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    
    ENV_NAME = 'BreakoutDeterministic-v0'
    
    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    window_length = 4
    nb_steps = 500000
    
    # Application mode: train, test
    # train: Training breakout deep qlearning network
    # test: Test breakout deep qlearning network with pre-trained model
    if len(args) == 0 or args[0] == 'test':
        app_mode = 'test'
    else:
        app_mode = 'train'
    
    INPUT_SHAPE = (84, 84)
    
    try: 
        np.random.seed(123)
        env.seed(123)
        nb_actions = env.action_space.n
        
        # Next, we build a very simple model.
        input_frame = Input(shape=(window_length,) + INPUT_SHAPE)
        dqn_out = Permute((2, 3, 1))(input_frame)
        dqn_out = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(dqn_out)
        dqn_out = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(dqn_out)
        dqn_out = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(dqn_out)
        dqn_out = Flatten()(dqn_out)
        dqn_out = Dense(512)(dqn_out)
        dqn_out = LeakyReLU()(dqn_out)
        dqn_out = Dense(nb_actions)(dqn_out)
        dqn_out = Activation('linear')(dqn_out)
        model = Model(inputs=[input_frame], outputs=[dqn_out])
        
        print(model.summary())
        
        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=nb_steps, window_length=window_length)
        policy = BoltzmannQPolicy()
        processor = AtariProcessor(input_shape=INPUT_SHAPE)
        
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=60,
                       processor=processor, target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=1e-5), metrics=['mae'])
        
        weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
        log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
        
        if app_mode == 'train':
            # Okay, now it's time to learn something! We visualize the training here for show, but this
            # slows down training quite a lot. You can always safely abort the training prematurely using
            # Ctrl + C.
            checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, 
                                                 interval=250000)]
            callbacks += [FileLogger(log_filename, interval=100)]
            dqn.fit(env, callbacks=callbacks, log_interval=10000,
                    nb_steps=nb_steps, visualize=True, verbose=2)
            
            # After training is done, we save the final weights.
            dqn.save_weights(weights_filename, overwrite=True)
            
            env.reset()
            
            # Finally, evaluate our algorithm for 5 episodes.
            dqn.test(env, nb_episodes=5, visualize=True)
            
        elif app_mode == 'test':
            awards_list = []
            #env.reset()
            csv_filename = 'dqn_' + ENV_NAME + '_test.csv'
            csv_logger = AwardLogger(csv_filename, dqn)
                        
            def print_test_logs(batch, logs):
                #print(batch)
                #print(logs)
                awards_list.append(logs['episode_reward'])
                
            callbacks = [LambdaCallback(on_episode_end=print_test_logs)]
            callbacks += [LambdaCallback(on_step_end=csv_logger.on_step_end)]
            callbacks += [LambdaCallback(on_episode_end=csv_logger.on_episode_end)]
            dqn.load_weights(weights_filename)
            dqn.test(env, callbacks=callbacks, nb_episodes=10, visualize=True)
            
            mean_award = np.mean(awards_list)
            print(f'Average awards: {mean_award:.2}')
            
            csv_logger.plot_award()
        
        else:
            print(f"Only support 'train' or 'test' mode")
    finally:
        if env is not None:
            env.close()

        
if __name__ == "__main__":
    main(sys.argv[1:])
