from PIL import Image
import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, LeakyReLU, Dense, Activation, Flatten
from keras.layers import Multiply, Lambda, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.callbacks import WandbLogger


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

ENV_NAME = 'BreakoutDeterministic-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
window_length = 4
nb_steps = 500000

# Application mode: train, test
# train: Training pong deep qlearning network
# test: Test pong deep qlearning network with pre-trained model
app_mode = "test"

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
    processor = AtariProcessor()
    
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
                                             interval=250000),
                     WandbLogger()]
        callbacks += [FileLogger(log_filename, interval=100)]
        dqn.fit(env, callbacks=callbacks, log_interval=10000,
                nb_steps=nb_steps, visualize=True, verbose=2)
        
        # After training is done, we save the final weights.
        dqn.save_weights(weights_filename, overwrite=True)
        
        env.reset()
        
        # Finally, evaluate our algorithm for 5 episodes.
        dqn.test(env, nb_episodes=5, visualize=True)
        
    elif app_mode == 'test':
        
        #env.reset()
        
        dqn.load_weights(weights_filename)
        dqn.test(env, nb_episodes=10, visualize=True)
    
    else:
        print(f"Only support 'train' or 'test' mode")
finally:
    if env is not None:
        env.close()