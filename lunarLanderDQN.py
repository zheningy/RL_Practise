import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os 
import random
import sys
import tensorflow as tf
from collections import deque

H1=256
H2=256
H3 = 512
BATCH_SIZE=32
GAMMA=0.9
INITIAL_EPSILON=0.9
END_EPSILON=0.1
REPLAY_SIZE=10000

log_dir=os.path.abspath('./logs/test_1')
save_dir=('logs/model')

class DQN():
    def __init__(self,env):
        self.replay_buffer=deque()
        self.time_step=0
        self.epsilon=INITIAL_EPSILON
        self.state_dim=8
        self.action_dim=4

        self.create_network()
        self.create_training_method()
        
        #self.saver=tf.train.Saver({"W1":self.W1,"W2":self.W2,
        #                           "b1":self.b1,"b2":self.b2})
        
        self.session=tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
            
        #merged=tf.summary.merge_all()
        
        self.writer=tf.summary.FileWriter(log_dir,self.session.graph)
        self.merged=tf.summary.merge_all()
        #self.merged=tf.summary.merge([self.W1_summary,self.W2_summary,self.loss_summary])
        
        #self.summary=self.session.run(self.merged)
     def create_network(self):
        self.state_input=tf.placeholder(tf.float32,[None,self.state_dim],name='state')

        self.W1=tf.get_variable("W1",[self.state_dim,H1])
        self.b1 = tf.Variable(tf.constant(0.01, shape=[H1,]))
        layer1=tf.nn.relu(tf.matmul(self.state_input,self.W1)+self.b1)

        self.W2=tf.get_variable("W2",[H1,H2])  
        self.b2 = tf.Variable(tf.constant(0.01, shape=[H2,]))        
        layer2=tf.nn.relu(tf.matmul(layer1,self.W2)+self.b2)

        self.W3=tf.get_variable("W3",[H2,H3])  
        self.b3 = tf.Variable(tf.constant(0.01, shape=[H3,]))        
        layer3=tf.nn.tanh(tf.matmul(layer2,self.W3)+self.b3)

        self.W4=tf.get_variable("W4",[H3,self.action_dim])  
        self.b4 = tf.Variable(tf.constant(0.01, shape=[self.action_dim,]))

        self.Q_value=tf.matmul(layer3,self.W4)+self.b4

    def create_training_method(self):
        self.action_input=tf.placeholder(tf.float32,                            [None,self.action_dim])
        self.y_input=tf.placeholder(tf.float32,[None])
        Q_action=tf.reduce_sum(tf.multiply(self.Q_value,\
                      self.action_input),reduction_indices=1)

        self.loss=tf.reduce_mean(tf.square(self.y_input-Q_action))

        self.optimizer=tf.train.AdamOptimizer(1e-5).minimize(self.loss)       
        
    def perceive(self,state,action,reward,next_state,done):
        one_hot_action=np.zeros(self.action_dim)
        one_hot_action[action]=1
        self.replay_buffer.append((state,one_hot_action,reward,
                                   next_state,done))
        if len(self.replay_buffer)>REPLAY_SIZE:
            self.replay_buffer.popleft()
            
        if len(self.replay_buffer)>BATCH_SIZE:
            self.train_network()
    
    def train_network(self):
        self.time_step+=1
        minibatch=random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch=[data[0] for data in minibatch]
        action_batch=[data[1] for data in minibatch]
        reward_batch=[data[2] for data in minibatch]
        next_state_batch=[data[3] for data in minibatch]

        y_batch=[]
        Q_value_batch=self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        for i in range(0,BATCH_SIZE):
            done=minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i]+GAMMA*np.max(Q_value_batch[i]))

            feed_dict={self.y_input:y_batch,
                       self.action_input:action_batch,
                       self.state_input:state_batch}
            self.session.run(self.optimizer,feed_dict)
        
        
    def egreedy_action(self,state):
        Q_value=self.Q_value.eval(feed_dict={self.state_input:[state]})[0]

        if self.epsilon>0.1:
            self.epsilon=self.epsilon-0.001
        else:
            self.epsilon*=DECAY_RATE

        if random.random()<=self.epsilon:
            return random.randint(0,self.action_dim-1)
        else:
            return np.argmax(Q_value)

    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict={self.state_input:[state]})[0])

    def store_data(self,state,action,reward,next_state,done):
        one_hot_action=np.zeros(self.action_dim)
        one_hot_action[action]=1
        self.replay_buffer.append((state,one_hot_action,reward,
                                   next_state,done))
        if len(self.replay_buffer)>REPLAY_SIZE:
            self.replay_buffer.popleft()


EPISODE=1000
STEP = 500 
TEST = 10

env=gym.envs.make("LunarLander-v2")
agent=DQN(env)

for episode in range(EPISODE):
    state=env.reset()
    for step in range(STEP):
        action=agent.egreedy_action(state)
        next_state,reward,done,_=env.step(action)
        agent.perceive(state,action,reward,next_state,done)
        state=next_state
        if done:
            break
            
    if episode%100==0:
        total_reward=0
        for i in range(TEST):
            state=env.reset()
            for j in range(STEP):
                action=agent.action(state)
                state,reward,done,_=env.step(action)
                total_reward+=reward
                if done:
                    break
        ave_reward=total_reward/TEST
        print('episode: ',episode,"Avarge Reward:",ave_reward)
        if ave_reward>=200:
            break
            
env.close()
