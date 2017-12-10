import gym
import numpy as np
import time
import matplotlib.pyplot as plt

def e_greedy(state,Q_table,epsilon):
    if np.random.rand()>epsilon:
        return np.argmax(Q_table[state])
    else:
        return np.random.randint(0,action_dim)

env=gym.make('FrozenLake8x8-v0')

state_dim=env.observation_space.n
action_dim=env.action_space.n

GAMMA=0.96
learning_rate=0.81
INITIAL_EPSILON=1
END_EPSILON=0.1
DECAY_RATE=0.98


EPISODE=100000
STEP=100
TEST=100
np.random.seed(1)
env.seed(1)

Q_table=np.random.rand(state_dim,action_dim)
#Q_table=np.zeros([state_dim,action_dim])
Q_table[19]=0
Q_table[29]=0
Q_table[35]=0
Q_table[41]=0
Q_table[42]=0
Q_table[46]=0
Q_table[49]=0
Q_table[52]=0
Q_table[54]=0
Q_table[59]=0
Q_table[63]=0
#print(Q_table)
eposide_time=0
epsilon=INITIAL_EPSILON

reward_sum=[]
episode_record=[]
for episode in range(EPISODE):
    #epsilon-=(INITIAL_EPSILON-END_EPSILON)/EPISODE
    epsilon*=DECAY_RATE
    state=env.reset()
    for step in range(STEP):
        #action=np.argmax(Q_table[state,:]  + np.random.randn(1, action_dim)*(1./(episode+1)))
        action=e_greedy(state,Q_table,epsilon)
        next_state,reward,done,_=env.step(action)
        Q_table[state][action]+=learning_rate*(reward+GAMMA*np.max(Q_table[next_state,:])-Q_table[state,action])
        state=next_state
        if done:
            break
            
    if episode%1000==0:
        total_reward=0
        for i in range(TEST):
            state=env.reset()
            pass_steps=100
            final_state=0
            for j in range(STEP):
                action=np.argmax(Q_table[state])
                state,reward,done,_=env.step(action)
                total_reward+=reward
                if done:
                    final_state=state
                    pass_steps=j
                    break
            #if final_state==15:
            #    print('episode:{},total steps:{}'.format(episode,pass_steps))
                
        avg_reward=total_reward/TEST
        print('episode: ',episode,"Average reward:",avg_reward)
        reward_sum.append(avg_reward)
        episode_record.append(episode)
            
print (Q_table)
plt.plot(episode_record,reward_sum)
plt.ylim(0,1)
plt.show()