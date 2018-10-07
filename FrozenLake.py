#import gym library and create Frozen Lake environment
import gym
import numpy as np
env=gym.make('FrozenLake-v0')

#Create a Q look up table with rows representing the states and columns representing the actions
Q=np.zeros([env.observation_space.n,env.action_space.n])

alpha=0.8
lamda=0.9
num_episodes=3000
rList=[]

for i in range(num_episodes):
	s=env.reset()
	rAll=0
	d=False
	for j in range(99):
		#greedily choose new action
		a = np.minimum(env.action_space.n,np.maximum(0,np.argmax(Q[s,:] + np.random.uniform(-env.action_space.n/2.0,env.action_space.n/2.0,4)*(1./((i+1))))))
		#get new state, reward and check for completion of episode
		s_next,r,d,_=env.step(a)
		Q[s,a]=Q[s,a]+alpha*(r+lamda*np.max(Q[s_next,:])-Q[s,a])
		rAll+=r
		s=s_next
		if(d):
			break
	rList.append(rAll)

#Display the score over time
print("Average Score over all episodes: "+ str(sum(rList)/num_episodes))
print("Score of first 10 episodes: "+str(rList[0:10]))
print("Score of last 10 episodes: "+str(rList[num_episodes-10:num_episodes]))
print("Final Q table values")
print(Q)


#once we have a Q table, let's calculate our average reward over say n_test episodes
n_test=100
rList2=[]
for i in range(n_test):
	s=env.reset()
	rAll=0
	d=False
	for j in range(99):
		a=np.argmax(Q[s,:])
		s,r,d,_=env.step(a)
		rAll+=r
		if(d):
			break
	rList2.append(rAll)
#print("Score of test episodes: "+str(rList2))
print("Average test Score: "+str(sum(rList2)/n_test))