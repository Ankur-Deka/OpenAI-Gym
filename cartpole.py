#Code to play cartpole using q learning
		
import gym
import time
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class cartpole_solver():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.history = []
        # parameters for bin sizes
        self.ang_params = {'high': 0.2, 'bins': 8}    # bins of the form [{one bin < -high}, {4 bins in -high:high} , {one bin > high}], total of 6 bins
        self.vel_params = {'high': 4, 'bins': 8}
        #create q value bins
        self.q = np.zeros((self.ang_params['bins'], self.vel_params['bins'],2))
        self.ang_bins = np.linspace(-self.ang_params['high'], self.ang_params['high'], self.ang_params['bins']-1)
        self.ang_bins = np.concatenate((np.array([-100]), self.ang_bins))
        self.ang_bins = np.concatenate((self.ang_bins, np.array([100])))

        self.vel_bins = np.linspace(-self.vel_params['high'], self.vel_params['high'], self.vel_params['bins']-1)
        self.vel_bins = np.concatenate((np.array([-100]), self.vel_bins))
        self.vel_bins = np.concatenate((self.vel_bins, np.array([100])))

        #define q learning parameters
        self.alpha = 0.3
        self.gama = 0.99

    # we will simplify the state space by using only angle and relative velocity
    def create_ss(self, observation):
        ang = observation[2]
        vel = observation[1] - observation[3]
        ss = [ang, vel]
        dss = [np.digitize(ss[0], self.ang_bins)-1, np.digitize(ss[1], self.vel_bins)-1]
        return(dss)

    def disp_ss(self, ss):
        print('Angle: '+str(ss[0])+' Velocity: '+str(ss[1]))

    # define a naive actor
    def naive_a(self, ss):
        ang = ss[0]
        action = 1 if ang > 0 else 0
        return(action)
    # define q learning based actor
    # discretized state space, iteration
    # iteration = -1 means testing, act optimally 
    def q_a(self, dss, iter):
        if iter > 0:
            explore = np.random.choice([0,1], 1, [1/iter, 1-1/iter])[0]
        else:
            explore = 0
        if explore:
            action = np.random.choice([0,1], 1, [0.5, 0.5])[0]
        else:
            action = 1 if self.q[dss[0],dss[1],0] < self.q[dss[0],dss[1],1] else 0
        return(action)

    # function to update q values
    def update_q(self, ss, action, nxt_ss, reward):
        self.q[ss[0], ss[1], action] = (1-self.alpha)*self.q[ss[0], ss[1], action] + (self.alpha)*(reward+self.gama*max(self.q[nxt_ss[0], nxt_ss[1], :]))

    # function to plot the sequence of actions over an s
    def plot_history(self):
        #normalize the values
        history = self.history
        m=[]
        for i in range(3):
            m.append(max([abs(his[i]) for his in history]))
        print(m)
        for i, his in enumerate(history):
            his = [his[j]/m[j] for j in range(3)]
            history[i] = his
        plt.plot(history)
        plt.show()

    # function to plot the q values
    def plot_q(self, save = False, name = 'q.png'): 
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # X = np.linspace(0,self.ang_params['bins'], self.ang_params['bins'], dtype = int)
        # Y = np.linspace(0,self.vel_params['bins'], self.vel_params['bins'], dtype = int)
        # Y, X = np.meshgrid(Y, X)
        # surf = ax.plot_surface(X, Y, np.argmax(self.q, axis=2), cmap=cm.plasma,
        #    linewidth=0, antialiased=False)
        
        if save:
            plt.imsave(name, np.argmax(self.q, axis=2))
        else:
            plt.imshow(np.argmax(self.q, axis=2), cmap='hot')
            plt.show()
    # function to solve the problem
    def solve(self, test = False):
        if test:
            episodes = 1
            solver.q = np.load('q.npy')
        else:
            episodes = 100
        iter = 0
        for i_episode in range(episodes):
            observation = self.env.reset()
            for t in range(200):
                self.env.render()
                ss = self.create_ss(observation)    # observation of the format cart position, cart velocity, pole angle, pole velocity at tip
                
                # act
                if test:
                    iter = -1
                else:
                    iter = iter+1
                action = self.q_a(ss, iter)
                observation, reward, done, info = self.env.step(action)

                nxt_ss = self.create_ss(observation)                

                if not test:
                    #update q
                    self.update_q(ss, action, nxt_ss, reward)

                #save history
                his = ss
                his.append(action)
                self.history.append(his)

                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
                # time.sleep(0.001)
            if i_episode%5 == 4:
                self.plot_q(save = True, name = str(i_episode))
        if not test:
            np.save('q.npy',self.q)



if __name__ == '__main__':
    solver = cartpole_solver()
    solver.solve(test = True)
    # solver.plot_history()q)
    solver.plot_q()