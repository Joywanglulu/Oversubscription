import numpy as np
import torch


#This version has slight modifications for the human in the loop module

class ExpertTraj:
    def __init__(self, data, train_ratio = 0.8):

        dstate = data[:,:-1]
        daction = data[:,-1]
        # initail_options = np.load('k-means_option.npy', allow_pickle=True)


        dstate = torch.Tensor(dstate)
        daction = torch.Tensor(daction)
        # initail_options = torch.Tensor(initail_options)
        sha = dstate.shape[0]
        self.full_state = dstate
        self.full_action = daction

        self.Tslice = int(train_ratio*sha)
        self.train_state = dstate[:int(train_ratio*sha)]
        self.train_action = daction[:int(train_ratio*sha)]
        self.n_transitions = self.train_state.shape[0]
        # self.train_initial_option = initail_options[:int(train_ratio * sha)]

        self.val_state = dstate[int(train_ratio*sha):]
        self.val_action = daction[int(train_ratio*sha):]
        # self.val_initial_option = initail_options[int(train_ratio * sha):]




    def eval(self):
        return np.array(self.val_state), np.array(self.val_action), self.Tslice

    def train(self):
        return np.array(self.train_state), np.array(self.train_action)

    def full(self):
        return np.array(self.full_state), np.array(self.full_action)

    def sample(self, batch_size):
        # print('self.n_transitions',self.n_transitions)
        indexes = np.random.randint(0, self.n_transitions, size=batch_size)
        state, action = [], []
        # print('self.train_state', self.train_state.shape)
        for i in indexes:
            s = self.train_state[i]
            a = self.train_action[i]
            # o = self.train_initial_option[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))

        return np.array(state), np.array(action)

# #
# expert = ExpertTraj( 'all_data.npy',train_ratio = 0.8)
# state, action = expert.sample(10)
# print('state', state.shape)
# print('action', action.shape)
#
