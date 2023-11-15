import torch
from collections import deque
import torch.nn as nn
import RL.MLP_RL as MLP_RL
import copy
import numpy as np
import random
import torch.nn.functional as F

class DQL:
    def __init__(self, state_size, action_size, batch_size, learning_rate = 0.01 , gamma = 0.9 , epsilon = 0.9, update_rate = 10, flag = False):
        # define the state size
        self.state_size = state_size # Number of weights (PCA) * (N+1)
        
        #define the action size
        self.action_size = action_size # Number of clients
        
        # define the experience replay
        self.replay_buffer = deque(maxlen=1000)
        
        # define the batch size
        self.batch_size = batch_size
        
        #define the discount factor
        self.gamma = gamma
        
        #define the epsilon value
        if (flag == False) :
            self.epsilon = epsilon
        else:
            self.epsilon = 0.2
 
        self.epsilon_end = 0.2
        self.epsilon_delta = (self.epsilon  -  self.epsilon_end) / 500
        # Update target network rate
        self.update_rate = update_rate
        
        # loss function, we use MSE loss function in DQL
        self.loss_func = nn.MSELoss()
        
        # fix the seed
        #self.seed = seed
        
        # learning rate
        self.learning_rate = learning_rate
        
        #define the main network
        self.main_network = MLP_RL.MLP(self.state_size, 512, self.action_size)
        
        # optimizer
        #self.optimizer = torch.optim.SGD(self.main_network.parameters(), lr= learning_rate , momentum = 0.99)
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr = self.learning_rate)
        
        #define the target 
        self.target_network =  MLP_RL.MLP(self.state_size, 512, self.action_size)
        
        #copy the weights of the main network to the target network
        self.target_network.load_state_dict(copy.deepcopy(self.main_network.state_dict()))
        
    
    def set_epsilon(self):
        self.epsilon =  self.epsilon - self.epsilon_delta
        if (self.epsilon <  self.epsilon_end):
            self.epsilon =  0.2
    # When we DQN by randomly sampling a minibatch of transitions from the
    #replay buffer. So, we define a function called store_transition which stores the transition information
    #into the replay buffer

    def store_transistion(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        

    def multiaction_selection(self, state, C, comm_round, mode):
        if (mode == "Mode1"):
            return self.epsilon_greedy(state, C, comm_round)
        else:
            return self.multinomial_selection(state, C, comm_round)
        
    
    # the selection stratey used in our solution
    def epsilon_greedy(self, state, C, comm_round):
        m = int(max(C*self.action_size, 1))
        
        self.main_network.eval()
        Q_values = self.main_network.forward(state.unsqueeze(0))
        
        # action_top_c will contain the indexes of the best top C actions for state

        actions_top_c = (torch.topk(Q_values[0], m).indices).tolist()
        # a subtraction to get the non-greedy shares
        actions_possible = list(set(np.arange(self.action_size)))
        
        clients_selected = []
        
        #print("the epsilon value is : ", self.epsilon)

        for step in range(m):
            if random.uniform(0,1) < self.epsilon:
                client = random.choice(actions_possible)
                clients_selected.append(client)
                actions_possible.remove(client)
                if client in actions_top_c:
                    actions_top_c.remove(client)
            else :
                clients_selected.append(actions_top_c[0])
                actions_possible.remove(actions_top_c[0])
                actions_top_c.remove(actions_top_c[0])
             
        self.set_epsilon()
           
        clients_selected.sort()
        
        return clients_selected
    
    
    # the selection strategy used in FedDRL
    def multinomial_selection(self, state, C, comm_round):
        m = int(max(C*self.action_size, 1))
        
        self.main_network.eval()
        Q_values = self.main_network.forward(state.unsqueeze(0))
        #actions_top_c = (torch.topk(Q_values[0], m).indices).tolist()
        
        
        actions_probs = F.softmax(Q_values, dim=1)
        clients_selected = (actions_probs.multinomial(num_samples= m).tolist())[0]
        return clients_selected
    
    def train(self, comm_round, mode):
        if (mode == "Mode1"):
            return self.trainV1_0(comm_round)
        else:
            return self.train_V2(comm_round)
        
    
    def trainV1_0(self, comm_round):
        #sample a mini batch of transition from the replay buffer
        if (len(self.replay_buffer) < self.batch_size):
            minibatch = random.sample(self.replay_buffer, len(self.replay_buffer))
        else:
            minibatch = random.sample(self.replay_buffer, self.batch_size)
            
        # get the index of the batch
        batch_index = range(0, len(minibatch))
        # get the states of the minibatch
        states_minibatch = torch.cat([sample[0].unsqueeze(0) for sample in minibatch], dim=0)

        # get the actions taked in each state of the minibatch
        actions_minibatch = torch.tensor(np.array([np.where(sample[1] == 1)[0] for sample in minibatch]))
        
        # get the reward recieved for each action in the minibatch
        rewards_minibatch = [sample[2] for sample in minibatch]
        # get the next states
        nextstates_minibatch = torch.cat([sample[3].unsqueeze(0) for sample in minibatch], dim=0)
        # get the boolean that tell us if the next state is terminal or not
        dones_minibatch = [sample[4] for sample in minibatch]
        
        self.main_network.eval()
        self.target_network.eval()

        # get Q values of each selected action in the samples of the batch
        Q_values = self.main_network.forward(states_minibatch)
        # Get the Q values of next state
        Q_values_next = self.target_network.forward(nextstates_minibatch)
        
        # Terminal state to 0
        Q_values_next = Q_values_next.clone()
        Q_values_next[dones_minibatch] = 0
        
        transformed_action = list(zip(*actions_minibatch))
        transformed_reward = list(zip(*rewards_minibatch))
        
        new_network = copy.deepcopy(self.main_network)
        
        list_loss = []
        for i in range(0, 5):
            # Get the updated values (True value)
            Q_target = Q_values.clone()
   
            Q_target[batch_index, transformed_action[i]] = torch.tensor(transformed_reward[i], dtype=torch.float) + self.gamma *Q_values_next[batch_index, torch.argmax(Q_values, dim = 1)]

            # Train the main network
            self.main_network.train()
            
            # Initialize the gradients to 0
            self.main_network.zero_grad()
            
            # Loss calculation
            loss = self.loss_func(Q_target.double(), Q_values.double())
            #print("loss of the DQL network is : ", loss.detach().item())
            list_loss.append(loss.detach().item())

            
            # Calculation of parameter gradients
            loss.backward()
            
            # Update of the parameters
            self.optimizer.step()
            
            Q_values =  new_network.forward(states_minibatch)
            Q_values_next = self.target_network.forward(nextstates_minibatch)
            # Terminal state to 0
            Q_values_next = Q_values_next.clone()
            Q_values_next[dones_minibatch] = 0
            
            
        #self.set_epsilon()
        #print(list_loss)
        return list_loss
    #train the network
    
    
    def train_V1(self, comm_round):
        
        #sample a mini batch of transition from the replay buffer
        #sample a mini batch of transition from the replay buffer
        if (len(self.replay_buffer) < self.batch_size):
            minibatch = random.sample(self.replay_buffer, len(self.replay_buffer))
        else:
            minibatch = random.sample(self.replay_buffer, self.batch_size)
        
        # get the index of the batch
        batch_index = range(0, len(minibatch))
        
        
        # get the states of the minibatch
        states_minibatch = torch.cat([sample[0].unsqueeze(0) for sample in minibatch], dim=0)
        # get the actions taked in each state of the minibatch
        actions_minibatch = torch.tensor(np.array([np.where(sample[1] == 1)[0] for sample in minibatch]))
        # get the reward recieved for each action in the minibatch
        rewards_minibatch = [sample[2] for sample in minibatch]
        # get the next states
        nextstates_minibatch = torch.cat([sample[3].unsqueeze(0) for sample in minibatch], dim=0)
        # get the boolean that tell us if the next state is terminal or not
        dones_minibatch = [sample[4] for sample in minibatch]

        self.main_network.eval()
        self.target_network.eval()
        
        # get Q values of each selected action in the samples of the batch
        Q_values_forupdate = self.main_network.forward(states_minibatch)
        Q_values = self.main_network.forward(states_minibatch)[torch.tensor(batch_index)[:, None], actions_minibatch]
        # Get the Q values of next state
        Q_values_next = self.target_network.forward(nextstates_minibatch)
        
        # Terminal state to 0
        Q_values_next = Q_values_next.clone()
        Q_values_next[dones_minibatch] = 0
        

        # Get the updated values (True value)
        #Q_target = torch.tensor(rewards_minibatch) + self.gamma * torch.max(Q_values_next, dim = 1).values.unsqueeze(1)
                                                  
        Q_target = torch.tensor(rewards_minibatch) + self.gamma *Q_values_next[batch_index, torch.argmax(Q_values_forupdate, dim = 1)].unsqueeze(1)
             
        
        # we will generate the same Q_target for all chosen actions :
        #Q_target =  Q_target[:, None].expand(-1, len(actions_minibatch[0]))
        # Train the main network
        self.main_network.train()
        
        # Initialize the gradients to 0
        self.main_network.zero_grad()
        
        
        # Loss calculation
        loss = self.loss_func(Q_target.double(), Q_values.double())
        #print("loss of the DQL network is : ", loss.detach().item())
        
        # Calculation of parameter gradients
        loss.backward()
        
        # Update of the parameters
        self.optimizer.step()

    def train_V2(self, comm_round):
        
        #sample a mini batch of transition from the replay buffer
        if (len(self.replay_buffer) < self.batch_size):
            minibatch = random.sample(self.replay_buffer, len(self.replay_buffer))
        else:
            minibatch = random.sample(self.replay_buffer, self.batch_size)
            
        # get the index of the batch
        batch_index = range(0, len(minibatch))
        # get the states of the minibatch
        states_minibatch = torch.cat([sample[0].unsqueeze(0) for sample in minibatch], dim=0)

        # get the actions taked in each state of the minibatch
        actions_minibatch = [sample[1] for sample in minibatch]
        # get the reward recieved for each action in the minibatch
        rewards_minibatch = [sample[2] for sample in minibatch]
        # get the next states
        nextstates_minibatch = torch.cat([sample[3].unsqueeze(0) for sample in minibatch], dim=0)
        # get the boolean that tell us if the next state is terminal or not
        dones_minibatch = [sample[4] for sample in minibatch]
        
        self.main_network.eval()
        self.target_network.eval()

        # get Q values of each selected action in the samples of the batch
        Q_values_forupdate = self.main_network.forward(states_minibatch)
        Q_values = self.main_network.forward(states_minibatch)[batch_index, actions_minibatch]
        # Get the Q values of next state
        Q_values_next = self.target_network.forward(nextstates_minibatch)
        
        # Terminal state to 0
        Q_values_next = Q_values_next.clone()
        Q_values_next[dones_minibatch] = 0
        
        # Get the updated values (True value)
        #Q_target = torch.tensor(rewards_minibatch) + self.gamma * torch.max(Q_values_next, dim = 1).values
        
        #print("Q_target 1 are : ", Q_target)
        #print("the max is : ", torch.argmax(Q_values))
        #print("The Q values : ", Q_values_forupdate)
        #print("The max is : ", torch.argmax(Q_values_forupdate, dim = 1))
        
        Q_target = torch.tensor(rewards_minibatch) + self.gamma *Q_values_next[batch_index, torch.argmax(Q_values_forupdate, dim = 1)]
        #print("Q_target 2 are : ", Q_target)
        
        # Train the main network
        self.main_network.train()
        
        # Initialize the gradients to 0
        self.main_network.zero_grad()
        
        # Loss calculation
        loss = self.loss_func(Q_target.double(), Q_values.double())
        print("loss of the DQL network is : ", loss.detach().item())
        
        # Calculation of parameter gradients
        loss.backward()
        
        # Update of the parameters
        self.optimizer.step()
        
        return loss.detach().item()
    
    
        #train the network
    def train_V3(self, comm_round):
        
        #sample a mini batch of transition from the replay buffer
        #sample a mini batch of transition from the replay buffer
        if (len(self.replay_buffer) < self.batch_size):
            minibatch = random.sample(self.replay_buffer, len(self.replay_buffer))
        else:
            minibatch = random.sample(self.replay_buffer, self.batch_size)
        
        # get the index of the batch
        batch_index = range(0, len(minibatch))
        
        
        # get the states of the minibatch
        states_minibatch = torch.cat([sample[0].unsqueeze(0) for sample in minibatch], dim=0)
        # get the actions taked in each state of the minibatch
        actions_minibatch = torch.tensor(np.array([np.where(sample[1] == 1)[0] for sample in minibatch]))
        # get the reward recieved for each action in the minibatch
        rewards_minibatch = [sample[2] for sample in minibatch]
        # get the next states
        nextstates_minibatch = torch.cat([sample[3].unsqueeze(0) for sample in minibatch], dim=0)
        # get the boolean that tell us if the next state is terminal or not
        dones_minibatch = [sample[4] for sample in minibatch]
        
        self.main_network.eval()
        self.target_network.eval()
        
        # get Q values of each selected action in the samples of the batch
        Q_values = self.main_network.forward(states_minibatch)[torch.tensor(batch_index)[:, None], actions_minibatch]
        # Get the Q values of next state
        Q_values_next = self.target_network.forward(nextstates_minibatch)
        
        # Terminal state to 0
        Q_values_next = Q_values_next.clone()
        Q_values_next[dones_minibatch] = 0
        
        # Get the updated values (True value)
        Q_values_forupdate = self.main_network.forward(states_minibatch)                                 
        Q_target = torch.tensor(rewards_minibatch) + self.gamma *Q_values_next[batch_index, torch.argmax(Q_values_forupdate, dim = 1)].unsqueeze(1)
        
        # we will generate the same Q_target for all chosen actions :
        Q_target =  Q_target[:, None].expand(-1, len(actions_minibatch[0]))

        # Train the main network
        self.main_network.train()
        
        # Initialize the gradients to 0
        self.main_network.zero_grad()
        
        
        # Loss calculation
        loss = self.loss_func(Q_target, Q_values.double())
        print("loss of the DQL network is : ", loss.detach().item())
        
        # Calculation of parameter gradients
        loss.backward()
        
        # Update of the parameters
        self.optimizer.step()
        
            
    # update the target network weights by copying from the main network
    def update_target_network(self):
        self.target_network.load_state_dict(copy.deepcopy(self.main_network.state_dict()))
        
        
    