
import torch 
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils

import gym_super_mario_bros
env = gym_super_mario_bros.make('SuperMarioBros-v0')


class Actor_net(torch.nn.Module):
    def __init__(self, state_dim,hidden_dim,action_dim):
        super(Actor_net,self).__init__()
        
        self.conv1 = torch.nn.Conv2d(state_dim,32,3,stride=2,padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.POOL1 = torch.nn.AdaptiveAvgPool2d(4*4)
        self.Linear2 = torch.nn.Linear(8192,action_dim)
        self.flatten =torch.nn.Flatten()
    def forward(self,x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.POOL1(x))

        x = F.relu(self.flatten(x))
        print(x.shape)
        return F.softmax(self.Linear2(x),dim=1)
    

class Value_net(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(Value_net,self).__init__()
        self.conv1 = torch.nn.Conv2d(state_dim,32,1,stride=2,padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 1, stride=2, padding=1)
        self.Linear2 = torch.nn.Linear(32**2,1)
        self.flatten =torch.nn.Flatten()

    def forward(self,x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.Linear2(x))
        return self.flatten(x)



class PPO:
    def __init__(self,state_dim,hidden_dim,action_dim,lr_a,lr_v,lmbda,epochs,eps,gamma,device):
        self.a_net = Actor_net(state_dim,hidden_dim,action_dim).to(device)
        self.v_net = Value_net(state_dim,action_dim).to(device)

        self.a_opt = torch.optim.Adam(self.a_net.parameters(),lr = lr_a)
        self.v_opt = torch.optim.Adam(self.v_net.parameters(),lr = lr_v)
        #self.lr_a = lr_a
        #`self.lr_v = lr_v

        self.lmbda = lmbda
        self.epochs = epochs
        
        self.eps = eps
        self.gamma = gamma
        self.device = device

    def take_action(self,state):

        states = torch.tensor([state],dtype=torch.float).to(self.device)
        actions =  self.a_net(states)
        print(actions.shape)
        action_dict = torch.distributions.Categorical(actions)

        action = action_dict.sample()
        #print()
        print(action.item())
        return action.item()

    def compute_advantage(self,gamma,lmbda,td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::1]:
            advantage = gamma*lmbda*advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()# 翻转
        return torch.tensor(advantage_list,dtype = torch.float)

    def update(self,transition_dict):
        state = torch.tensor(transition_dict['states'],dtype = torch.float).to(self.device)
        action = torch.tensor(transition_dict['actions'],dtype=torch.float).view(-1,1).to(self.device)
        reward = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        next_state = torch.tensor(transition_dict['next_states'],dtype = torch.float).to(self.device)
        done = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)

        td_target = reward+self.gamma *self.v_net(next_state)*(1-done)
        td_delta = td_target - self.v_net(state)

        advantage = self.compute_advantage(self.gamma,self.lmbda,td_delta.cpu()).to(self.device)


        old_log_prob = torch.log(self.a_net(state).gather(1,action)).detach()
        for _ in range(self.epochs):
            log_prob = torch.log(self.a_net(state).gather(1,action))
            ratio = torch.exp(log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio,1-self.eps,1+self.eps) * advantage

            a_loss = torch.mean(-torch.min(surr1,surr2))
            v_loss = torch.mean(F.mse_loss(self.a_net(state),td_target.detach()))


            self.a_opt.zero_grad()
            self.v_opt.zero_grad()
            a_loss.backward()
            v_loss.backward()

            self.a_net.step()
            self.v_opt.step()

    

            
actor_lr = 1e-3
c_le = 1e-2

num_epis = 500
hidden_dim = 256
gamma = 0.98
eps = 0.3
lmbda = 0.95

epoch = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")




state_dim = env.observation_space.shape[0]


action_dim = env.action_space.n


agent = PPO(state_dim,hidden_dim,action_dim,actor_lr,c_le,lmbda,epoch,eps,gamma,device)


return_list  = rl_utils.train_on_policy_agent(env=env,agent=agent,num_episodes=num_epis)




