import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class policy_net(torch.nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim):
        super(policy_net,self).__init__()
        self.f1 = torch.nn.Linear(state_dim,hidden_dim)
        self.f2 = torch.nn.Linear(hidden_dim,action_dim)
    def forward(self,x):
        x = F.relu(self.f1(x))
        return F.softmax(self.f2(x),dim =1)

class value_net(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(value_net,self).__init__()
        self.f1 = torch.nn.Linear(state_dim,hidden_dim)
        self.f2 = torch.nn.Linear(hidden_dim,1)
    def forward(self,x):
        x = F.relu(self.f1(x))
        return self.f2(x)

class PPO:
    def __init__(self,state_dim,action_dim,hidden_dim,lr_p,lr_v,lmbda,epochs,eps,gamma,device):


        self.action_net = policy_net(state_dim,action_dim,hidden_dim)
        self.critic_net = value_net(state_dim,hidden_dim)
        self.actor_opt = torch.optim.Adam(self.action_net.parameters(),lr=lr_a)

        self.cri_opt = torch.optim.Adam(self.critic_net.parameters(),lr=lr_c)
        self.lr_a = lr_p
        self.lr_c = lr_v

        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps #截断的数值
        
    def take_action(self,state):
        state = torch.tensor([state],torch.float).to(self.device)
        prob = self.action_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self,tmp):
        states = torch.tensor(tmp['states'],dtype = torch.float).to(self.device)
        rewards = torch.tensor(tmp['rewards'],dtype = torch.float).view(-1,1).to(self.device)
        actions = torch.tensor(tmp['actions'],dtype = torch.float).views(-1,1).to(self.device)
                
        dones= torch.tensor(tmp['dones'],dtype = torch.float).views(-1,1).to(self.device)
        next_states = torch.tensor(tmp['next_states'],dtype = torch.float).to(self.device)
        td_target = rewards + self.gamma *self.critic_net(next_states) *(1-dones)
        td_delta = td_target - critic_net(states)
        adv = self.compute_advantage(gamma=self.gamma,lmbda = self.lmbda,td_delta)


        old_log_probs = torch.log(self.action_net(states).gather(1,actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.action_net(states).gather(1,actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 =ratio * adv
            surr2 = torch.clamp(ratio,1-self.eps,1+self.eps)*adv

            actor_loss = torch.mean(-torch.min(surr1,surr2))
            cri_loss = torch.mean(F.mse_loss(self.critic_net(state),td_target.detach()))

            self.actor_opt.zero_grad()
            self.critic_net.zero_grad()

            actor_loss.backward()
            cri_loss,backward()
            self.actor_opt.step()
            self.cri_opt.step()

    


            




    def compute_advantage(gamma,lmbda,td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::1]:
            advantage = gamma*lmbda*advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()# 翻转
        return torch.tensor(advantage_list,dtype = torch.float)
