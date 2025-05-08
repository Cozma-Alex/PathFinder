import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from collections import deque
import random
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

class NavigationTrainer:
    def __init__(self, model, device='cpu', max_steps=50):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.epsilon = self.eps_start
        self.target_update = 10
        self.steps_done = 0
        self.max_steps = max_steps
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def update_policy(self, state, action, reward, next_state, done):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state).to(self.device)
        
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.BoolTensor([done]).to(self.device)
        
        curr_Q = self.model.policy_net(state)[0]
        curr_Q = curr_Q.gather(1, action.unsqueeze(1))
        
        next_Q = self.model.target_net(next_state)[0]
        max_next_Q = next_Q.max(1)[0].detach()
        expected_Q = (max_next_Q * self.gamma * (~done)) + reward
        
        loss = F.smooth_l1_loss(curr_Q, expected_Q.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        num_steps = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            batch = batch.to(self.device)
            current_state = batch.clone()
            
            for step in range(self.max_steps):
                action, _ = self.model.get_action(
                    current_state.state,
                    epsilon=self.epsilon
                )
                
                next_state = self.get_next_state(current_state, action)
                reward = self.compute_reward(next_state, batch)
                done = self.check_done(next_state, batch)
                
                loss = self.update_policy(current_state.state, action, reward, next_state.state, done)
                total_loss += loss
                num_steps += 1
                
                if done:
                    break
                    
                current_state = next_state
            
            pbar.set_postfix({'loss': total_loss / num_steps, 'epsilon': self.epsilon})
            
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        if epoch % self.target_update == 0:
            self.model.update_target_net()
            
        return total_loss / num_steps if num_steps > 0 else 0.0
        
    def get_next_state(self, current_state, action):
        next_state = current_state.clone()
        
        if next_state.state.dim() == 2:
            H = int(np.sqrt(next_state.state.size(1)))
            next_state.state = next_state.state.view(-1, H, H)
            
        for i in range(3):
            next_state_i = next_state.state[i]
            if next_state_i.dim() == 1:
                H = int(np.sqrt(next_state_i.size(0)))
                next_state.state[i] = next_state_i.view(H, H)
                
        agent_state = next_state.state[1]
        agent_pos = torch.nonzero(agent_state)
        if len(agent_pos) == 0:
            return next_state
            
        curr_y, curr_x = agent_pos[0]
        next_x, next_y = curr_x.item(), curr_y.item()
        
        if action == 0:  # Up
            next_y = curr_y - 1
        elif action == 1:  # Right
            next_x = curr_x + 1
        elif action == 2:  # Down
            next_y = curr_y + 1
        else:  # Left
            next_x = curr_x - 1
            
        grid = next_state.state[0]
        if self.is_valid_move(grid, (next_x, next_y)):
            next_state.state[1].zero_()
            next_state.state[1][next_y, next_x] = 1.0
            
        return Data(state=next_state.state)
        
    def is_valid_move(self, grid_map, pos):
        x, y = pos
        if not isinstance(grid_map, torch.Tensor):
            grid_map = torch.tensor(grid_map)
            
        if grid_map.dim() == 1:
            H = int(np.sqrt(grid_map.size(0)))
            grid_map = grid_map.view(H, H)
            
        if x < 0 or y < 0 or x >= grid_map.size(1) or y >= grid_map.size(0):
            return False
            
        return grid_map[y, x].item() == 1
        
    def compute_reward(self, state, target):
        reward = torch.zeros(1, device=self.device)
        
        agent_pos = torch.nonzero(state.state[1])
        goal_pos = torch.nonzero(target.state[2])
        
        if len(agent_pos) == 0 or len(goal_pos) == 0:
            reward[0] = -1
            return reward
            
        agent_pos = agent_pos[0]
        goal_pos = goal_pos[0]
        
        dist = torch.sqrt(
            (agent_pos[0] - goal_pos[0]).float() ** 2 +
            (agent_pos[1] - goal_pos[1]).float() ** 2
        )
        
        if dist == 0:
            reward[0] = 1
        else:
            reward[0] = -0.01
            
        return reward.item()
        
    def check_done(self, state, target):
        agent_pos = torch.nonzero(state.state[1])
        goal_pos = torch.nonzero(target.state[2])
        
        if len(agent_pos) == 0 or len(goal_pos) == 0:
            return True
            
        agent_pos = agent_pos[0]
        goal_pos = goal_pos[0]
        
        return torch.all(agent_pos.eq(goal_pos)).item()