import torch
import torch.nn.functional as F
from collections import deque
from application.thor_controller.thor_controller import Action


class NavigationTrainer:
    def __init__(self, model, device="cpu", max_steps=50):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.policy_net.parameters(), lr=0.0003)
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
        self.action_to_idx = {
            Action.MOVE_FORWARD: 0,
            Action.FACE_EAST: 1,
            Action.MOVE_BACK: 2,
            Action.FACE_WEST: 3
        }

    def store_transition(self, state, action, reward, next_state, done):
        action_idx = self.action_to_idx[action]
        self.memory.append((state, action_idx, reward, next_state, done))

    def update_policy(self, state, action, reward, next_state, done):
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action_idx = self.action_to_idx[action] if isinstance(action, Action) else action
        action = torch.tensor([action_idx], device=self.device).unsqueeze(1)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], dtype=torch.bool, device=self.device)

        policy, value, _ = self.model.policy_net(state.unsqueeze(0))
        policy_value = policy.gather(1, action)

        with torch.no_grad():
            next_policy, next_value, _ = self.model.target_net(next_state.unsqueeze(0))
            max_next_value = next_policy.max(1)[0]
            expected_value = reward + self.gamma * max_next_value * (~done)

        loss = F.smooth_l1_loss(policy_value, expected_value.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, data_loader, epoch):
        self.model.policy_net.train()
        total_loss = 0
        batch_count = 0

        for batch in data_loader:
            batch_loss = 0
            batch_count += 1

            for state in batch:
                state = state.to(self.device)
                current_state = state.clone()

                for step in range(self.max_steps):
                    action, _ = self.model.get_action(
                        current_state, epsilon=self.epsilon
                    )
                    
                    next_state, reward, done = self.simulate_step(
                        current_state, action
                    )
                    loss = self.update_policy(
                        current_state, action, reward, next_state, done
                    )
                    batch_loss += loss

                    if done:
                        break

                    current_state = next_state

            total_loss += batch_loss / len(batch)

            if batch_count % 10 == 0:
                print(
                    f"  Batch {batch_count}: Loss = {batch_loss / len(batch):.4f}, Epsilon = {self.epsilon:.4f}"
                )

        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

        if epoch % self.target_update == 0:
            self.model.update_target_net()

        return total_loss / batch_count if batch_count > 0 else 0

    def simulate_step(self, state, action):
        next_state = state.clone()

        grid_map = state[0]
        agent_mask = state[1]
        goal_mask = state[2]

        agent_pos = torch.nonzero(agent_mask)[0]
        goal_pos = torch.nonzero(goal_mask)[0]

        next_pos = self.get_next_position(agent_pos, action)
        valid = self.is_valid_position(next_pos, grid_map)

        if valid:
            next_state[1].zero_()
            next_state[1, next_pos[0], next_pos[1]] = 1.0

            goal_reached = (next_pos == goal_pos).all()

            if goal_reached:
                reward = 1.0
                done = True
            else:
                reward = -0.01
                done = False
        else:
            reward = -0.1
            done = False

        return next_state, reward, done

    def get_next_position(self, position, action):
        row, col = position

        if action == Action.MOVE_FORWARD:
            return torch.tensor([max(0, row - 1), col])
        elif action == Action.FACE_EAST:
            return torch.tensor([row, col + 1])
        elif action == Action.MOVE_BACK:
            return torch.tensor([row + 1, col])
        else:  # Action.FACE_WEST
            return torch.tensor([row, max(0, col - 1)])

    def is_valid_position(self, position, grid_map):
        row, col = position

        if row < 0 or row >= grid_map.size(0) or col < 0 or col >= grid_map.size(1):
            return False

        return grid_map[row, col] > 0