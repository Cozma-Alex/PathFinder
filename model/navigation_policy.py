import torch
import torch.nn as nn
import torch.nn.functional as F


class NavigationPolicy(nn.Module):
    def __init__(self, input_channels=3, hidden_size=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, 4)
        self.value_head = nn.Linear(128, 1)

        self.lstm = nn.LSTM(64 * 8 * 8, hidden_size, batch_first=True)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(batch_size, -1)
        x = x.unsqueeze(1)

        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)

        x = x.squeeze(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy = F.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)

        return policy, value, hidden


class NavigationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = NavigationPolicy()
        self.target_net = NavigationPolicy()
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(self, state, hidden=None, epsilon=0.0):
        if torch.rand(1) < epsilon:
            return torch.randint(0, 4, (1,)), hidden

        with torch.no_grad():
            policy, _, new_hidden = self.policy_net(state.unsqueeze(0), hidden)
            action = torch.argmax(policy, dim=-1)
            return action, new_hidden
