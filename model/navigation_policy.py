import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import enum

# Add parent directory to path to allow importing from application
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from application.thor_controller.thor_controller import Action
except ImportError:
    # Fallback if application module can't be imported
    class Action(enum.Enum):
        MOVE_FORWARD = "MoveAhead"
        MOVE_BACK = "MoveBack"
        FACE_NORTH = "RotateAgent"  # Will use with rotation parameter y=180
        FACE_SOUTH = "RotateAgent"  # Will use with rotation parameter y=0
        FACE_EAST = "RotateAgent"   # Will use with rotation parameter y=90
        FACE_WEST = "RotateAgent"   # Will use with rotation parameter y=270


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
        self.action_mapping = {
            0: Action.MOVE_FORWARD,
            1: Action.FACE_EAST,
            2: Action.MOVE_BACK,
            3: Action.FACE_WEST
        }

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(self, state, hidden=None, epsilon=0.0):
        if torch.rand(1) < epsilon:
            action_idx = torch.randint(0, 4, (1,))
            return self.action_mapping[action_idx.item()], hidden

        with torch.no_grad():
            policy, _, new_hidden = self.policy_net(
                state.unsqueeze(0) if state.dim() == 3 else state, hidden
            )
            action_idx = torch.argmax(policy, dim=-1).item()
            return self.action_mapping[action_idx], new_hidden

    def load_state_dict(self, state_dict):
        missing_keys, unexpected_keys = self.policy_net.load_state_dict(
            state_dict, strict=False
        )
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")