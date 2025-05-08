import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class NavigationGNN(nn.Module):
    def __init__(self, input_channels=3, hidden_size=256):
        super().__init__()
        self.conv1 = GCNConv(input_channels, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, hidden_size)

        self.spatial_conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.spatial_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.spatial_conv3 = nn.Conv2d(64, hidden_size, 3, padding=1)

        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, 4)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, edge_index=None, batch=None, hidden=None):
        batch_size = x.size(0) if len(x.size()) > 3 else 1
        if len(x.size()) == 3:
            x = x.unsqueeze(0)

        spatial_x = F.relu(self.spatial_conv1(x))
        spatial_x = F.max_pool2d(spatial_x, 2)
        spatial_x = F.relu(self.spatial_conv2(spatial_x))
        spatial_x = F.max_pool2d(spatial_x, 2)
        spatial_x = F.relu(self.spatial_conv3(spatial_x))
        spatial_x = F.adaptive_avg_pool2d(spatial_x, (1, 1))
        spatial_features = spatial_x.view(batch_size, -1)

        if edge_index is not None:
            graph_x = F.relu(self.conv1(x.view(-1, x.size(-1)), edge_index))
            graph_x = F.relu(self.conv2(graph_x, edge_index))
            graph_x = self.conv3(graph_x, edge_index)

            if batch is not None:
                graph_features = global_mean_pool(graph_x, batch)
            else:
                graph_features = graph_x.mean(dim=0).unsqueeze(0).expand(batch_size, -1)
        else:
            graph_features = torch.zeros(
                batch_size, spatial_features.size(1), device=x.device
            )

        combined_features = torch.cat([spatial_features, graph_features], dim=1)
        combined_features = combined_features.unsqueeze(1)

        if hidden is None:
            lstm_out, new_hidden = self.lstm(combined_features)
        else:
            lstm_out, new_hidden = self.lstm(combined_features, hidden)

        x = F.relu(self.fc1(lstm_out.squeeze(1)))
        x = F.relu(self.fc2(x))

        policy = F.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)

        return policy, value, new_hidden


class NavigationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = NavigationGNN()
        self.target_net = NavigationGNN()
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(self, state, edge_index=None, batch=None, hidden=None, epsilon=0.0):
        if torch.rand(1) < epsilon:
            return torch.randint(0, 4, (1,)), hidden

        with torch.no_grad():
            policy, _, new_hidden = self.policy_net(
                state.unsqueeze(0) if state.dim() == 3 else state,
                edge_index,
                batch,
                hidden,
            )
            action = torch.argmax(policy, dim=-1)
            return action, new_hidden
