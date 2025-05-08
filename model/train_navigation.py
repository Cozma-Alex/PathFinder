import os
import torch
from datetime import datetime
from navigation_policy import NavigationNet
from navigation_trainer import NavigationTrainer
import numpy as np


def create_synthetic_data(batch_size=16, grid_size=32):
    data = []
    for _ in range(batch_size):
        grid_map = np.ones((grid_size, grid_size), dtype=np.float32)

        obstacles = np.random.randint(0, grid_size, size=(grid_size // 4, 2))
        for i, j in obstacles:
            if 0 <= i < grid_size and 0 <= j < grid_size:
                grid_map[i, j] = 0

        # Random start and goal positions
        valid_positions = np.array(np.where(grid_map == 1)).T
        idx = np.random.choice(len(valid_positions), 2, replace=False)
        start_pos = valid_positions[idx[0]]
        goal_pos = valid_positions[idx[1]]

        # Create state representation
        state = np.zeros((3, grid_size, grid_size), dtype=np.float32)
        state[0] = grid_map
        state[1, start_pos[0], start_pos[1]] = 1.0
        state[2, goal_pos[0], goal_pos[1]] = 1.0

        data.append(torch.tensor(state))

    return data


class SyntheticDataLoader:
    def __init__(self, batch_size=16, grid_size=32, num_batches=100):
        self.batch_size = batch_size
        self.grid_size = grid_size
        self.num_batches = num_batches

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch < self.num_batches:
            self.current_batch += 1
            return create_synthetic_data(self.batch_size, self.grid_size)
        else:
            raise StopIteration


def train_model(epochs=10, save_path="trained_models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NavigationNet().to(device)
    trainer = NavigationTrainer(model, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f"nav_model_{timestamp}.pt")

    print("Starting training...")
    train_loader = SyntheticDataLoader(batch_size=16, grid_size=32, num_batches=50)

    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader, epoch)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

    print(f"Training completed. Saving model to {model_path}")
    torch.save(model.policy_net.state_dict(), model_path)

    return model_path


if __name__ == "__main__":
    train_model(epochs=5)
