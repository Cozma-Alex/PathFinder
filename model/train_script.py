import os
import torch
import logging
from datetime import datetime
from navigation_policy import NavigationNet
from navigation_trainer import NavigationTrainer
from s3dis_room_database import S3DISRoomDataset, S3DISNavigationDataset
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def setup_logging(save_dir):
    log_file = os.path.join(save_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("training")


def save_config(config, save_dir):
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def manual_resize_tensor(tensor, target_shape):
    _, h, w = tensor.shape
    target_h, target_w = target_shape

    result = torch.zeros(3, target_h, target_w, dtype=tensor.dtype)

    agent_pos = None
    goal_pos = None

    agent_indices = tensor[1].nonzero()
    if len(agent_indices) > 0:
        agent_pos = agent_indices[0]

    goal_indices = tensor[2].nonzero()
    if len(goal_indices) > 0:
        goal_pos = goal_indices[0]

    for i in range(target_h):
        for j in range(target_w):
            src_i = min(int(i * h / target_h), h - 1)
            src_j = min(int(j * w / target_w), w - 1)
            result[0, i, j] = tensor[0, src_i, src_j]

    if agent_pos is not None:
        new_agent_i = min(int(agent_pos[0] * target_h / h), target_h - 1)
        new_agent_j = min(int(agent_pos[1] * target_w / w), target_w - 1)
        result[1, new_agent_i, new_agent_j] = 1.0
    else:
        result[1, 5, 5] = 1.0

    if goal_pos is not None:
        new_goal_i = min(int(goal_pos[0] * target_h / h), target_h - 1)
        new_goal_j = min(int(goal_pos[1] * target_w / w), target_w - 1)
        result[2, new_goal_i, new_goal_j] = 1.0
    else:
        result[2, target_h - 6, target_w - 6] = 1.0

    return result


class S3DISDataIterator:
    def __init__(self, dataset, batch_size=32, shuffle=True, target_size=(64, 64)):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_size = target_size
        self.indices = list(range(len(dataset)))
        if shuffle:
            import random
            random.shuffle(self.indices)

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[self.current : self.current + self.batch_size]
        batch_indices = batch_indices[: min(len(batch_indices), self.batch_size)]
        self.current += self.batch_size

        if len(batch_indices) == 0:
            raise StopIteration

        batch_states = []
        for idx in batch_indices:
            data = self.dataset[idx]
            resized_state = manual_resize_tensor(data.state, self.target_size)
            batch_states.append(resized_state)

        return torch.stack(batch_states)


def train_s3dis(model, trainer, config, save_dir, logger):
    logger.info("Loading S3DIS dataset...")
    
    try:
        train_room_dataset = S3DISRoomDataset(
            root_dir=config["data_root"],
            area_ids=config["train_areas"]
        )
        
        val_room_dataset = S3DISRoomDataset(
            root_dir=config["data_root"],
            area_ids=config["val_areas"]
        )
        
        train_nav_dataset = S3DISNavigationDataset(
            train_room_dataset,
            sequence_length=config["max_steps"]
        )
        
        val_nav_dataset = S3DISNavigationDataset(
            val_room_dataset,
            sequence_length=config["max_steps"]
        )
        
        logger.info(f"Loaded {len(train_nav_dataset)} training samples and {len(val_nav_dataset)} validation samples")
        
        train_loader = S3DISDataIterator(
            train_nav_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            target_size=config["target_size"]
        )
        
        val_loader = S3DISDataIterator(
            val_nav_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            target_size=config["target_size"]
        )
        
    except Exception as e:
        logger.error(f"Failed to load S3DIS dataset: {str(e)}")
        logger.info("Falling back to synthetic data...")
        return train_synthetic(model, trainer, config, save_dir, logger)

    logger.info("Starting training on S3DIS data...")
    best_loss = float("inf")
    best_val_loss = float("inf")
    
    for epoch in range(config["num_epochs"]):
        train_loss = trainer.train_epoch(train_loader, epoch)
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        
        if epoch % 5 == 0:
            val_loss = evaluate_model(model, val_loader, trainer, logger)
            logger.info(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(save_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model with validation loss {best_val_loss:.4f}")

        if epoch % config["save_interval"] == 0 or epoch == config["num_epochs"] - 1:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "loss": train_loss,
                    "epsilon": trainer.epsilon,
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            if train_loss < best_loss:
                best_loss = train_loss

    final_model_path = os.path.join(save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed. Final model saved to {final_model_path}")

    return final_model_path


def evaluate_model(model, data_loader, trainer, logger):
    model.policy_net.eval()
    total_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch_loss = 0
            batch_count += 1
            
            for state in batch:
                state = state.to(trainer.device)
                current_state = state.clone()
                
                for step in range(min(10, trainer.max_steps)):
                    action, _ = model.get_action(current_state, epsilon=0.0)
                    next_state, reward, done = trainer.simulate_step(current_state, action)
                    
                    policy, value, _ = model.policy_net(current_state.unsqueeze(0))
                    target_policy, target_value, _ = model.target_net(next_state.unsqueeze(0))
                    
                    action_idx = trainer.action_to_idx[action] if hasattr(trainer, 'action_to_idx') else 0
                    loss = torch.nn.functional.mse_loss(
                        policy[0, action_idx], 
                        reward + trainer.gamma * target_policy.max(1)[0]
                    )
                    batch_loss += loss.item()
                    
                    if done:
                        break
                    current_state = next_state
            
            total_loss += batch_loss / len(batch)
            
            if batch_count >= 10:
                break
    
    model.policy_net.train()
    return total_loss / batch_count if batch_count > 0 else 0


def train_synthetic(model, trainer, config, save_dir, logger):
    import torch
    import numpy as np

    logger.info("Generating synthetic training data...")

    class SyntheticIterator:
        def __init__(self, batch_size=32, num_batches=10, grid_size=64):
            self.batch_size = batch_size
            self.num_batches = num_batches
            self.grid_size = grid_size

        def __iter__(self):
            self.current_batch = 0
            return self

        def __next__(self):
            if self.current_batch >= self.num_batches:
                raise StopIteration

            self.current_batch += 1

            batch = torch.zeros(self.batch_size, 3, self.grid_size, self.grid_size)

            for b in range(self.batch_size):
                grid = torch.ones(self.grid_size, self.grid_size)

                num_obstacles = np.random.randint(5, 15)
                for _ in range(num_obstacles):
                    x = np.random.randint(5, self.grid_size - 5)
                    y = np.random.randint(5, self.grid_size - 5)
                    size = np.random.randint(3, 8)
                    grid[
                        max(0, y - size) : min(self.grid_size, y + size),
                        max(0, x - size) : min(self.grid_size, x + size),
                    ] = 0

                grid[
                    self.grid_size // 4 : self.grid_size * 3 // 4,
                    self.grid_size // 4 : self.grid_size * 3 // 4,
                ] = 1

                valid_positions = torch.nonzero(grid)
                if len(valid_positions) > 1:
                    idx1, idx2 = torch.randint(0, len(valid_positions), (2,))
                    agent_pos = valid_positions[idx1]
                    goal_pos = valid_positions[idx2]
                else:
                    agent_pos = torch.tensor([10, 10])
                    goal_pos = torch.tensor([self.grid_size - 10, self.grid_size - 10])

                batch[b, 0] = grid
                batch[b, 1, agent_pos[0], agent_pos[1]] = 1.0
                batch[b, 2, goal_pos[0], goal_pos[1]] = 1.0

            return batch

    train_loader = SyntheticIterator(
        batch_size=config["batch_size"],
        num_batches=50,
        grid_size=config["target_size"][0],
    )

    logger.info("Starting training on synthetic data...")
    best_loss = float("inf")
    for epoch in range(config["num_epochs"]):
        train_loss = trainer.train_epoch(train_loader, epoch)
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        if epoch % config["save_interval"] == 0 or epoch == config["num_epochs"] - 1:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "loss": train_loss,
                    "epsilon": trainer.epsilon,
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            if train_loss < best_loss:
                best_loss = train_loss
                best_model_path = os.path.join(save_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model with loss {best_loss:.4f}")

    final_model_path = os.path.join(save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed. Final model saved to {final_model_path}")

    return final_model_path


def main():
    config = {
        "data_root": "/home/acozma/PathFinder/S3DIS/Stanford3dDataset_v1.2_Aligned_Version",
        "train_areas": [1, 2, 3, 4],
        "val_areas": [5, 6],
        "batch_size": 4,
        "num_epochs": 50,
        "max_steps": 25,
        "learning_rate": 0.0001,
        "save_interval": 10,
        "target_size": (64, 64),
        "use_synthetic": False,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("trained_models", f"nav_model_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logging(save_dir)
    save_config(config, save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Initializing model...")
    model = NavigationNet().to(device)
    trainer = NavigationTrainer(model, device=device, max_steps=config["max_steps"])

    if config["use_synthetic"]:
        logger.info("Using synthetic data for training")
        train_synthetic(model, trainer, config, save_dir, logger)
    else:
        logger.info("Using S3DIS data for training")
        train_s3dis(model, trainer, config, save_dir, logger)

    logger.info(f"Training completed. Final model saved to {save_dir}")

    with open(os.path.join(save_dir, "training_complete.txt"), "w") as f:
        f.write(
            f'Training completed at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        )


if __name__ == "__main__":
    main()