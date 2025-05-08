import os
import torch
import logging
from datetime import datetime
from s3dis_room_database import S3DISRoomDataset, S3DISNavigationDataset
from NavigationGNN import NavigationNet
from navigation_trainer import NavigationTrainer
import json


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


def main():
    config = {
        "data_root": "../S3DIS/Stanford3dDataset_v1.2_Aligned_Version",
        "train_areas": [1, 2, 3, 4],
        "val_areas": [5],
        "batch_size": 32,
        "num_epochs": 100,
        "max_steps": 4000,
        "learning_rate": 0.00003,
        "save_interval": 10,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("trained_models", f"nav_model_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logging(save_dir)
    save_config(config, save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading datasets...")
    train_room_dataset = S3DISRoomDataset(
        config["data_root"], area_ids=config["train_areas"]
    )
    val_room_dataset = S3DISRoomDataset(
        config["data_root"], area_ids=config["val_areas"]
    )

    train_dataset = S3DISNavigationDataset(train_room_dataset)
    val_dataset = S3DISNavigationDataset(val_room_dataset)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    logger.info("Initializing model...")
    model = NavigationNet().to(device)
    trainer = NavigationTrainer(model, device=device)

    logger.info("Starting training...")
    best_loss = float("inf")
    for epoch in range(config["num_epochs"]):
        train_loss = trainer.train_epoch(train_dataset, epoch)
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        if epoch % config["save_interval"] == 0:
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

    with open(os.path.join(save_dir, "training_complete.txt"), "w") as f:
        f.write(
            f'Training completed at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        )
        f.write(f"Best loss: {best_loss:.4f}\n")


if __name__ == "__main__":
    main()
