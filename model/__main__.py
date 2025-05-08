import torch
from s3dis_processor import S3DISProcessor, NavigationDataset
from navigation_model import NavigationNet
from trainer import NavigationTrainer
from thor_tester import ThorTester


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s3dis_root = "PathFinder/S3DIS"
    processor = S3DISProcessor(s3dis_root)

    all_rooms = []
    for area in processor.areas:
        rooms = processor.load_area(area)
        all_rooms.extend(rooms)

    train_rooms = all_rooms[: int(0.8 * len(all_rooms))]
    val_rooms = all_rooms[int(0.8 * len(all_rooms)) :]

    train_dataset = NavigationDataset(train_rooms)
    val_dataset = NavigationDataset(val_rooms)

    model = NavigationNet()
    trainer = NavigationTrainer(model, device)

    num_epochs = 100
    trainer.train_s3dis(train_dataset, num_epochs)

    torch.save(model.state_dict(), "navigation_model.pth")

    thor_scenes = [
        "FloorPlan_Train1_1",
        "FloorPlan_Train2_1",
        "FloorPlan_Train3_1",
        "FloorPlan_Train4_1",
        "FloorPlan_Val1_1",
        "FloorPlan_Val2_1",
    ]

    tester = ThorTester(model, device)
    results = tester.evaluate_model(thor_scenes)

    print("\nTesting Results:")
    for scene_result in results:
        print(f"\nScene: {scene_result['scene']}")
        print(f"Success Rate: {scene_result['success_rate']:.2f}")
        print(f"Average Path Length: {scene_result['avg_path_length']:.2f}")
        print(f"Min Path Length: {scene_result['min_path_length']}")
        print(f"Max Path Length: {scene_result['max_path_length']}")


if __name__ == "__main__":
    main()
