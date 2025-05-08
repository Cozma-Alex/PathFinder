import numpy as np
import os
from torch.utils.data import Dataset
import torch


class S3DISProcessor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.areas = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]
        self.grid_size = 0.25

    def load_area(self, area_name):
        area_path = os.path.join(self.root_dir, area_name)
        rooms = []
        for room in os.listdir(area_path):
            if os.path.isdir(os.path.join(area_path, room)):
                point_cloud = np.load(os.path.join(area_path, room, "point_cloud.npy"))
                semantic_labels = np.load(
                    os.path.join(area_path, room, "semantic_labels.npy")
                )
                rooms.append(
                    {
                        "name": room,
                        "points": point_cloud,
                        "labels": semantic_labels,
                        "grid": self.create_navigation_grid(point_cloud),
                    }
                )
        return rooms

    def create_navigation_grid(self, points):
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])

        grid_width = int((max_x - min_x) / self.grid_size) + 1
        grid_height = int((max_z - min_z) / self.grid_size) + 1

        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

        for point in points:
            grid_x = int((point[0] - min_x) / self.grid_size)
            grid_z = int((point[2] - min_z) / self.grid_size)
            if 0 <= grid_x < grid_width and 0 <= grid_z < grid_height:
                grid[grid_z, grid_x] = 1

        return grid


class NavigationDataset(Dataset):
    def __init__(self, rooms, sequence_length=32):
        self.rooms = rooms
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.rooms)

    def __getitem__(self, idx):
        room = self.rooms[idx]
        grid = room["grid"]

        start_x = np.random.randint(0, grid.shape[1])
        start_z = np.random.randint(0, grid.shape[0])
        goal_x = np.random.randint(0, grid.shape[1])
        goal_z = np.random.randint(0, grid.shape[0])

        while not grid[start_z, start_x] or not grid[goal_z, goal_x]:
            start_x = np.random.randint(0, grid.shape[1])
            start_z = np.random.randint(0, grid.shape[0])
            goal_x = np.random.randint(0, grid.shape[1])
            goal_z = np.random.randint(0, grid.shape[0])

        state = np.stack(
            [
                grid,
                np.zeros_like(grid, dtype=np.float32),
                np.zeros_like(grid, dtype=np.float32),
            ]
        )

        state[1, start_z, start_x] = 1.0
        state[2, goal_z, goal_x] = 1.0

        return {
            "state": torch.FloatTensor(state),
            "start": (start_x, start_z),
            "goal": (goal_x, goal_z),
            "grid": torch.FloatTensor(grid),
        }
