import numpy as np
import os
import torch
from torch_geometric.data import Dataset, Data


class S3DISRoomDataset(Dataset):
    def __init__(self, root_dir, area_ids=None, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.areas = (
            [f"Area_{i}" for i in range(1, 7)]
            if area_ids is None
            else [f"Area_{i}" for i in area_ids]
        )
        self.rooms = self._get_room_list()
        self.grid_size = 0.25

    def _get_room_list(self):
        rooms = []
        for area in self.areas:
            area_dir = os.path.join(self.root_dir, area)
            for room in os.listdir(area_dir):
                room_dir = os.path.join(area_dir, room)
                if os.path.isdir(room_dir):
                    rooms.append((area, room))
        return rooms

    def len(self):
        return len(self.rooms)

    def get(self, idx):
        area, room = self.rooms[idx]
        room_dir = os.path.join(self.root_dir, area, room, "Annotations")

        points = []
        labels = []
        object_id = 0

        for file in os.listdir(room_dir):
            if file.endswith(".txt"):
                object_type = "_".join(file.split("_")[:-1])
                with open(os.path.join(room_dir, file), "r") as f:
                    for line in f:
                        x, y, z = map(float, line.strip().split()[:3])
                        points.append([x, y, z])
                        labels.append(object_id)
                object_id += 1

        points = np.array(points)
        labels = np.array(labels)

        grid_map = self._create_navigation_grid(points)

        pos = torch.from_numpy(points).float()
        y = torch.from_numpy(labels).long()
        grid_map = torch.from_numpy(grid_map).float()

        data = Data(pos=pos, y=y, grid_map=grid_map, room_id=f"{area}/{room}")

        if self.transform:
            data = self.transform(data)

        return data

    def _create_navigation_grid(self, points):
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])

        grid_width = int((max_x - min_x) / self.grid_size) + 1
        grid_height = int((max_z - min_z) / self.grid_size) + 1

        grid = np.zeros((grid_height, grid_width), dtype=np.float32)

        for point in points:
            grid_x = int((point[0] - min_x) / self.grid_size)
            grid_z = int((point[2] - min_z) / self.grid_size)
            if 0 <= grid_x < grid_width and 0 <= grid_z < grid_height:
                grid[grid_z, grid_x] = 1

        return grid


class S3DISNavigationDataset(Dataset):
    def __init__(self, room_dataset, sequence_length=32):
        super().__init__()
        self.room_dataset = room_dataset
        self.sequence_length = sequence_length

    def len(self):
        return len(self.room_dataset)

    def get(self, idx):
        room_data = self.room_dataset[idx]
        grid_map = room_data.grid_map

        valid_positions = torch.nonzero(grid_map)
        num_valid = valid_positions.size(0)

        start_idx = torch.randint(0, num_valid, (1,))
        goal_idx = torch.randint(0, num_valid, (1,))
        while goal_idx == start_idx:
            goal_idx = torch.randint(0, num_valid, (1,))

        start_pos = valid_positions[start_idx[0]]
        goal_pos = valid_positions[goal_idx[0]]

        state = torch.zeros(3, grid_map.size(0), grid_map.size(1))
        state[0] = grid_map
        state[1, start_pos[0], start_pos[1]] = 1.0
        state[2, goal_pos[0], goal_pos[1]] = 1.0

        return Data(
            state=state,
            grid_map=grid_map,
            start_pos=start_pos,
            goal_pos=goal_pos,
            room_id=room_data.room_id,
        )
