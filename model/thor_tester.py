import torch
import numpy as np
from gui.map_thor_controller import MapThorController
from application.thor_controller.thor_controller import Action


class ThorTester:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def test_scene(self, scene_name, start_pos=None, goal_pos=None):
        thor_controller = MapThorController(scene=scene_name)
        thor_controller.start()

        grid_map = thor_controller.get_grid_map()
        if start_pos is None or goal_pos is None:
            start_pos, goal_pos = self.get_random_positions(grid_map)

        state = self.create_initial_state(grid_map, start_pos, goal_pos)
        current_pos = start_pos
        hidden = None
        done = False
        path = [start_pos]

        thor_controller.teleport_to_grid_position(*start_pos)

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, hidden = self.model.get_action(state_tensor, hidden)

            next_pos = self.get_next_position(current_pos, action)

            if not self.is_valid_position(next_pos, grid_map):
                break

            thor_controller.teleport_to_grid_position(*next_pos)
            path.append(next_pos)

            if next_pos == goal_pos:
                done = True
                success = True
            elif len(path) > 100:
                done = True
                success = False

            state = self.update_state(state, next_pos)
            current_pos = next_pos

        thor_controller.stop()
        return {
            "path": path,
            "success": success,
            "path_length": len(path),
            "start": start_pos,
            "goal": goal_pos,
        }

    def get_random_positions(self, grid_map):
        valid_positions = np.where(grid_map == 1)
        valid_indices = range(len(valid_positions[0]))

        start_idx = np.random.choice(valid_indices)
        start_pos = (valid_positions[1][start_idx], valid_positions[0][start_idx])

        goal_idx = np.random.choice(valid_indices)
        while goal_idx == start_idx:
            goal_idx = np.random.choice(valid_indices)
        goal_pos = (valid_positions[1][goal_idx], valid_positions[0][goal_idx])

        return start_pos, goal_pos

    def create_initial_state(self, grid_map, start_pos, goal_pos):
        state = np.zeros((3, grid_map.shape[0], grid_map.shape[1]), dtype=np.float32)
        state[0] = grid_map
        state[1, start_pos[1], start_pos[0]] = 1.0
        state[2, goal_pos[1], goal_pos[0]] = 1.0
        return state

    def get_next_position(self, current_pos, action):
        x, z = current_pos
        if action == Action.MOVE_FORWARD:
            return (x, z - 1)
        elif action == Action.FACE_EAST:
            return (x + 1, z)
        elif action == Action.MOVE_BACK:
            return (x, z + 1)
        elif action == Action.FACE_WEST:
            return (x - 1, z)
        else:
            return current_pos

    def is_valid_position(self, pos, grid_map):
        x, z = pos
        if x < 0 or x >= grid_map.shape[1] or z < 0 or z >= grid_map.shape[0]:
            return False
        return grid_map[z, x] == 1

    def update_state(self, state, new_pos):
        new_state = state.copy()
        new_state[1] = np.zeros_like(new_state[1])
        new_state[1, new_pos[1], new_pos[0]] = 1.0
        return new_state

    def evaluate_model(self, scenes, num_episodes_per_scene=10):
        results = []
        for scene in scenes:
            scene_results = []
            for _ in range(num_episodes_per_scene):
                result = self.test_scene(scene)
                scene_results.append(result)

            successes = [r["success"] for r in scene_results]
            path_lengths = [r["path_length"] for r in scene_results]

            scene_metrics = {
                "scene": scene,
                "success_rate": np.mean(successes),
                "avg_path_length": np.mean(path_lengths),
                "min_path_length": np.min(path_lengths),
                "max_path_length": np.max(path_lengths),
            }
            results.append(scene_metrics)

        return results
