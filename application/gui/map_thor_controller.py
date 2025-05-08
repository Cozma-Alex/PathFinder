import numpy as np
from typing import Tuple, Optional, Dict, Any
from thor_controller.thor_controller import ThorController, Action

class MapThorController(ThorController):
    def __init__(self, scene: str = "FloorPlan_Train1_1", headless: bool = False):
        super().__init__(scene, headless)
        self.grid_size = 0.25
        self.reachable_positions = None
        self.grid_map = None
        self.map_dimensions = None
        self.current_grid_position = None
        self._initialize_map()

    def _initialize_map(self):
        if not self.controller:
            self.init_controller()
            
        event = self.controller.step(action="GetReachablePositions")
        if not event:
            return
            
        self.reachable_positions = event.metadata["actionReturn"]
        if not self.reachable_positions:
            return
            
        x_coords = [p["x"] for p in self.reachable_positions]
        z_coords = [p["z"] for p in self.reachable_positions]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_z, max_z = min(z_coords), max(z_coords)
        
        grid_width = int((max_x - min_x) / self.grid_size) + 1
        grid_height = int((max_z - min_z) / self.grid_size) + 1
        
        self.map_dimensions = {
            'min_x': min_x,
            'max_x': max_x,
            'min_z': min_z,
            'max_z': max_z,
            'width': grid_width,
            'height': grid_height
        }
        
        self.grid_map = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        for pos in self.reachable_positions:
            grid_x, grid_z = self.world_to_grid(pos["x"], pos["z"])
            if 0 <= grid_x < grid_width and 0 <= grid_z < grid_height:
                self.grid_map[grid_z, grid_x] = 1
                
    def world_to_grid(self, world_x: float, world_z: float) -> Tuple[int, int]:
        if not self.map_dimensions:
            return (0, 0)
        grid_x = int((world_x - self.map_dimensions['min_x']) / self.grid_size)
        grid_z = int((world_z - self.map_dimensions['min_z']) / self.grid_size)
        return grid_x, grid_z
    
    def grid_to_world(self, grid_x: int, grid_z: int) -> Tuple[float, float]:
        if not self.map_dimensions:
            return (0.0, 0.0)
        world_x = grid_x * self.grid_size + self.map_dimensions['min_x']
        world_z = grid_z * self.grid_size + self.map_dimensions['min_z']
        return float(world_x), float(world_z)
        
    def teleport_to_grid_position(self, grid_x: int, grid_z: int) -> Optional[Dict[str, Any]]:
        if not self.controller or not self.reachable_positions:
            return None
                
        if not self.is_position_reachable(grid_x, grid_z):
            return None
                
        world_x, world_z = self.grid_to_world(grid_x, grid_z)
        position = {"x": world_x, "y": 0.9009997248649597, "z": world_z}
        rotation = {"x": 0, "y": 0, "z": 0}
        
        try:
            return self.controller.step(
                action="TeleportFull",
                position=position,
                rotation=rotation,
                horizon=30.0,
                forceAction=True
            )
        except Exception as e:
            print(f"Teleport error: {str(e)}")
            return None
    
    def get_current_grid_position(self) -> Tuple[int, int]:
        if not self.controller:
            return (0, 0)
        try:
            event = self.get_current_state()
            if not event or not event.metadata:
                return (0, 0)
            agent_pos = event.metadata["agent"]["position"]
            return self.world_to_grid(agent_pos["x"], agent_pos["z"])
        except:
            return (0, 0)
    
    def get_grid_map(self) -> np.ndarray:
        if self.grid_map is None:
            self._initialize_map()
        return self.grid_map.copy() if self.grid_map is not None else np.zeros((1, 1), dtype=np.uint8)
    
    def is_position_reachable(self, grid_x: int, grid_z: int) -> bool:
        if self.grid_map is None or not self.map_dimensions:
            return False
        if 0 <= grid_x < self.map_dimensions['width'] and 0 <= grid_z < self.map_dimensions['height']:
            return bool(self.grid_map[grid_z, grid_x])