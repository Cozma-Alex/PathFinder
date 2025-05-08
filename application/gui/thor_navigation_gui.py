from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent
from gui.modern_styles import ModernButton, ModernFrame, MODERN_WINDOW_STYLE
from thor_controller.thor_controller import Action
import numpy as np
import torch
import cv2


class ViewPanel(ModernFrame):
    def __init__(self, title):
        super().__init__()

        layout = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        self.content = QLabel()
        self.content.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content.setMinimumSize(500, 500)
        layout.addWidget(self.content)

        self.setLayout(layout)


class ThorNavigationGUI(QMainWindow):
    def __init__(self, thor_controller, model=None):
        super().__init__()
        self.thor_controller = thor_controller
        self.model = model
        self.hidden_state = None
        self.initUI()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_views)
        self.timer.start(100)

        self.selecting_start = False
        self.selecting_goal = False
        self.start_pos = None
        self.goal_pos = None
        self.current_path = []
        self.navigation_active = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def initUI(self):
        self.setWindowTitle("Thor Navigation System")
        self.setGeometry(100, 100, 800, 900)
        self.setStyleSheet(MODERN_WINDOW_STYLE)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_widget.setLayout(main_layout)

        self.map_panel = ViewPanel("Environment Map")
        self.map_view = self.map_panel.content
        self.map_view.mousePressEvent = self.map_click
        main_layout.addWidget(self.map_panel)

        control_panel = ModernFrame()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setSpacing(15)

        self.status_label = QLabel("Select start position")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: 500;")
        self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label, 1)

        self.start_button = ModernButton("Set Start")
        self.goal_button = ModernButton("Set Goal")
        self.run_button = ModernButton("Run Manual")
        self.auto_button = ModernButton("Run Auto")
        self.stop_button = ModernButton("Stop")

        self.start_button.clicked.connect(self.start_selection)
        self.goal_button.clicked.connect(self.goal_selection)
        self.run_button.clicked.connect(self.run_manual_navigation)
        self.auto_button.clicked.connect(self.run_auto_navigation)
        self.stop_button.clicked.connect(self.stop_navigation)

        self.auto_button.setEnabled(self.model is not None)

        for button in [
            self.start_button,
            self.goal_button,
            self.run_button,
            self.auto_button,
            self.stop_button,
        ]:
            button.setMinimumWidth(110)
            control_layout.addWidget(button)

        main_layout.addWidget(control_panel)

    def keyPressEvent(self, event: QKeyEvent):
        if not self.thor_controller or not self.navigation_active:
            return

        key = event.key()
        current_pos = self.thor_controller.get_current_grid_position()
        
        if key == Qt.Key.Key_W:
            # Move forward - first get the action, then update the path
            self.thor_controller.execute_action(Action.MOVE_FORWARD)
            # Update path after movement
            new_pos = self.thor_controller.get_current_grid_position()
            if new_pos != current_pos:
                self.current_path.append(new_pos)
                
        elif key == Qt.Key.Key_S:
            # Move backward
            self.thor_controller.execute_action(Action.MOVE_BACK) 
            # Update path after movement
            new_pos = self.thor_controller.get_current_grid_position()
            if new_pos != current_pos:
                self.current_path.append(new_pos)
                
        elif key == Qt.Key.Key_A:
            # Rotate west and then try to move west
            self.thor_controller.execute_action(Action.FACE_WEST)
            
            # After rotation, try to move in that direction
            west_pos = (current_pos[0] - 1, current_pos[1])
            if self.thor_controller.is_position_reachable(*west_pos):
                self.thor_controller.teleport_to_grid_position(*west_pos)
                self.current_path.append(west_pos)
                
        elif key == Qt.Key.Key_D:
            # Rotate east and then try to move east
            self.thor_controller.execute_action(Action.FACE_EAST)
            
            # After rotation, try to move in that direction
            east_pos = (current_pos[0] + 1, current_pos[1])
            if self.thor_controller.is_position_reachable(*east_pos):
                self.thor_controller.teleport_to_grid_position(*east_pos)
                self.current_path.append(east_pos)
                
        elif key == Qt.Key.Key_Q:
            # Rotate north and then try to move north
            self.thor_controller.execute_action(Action.FACE_NORTH)
            
            # After rotation, try to move in that direction
            north_pos = (current_pos[0], current_pos[1] - 1)
            if self.thor_controller.is_position_reachable(*north_pos):
                self.thor_controller.teleport_to_grid_position(*north_pos)
                self.current_path.append(north_pos)
                
        elif key == Qt.Key.Key_E:
            # Rotate south and then try to move south
            self.thor_controller.execute_action(Action.FACE_SOUTH)
            
            # After rotation, try to move in that direction
            south_pos = (current_pos[0], current_pos[1] + 1)
            if self.thor_controller.is_position_reachable(*south_pos):
                self.thor_controller.teleport_to_grid_position(*south_pos)
                self.current_path.append(south_pos)

    def update_views(self):
        grid_map = self.thor_controller.get_grid_map()
        current_pos = self.thor_controller.get_current_grid_position()

        vis_map = np.zeros((grid_map.shape[0], grid_map.shape[1], 3), dtype=np.uint8)
        vis_map[grid_map == 1] = [245, 245, 245]
        vis_map[grid_map == 0] = [40, 40, 40]

        for path_pos in self.current_path:
            if (
                path_pos != self.start_pos
                and path_pos != self.goal_pos
                and path_pos != current_pos
            ):
                vis_map[path_pos[1], path_pos[0]] = [
                    180,
                    180,
                    180,
                ]

        if self.start_pos:
            vis_map[self.start_pos[1], self.start_pos[0]] = [
                52,
                152,
                219,
            ]

        if self.goal_pos:
            vis_map[self.goal_pos[1], self.goal_pos[0]] = [231, 76, 60]

        if current_pos:
            vis_map[current_pos[1], current_pos[0]] = [
                46,
                204,
                113,
            ]

        h, w = vis_map.shape[:2]
        qt_map = QImage(vis_map.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self.map_view.setPixmap(
            QPixmap.fromImage(qt_map).scaled(
                self.map_view.size(), Qt.AspectRatioMode.KeepAspectRatio
            )
        )

    def map_click(self, event):
        if not (self.selecting_start or self.selecting_goal):
            return

        pixmap = self.map_view.pixmap()
        if not pixmap:
            return

        view_rect = self.map_view.rect()
        scaled_pixmap = pixmap.scaled(
            view_rect.size(), Qt.AspectRatioMode.KeepAspectRatio
        )

        x_offset = (view_rect.width() - scaled_pixmap.width()) / 2
        y_offset = (view_rect.height() - scaled_pixmap.height()) / 2

        pos = event.pos()
        adjusted_x = pos.x() - x_offset
        adjusted_y = pos.y() - y_offset

        if adjusted_x < 0 or adjusted_y < 0:
            return

        grid_x = int(
            adjusted_x
            * self.thor_controller.map_dimensions["width"]
            / scaled_pixmap.width()
        )
        grid_z = int(
            adjusted_y
            * self.thor_controller.map_dimensions["height"]
            / scaled_pixmap.height()
        )

        if self.thor_controller.is_position_reachable(grid_x, grid_z):
            if self.selecting_start:
                self.start_pos = (grid_x, grid_z)
                self.selecting_start = False
                self.start_button.setChecked(False)
                self.status_label.setText(
                    "Start position set! Now select goal position."
                )
            elif self.selecting_goal:
                self.goal_pos = (grid_x, grid_z)
                self.selecting_goal = False
                self.goal_button.setChecked(False)
                self.status_label.setText("Goal position set! Ready to run.")

    def start_selection(self):
        self.current_path = []
        self.selecting_start = True
        self.selecting_goal = False
        self.status_label.setText("Click on the map to set start position")

    def goal_selection(self):
        self.selecting_start = False
        self.selecting_goal = True
        self.status_label.setText("Click on the map to set goal position")

    def run_manual_navigation(self):
        try:
            if self.start_pos and self.goal_pos:
                event = self.thor_controller.teleport_to_grid_position(*self.start_pos)
                if event and not event.metadata["lastActionSuccess"]:
                    self.status_label.setText(
                        f"Teleport failed: {event.metadata['errorMessage']}"
                    )
                    return

                self.current_path = [self.start_pos]
                self.navigation_active = True
                self.status_label.setText(
                    "Manual navigation active. Use WASD keys to move."
                )
        except Exception as e:
            self.status_label.setText(f"Error during navigation: {str(e)}")

    def prepare_input_for_model(self, state_tensor):
        try:
            with torch.no_grad():
                test_input = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
                self.model.get_action(test_input)

            if state_tensor.shape[1] != 64 or state_tensor.shape[2] != 64:
                resized_state = torch.zeros((3, 64, 64), dtype=torch.float32)
                for i in range(3):
                    channel = state_tensor[i]
                    resized_channel = torch.nn.functional.interpolate(
                        channel.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="nearest"
                    ).squeeze()
                    resized_state[i] = resized_channel
                return resized_state
            return state_tensor
        except:
            return state_tensor

    def run_auto_navigation(self):
        if not self.model or not self.start_pos or not self.goal_pos:
            self.status_label.setText("Model or positions not set")
            return

        try:
            grid_map = self.thor_controller.get_grid_map()
            event = self.thor_controller.teleport_to_grid_position(*self.start_pos)
            if event and not event.metadata["lastActionSuccess"]:
                self.status_label.setText(
                    f"Teleport failed: {event.metadata['errorMessage']}"
                )
                return

            self.current_path = [self.start_pos]
            self.navigation_active = True
            self.status_label.setText("Automatic navigation in progress...")

            state = self.create_initial_state(grid_map)
            self.navigate_with_model(state)

        except Exception as e:
            self.status_label.setText(f"Error during auto navigation: {str(e)}")
            self.navigation_active = False

    def create_initial_state(self, grid_map):
        state = np.zeros((3, grid_map.shape[0], grid_map.shape[1]), dtype=np.float32)
        state[0] = grid_map
        state[1, self.start_pos[1], self.start_pos[0]] = 1.0
        state[2, self.goal_pos[1], self.goal_pos[0]] = 1.0
        return torch.FloatTensor(state)

    def navigate_with_model(self, state):
        if not self.navigation_active:
            return

        current_pos = self.thor_controller.get_current_grid_position()
        if current_pos == self.goal_pos:
            self.status_label.setText("Goal reached!")
            self.navigation_active = False
            return

        if len(self.current_path) > 100:
            self.status_label.setText("Navigation stopped: path too long")
            self.navigation_active = False
            return

        try:
            processed_state = self.prepare_input_for_model(state)

            action, self.hidden_state = self.model.get_action(
                processed_state.unsqueeze(0), self.hidden_state
            )

            # First execute the rotation action if it's a facing direction
            if action in [Action.FACE_NORTH, Action.FACE_SOUTH, Action.FACE_EAST, Action.FACE_WEST]:
                self.thor_controller.execute_action(action)

            # Then determine next position based on the action
            next_pos = self.get_next_position(current_pos, action)
            
            # Check if position is valid
            if self.thor_controller.is_position_reachable(*next_pos):
                # Now teleport to the position
                event = self.thor_controller.teleport_to_grid_position(*next_pos)
                if event and event.metadata["lastActionSuccess"]:
                    self.current_path.append(next_pos)

                    # Update the state to reflect the new position
                    state[1].zero_()
                    state[1, next_pos[1], next_pos[0]] = 1.0

                    # Continue navigation after a short delay
                    QTimer.singleShot(300, lambda: self.navigate_with_model(state))
                else:
                    # If teleport failed, try an alternative approach - execute the action directly
                    if action == Action.MOVE_FORWARD:
                        self.thor_controller.execute_action(action)
                        new_pos = self.thor_controller.get_current_grid_position()
                        if new_pos != current_pos:
                            self.current_path.append(new_pos)
                            # Update state for next iteration
                            state[1].zero_()
                            state[1, new_pos[1], new_pos[0]] = 1.0
                            QTimer.singleShot(300, lambda: self.navigate_with_model(state))
                        else:
                            self.status_label.setText("Navigation paused: Unable to move forward")
                            self.navigation_active = False
                    else:
                        self.status_label.setText("Navigation error: Invalid move")
                        self.navigation_active = False
            else:
                # Try to find a valid adjacent position
                valid_neighbors = []
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # N, E, S, W
                    neighbor = (current_pos[0] + dx, current_pos[1] + dy)
                    if self.thor_controller.is_position_reachable(*neighbor):
                        valid_neighbors.append(neighbor)
                
                if valid_neighbors:
                    # Find closest neighbor to goal
                    if self.goal_pos:
                        closest = min(valid_neighbors, key=lambda pos: 
                            (pos[0] - self.goal_pos[0])**2 + (pos[1] - self.goal_pos[1])**2)
                        
                        event = self.thor_controller.teleport_to_grid_position(*closest)
                        if event and event.metadata["lastActionSuccess"]:
                            self.current_path.append(closest)
                            state[1].zero_()
                            state[1, closest[1], closest[0]] = 1.0
                            QTimer.singleShot(300, lambda: self.navigate_with_model(state))
                            return
                
                self.status_label.setText("Navigation error: No valid path found")
                self.navigation_active = False
        except Exception as e:
            self.status_label.setText(f"Error during auto navigation: {str(e)}")
            self.navigation_active = False

    def get_next_position(self, current_pos, action):
        x, z = current_pos
        
        agent_rotation = 0
        try:
            event = self.thor_controller.get_current_state()
            if event and event.metadata and 'agent' in event.metadata:
                agent_rotation = event.metadata['agent']['rotation']['y']
        except:
            pass

        if action == Action.MOVE_FORWARD:
            if agent_rotation == 0:
                return (x, z + 1)
            elif agent_rotation == 90:
                return (x + 1, z)
            elif agent_rotation == 180:
                return (x, z - 1)
            elif agent_rotation == 270:
                return (x - 1, z)
        elif action == Action.MOVE_BACK:
            if agent_rotation == 0:
                return (x, z - 1)
            elif agent_rotation == 90:
                return (x - 1, z)
            elif agent_rotation == 180: 
                return (x, z + 1)
            elif agent_rotation == 270:
                return (x + 1, z)
        elif action == Action.FACE_EAST:
            return (x + 1, z)
        elif action == Action.FACE_WEST:
            return (x - 1, z)
        elif action == Action.FACE_NORTH:
            return (x, z - 1)
        elif action == Action.FACE_SOUTH:
            return (x, z + 1)
            
        return current_pos

    def stop_navigation(self):
        self.navigation_active = False
        self.status_label.setText("Navigation stopped.")

    def closeEvent(self, event):
        self.navigation_active = False
        if self.thor_controller:
            self.thor_controller.stop()
        event.accept()