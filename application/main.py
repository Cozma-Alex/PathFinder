from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtWidgets import QLabel, QFileDialog, QComboBox
from PyQt6.QtCore import Qt
from gui.modern_styles import ModernButton, ModernFrame, MODERN_WINDOW_STYLE
from gui.thor_navigation_gui import ThorNavigationGUI
from gui.map_thor_controller import MapThorController
import torch
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("pathfinder")

# Add the project root directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class NavigationNet:
    """Custom wrapper to handle model loading regardless of structure."""

    def __init__(self):
        self.policy_net = None
        self.hidden = None

    def load_state_dict(self, state_dict):
        try:
            logger.info("Attempting to load model...")
            # Create a simple default network that can handle basic navigation
            import torch.nn as nn
            import torch.nn.functional as F

            class SimplePolicy(nn.Module):
                def __init__(self, input_channels=3):
                    super().__init__()
                    self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.fc = nn.Linear(256, 4)

                def forward(self, x, hidden=None):
                    if x.dim() == 3:
                        x = x.unsqueeze(0)

                    # Ensure input has proper dimensions
                    if x.shape[2] != 64 or x.shape[3] != 64:
                        x = F.interpolate(x, size=(64, 64), mode="nearest")

                    x = F.relu(self.conv1(x))
                    x = F.max_pool2d(x, 2)
                    x = F.relu(self.conv2(x))
                    x = F.adaptive_avg_pool2d(x, (2, 2))
                    x = x.view(x.size(0), -1)
                    actions = F.softmax(self.fc(x), dim=1)
                    return actions, None, hidden

            self.policy_net = SimplePolicy()

            # Check if state_dict is a complete model or just network weights
            if isinstance(state_dict, dict) and "policy_net.conv1.weight" in state_dict:
                # Full model state dict
                new_state_dict = {}
                # Remove prefixes like 'policy_net.'
                for k, v in state_dict.items():
                    if k.startswith("policy_net."):
                        new_key = k[len("policy_net.") :]
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v
                state_dict = new_state_dict

            missing, unexpected = self.policy_net.load_state_dict(
                state_dict, strict=False
            )
            if missing:
                logger.info(f"Model loaded with missing keys: {missing[:5]}...")
            if unexpected:
                logger.info(f"Model loaded with unexpected keys: {unexpected[:5]}...")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def get_action(self, state, hidden=None, epsilon=0.1):
        # Add random exploration to make navigation more robust
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, 4, (1,)), hidden

        if self.policy_net is not None:
            with torch.no_grad():
                try:
                    policy, _, new_hidden = self.policy_net(state, hidden)
                    action = torch.argmax(policy, dim=1)
                    return action, new_hidden
                except Exception as e:
                    logger.error(f"Error during prediction: {str(e)}")
                    # Return a random action as fallback
                    return torch.randint(0, 4, (1,)), hidden

        # Fallback to random action if model loading failed
        return torch.randint(0, 4, (1,)), hidden


class PathFinderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thor_controller = None
        self.navigation_window = None
        self.model = None
        self.model_path = None
        self.scenes = [
            "FloorPlan_Train1_1",
            "FloorPlan_Train1_2",
            "FloorPlan_Train1_3",
            "FloorPlan_Train1_4",
            "FloorPlan_Train2_1",
            "FloorPlan_Train2_2",
            "FloorPlan_Train2_3",
            "FloorPlan_Train2_4",
            "FloorPlan_Train3_1",
            "FloorPlan_Train3_2",
            "FloorPlan_Train3_3",
            "FloorPlan_Train3_4",
            "FloorPlan_Train4_1",
            "FloorPlan_Train4_2",
            "FloorPlan_Train4_3",
            "FloorPlan_Train4_4",
            "FloorPlan_Val1_1",
            "FloorPlan_Val1_2",
            "FloorPlan_Val1_3",
            "FloorPlan_Val1_4",
            "FloorPlan_Val2_1",
            "FloorPlan_Val2_2",
            "FloorPlan_Val2_3",
            "FloorPlan_Val2_4",
        ]
        self.current_scene = "FloorPlan_Train2_2"
        self.initUI()

    def initUI(self):
        self.setWindowTitle("PathFinder Navigation System")
        self.setGeometry(100, 100, 600, 250)
        self.setStyleSheet(MODERN_WINDOW_STYLE)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        model_frame = ModernFrame()
        model_layout = QVBoxLayout(model_frame)

        # Model selection section
        model_section = QHBoxLayout()
        model_label = QLabel("Navigation Model:")
        self.model_path_label = QLabel("No model selected")
        self.model_path_label.setStyleSheet("color: #7f8c8d;")
        self.browse_button = ModernButton("Browse")
        self.browse_button.clicked.connect(self.browse_model)

        model_section.addWidget(model_label)
        model_section.addWidget(self.model_path_label, 1)
        model_section.addWidget(self.browse_button)
        model_layout.addLayout(model_section)

        # Level selection section
        level_section = QHBoxLayout()
        level_label = QLabel("Environment Level:")
        self.level_combo = QComboBox()
        self.level_combo.addItems(self.scenes)
        self.level_combo.setCurrentText(self.current_scene)
        self.level_combo.currentTextChanged.connect(self.update_scene)

        level_section.addWidget(level_label)
        level_section.addWidget(self.level_combo, 1)
        model_layout.addLayout(level_section)

        # Start button
        start_section = QHBoxLayout()
        start_section.addStretch()
        self.launch_button = ModernButton("Launch Navigation")
        self.launch_button.setMinimumWidth(200)
        self.launch_button.clicked.connect(self.launch_navigation)

        # Enable launch button even without model - we'll use random navigation as fallback
        self.launch_button.setEnabled(True)

        start_section.addWidget(self.launch_button)
        start_section.addStretch()

        model_layout.addLayout(start_section)

        # Status display
        self.status_label = QLabel(
            "Select a model file or click Launch for random navigation"
        )
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: #7f8c8d;")
        model_layout.addWidget(self.status_label)

        main_layout.addWidget(model_frame)

    def browse_model(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Navigation Model",
            "",
            "Model Files (*.pt *.pth);;All Files (*)",
        )

        if file_path:
            self.model_path = file_path
            self.model_path_label.setText(os.path.basename(file_path))
            self.status_label.setText("Model selected. Ready to launch.")
            self.launch_button.setEnabled(True)

    def update_scene(self, scene_name):
        self.current_scene = scene_name

    def load_model(self):
        self.model = NavigationNet()

        if not self.model_path:
            self.status_label.setText("No model selected. Using random navigation.")
            logger.info("No model selected. Using random navigation.")
            return True

        try:
            self.status_label.setText("Loading model...")

            # Use weights_only=True to avoid pickle security issues
            state_dict = torch.load(
                self.model_path,
                map_location=torch.device("cpu"),
                weights_only=True,  # Security improvement
            )

            if self.model.load_state_dict(state_dict):
                self.status_label.setText("Model loaded successfully.")
                return True
            else:
                self.status_label.setText(
                    "Error loading model. Using random navigation."
                )
                return True  # Still return True to continue with fallback
        except Exception as e:
            self.status_label.setText(f"Error loading model: {str(e)}")
            logger.error(f"Error loading model: {str(e)}")
            return True  # Still return True to continue with fallback

    def launch_navigation(self):
        if self.load_model():
            try:
                self.status_label.setText("Initializing environment...")

                # Create controller for the selected scene
                self.thor_controller = MapThorController(scene=self.current_scene)

                # Start the controller
                self.thor_controller.start()

                if not self.thor_controller.controller:
                    self.status_label.setText("Failed to initialize Thor controller")
                    return

                # Create and show navigation window
                self.status_label.setText("Starting navigation...")
                self.navigation_window = ThorNavigationGUI(
                    self.thor_controller, self.model
                )
                self.navigation_window.show()
                self.hide()
            except Exception as e:
                self.status_label.setText(f"Error starting navigation: {str(e)}")
                logger.error(f"Error starting navigation: {str(e)}")
                if self.thor_controller:
                    self.thor_controller.stop()
                    self.thor_controller = None

    def closeEvent(self, event):
        if self.navigation_window:
            self.navigation_window.close()
        if self.thor_controller:
            self.thor_controller.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = PathFinderApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
