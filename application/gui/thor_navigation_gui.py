from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent
from gui.modern_styles import ModernButton, ModernFrame, MODERN_WINDOW_STYLE
import numpy as np
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
    def __init__(self, thor_controller):
        super().__init__()
        self.thor_controller = thor_controller
        self.initUI()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_views)
        self.timer.start(100)
        
        self.selecting_start = False
        self.selecting_goal = False
        self.start_pos = None
        self.goal_pos = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
    def initUI(self):
        self.setWindowTitle('Thor Navigation System')
        self.setGeometry(100, 100, 800, 900)
        self.setStyleSheet(MODERN_WINDOW_STYLE)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_widget.setLayout(main_layout)
        
        self.map_panel = ViewPanel('Environment Map')
        self.map_view = self.map_panel.content
        self.map_view.mousePressEvent = self.map_click
        main_layout.addWidget(self.map_panel)
        
        control_panel = ModernFrame()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setSpacing(15)
        
        self.status_label = QLabel("Select start position")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: 500;")
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        
        self.start_button = ModernButton('Set Start')
        self.goal_button = ModernButton('Set Goal')
        self.run_button = ModernButton('Run')
        self.stop_button = ModernButton('Stop')
        
        self.start_button.clicked.connect(self.start_selection)
        self.goal_button.clicked.connect(self.goal_selection)
        self.run_button.clicked.connect(self.run_navigation)
        self.stop_button.clicked.connect(self.stop_navigation)
        
        for button in [self.start_button, self.goal_button, self.run_button, self.stop_button]:
            control_layout.addWidget(button)
            button.setMinimumWidth(120)
        
        main_layout.addWidget(control_panel)

    def keyPressEvent(self, event: QKeyEvent):
        if not self.thor_controller:
            return
            
        key = event.key()
        if key == Qt.Key.Key_W:
            self.thor_controller.execute_action(Action.MOVE_FORWARD)
        elif key == Qt.Key.Key_S:
            self.thor_controller.execute_action(Action.MOVE_BACK)
        elif key == Qt.Key.Key_A:
            self.thor_controller.execute_action(Action.FACE_WEST)
        elif key == Qt.Key.Key_D:
            self.thor_controller.execute_action(Action.FACE_EAST)
        elif key == Qt.Key.Key_Q:
            self.thor_controller.execute_action(Action.FACE_NORTH)
        elif key == Qt.Key.Key_E:
            self.thor_controller.execute_action(Action.FACE_SOUTH)
    
    def update_views(self):
        grid_map = self.thor_controller.get_grid_map()
        current_pos = self.thor_controller.get_current_grid_position()
        
        vis_map = np.zeros((grid_map.shape[0], grid_map.shape[1], 3), dtype=np.uint8)
        vis_map[grid_map == 1] = [245, 245, 245]
        vis_map[grid_map == 0] = [40, 40, 40]
        
        marker_size = 1
        thickness = 1
        
        if current_pos:
            cv2.drawMarker(vis_map, current_pos, (52, 152, 219), 
                          cv2.MARKER_DIAMOND, marker_size, thickness)
        
        if self.start_pos:
            cv2.drawMarker(vis_map, self.start_pos, (46, 204, 113), 
                          cv2.MARKER_STAR, marker_size, thickness)
        
        if self.goal_pos:
            cv2.drawMarker(vis_map, self.goal_pos, (231, 76, 60), 
                          cv2.MARKER_CROSS, marker_size, thickness)
        
        h, w = vis_map.shape[:2]
        qt_map = QImage(vis_map.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self.map_view.setPixmap(QPixmap.fromImage(qt_map).scaled(
            self.map_view.size(), Qt.AspectRatioMode.KeepAspectRatio))
    
    def map_click(self, event):
        if not (self.selecting_start or self.selecting_goal):
            return
            
        pixmap = self.map_view.pixmap()
        if not pixmap:
            return
            
        view_rect = self.map_view.rect()
        scaled_pixmap = pixmap.scaled(view_rect.size(), Qt.AspectRatioMode.KeepAspectRatio)
        
        x_offset = (view_rect.width() - scaled_pixmap.width()) / 2
        y_offset = (view_rect.height() - scaled_pixmap.height()) / 2
        
        pos = event.pos()
        adjusted_x = pos.x() - x_offset
        adjusted_y = pos.y() - y_offset
        
        if adjusted_x < 0 or adjusted_y < 0:
            return
            
        grid_x = int(adjusted_x * self.thor_controller.map_dimensions['width'] / scaled_pixmap.width())
        grid_z = int(adjusted_y * self.thor_controller.map_dimensions['height'] / scaled_pixmap.height())
        
        if self.thor_controller.is_position_reachable(grid_x, grid_z):
            if self.selecting_start:
                self.start_pos = (grid_x, grid_z)
                self.selecting_start = False
                self.start_button.setChecked(False)
                self.status_label.setText("Start position set! Now select goal position.")
            elif self.selecting_goal:
                self.goal_pos = (grid_x, grid_z)
                self.selecting_goal = False
                self.goal_button.setChecked(False)
                self.status_label.setText("Goal position set! Ready to run.")
    
    def start_selection(self):
        self.selecting_start = True
        self.selecting_goal = False
        self.status_label.setText("Click on the map to set start position")
    
    def goal_selection(self):
        self.selecting_start = False
        self.selecting_goal = True
        self.status_label.setText("Click on the map to set goal position")
    
    def run_navigation(self):
        try:
            if self.start_pos and self.goal_pos:
                event = self.thor_controller.teleport_to_grid_position(*self.start_pos)
                if event and not event.metadata['lastActionSuccess']:
                    print(f"Teleport failed: {event.metadata['errorMessage']}")
                self.status_label.setText("Navigation in progress...")
        except Exception as e:
            print(f"Error during navigation: {str(e)}")
            self.status_label.setText("Navigation failed.")
    
    def stop_navigation(self):
        self.status_label.setText("Navigation stopped.")