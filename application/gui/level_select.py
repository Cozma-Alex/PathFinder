from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QScrollArea, QGridLayout
from PyQt6.QtCore import Qt
from gui.modern_styles import ModernButton, MODERN_WINDOW_STYLE
from gui.thor_navigation_gui import ThorNavigationGUI
from gui.map_thor_controller import MapThorController

class LevelSelectWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thor_controller = None
        self.navigation_window = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Select Level')
        self.setGeometry(200, 200, 900, 700)
        self.setStyleSheet(MODERN_WINDOW_STYLE)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(15)

        levels = [
            "FloorPlan_Train1_1", "FloorPlan_Train1_2", "FloorPlan_Train1_3", "FloorPlan_Train1_4",
            "FloorPlan_Train2_1", "FloorPlan_Train2_2", "FloorPlan_Train2_3", "FloorPlan_Train2_4",
            "FloorPlan_Train3_1", "FloorPlan_Train3_2", "FloorPlan_Train3_3", "FloorPlan_Train3_4",
            "FloorPlan_Train4_1", "FloorPlan_Train4_2", "FloorPlan_Train4_3", "FloorPlan_Train4_4",
            "FloorPlan_Val1_1", "FloorPlan_Val1_2", "FloorPlan_Val1_3", "FloorPlan_Val1_4",
            "FloorPlan_Val2_1", "FloorPlan_Val2_2", "FloorPlan_Val2_3", "FloorPlan_Val2_4"
        ]

        row = 0
        col = 0
        columns = 3

        for level in levels:
            btn = ModernButton(level)
            btn.clicked.connect(lambda checked, l=level: self.start_level(l))
            grid.addWidget(btn, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1

        scroll.setWidget(container)
        
    def start_level(self, level):
        try:
            self.thor_controller = MapThorController(scene=level)
            self.thor_controller.start()
            
            if not self.thor_controller.controller:
                raise RuntimeError("Failed to initialize Thor controller")
                
            self.navigation_window = ThorNavigationGUI(self.thor_controller)
            self.navigation_window.show()
        except Exception as e:
            print(f"Error starting level: {str(e)}")
            if self.thor_controller:
                self.thor_controller.stop()
                self.thor_controller = None

    def closeEvent(self, event):
        if self.navigation_window:
            self.navigation_window.close()
        if self.thor_controller:
            self.thor_controller.stop()
        event.accept()