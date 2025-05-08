from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt
from gui.modern_styles import ModernButton, MODERN_WINDOW_STYLE
from gui.level_select import LevelSelectWindow


class MenuWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("AI Navigation System")
        self.setGeometry(300, 300, 400, 300)
        self.setStyleSheet(MODERN_WINDOW_STYLE)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 40, 30, 40)

        choose_level_btn = ModernButton("Choose Level")
        training_btn = ModernButton("Training")

        choose_level_btn.clicked.connect(self.start_navigation)
        training_btn.clicked.connect(self.start_training)

        layout.addWidget(choose_level_btn)
        layout.addWidget(training_btn)

    def start_navigation(self):
        self.level_select = LevelSelectWindow()
        self.level_select.show()
        self.hide()

    def start_training(self):
        pass
