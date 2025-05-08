from PyQt6.QtWidgets import QPushButton, QFrame
from PyQt6.QtGui import QFont


class ModernButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setMinimumHeight(50)
        self.setFont(QFont("Segoe UI", 11))
        self.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                margin: 6px;
                font-weight: 500;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2471a3;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)


class ModernFrame(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 15px;
                border: 1px solid #e0e0e0;
                padding: 15px;
            }
        """)


MODERN_WINDOW_STYLE = """
    QMainWindow {
        background-color: #f5f6fa;
    }
    QLabel {
        color: #2c3e50;
        font-family: 'Segoe UI';
        font-size: 12px;
    }
    QLabel#title {
        font-size: 16px;
        font-weight: bold;
        padding: 10px;
    }
    QScrollArea {
        border: none;
        background-color: transparent;
    }
    QScrollBar:vertical {
        border: none;
        background-color: #f0f0f0;
        width: 10px;
        margin: 0;
    }
    QScrollBar::handle:vertical {
        background-color: #c0c0c0;
        border-radius: 5px;
        min-height: 20px;
    }
    QScrollBar::handle:vertical:hover {
        background-color: #a0a0a0;
    }
    QComboBox {
        padding: 6px 12px;
        border: 1px solid #ccc;
        border-radius: 4px;
        background-color: white;
        min-height: 30px;
    }
    QComboBox::drop-down {
        width: 20px;
        border-left: 1px solid #ccc;
    }
    QLineEdit {
        padding: 6px 12px;
        border: 1px solid #ccc;
        border-radius: 4px;
        background-color: white;
        min-height: 30px;
    }
"""
