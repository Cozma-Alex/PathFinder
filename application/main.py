from PyQt6.QtWidgets import QApplication
import sys
from gui.menu import MenuWindow

def main():
    app = QApplication(sys.argv)
    window = MenuWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()