# PyQt6 app entry point for raymarch preview

import sys
from PyQt6.QtWidgets import QApplication
from fractal_printer.preview.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
