# PyQt6 app entry point for raymarch preview

import sys
import traceback
from PyQt6.QtWidgets import QApplication
from fractal_printer.preview.main_window import MainWindow


def _excepthook(exc_type, exc, tb):
    print("Uncaught exception:", file=sys.stderr)
    traceback.print_exception(exc_type, exc, tb)
    sys.stderr.flush()
    sys.stdout.flush()

sys.excepthook = _excepthook

def main():
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        print("Starting...", flush=True)
        window.show()
        ret = app.exec()
        print("Quitting with code", ret, flush=True)
        sys.exit(ret)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
