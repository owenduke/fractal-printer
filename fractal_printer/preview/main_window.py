# Main window for the raymarch preview app
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFrame, QPushButton, QProgressBar, QLabel
from PyQt6.QtCore import pyqtSignal
from fractal_printer.preview.modern_gl_widget import ModernGLWidget
from fractal_printer.preview.controls_panel import ControlsPanel
import pyperclip
import json
import random

RANDOM_ORDER = 3

DEFAULT_SETTINGS = {
    "coefficients": [
        [-0.381, 0.625, 0.794, 0],
        [0,0,0,0],
        [1.0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]
    ],
    "power": 2,
    "slice": 0.02100000000000013,
    "offset": 0.01,
    "iterations": 10,
    "bailout": 100
}

def _format_settings(settings):
    keys = list(settings.keys())
    lines = ["{"]
    for idx, key in enumerate(keys):
        comma = "," if idx < len(keys) - 1 else ""
        value = settings[key]
        if key == "coefficients":
            lines.append('    "coefficients": [')
            for i, c in enumerate(value):
                c_comma = "," if i < len(value) - 1 else ""
                lines.append(f"        {json.dumps(c)}{c_comma}")
            lines.append(f"    ]{comma}")
        else:
            lines.append(f'    "{key}": {json.dumps(value)}{comma}')
    lines.append("}")
    return "\n".join(lines)


class MainWindow(QMainWindow):
    viewportResized = pyqtSignal(int, int)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quaternionic Julia Preview")
        self.resize(800, 600)
        
        # Central widget and layout
        central = QWidget()
        layout = QHBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # ModernGLWidget for GLSL rendering (left panel)
        self.gl_widget = ModernGLWidget()
        self.gl_widget.setMinimumWidth(800)
        layout.addWidget(self.gl_widget, stretch=3)

        # Connect viewport resize signal
        self.viewportResized.connect(self.gl_widget.resizeGL)

        # Container panel for controls
        self.sidebar = QFrame()
        sidebar_layout = QVBoxLayout()
        self.sidebar.setLayout(sidebar_layout)
        layout.addWidget(self.sidebar, stretch=0)

        # Shader options
        self.controls_panel = ControlsPanel()
        self.controls_panel.controlsChanged.connect(self.gl_widget.updateControls)
        self.gl_widget.updateControls(self.controls_panel.get_controls())
        self.controls_panel.setMinimumWidth(300)
        sidebar_layout.addWidget(self.controls_panel, stretch=1)

        # Bottom box for mesh generation
        self.bottom_box = QFrame()
        self.bottom_box.setFrameShape(QFrame.Shape.StyledPanel)
        self.bottom_box.setFixedHeight(80)
        self.bottom_layout = QHBoxLayout()
        self.bottom_box.setLayout(self.bottom_layout)
        sidebar_layout.addWidget(self.bottom_box)

        # Buttons
        big_label_style = "font-size: 14pt; font-weight: bold;"

        self.copy_btn = QPushButton("Copy Settings")
        self.copy_btn.setStyleSheet(big_label_style)
        self.copy_btn.clicked.connect(self.on_copy_settings)
        self.paste_btn = QPushButton("Paste Settings")
        self.paste_btn.setStyleSheet(big_label_style)
        self.paste_btn.clicked.connect(self.on_paste_settings)
        self.randomize_btn = QPushButton("Randomize")
        self.randomize_btn.setStyleSheet(big_label_style)
        self.randomize_btn.clicked.connect(self.on_randomize_settings)

        self.bottom_layout.addWidget(self.copy_btn)
        self.bottom_layout.addWidget(self.paste_btn)
        self.bottom_layout.addWidget(self.randomize_btn)

        # Put in the default values
        self.controls_panel.set_controls(DEFAULT_SETTINGS)
        
        # Call the resize handler to initialize the viewport
        print("Calling the resize event")
        self.resizeEvent(None)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Emit signal with new size of the GL widget
        if self.gl_widget:
            w = self.gl_widget.width()
            h = self.gl_widget.height()
            self.viewportResized.emit(w, h)

    def on_generate_mesh(self):
        self.progress.setVisible(True)
        # Here, launch a process to print controls (stub for now)
        import multiprocessing, time
        controls = self.controls_panel.get_controls()
        def dummy_mesh_job(ctrls):
            print("[Dummy Mesh Process] Controls:", ctrls)
            time.sleep(2)
        p = multiprocessing.Process(target=dummy_mesh_job, args=(controls,))
        p.start()
        # Simulate done after a delay (in real app, use QProcess or signals)
        from threading import Timer
        def finish():
            self.progress.setVisible(False)
        Timer(2.2, finish).start()

    def on_copy_settings(self):
        data_string = _format_settings(self.controls_panel.get_controls())
        print(data_string)
        pyperclip.copy(data_string)

    def on_paste_settings(self):
        data_string = pyperclip.paste()
        print(data_string)
        self.controls_panel.set_controls(json.loads(data_string))

    def on_randomize_settings(self):
        settings = json.loads(json.dumps(DEFAULT_SETTINGS))

        settings["coefficients"] = [
            [round(random.uniform(-1, 1), 3) for _ in range(4)] if n <= RANDOM_ORDER else [0, 0, 0, 0]
            for n in range(len(settings["coefficients"]))
        ]
        settings["power"] = RANDOM_ORDER
        self.controls_panel.set_controls(settings)
