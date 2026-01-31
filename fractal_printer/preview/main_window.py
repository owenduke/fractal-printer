# Main window for the raymarch preview app
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFrame, QPushButton, QProgressBar, QLabel
from PyQt6.QtCore import pyqtSignal
from fractal_printer.preview.modern_gl_widget import ModernGLWidget
from fractal_printer.preview.controls_panel import ControlsPanel

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

        # Container panel for controlls
        self.sidebar = QFrame()
        sidebar_layout = QVBoxLayout()
        self.sidebar.setLayout(sidebar_layout)
        layout.addWidget(self.sidebar, stretch=0)

        # Shader options
        controls_json = {
            'width': {'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 1.0},
            'height': {'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 1.0},
            'depth': {'min': 0.0, 'max': 1.0, 'step': 0.01, 'default': 1.0},
        }
        self.controls_panel = ControlsPanel(controls_json)
        self.controls_panel.controlsChanged.connect(self.gl_widget.updateControls)
        self.controls_panel.setMinimumWidth(300)
        sidebar_layout.addWidget(self.controls_panel, stretch=1)

        # Bottom box for mesh generation
        self.bottom_box = QFrame()
        self.bottom_box.setFrameShape(QFrame.Shape.StyledPanel)
        self.bottom_box.setFixedHeight(80)
        self.bottom_layout = QHBoxLayout()
        self.bottom_box.setLayout(self.bottom_layout)
        sidebar_layout.addWidget(self.bottom_box)

        # Button
        self.generate_btn = QPushButton("Generate Mesh")
        self.generate_btn.clicked.connect(self.on_generate_mesh)

        # Progress Bar
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(1) 
        self.progress.setVisible(False)
        self.bottom_layout.addWidget(self.generate_btn)
        self.bottom_layout.addWidget(self.progress)

        # Call the resize handler to initialize the viewport
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
