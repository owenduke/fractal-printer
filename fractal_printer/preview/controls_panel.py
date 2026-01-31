import json
from PyQt6.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QFormLayout, QSlider, QLineEdit, QHBoxLayout, QLabel
from PyQt6.QtCore import pyqtSignal, Qt


class ControlsPanel(QScrollArea):
    controlsChanged = pyqtSignal(dict)
    def __init__(self, controls_json, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.controls = {}
        self.sliders = {}
        self.lineedits = {}
        self.widget = QWidget()
        self.layout = QFormLayout()
        self.widget.setLayout(self.layout)
        self.setWidget(self.widget)
        self._build_controls(controls_json)

    def _build_controls(self, controls_json):
        for name, spec in controls_json.items():
            hbox = QHBoxLayout()
            slider = QSlider(Qt.Orientation.Horizontal)
            #slider.setOrientation(Orientation.Horizontal)  # Qt.Horizontal

            # Quantize slider to integer steps, map to float
            steps = int((spec['max'] - spec['min']) / spec.get('step', 0.01))
            slider.setMinimum(0)
            slider.setMaximum(steps)
            default_val = spec.get('default', spec['min'])
            slider.setValue(int((default_val - spec['min']) / spec.get('step', 0.01)))
            lineedit = QLineEdit()
            lineedit.setFixedWidth(100)
            lineedit.setText(f"{default_val:.3f}")
            hbox.addWidget(slider)
            hbox.addWidget(lineedit)
            self.layout.addRow(QLabel(name), hbox)
            self.sliders[name] = (slider, spec)
            self.lineedits[name] = (lineedit, spec)
            self.controls[name] = default_val

            # Connect slider and lineedit
            slider.valueChanged.connect(lambda val, n=name, s=spec: self._slider_changed(n, val, s))
            lineedit.editingFinished.connect(lambda n=name, le=lineedit, s=spec: self._lineedit_changed(n, le, s))

    def _slider_changed(self, name, value, spec):
        float_val = spec['min'] + value * spec.get('step', 0.01)
        self.controls[name] = float_val
        lineedit, _ = self.lineedits[name]
        lineedit.setText(f"{float_val:.3f}")
        self.controlsChanged.emit(self.controls.copy())

    def _lineedit_changed(self, name, lineedit, spec):
        try:
            val = float(lineedit.text())
        except ValueError:
            val = spec['min']
        val = max(spec['min'], min(spec['max'], val))
        self.controls[name] = val
        slider, _ = self.sliders[name]
        slider.setValue(int((val - spec['min']) / spec.get('step', 0.01)))
        self.controlsChanged.emit(self.controls.copy())


    def _emit_change(self):
        pass  # Not used anymore

    def get_controls(self):
        return self.controls.copy()

    def set_controls(self, controls):
        for name, value in controls.items():
            if name in self.sliders:
                slider, spec = self.sliders[name]
                slider.setValue(int((value - spec['min']) / spec.get('step', 0.01)))
            if name in self.lineedits:
                lineedit, _ = self.lineedits[name]
                lineedit.setText(f"{value:.3f}")
