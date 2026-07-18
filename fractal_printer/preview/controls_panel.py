import json
from PyQt6.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QFormLayout, QSlider, QLineEdit, QHBoxLayout, QLabel, QGroupBox, QGridLayout, QCheckBox, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt, QSize

class CoupledBox(QWidget):
    value_changed = pyqtSignal(float)
    def __init__(self, name: str, parent = None, max = 1, min = -1, step = 0.001, default = 0):
        super().__init__(parent)
        self.setLayout(QHBoxLayout())

        self.label = QLabel(name)
        self.layout().addWidget(self.label)

        self.max = max
        self.min = min
        self.step = step
        self.default = default
        self.value = default

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.layout().addWidget(self.slider)
        steps = int((max - min)/step)
        self.slider.setMinimum(0)
        self.slider.setMaximum(steps)
        self.slider.setValue(int((self.value - min) / step))

        self.lineedit = QLineEdit()
        self.layout().addWidget(self.lineedit)
        self.lineedit.setFixedWidth(100)
        self.lineedit.setText(f"{default:.3f}")

        # Connect slider and lineedit
        self.slider.valueChanged.connect(self.update_value)
        self.lineedit.editingFinished.connect(self.update_value)

    def update_value(self, new_state = None):
        if isinstance(new_state, str):
            # Change is coming from the text box
            try:
                value = float(new_state)
            except ValueError:
                value = self.value
        elif isinstance(new_state, int):
            # Change is coming from the slider
            value = self.min + new_state * self.step
        elif new_state is None:
            value = self.value

        self.lineedit.setText(f"{value:.3f}")
        self.slider.setValue(int((value - self.min)/self.step))

        if self.value != value:
            self.value = value
            self.value_changed.emit(value)
    
    def reset(self):
        self.value = self.default
        self.update_value()

    def toggle(self, state):
        for w in [self.label, self.slider, self.lineedit]:
            w.setVisible(state)
        self.value_changed.emit(self.value)

class QuaternionSelector(QWidget):
    value_changed = pyqtSignal(list)
    def __init__(self, name: str, parent = None):
        super().__init__(parent)
        self.setMinimumSize(QSize(0,0))
        self.setLayout(QFormLayout())
        self.layout().setVerticalSpacing(10)


        titlebox = QHBoxLayout()
        
        titlebox.addWidget(QLabel(name))

        self.enabledbox = QCheckBox("Enable")
        self.enabledbox.checkStateChanged.connect(self.toggle)
        titlebox.addWidget(self.enabledbox)

        self.zerobutton = QPushButton("Clear")
        self.zerobutton.clicked.connect(self.clear)
        titlebox.addWidget(self.zerobutton)

        self.layout().addRow(titlebox)

        self.controls = {}
        self.value = [0]*4
        for axis in ["x","y","z","w"]:
            c = CoupledBox(axis, parent = self)
            c.value_changed.connect(self.update_value)
            self.controls[axis] = c
            self.layout().addRow(c)
            
    def update_value(self):
        for i, n in enumerate(self.controls):
            self.value[i] = self.controls[n].value
        self.value_changed.emit(self.value)

    def set_value(self, value):
        self.value = value
        for i, n in enumerate(self.controls):
            self.controls[n].value = value[i]
            self.controls[n].update_value()
        
    def toggle(self):
        state = self.enabledbox.isChecked()
        for n in self.controls:
            self.controls[n].toggle(state)
    
    def clear(self):
        for n in self.controls:
            self.controls[n].reset()


class ControlsPanel(QScrollArea):
    controlsChanged = pyqtSignal(dict)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.widget = QWidget()
        self.layout = QFormLayout()
        self.widget.setLayout(self.layout)
        self.setWidget(self.widget)

        # Add the coefficient controls
        self.coeffs = [QuaternionSelector(f"C{n}") for n in range(9)]

        # Add the other controls
        self.controls = {
            "slice"      : CoupledBox("Slice"),
            "offset"     : CoupledBox("Offset", min=0.001, max = 0.5, step = 0.001, default = 0.01),
            "iterations" : CoupledBox("Iterations", min=1, max = 100, step = 1, default = 10),
            "bailout"    : CoupledBox("Bailout", min = 1, max = 100000, step = 1, default = 100)
        }

        for c in self.coeffs:
            self.layout.addRow(c)
            c.value_changed.connect(self.update_controls)
            c.enabledbox.setChecked(False)
            c.toggle()

        for name, c in self.controls.items():
            self.layout.addRow(c)
            c.value_changed.connect(self.update_controls)
        
        self.update_controls()

    def update_controls(self):
        settings = {}
        power = 0
        coefficients = []
        for n, c in enumerate(self.coeffs):
            if c.enabledbox.isChecked():
                if power < n:
                    power = n
                coefficients.append(c.value)
            else:
                coefficients.append([0]*4)

        settings["coefficients"] = coefficients
        settings["power"] = power

        for name, control in self.controls.items():
            settings[name] = control.value

        self.controlsChanged.emit(settings)
        self.settings = settings

    def get_controls(self):
        return self.settings
    
    def set_controls(self, settings):
        for i, c in enumerate(settings["coefficients"]):
            self.coeffs[i].set_value(c)
            self.coeffs[i].enabledbox.setChecked(any(c))
            self.coeffs[i].toggle()

        for name, control in self.controls.items():
            try:
                control.value = settings[name]
                control.update_value()
            except KeyError:
                pass

