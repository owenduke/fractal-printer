# ModernGLWidget: OpenGL rendering widget for raymarching
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import pyqtSignal, Qt, QEvent
from PyQt6.QtWidgets import QMessageBox, QPinchGesture
import moderngl
import numpy as np
import os


class ModernGLWidget(QOpenGLWidget):
    cameraMoved = pyqtSignal(float, float)
    zoomChanged = pyqtSignal(float)
    controlsChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctx = None
        self.prog = None
        self.vbo = None
        self.vao = None
        self.mouse_pos = (0, 0)
        self.camera = {'theta': 0.0, 'phi': np.pi / 2, 'distance': 5.0}
        self.controls = {}
        self._shader_error_shown = False

        # Attempt to grab pinch gesture (different enum locations across Qt versions)
        try:
            self.grabGesture(Qt.PinchGesture)
        except Exception:
            try:
                self.grabGesture(Qt.GestureType.PinchGesture)
            except Exception:
                pass

    def initializeGL(self):
        try:
            self.ctx = moderngl.create_context(standalone=False)
        except Exception:
            import traceback as _tb
            _tb.print_exc()
            self.ctx = None
            return

        vert_shader = '''
        #version 330
        in vec2 in_vert;
        out vec2 uv;
        void main() {
            uv = in_vert * 0.5 + 0.5;
            gl_Position = vec4(in_vert, 0.0, 1.0);
        }
        '''

        # Load fragment shader from file

        shader_path = os.path.join(os.path.dirname(__file__), 'shaders', 'raymarch_julia.frag')
        with open(shader_path, 'r') as f:
            frag_shader = f.read()

        self.prog = self.ctx.program(
            vertex_shader=vert_shader,
            fragment_shader=frag_shader
        )
        print(self.ctx.error)
        # Check for errors
        # if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        #     info_log = glGetShaderInfoLog(shader).decode()
        #     print("Shader Compilation Error:\n", info_log) # Prints error
        #     raise Exception()

        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

        self.ctx.viewport = (0, 0, self.width(), self.height())

    def paintGL(self):
        self.ctx.clear(0.0,1.0,0.0)

        for key in self.controls:
            if key in self.prog:
                self.prog[key].value = self.controls[key]

        self.prog['camera'].value = tuple([
            self.camera['theta'],
            self.camera['phi'],
            self.camera['distance']
        ])

        self.vao.render(mode = moderngl.TRIANGLE_STRIP)
        if self.ctx.error != "GL_NO_ERROR":
            print(f"OpenGL Error: {self.ctx.error}")

    def mouseMoveEvent(self, event):
        dx = event.position().x() - self.mouse_pos[0]
        dy = event.position().y() - self.mouse_pos[1]
        self.mouse_pos = (event.position().x(), event.position().y())
        self.camera['theta'] += dx * 0.01
        self.camera['phi'] += dy * 0.01
        self.cameraMoved.emit(self.camera['theta'], self.camera['phi'])
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_pos = (event.position().x(), event.position().y())

    def wheelEvent(self, event):
        try:
            # angleDelta is preferred; fallback to pixelDelta
            ad = event.angleDelta()
            delta = ad.y() if not ad.isNull() else event.pixelDelta().y()
            # One notch (120) -> change by 0.5 units
            change = (delta / 120.0) * 0.5
            new_dist = self.camera['distance'] - change
            new_dist = max(1.0, min(20.0, new_dist))
            if new_dist != self.camera['distance']:
                self.camera['distance'] = new_dist
                self.zoomChanged.emit(self.camera['distance'])
                self.update()
        except Exception:
            import traceback as _tb
            _tb.print_exc()

    def event(self, event):
        # Intercept gesture events and dispatch to gesture handler
        try:
            if event.type() == QEvent.Type.Gesture:
                return self.gestureEvent(event)
        except Exception:
            pass
        return super().event(event)

    def gestureEvent(self, event):
        try:
            for g in event.gestures():
                if isinstance(g, QPinchGesture):
                    pinch = g
                    last = pinch.lastScaleFactor()
                    cur = pinch.scaleFactor()
                    if last <= 0.0:
                        last = 1.0
                    if cur <= 0.0:
                        cur = 1.0
                    ratio = cur / last
                    # Scale camera distance inversely to pinch scale
                    new_dist = self.camera['distance'] / ratio
                    new_dist = max(1.0, min(20.0, new_dist))
                    if new_dist != self.camera['distance']:
                        self.camera['distance'] = new_dist
                        self.zoomChanged.emit(self.camera['distance'])
                        self.update()
                        #print(f"New distance: {new_dist:.3f}")
                    return True
        except Exception:
            import traceback as _tb
            _tb.print_exc()
        return False

    def updateControls(self, controls):
        self.controls.update(controls)
        self.controlsChanged.emit(self.controls)
        self.update()

    def resizeGL(self, w, h):
        if self.ctx:
            self.ctx.viewport = (0, 0, w, h)
            self.ctx.detect_framebuffer().use()
