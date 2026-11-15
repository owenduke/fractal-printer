# ModernGLWidget: OpenGL rendering widget for raymarching
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import pyqtSignal, Qt, QEvent
from PyQt6.QtWidgets import QMessageBox, QPinchGesture
import moderngl
import numpy as np
import os
import quaternion


class ModernGLWidget(QOpenGLWidget):
    #cameraMoved = pyqtSignal(float, float)
    #zoomChanged = pyqtSignal(float)
    controlsChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctx = None
        self.prog = None
        self.vbo = None
        self.vao = None
        self.mouse_pos = (0, 0)
        self.camera = quaternion.quaternion(0,3,0,0)
        self.up     = quaternion.quaternion(0,0,0,1)
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

        self.prog['camera'].value = self.camera.components[1:]
        self.prog['up'].value = self.up.components[1:]

        self.prog['aspect'].value = self.ctx.viewport[2]/self.ctx.viewport[3]

        self.vao.render(mode = moderngl.TRIANGLE_STRIP)
        if self.ctx.error != "GL_NO_ERROR":
            print(f"OpenGL Error: {self.ctx.error}")

    def mouseMoveEvent(self, event):
        dx = event.position().x() - self.mouse_pos[0]
        dy = event.position().y() - self.mouse_pos[1]
        self.mouse_pos = (event.position().x(), event.position().y())

        angle = -np.sqrt(dx**2 + dy**2) / 1e4
        right = (self.up * self.camera - self.camera * self.up)
        right = right / np.abs(right)
        vector = self.up * dx + right * dy
        rotation = np.cos(angle/2) * quaternion.quaternion(1,0,0,0) + np.sin(angle/2) * vector

        self.camera = rotation * self.camera / rotation
        self.up     = rotation * self.up / rotation

        # Project up back onto the perpendicular plane and normalize
        #print(np.dot(np.array(self.camera.components), np.array(self.up.components)), np.abs(self.up))


        #self.cameraMoved.emit(self.camera)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_pos = (event.position().x(), event.position().y())

    def wheelEvent(self, event):
        try:
            # angleDelta is preferred; fallback to pixelDelta
            ad = event.angleDelta()
            delta = ad.y() if not ad.isNull() else event.pixelDelta().y()
            self.camera = self.camera * np.exp(-delta / 1200.0)
            self.update()

        except Exception:
            import traceback as _tb
            _tb.print_exc()

    def updateControls(self, controls):
        self.controls.update(controls)
        self.controlsChanged.emit(self.controls)
        self.update()

    def resizeGL(self, w, h):
        if self.ctx:
            self.ctx.viewport = (0, 0, w, h)
            self.ctx.detect_framebuffer().use()
