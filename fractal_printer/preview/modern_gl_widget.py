# ModernGLWidget: OpenGL rendering widget for raymarching
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import pyqtSignal, Qt
import moderngl
import numpy as np
import os

class BasicGLWidget(QOpenGLWidget):

        def __init__(self, parent=None):
            super().__init__(parent)
            self.ctx = None
            self.prog = None
            self.vbo = None
            self.vao = None

        def initializeGL(self):
            self.ctx = moderngl.create_context()
            vert_shader = '''
            #version 330
            in vec2 in_vert;
            out vec2 uv;
            void main() {
                uv = in_vert * 0.5 + 0.5;
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
            '''
            frag_shader = '''
            #version 330 core
            in vec2 uv;
            out vec4 fragColor;
            void main() {
                fragColor = vec4(uv, 0.5 + 0.5 * sin(uv.x * 10.0), 1.0);
            }
            '''
            self.prog = self.ctx.program(
                vertex_shader=vert_shader,
                fragment_shader=frag_shader
            )
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
            #self.ctx.screen.use()
            self.ctx.clear(0.5, 0.5, 0.5)
            self.vao.render(mode=moderngl.TRIANGLE_STRIP)
            print(f"Paint Called. Error: {self.ctx.error}")

        def resizeGL(self, w, h):
            if self.ctx:
                self.ctx.viewport = (0, 0, w, h)
                self.ctx.detect_framebuffer().use()

class ModernGLWidget(QOpenGLWidget):
    cameraMoved = pyqtSignal(float, float)
    controlsChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctx = None
        self.prog = None
        self.vbo = None
        self.vao = None
        self.mouse_pos = (0, 0)
        self.camera = {'theta': 0.0, 'phi': 0.0, 'distance': 3.0}
        self.controls = {'width': 1.0, 'height': 1.0, 'depth': 1.0}

    def initializeGL(self):
        self.ctx = moderngl.create_context(standalone=False)
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

        shader_path = os.path.join(os.path.dirname(__file__), 'shaders', 'raymarch_prism.frag')
        with open(shader_path, 'r') as f:
            frag_shader = f.read()
        # frag_shader = '''
        # #version 330 core
        # in vec2 uv;
        # out vec4 fragColor;
        # uniform float width;
        # uniform float height;
        # uniform float depth;
        # uniform vec3 camera; // theta, phi, distance

        # void main() {
        #     camera + vec3(1,0,0);
        #     fragColor = vec4(uv, (width + height + depth)/3, 1.0);
        # }
        # '''
        self.prog = self.ctx.program(
            vertex_shader=vert_shader,
            fragment_shader=frag_shader
        )
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
        self.prog['width'].value = self.controls['width']
        self.prog['height'].value = self.controls['height']
        self.prog['depth'].value = self.controls['depth']
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

    def updateControls(self, controls):
        self.controls.update(controls)
        self.controlsChanged.emit(self.controls)
        self.update()

    def resizeGL(self, w, h):
        if self.ctx:
            self.ctx.viewport = (0, 0, w, h)
            self.ctx.detect_framebuffer().use()
