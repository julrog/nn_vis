import glfw
from OpenGL.GL import *
from pyrr import Vector3

from edge import EdgeHandler
from render_helper import VertexDataHandler
from shader import ShaderHandler
from window import WindowHandler

WIDTH, HEIGHT = 1280, 720


def window_resize_clb(glfw_window, width, height):
    pass


def mouse_look_clb(glfw_window, xpos, ypos):
    pass


def key_input_clb(glfw_window, key, scancode, action, mode):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(glfw_window, True)


window_handler = WindowHandler()
window = window_handler.create_window("Testing", WIDTH, HEIGHT)
window.set_position(400, 200)
window.set_callbacks(window_resize_clb, mouse_look_clb, key_input_clb)
window.activate()

layer_one = [-0.5, -0.3, 0.0,
             -0.5, 0, 0.0,
             -0.5, 0.3, 0.0]
layer_two = [0.5, -0.3, 0.0,
             0.5, 0, 0.0,
             0.5, 0.3, 0.0]

sample_length = (Vector3([-0.5, -0.3, 0.0]) - Vector3([0.5, -0.3, 0.0])).length / 100.0
edge_handler = EdgeHandler(sample_length, True)
edge_handler.set_data(layer_one, layer_two)
edge_handler.sample_edges()
# edge_handler.sample_noise(1.0)
#sample_length = (Vector3([-0.5, -0.3, 0.0]) - Vector3([0.5, -0.3, 0.0])).length / 100.0
# edge_handler.resample(sample_length)
# for _ in range(10):
    # edge_handler.resample()
    # edge_handler.sample_noise(0.5)

sampled_point_buffer = edge_handler.generate_buffer_data()
sampled_points = edge_handler.get_points()

# edge_handler.resample()

resampled_point_buffer = edge_handler.generate_buffer_data()
resampled_points = edge_handler.get_points()

shader_handler = ShaderHandler()
shader = shader_handler.create("base", "base.vert", "base.frag")

sampled_vertex_data_handler = VertexDataHandler()
resampled_vertex_data_handler = VertexDataHandler()

sampled_vertex_data_handler.load_data(sampled_point_buffer)
resampled_vertex_data_handler.load_data(resampled_point_buffer)

print(glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0))
print(glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1))
print(glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2))
print(glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0))
print(glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1))
print(glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2))
print(glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS))
print(glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE))

'''compute_shader_handler = ComputeShaderHandler()
compute_shader = compute_shader_handler.create("test", "test.comp")
compute_shader_texture = Texture(10, 10)
compute_shader_texture.setup()
compute_shader.set_textures([(compute_shader_texture, "write")])
compute_shader.set_uniform_data([('alpha', 0.42, 'float')])
compute_shader.use(10)

print(compute_shader_texture.read())'''

glUseProgram(shader.shader_handle)
point_color_loc = glGetUniformLocation(shader.shader_handle, "point_color")

glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# the main application loop
while window.active():
    window_handler.loop()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    #edge_handler.sample_noise(0.5)

    sampled_point_buffer = edge_handler.generate_buffer_data()
    sampled_points = edge_handler.get_points()

    #edge_handler.resample()

    resampled_point_buffer = edge_handler.generate_buffer_data()
    resampled_points = edge_handler.get_points()

    glUseProgram(shader.shader_handle)

    sampled_vertex_data_handler.load_data(sampled_point_buffer)
    resampled_vertex_data_handler.load_data(resampled_point_buffer)

    glPointSize(3.0)
    glUniform3fv(point_color_loc, 1, [1.0, 0.0, 0.0])
    sampled_vertex_data_handler.set()
    glDrawArrays(GL_POINTS, 0, sampled_points)

    glPointSize(1.0)
    glUniform3fv(point_color_loc, 1, [0.0, 1.0, 0.0])
    resampled_vertex_data_handler.set()
    glPrimitiveRestartIndex(1)
    glDrawArrays(GL_LINE_STRIP, 0, resampled_points)

    # glDrawElementsInstanced(GL_TRIANGLES, len(point_buffer), GL_UNSIGNED_INT, None, len_of_instance_array)

    window.swap()

window_handler.destroy()
