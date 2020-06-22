import pyrr
from pyrr import Vector3, Matrix44, vector, vector3
from math import sin, cos, radians, asin, degrees, acos


def look_at(position: Vector3, target: Vector3, world_up: Vector3) -> Matrix44:
    z_axis: Vector3 = vector.normalise(position - target)
    x_axis: Vector3 = vector.normalise(vector3.cross(vector.normalise(world_up), z_axis))
    y_axis: Vector3 = vector3.cross(z_axis, x_axis)

    translation: Matrix44 = Matrix44.identity()
    translation[3][0] = -position.x
    translation[3][1] = -position.y
    translation[3][2] = -position.z

    rotation: Matrix44 = Matrix44.identity()
    rotation[0][0] = x_axis[0]
    rotation[1][0] = x_axis[1]
    rotation[2][0] = x_axis[2]
    rotation[0][1] = y_axis[0]
    rotation[1][1] = y_axis[1]
    rotation[2][1] = y_axis[2]
    rotation[0][2] = z_axis[0]
    rotation[1][2] = z_axis[1]
    rotation[2][2] = z_axis[2]

    return rotation * translation


class Camera:
    def __init__(self, width: float, height: float, base: Vector3, move_speed: float = 0.1,
                 rotation_speed: float = 0.25):
        self.base: Vector3 = base
        self.camera_pos: Vector3 = self.base + Vector3([-3.0, 0.0, 0.0])
        self.camera_front: Vector3 = Vector3([1.0, 0.0, 0.0])
        self.camera_up: Vector3 = Vector3([0.0, 1.0, 0.0])
        self.camera_right: Vector3 = Vector3([0.0, 0.0, 1.0])

        self.move_vector: Vector3 = Vector3([0, 0, 0])
        self.move_speed: float = move_speed

        self.yaw: float = 0.0
        self.pitch: float = 0.0
        self.rotation_speed: float = rotation_speed

        self.projection: Matrix44 = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100)
        self.view: Matrix44 = self.generate_view_matrix()

    def update(self):
        self.update_camera_vectors()
        self.generate_view_matrix()

    def generate_view_matrix(self) -> Matrix44:
        self.view = look_at(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)
        return self.view

    def process_mouse_movement(self, x_offset: float, y_offset: float, constrain_pitch: bool = True):
        self.yaw += x_offset * self.rotation_speed
        self.pitch += y_offset * self.rotation_speed

        if constrain_pitch:
            if self.pitch > 45.0:
                self.pitch = 45.0
            if self.pitch < -45.0:
                self.pitch = -45.0

    def update_camera_vectors(self):
        front: Vector3 = Vector3([0.0, 0.0, 0.0])
        front.x = cos(radians(self.yaw)) * cos(radians(self.pitch))
        front.y = sin(radians(self.pitch))
        front.z = sin(radians(self.yaw)) * cos(radians(self.pitch))

        self.camera_front = vector.normalize(front)
        self.camera_right = vector.normalize(vector3.cross(self.camera_front, Vector3([0.0, 1.0, 0.0])))
        self.camera_up = vector.normalise(vector3.cross(self.camera_right, self.camera_front))
        self.camera_pos = self.camera_pos + self.camera_right * self.move_vector.x * self.move_speed
        self.camera_pos = self.camera_pos + self.camera_up * self.move_vector.y * self.move_speed
        self.camera_pos = self.camera_pos + self.camera_front * self.move_vector.z * self.move_speed

    def set_size(self, width: float, height: float):
        self.projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100)
        pass

    def move(self, direction: Vector3):
        self.move_vector.x = self.move_vector.x if self.move_vector.x != 0 else direction.x
        self.move_vector.y = self.move_vector.y if self.move_vector.y != 0 else direction.y
        self.move_vector.z = self.move_vector.z if self.move_vector.z != 0 else direction.z

    def stop(self, direction: Vector3):
        self.move_vector.x = 0 if self.move_vector.x == direction.x else self.move_vector.x
        self.move_vector.y = 0 if self.move_vector.y == direction.y else self.move_vector.y
        self.move_vector.z = 0 if self.move_vector.z == direction.z else self.move_vector.z

    def set_position(self, camera_position_index: int):
        if camera_position_index == 2:
            self.camera_pos = Vector3([4.0, 0.0, 0.0])
            self.camera_front = Vector3(vector.normalise(Vector3([0.0, 0.0, 0.0]) - self.camera_pos))
            self.camera_pos = self.base + self.camera_pos
            self.camera_up = Vector3([0.0, 1.0, 0.0])
            self.camera_right = Vector3([0.0, 0.0, -1.0])
            self.set_yaw_pitch_from_front()
        elif camera_position_index == 3:
            self.camera_pos = Vector3([0.0, 0.0, 3.5])
            self.camera_front = Vector3(vector.normalise(Vector3([0.0, 0.0, 0.0]) - self.camera_pos))
            self.camera_pos = self.base + self.camera_pos
            self.camera_up = Vector3([0.0, 1.0, 0.0])
            self.camera_right = Vector3([-1.0, 0.0, 0.0])
            self.set_yaw_pitch_from_front(False)
        elif camera_position_index == 4:
            self.camera_pos = Vector3([0.0, 0.0, -3.5])
            self.camera_front = Vector3(vector.normalise(Vector3([0.0, 0.0, 0.0]) - self.camera_pos))
            self.camera_pos = self.base + self.camera_pos
            self.camera_up = Vector3([0.0, 1.0, 0.0])
            self.camera_right = Vector3([1.0, 0.0, 0.0])
            self.set_yaw_pitch_from_front(False)
        elif camera_position_index == 5:
            self.camera_pos = Vector3([-3.5, 0.0, -3.5])
            self.camera_front = Vector3(vector.normalise(Vector3([0.0, 0.0, 0.0]) - self.camera_pos))
            self.camera_pos = self.base + self.camera_pos
            self.camera_up = Vector3([0.0, 1.0, 0.0])
            self.camera_right = Vector3([1.0, 0.0, 1.0])
            self.set_yaw_pitch_from_front(False)
        elif camera_position_index == 6:
            self.camera_pos = Vector3([-3.5, 0.0, 3.5])
            self.camera_front = Vector3(vector.normalise(Vector3([0.0, 0.0, 0.0]) - self.camera_pos))
            self.camera_pos = self.base + self.camera_pos
            self.camera_up = Vector3([0.0, 1.0, 0.0])
            self.camera_right = Vector3([1.0, 0.0, 1.0])
            self.set_yaw_pitch_from_front(False)
        elif camera_position_index == 7:
            self.camera_pos = Vector3([-3.0, 3.0, -3.0])
            self.camera_front = Vector3(vector.normalise(Vector3([0.0, 0.0, 0.0]) - self.camera_pos))
            self.camera_pos = self.base + self.camera_pos
            self.camera_up = Vector3([0.0, 1.0, 0.0])
            self.camera_right = Vector3([1.0, 0.0, 1.0])
            self.set_yaw_pitch_from_front(False)
        elif camera_position_index == 8:
            self.camera_pos = Vector3([-3.0, 3.0, 3.0])
            self.camera_front = Vector3(vector.normalise(Vector3([0.0, 0.0, 0.0]) - self.camera_pos))
            self.camera_pos = self.base + self.camera_pos
            self.camera_up = Vector3([0.0, 1.0, 0.0])
            self.camera_right = Vector3([1.0, 0.0, 1.0])
            self.set_yaw_pitch_from_front(False)
        else:
            self.camera_pos = Vector3([-4.0, 0.0, 0.0])
            self.camera_front = Vector3(vector.normalise(Vector3([0.0, 0.0, 0.0]) - self.camera_pos))
            self.camera_pos = self.base + self.camera_pos
            self.camera_up = Vector3([0.0, 1.0, 0.0])
            self.camera_right = Vector3([0.0, 0.0, 1.0])
            self.set_yaw_pitch_from_front()

    def set_yaw_pitch_from_front(self, use_x: bool = True):
        self.pitch = degrees(asin(self.camera_front.y))
        if use_x:
            self.yaw = degrees(acos(self.camera_front.x / cos(radians(self.pitch))))
        else:
            self.yaw = degrees(asin(self.camera_front.z / cos(radians(self.pitch))))
