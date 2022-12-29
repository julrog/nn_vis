#version 410

layout(points) in;
in vec4 vs_next_position[];
in vec3 vs_normal[];
in float vs_discard[];
in float vs_importance[];
in vec4 vs_color[];
in float vs_edge[];


layout(triangle_strip, max_vertices = 14) out;
flat out mat4 gs_local_ellipsoid_transformation;
flat out mat4 gs_local_ellipsoid_transformation_inverse;
flat out vec3 gs_local_ray_origin;
flat out vec3 gs_ellipsoid_radius;
flat out vec4 gs_color;
out vec3 gs_local_cuboid_hit_position;

uniform mat4 projection;
uniform float object_radius;
uniform float scale;

void draw_vertex(vec3 position, vec3 right, vec3 up, vec3 front, vec3 offset)
{
    vec3 hit_position = position;
    hit_position = hit_position + (offset.x) * right + (offset.x * normalize(right) + offset.y * up + offset.z * front) * object_radius * scale * vs_importance[0];
    gs_local_cuboid_hit_position = vec3(gs_local_ellipsoid_transformation * vec4(hit_position, 1.0));// output
    gl_Position = projection * vec4(hit_position, 1.0);// output

    EmitVertex();
}

void main()
{
    if (vs_discard[0] == 0.0) {
        gs_color = vs_color[0];

        vec3 position_cam_a = gl_in[0].gl_Position.xyz;
        vec3 position_cam_b = vs_next_position[0].xyz;

        vec3 ellipsoid_position = (position_cam_a + position_cam_b)/2.0;// output
        vec3 line_right = normalize(ellipsoid_position - position_cam_a);
        vec3 line_up = vs_normal[0];
        vec3 line_front = normalize(cross(line_right, line_up));
        line_up = normalize(cross(line_right, line_front));

        float line_direction_scale = 1.0 + 1.5 * vs_importance[0];
        if(vs_edge[0] == 1.0) line_direction_scale = 1.0;

        vec3 line_right_offset = (ellipsoid_position - position_cam_a) * line_direction_scale;
        line_right_offset = (length(line_right_offset) + object_radius * scale * vs_importance[0]) * normalize(line_right_offset);

        mat4 local_ellipsoid_coord = mat4(0.0);
        local_ellipsoid_coord[0][0] = line_right.x;
        local_ellipsoid_coord[1][0] = line_right.y;
        local_ellipsoid_coord[2][0] = line_right.z;
        local_ellipsoid_coord[0][1] = line_up.x;
        local_ellipsoid_coord[1][1] = line_up.y;
        local_ellipsoid_coord[2][1] = line_up.z;
        local_ellipsoid_coord[0][2] = line_front.x;
        local_ellipsoid_coord[1][2] = line_front.y;
        local_ellipsoid_coord[2][2] = line_front.z;
        local_ellipsoid_coord[3][3] = 1;

        mat4 local_ellipsoid_translation = mat4(0.0);
        local_ellipsoid_translation[0][0] = 1;
        local_ellipsoid_translation[1][1] = 1;
        local_ellipsoid_translation[2][2] = 1;
        local_ellipsoid_translation[3][0] = -ellipsoid_position.x;
        local_ellipsoid_translation[3][1] = -ellipsoid_position.y;
        local_ellipsoid_translation[3][2] = -ellipsoid_position.z;
        local_ellipsoid_translation[3][3] = 1;

        mat4 from_local_ellipsoid_coord = mat4(0.0);
        from_local_ellipsoid_coord[0][0] = line_right.x;
        from_local_ellipsoid_coord[1][0] = line_up.x;
        from_local_ellipsoid_coord[2][0] = line_front.x;
        from_local_ellipsoid_coord[0][1] = line_right.y;
        from_local_ellipsoid_coord[1][1] = line_up.y;
        from_local_ellipsoid_coord[2][1] = line_front.y;
        from_local_ellipsoid_coord[0][2] = line_right.z;
        from_local_ellipsoid_coord[1][2] = line_up.z;
        from_local_ellipsoid_coord[2][2] = line_front.z;
        from_local_ellipsoid_coord[3][3] = 1;

        mat4 from_local_ellipsoid_translation = mat4(0.0);
        from_local_ellipsoid_translation[0][0] = 1;
        from_local_ellipsoid_translation[1][1] = 1;
        from_local_ellipsoid_translation[2][2] = 1;
        from_local_ellipsoid_translation[3][0] = ellipsoid_position.x;
        from_local_ellipsoid_translation[3][1] = ellipsoid_position.y;
        from_local_ellipsoid_translation[3][2] = ellipsoid_position.z;
        from_local_ellipsoid_translation[3][3] = 1;

        gs_local_ellipsoid_transformation = local_ellipsoid_coord * local_ellipsoid_translation;// output
        gs_local_ellipsoid_transformation_inverse = from_local_ellipsoid_translation * from_local_ellipsoid_coord;// output
        gs_local_ray_origin = vec3(gs_local_ellipsoid_transformation * vec4(0.0, 0.0, 0.0, 1.0));// output
        gs_ellipsoid_radius = vec3(length(line_right_offset), object_radius * scale * vs_importance[0], object_radius * scale * vs_importance[0]);// output

        if (gs_ellipsoid_radius.x > object_radius * scale * vs_importance[0]) {
            draw_vertex(ellipsoid_position, line_right_offset, line_up, line_front, vec3(-1.0, 1.0, -1.0));
            draw_vertex(ellipsoid_position, line_right_offset, line_up, line_front, vec3(1.0, 1.0, -1.0));
            draw_vertex(ellipsoid_position, line_right_offset, line_up, line_front, vec3(-1.0, -1.0, -1.0));
            draw_vertex(ellipsoid_position, line_right_offset, line_up, line_front, vec3(1.0, -1.0, -1.0));
            draw_vertex(ellipsoid_position, line_right_offset, line_up, line_front, vec3(1.0, -1.0, 1.0));
            draw_vertex(ellipsoid_position, line_right_offset, line_up, line_front, vec3(1.0, 1.0, -1.0));
            draw_vertex(ellipsoid_position, line_right_offset, line_up, line_front, vec3(1.0, 1.0, 1.0));
            draw_vertex(ellipsoid_position, line_right_offset, line_up, line_front, vec3(-1.0, 1.0, -1.0));
            draw_vertex(ellipsoid_position, line_right_offset, line_up, line_front, vec3(-1.0, 1.0, 1.0));
            draw_vertex(ellipsoid_position, line_right_offset, line_up, line_front, vec3(-1.0, -1.0, -1.0));
            draw_vertex(ellipsoid_position, line_right_offset, line_up, line_front, vec3(-1.0, -1.0, 1.0));
            draw_vertex(ellipsoid_position, line_right_offset, line_up, line_front, vec3(1.0, -1.0, 1.0));
            draw_vertex(ellipsoid_position, line_right_offset, line_up, line_front, vec3(-1.0, 1.0, 1.0));
            draw_vertex(ellipsoid_position, line_right_offset, line_up, line_front, vec3(1.0, 1.0, 1.0));
        }
    }

    EndPrimitive();
}
