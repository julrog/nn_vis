#version 440

layout(lines) in;
in float       vs_radius[];
in vec3        vs_normal[];
in float       vs_discard[];

layout(triangle_strip, max_vertices = 14) out;
flat out vec3  gs_position_cam_a;
flat out vec3  gs_position_cam_b;
flat out vec3  ellipsoid_position;
flat out float gs_radius;
flat out vec3  gs_normal;
flat out mat4  local_ellipsoid_transformation;
flat out mat4  local_ellipsoid_transformation_inverse;
flat out vec3  local_ray_origin;
flat out vec3  ellipsoid_radius;
out vec3       local_hit_position;


uniform mat4   projection;


void draw_vertex(vec3 mid, vec3 right, vec3 up, vec3 front, vec3 offset)
{
    vec3 hit_position = mid;
    hit_position = hit_position + (offset.x) * right + (offset.x * normalize(right) + offset.y * up + offset.z * front) * gs_radius;
    local_hit_position = vec3(local_ellipsoid_transformation * vec4(hit_position, 1.0));
    gl_Position = projection * vec4(hit_position, 1.0);

    EmitVertex();
}

void main()
{
    gs_position_cam_a = gl_in[0].gl_Position.xyz;
    gs_position_cam_b = gl_in[1].gl_Position.xyz;
    gs_radius = vs_radius[0];
    gs_normal = vs_normal[0];

    ellipsoid_position = (gs_position_cam_a + gs_position_cam_b)/2.0;
    vec3 line_right = normalize(ellipsoid_position - gs_position_cam_a);
    vec3 line_up = gs_normal;
    vec3 line_front = normalize(cross(line_right, line_up));
    line_up = normalize(cross(line_right, line_front));
    vec3 right_offset = (ellipsoid_position - gs_position_cam_a);
    right_offset = (length(right_offset) + gs_radius) * normalize(right_offset);

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

    local_ellipsoid_transformation = local_ellipsoid_coord * local_ellipsoid_translation;
    local_ellipsoid_transformation_inverse = from_local_ellipsoid_translation * from_local_ellipsoid_coord;
    local_ray_origin = vec3(local_ellipsoid_transformation * vec4(0.0, 0.0, 0.0, 1.0));
    ellipsoid_radius = vec3(length(right_offset), gs_radius, gs_radius);

    if (vs_discard[1] == 0.0 && ellipsoid_radius.x > gs_radius) {
        draw_vertex(ellipsoid_position, right_offset, line_up, line_front, vec3(-1.0, 1.0, -1.0));
        draw_vertex(ellipsoid_position, right_offset, line_up, line_front, vec3(1.0, 1.0, -1.0));
        draw_vertex(ellipsoid_position, right_offset, line_up, line_front, vec3(-1.0, -1.0, -1.0));
        draw_vertex(ellipsoid_position, right_offset, line_up, line_front, vec3(1.0, -1.0, -1.0));
        draw_vertex(ellipsoid_position, right_offset, line_up, line_front, vec3(1.0, -1.0, 1.0));
        draw_vertex(ellipsoid_position, right_offset, line_up, line_front, vec3(1.0, 1.0, -1.0));
        draw_vertex(ellipsoid_position, right_offset, line_up, line_front, vec3(1.0, 1.0, 1.0));
        draw_vertex(ellipsoid_position, right_offset, line_up, line_front, vec3(-1.0, 1.0, -1.0));
        draw_vertex(ellipsoid_position, right_offset, line_up, line_front, vec3(-1.0, 1.0, 1.0));
        draw_vertex(ellipsoid_position, right_offset, line_up, line_front, vec3(-1.0, -1.0, -1.0));
        draw_vertex(ellipsoid_position, right_offset, line_up, line_front, vec3(-1.0, -1.0, 1.0));
        draw_vertex(ellipsoid_position, right_offset, line_up, line_front, vec3(1.0, -1.0, 1.0));
        draw_vertex(ellipsoid_position, right_offset, line_up, line_front, vec3(-1.0, 1.0, 1.0));
        draw_vertex(ellipsoid_position, right_offset, line_up, line_front, vec3(1.0, 1.0, 1.0));
    }

    EndPrimitive();
}
