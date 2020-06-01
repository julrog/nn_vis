#version 440

flat in vec3  gs_sphere_position;
flat in float gs_sphere_radius;
flat in vec3 gs_color;
in vec3 gs_cube_hit_position;

out vec4 frag_color;
layout (depth_greater) out float gl_FragDepth;

uniform mat4  projection;

const vec3 light_direction_cam = normalize(vec3(-1.0, 1.0, -1.0));
const vec3 atom_color_diffuse  = vec3(0.8, 0.8, 0.8);
const vec3 atom_color_ambient  = vec3(0.2, 0.2, 0.2);
const vec3 atom_color_specular = vec3(0.1, 0.1, 0.1);


vec2 sphereIntersection(vec3 ray_direction)
{
    float a = dot(ray_direction, -gs_sphere_position);
    float b = a * a - (dot(gs_sphere_position, gs_sphere_position) - gs_sphere_radius * gs_sphere_radius);

    if (b < 0) discard; // no intersections

    float d = -a;
    float e = sqrt(b);
    return vec2(d - e, d + e);
}

void main()
{
    vec3 ray_direction = normalize(gs_cube_hit_position);
    vec2 ray_intersection_distance = sphereIntersection(ray_direction);
    vec3 real_hit_position = ray_direction * ray_intersection_distance.x;
    vec3 normal = normalize(real_hit_position - gs_sphere_position);

    frag_color = vec4(atom_color_ambient +
    clamp(gs_color * max(dot(normal, light_direction_cam), 0.0), 0.0, 1.0) +
    clamp(atom_color_specular * pow(max(dot(reflect(light_direction_cam, normal), ray_direction), 0.0), 4.0), 0.0, 1.0), 1.0);

    vec4 real_position_screen = projection * vec4(real_hit_position, 1.0);
    gl_FragDepth = 0.5 * (real_position_screen.z / real_position_screen.w) + 0.5;
}

