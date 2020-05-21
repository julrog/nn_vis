#version 440

flat in vec3  gs_sphere_position;
flat in float gs_sphere_radius;
in vec3 gs_cube_hit_position;

out vec4 frag_color;
layout (depth_greater) out float gl_FragDepth;

uniform mat4  projection;
uniform float farthest_point_view_z;
uniform float nearest_point_view_z;

const float base_opacity = 0.1;
const float base_shpere_opacity = 0.2;
const float opacity_exponent = 1.0;

vec2 sphereIntersection(vec3 ray_direction)
{
    float a = dot(ray_direction, -gs_sphere_position);
    float b = a * a - (dot(gs_sphere_position, gs_sphere_position) - gs_sphere_radius * gs_sphere_radius);

    if (b < 0) discard;// no intercections

    float d = -a;
    float e = sqrt(b);
    return vec2(d - e, d + e);
}

vec4 calculate_transparency_color(float depth, float density)
{
    float overall_opacity = pow(density, opacity_exponent);

    vec4 real_position_screen_furthest = projection * vec4(0.0, 0.0, farthest_point_view_z, 1.0);
    vec4 real_position_screen_nearest = projection * vec4(0.0, 0.0, nearest_point_view_z, 1.0);
    float relative_depth = (depth - real_position_screen_furthest.z)/(real_position_screen_nearest.z - real_position_screen_furthest.z);

    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    color = vec4(color.x, color.y, color.z, (base_shpere_opacity + relative_depth * (1.0 - base_shpere_opacity)) * (overall_opacity - base_opacity) + base_opacity);
    color = vec4(color.x * color.w + (1.0 - color.w), color.y * color.w + (1.0 - color.w), color.z * color.w + (1.0 - color.w), color.w);
    return color;
}

void main()
{
    vec3 ray_direction = normalize(gs_cube_hit_position);
    vec2 ray_intersection_distance = sphereIntersection(ray_direction);
    vec3 real_hit_position = ray_direction * ray_intersection_distance.x;
    vec3 real_hit_position_back = ray_direction * ray_intersection_distance.y;

    vec4 real_position_screen = projection * vec4(real_hit_position, 1.0);
    gl_FragDepth = 0.5 * (real_position_screen.z / real_position_screen.w) + 0.5;

    float ray_depth = distance(real_hit_position, real_hit_position_back) / (2.0 * gs_sphere_radius);
    frag_color = calculate_transparency_color(real_position_screen.z, ray_depth);
}

