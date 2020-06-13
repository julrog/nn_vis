#version 440

flat in mat4 gs_local_ellipsoid_transformation;
flat in mat4 gs_local_ellipsoid_transformation_inverse;
flat in vec3 gs_local_ray_origin;
flat in vec3 gs_ellipsoid_radius;
flat in vec4 gs_color;
in vec3 gs_local_cuboid_hit_position;

out vec4 frag_color;
layout (depth_greater) out float gl_FragDepth;

uniform mat4 projection;
uniform mat4 view;
uniform float farthest_point_view_z;
uniform float nearest_point_view_z;
uniform float base_opacity = 0.1;
uniform float base_shpere_opacity = 0.2;
uniform float opacity_exponent = 1.0;

vec3 ellipsoidNormal(vec3 position, vec3 ellipsoid_radius)
{
    return normalize(position/ellipsoid_radius);
}

vec2 ellipsoidIntersect(vec3 ray_origin, vec3 ray_direction, vec3 ellipsoid_radius)
{
    vec3 ray_origin_scaled = ray_origin / ellipsoid_radius;
    vec3 ray_direction_scaled = ray_direction / ellipsoid_radius;

    float a = dot(ray_direction_scaled, ray_direction_scaled);
    float b = dot(ray_origin_scaled, ray_direction_scaled);
    float c = dot(ray_origin_scaled, ray_origin_scaled);
    float h = b * b - a * (c - 1.0);
    if (h < 0.0) discard; // no intersections
    float e = sqrt(h);
    return vec2((-b - e)/a, (-b + e)/a);
}

vec4 calculate_transparency_color(float depth, float density)
{
    float overall_opacity = pow(density, opacity_exponent);

    vec4 real_position_screen_furthest = projection * vec4(0.0, 0.0, farthest_point_view_z, 1.0);
    vec4 real_position_screen_nearest = projection * vec4(0.0, 0.0, nearest_point_view_z, 1.0);
    float relative_depth = (depth - real_position_screen_furthest.z)/(real_position_screen_nearest.z - real_position_screen_furthest.z);

    vec4 color = gs_color;
    color = vec4(color.x, color.y, color.z, (base_shpere_opacity * color.w + relative_depth * (1.0 - base_shpere_opacity)) * (overall_opacity - base_opacity) + base_opacity);
    color = vec4(color.x * color.w + (1.0 - color.w), color.y * color.w + (1.0 - color.w), color.z * color.w + (1.0 - color.w), color.w);
    return color;
}

void main()
{
    vec3 local_ray_direction = normalize(gs_local_cuboid_hit_position - gs_local_ray_origin);
    vec2 ray_intersection_distance = ellipsoidIntersect(gs_local_ray_origin, local_ray_direction, gs_ellipsoid_radius);
    vec3 local_hit_position = gs_local_ray_origin + ray_intersection_distance.x * local_ray_direction;
    vec3 local_hit_position_back = gs_local_ray_origin + ray_intersection_distance.y * local_ray_direction;

    vec3 real_hit_position = vec3(gs_local_ellipsoid_transformation_inverse * vec4(local_hit_position, 1.0));
    vec4 real_position_screen = projection * vec4(real_hit_position, 1.0);
    gl_FragDepth = 0.5 * (real_position_screen.z / real_position_screen.w) + 0.5;

    float density_view = dot(normalize(local_ray_direction/gs_ellipsoid_radius), normalize((-local_hit_position)/gs_ellipsoid_radius));
    float ellipsoid_density = clamp(density_view, 0.0, 1.0);

    frag_color = calculate_transparency_color(real_position_screen.z, ellipsoid_density);
}

