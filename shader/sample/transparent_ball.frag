#version 440

flat in vec3  gs_atom_position_cam;
flat in float gs_atom_radius;
flat in vec3  gs_normal;
in vec3 gs_hit_position_cam;

out vec4 frag_color;
layout (depth_greater) out float gl_FragDepth;

uniform mat4  projection;
uniform mat4  view;
uniform float farthest_point_view_z;
uniform float nearest_point_view_z;


vec3 sphereIntersection(bool back)
{
    vec3 ray_direction = normalize(gs_hit_position_cam);

    float a = dot(ray_direction, -gs_atom_position_cam);
    float b = a * a - length(gs_atom_position_cam) * length(gs_atom_position_cam) + gs_atom_radius * gs_atom_radius;

    if (b < 0) discard;// no intercections

    float d = -a;
    if (back) d -= sqrt(b); // + for backside
    else d -= sqrt(b);  // - for frontside
    return d * ray_direction;
}

void main()
{
    vec3 ray_direction = normalize(gs_hit_position_cam);
    vec3 real_hit_position_cam = sphereIntersection(false);
    vec3 normal = normalize(real_hit_position_cam - gs_atom_position_cam);

    vec4 real_position_screen = projection * vec4(real_hit_position_cam, 1.0);
    gl_FragDepth = 0.5 * (real_position_screen.z / real_position_screen.w) + 0.5;

    frag_color = vec4(0.0, 0.0, 0.0, 1.0);
    float base_opacity = 0.1;
    float base_shpere_opacity = 0.2;
    float sphere_density = clamp((normal.z), 0.0, 1.0);
    vec4 real_position_screen_furthest = projection * vec4(0.0, 0.0, farthest_point_view_z, 1.0);
    vec4 real_position_screen_nearest = projection * vec4(0.0, 0.0, nearest_point_view_z, 1.0);
    float depth = (real_position_screen.z - real_position_screen_furthest.z)/(real_position_screen_nearest.z - real_position_screen_furthest.z);
    float overall_opacity = pow(sphere_density, 0.5);

    frag_color = vec4(frag_color.x, frag_color.y, frag_color.z, (base_shpere_opacity + depth * (1.0 - base_shpere_opacity)) * (overall_opacity - base_opacity) + base_opacity);
    frag_color = vec4(frag_color.x * frag_color.w + (1.0 - frag_color.w), frag_color.y * frag_color.w + (1.0 - frag_color.w), frag_color.z * frag_color.w + (1.0 - frag_color.w), frag_color.w);
}

