#version 440

uniform mat4  projection;
uniform mat4  view;
uniform float farthest_point_view_z;
uniform float nearest_point_view_z;

flat in vec3  gs_position_cam_a;
flat in vec3  gs_position_cam_b;
flat in vec3  ellipsoid_position;
flat in float gs_radius;
flat in vec3  gs_normal;
flat in mat4  local_ellipsoid_transformation;
flat in mat4  local_ellipsoid_transformation_inverse;
flat in vec3  local_ray_origin;
flat in vec3  ellipsoid_radius;
in vec3       local_hit_position;

out vec4 frag_color;
layout (depth_greater) out float gl_FragDepth;


const vec3  light_direction_cam = normalize(vec3(-1.0, 1.0, 1.0));
const vec3  atom_color_diffuse  = vec3(0.8, 0.8, 0.8);
const vec3  atom_color_ambient  = vec3(0.2, 0.2, 0.2);
const vec3  atom_color_specular = vec3(0.3, 0.3, 0.3);

struct Ellipsoid
{
    vec3 cen;
    vec3 rad;
};

vec3 eliNormal(in vec3 pos, in Ellipsoid sph)
{
    return normalize((pos-sph.cen)/sph.rad);
}

vec2 eliIntersect(in vec3 ro, in vec3 rd, in Ellipsoid sph)
{
    vec3 oc = ro - sph.cen;

    vec3 ocn = oc / sph.rad;
    vec3 rdn = rd / sph.rad;

    float a = dot(rdn, rdn);
    float b = dot(ocn, rdn);
    float c = dot(ocn, ocn);
    float h = b*b - a*(c-1.0);
    if (h<0.0) discard;
    float e = sqrt(h);
    return vec2((-b - e)/a, (-b + e)/a);
}

void main()
{
    vec3 local_ray_direction = normalize(local_hit_position - local_ray_origin);
    Ellipsoid ellipsoid = Ellipsoid(vec3(0.0, 0.0, 0.0), ellipsoid_radius);
    vec2 t2 = eliIntersect(local_ray_origin, local_ray_direction, ellipsoid);
    vec3 local_real_hit_position = local_ray_origin + t2.x * local_ray_direction;
    vec3 local_real_hit_position_back = local_ray_origin + t2.y * local_ray_direction;
    vec3 local_normal = eliNormal(local_real_hit_position, ellipsoid);

    vec3 real_hit_position = vec3(local_ellipsoid_transformation_inverse * vec4(local_real_hit_position, 1.0));
    float density_view = distance(local_real_hit_position, local_real_hit_position_back);
    float max_density_view = ellipsoid_radius.x;

    vec4 real_position_screen = projection * vec4(real_hit_position, 1.0);
    gl_FragDepth = 0.5 * (real_position_screen.z / real_position_screen.w) + 0.5;

    float base_opacity = 0.1;
    float base_shpere_opacity = 0.2;
    float sphere_density = clamp(density_view/max_density_view, 0.0, 1.0);
    vec4 real_position_screen_furthest = projection * vec4(0.0, 0.0, farthest_point_view_z, 1.0);
    vec4 real_position_screen_nearest = projection * vec4(0.0, 0.0, nearest_point_view_z, 1.0);
    float depth = (real_position_screen.z - real_position_screen_furthest.z)/(real_position_screen_nearest.z - real_position_screen_furthest.z);
    float overall_opacity = pow(sphere_density, 0.5);

    frag_color = vec4(frag_color.x, frag_color.y, frag_color.z, (base_shpere_opacity + depth * (1.0 - base_shpere_opacity)) * (overall_opacity - base_opacity) + base_opacity);
    frag_color = vec4(frag_color.x * frag_color.w + (1.0 - frag_color.w), frag_color.y * frag_color.w + (1.0 - frag_color.w), frag_color.z * frag_color.w + (1.0 - frag_color.w), frag_color.w);
}

