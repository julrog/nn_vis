#version 440

uniform mat4  projection;
uniform mat4  view;

flat in vec3  gs_atom_position_cam;
flat in float gs_atom_radius;
flat in vec3  gs_normal;

in vec3 gs_hit_position_cam;

out vec4 frag_color;
layout (depth_greater) out float gl_FragDepth;


const vec3  light_direction_cam = normalize(vec3(-1.0, 1.0, 1.0));
const vec3  atom_color_diffuse  = vec3(0.8, 0.8, 0.8);
const vec3  atom_color_ambient  = vec3(0.2, 0.2, 0.2);
const vec3  atom_color_specular = vec3(0.3, 0.3, 0.3);


vec3 sphereIntersection(bool back)
{
    vec3 ray_direction = normalize(gs_hit_position_cam);

    float a = dot(ray_direction, -gs_atom_position_cam);
    float b = a * a - (dot(gs_atom_position_cam, gs_atom_position_cam) - gs_atom_radius * gs_atom_radius);

    if (b < 0) discard;// no intersections

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

    frag_color = vec4(atom_color_ambient +
    clamp(atom_color_diffuse * max(dot(normal, light_direction_cam), 0.0), 0.0, 1.0) +
    clamp(atom_color_specular * pow(max(dot(reflect(-light_direction_cam, normal), ray_direction), 0.0), 2.0), 0.0, 1.0), 1.0);

    vec4 real_position_screen = projection * vec4(real_hit_position_cam, 1.0);
    gl_FragDepth = 0.5 * (real_position_screen.z / real_position_screen.w) + 0.5;
}

