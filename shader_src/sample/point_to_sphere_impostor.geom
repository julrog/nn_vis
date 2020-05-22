#version 440

layout(points) in;
in vec3 vs_normal[];
in float vs_discard[];

layout(triangle_strip, max_vertices = 14) out;
flat out vec3 gs_sphere_position;
out vec3 gs_cube_hit_position;

uniform mat4 projection;
uniform float sphere_radius;

void draw_vertex(float sphere_radius, vec3 offset)
{
    gs_cube_hit_position = gs_sphere_position + sphere_radius * offset;// output
    gl_Position = projection * vec4(gs_cube_hit_position, 1.0);// output

    EmitVertex();
}

void main()
{
    gs_sphere_position = gl_in[0].gl_Position.xyz;// output

    if (vs_discard[0] == 0.0) {
        draw_vertex(sphere_radius, vec3(-1.0, 1.0, -1.0));
        draw_vertex(sphere_radius, vec3(1.0, 1.0, -1.0));
        draw_vertex(sphere_radius, vec3(-1.0, -1.0, -1.0));
        draw_vertex(sphere_radius, vec3(1.0, -1.0, -1.0));
        draw_vertex(sphere_radius, vec3(1.0, -1.0, 1.0));
        draw_vertex(sphere_radius, vec3(1.0, 1.0, -1.0));
        draw_vertex(sphere_radius, vec3(1.0, 1.0, 1.0));
        draw_vertex(sphere_radius, vec3(-1.0, 1.0, -1.0));
        draw_vertex(sphere_radius, vec3(-1.0, 1.0, 1.0));
        draw_vertex(sphere_radius, vec3(-1.0, -1.0, -1.0));
        draw_vertex(sphere_radius, vec3(-1.0, -1.0, 1.0));
        draw_vertex(sphere_radius, vec3(1.0, -1.0, 1.0));
        draw_vertex(sphere_radius, vec3(-1.0, 1.0, 1.0));
        draw_vertex(sphere_radius, vec3(1.0, 1.0, 1.0));
    }

    EndPrimitive();
}
