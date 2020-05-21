#version 440

layout(points) in;
in vec3 vs_normal[];
in float vs_discard[];
in float vs_size[];

layout(triangle_strip, max_vertices = 14) out;
flat out vec3 gs_sphere_position;
flat out float gs_sphere_radius;
out vec3 gs_cube_hit_position;

uniform mat4 projection;
uniform float sphere_radius;

void draw_vertex(vec3 offset)
{
    gs_cube_hit_position = gs_sphere_position + gs_sphere_radius * offset; // output
    gl_Position = projection * vec4(gs_cube_hit_position, 1.0); // output

    EmitVertex();
}

void main()
{
    gs_sphere_position = gl_in[0].gl_Position.xyz; // output
    gs_sphere_radius = sphere_radius * vs_size[0];

    if (vs_discard[0] == 0.0) {
        draw_vertex(vec3(-1.0, 1.0, -1.0));
        draw_vertex(vec3(1.0, 1.0, -1.0));
        draw_vertex(vec3(-1.0, -1.0, -1.0));
        draw_vertex(vec3(1.0, -1.0, -1.0));
        draw_vertex(vec3(1.0, -1.0, 1.0));
        draw_vertex(vec3(1.0, 1.0, -1.0));
        draw_vertex(vec3(1.0, 1.0, 1.0));
        draw_vertex(vec3(-1.0, 1.0, -1.0));
        draw_vertex(vec3(-1.0, 1.0, 1.0));
        draw_vertex(vec3(-1.0, -1.0, -1.0));
        draw_vertex(vec3(-1.0, -1.0, 1.0));
        draw_vertex(vec3(1.0, -1.0, 1.0));
        draw_vertex(vec3(-1.0, 1.0, 1.0));
        draw_vertex(vec3(1.0, 1.0, 1.0));
    }

    EndPrimitive();
}
