#version 410

layout(points) in;
in vec3 vs_normal[];
in float vs_discard[];
in float vs_size[];
in vec4 vs_color[];

layout(triangle_strip, max_vertices = 14) out;
flat out vec3 gs_sphere_position;
flat out float gs_sphere_radius;
flat out vec4 gs_color;
out vec3 gs_cube_hit_position;

uniform mat4 projection;
uniform float object_radius;

void draw_vertex(vec3 offset)
{
    gs_cube_hit_position = gs_sphere_position + gs_sphere_radius * offset; // output
    gl_Position = projection * vec4(gs_cube_hit_position, 1.0); // output

    EmitVertex();
}

void main()
{
    gs_sphere_position = gl_in[0].gl_Position.xyz; // output
    gs_sphere_radius = object_radius * vs_size[0];
    gs_color = vs_color[0];

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
