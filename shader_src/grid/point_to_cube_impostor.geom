#version 410

layout(points) in;
in float vs_density[];
in float vs_discard[];

layout(triangle_strip, max_vertices = 14) out;
flat out float gs_density;

uniform mat4 projection;
uniform mat4 view;
uniform float scale;

void draw_vertex(vec3 position, vec3 offset)
{
    gl_Position = projection * view * vec4((position + gs_density * offset) * scale, 1.0);

    EmitVertex();
}

void main()
{
    vec3 position = gl_in[0].gl_Position.xyz;
    gs_density = vs_density[0];

    if (vs_discard[0] == 0.0) {
        draw_vertex(position, vec3(-1.0, 1.0, -1.0));
        draw_vertex(position, vec3(1.0, 1.0, -1.0));
        draw_vertex(position, vec3(-1.0, -1.0, -1.0));
        draw_vertex(position, vec3(1.0, -1.0, -1.0));
        draw_vertex(position, vec3(1.0, -1.0, 1.0));
        draw_vertex(position, vec3(1.0, 1.0, -1.0));
        draw_vertex(position, vec3(1.0, 1.0, 1.0));
        draw_vertex(position, vec3(-1.0, 1.0, -1.0));
        draw_vertex(position, vec3(-1.0, 1.0, 1.0));
        draw_vertex(position, vec3(-1.0, -1.0, -1.0));
        draw_vertex(position, vec3(-1.0, -1.0, 1.0));
        draw_vertex(position, vec3(1.0, -1.0, 1.0));
        draw_vertex(position, vec3(-1.0, 1.0, 1.0));
        draw_vertex(position, vec3(1.0, 1.0, 1.0));
    }

    EndPrimitive();
}
