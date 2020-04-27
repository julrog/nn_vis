#version 440

layout(points) in;
in float       vs_density[];
in float       vs_discard[];

layout(triangle_strip, max_vertices = 14) out;
flat out vec3  gs_position;
flat out float gs_density;


uniform mat4   projection;
uniform mat4   view;



void draw_vertex(vec3 offset)
{
    vec3 position = gs_position;
    position += gs_density * offset;

    gl_Position = projection * view * vec4(position, 1.0);

    EmitVertex();
}

void main()
{
    gs_position = gl_in[0].gl_Position.xyz;
    gs_density = vs_density[0];

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
