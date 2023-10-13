#version 410

layout(points) in;
in vec3 vs_normal[];
in float vs_discard[];
in vec4 vs_next_position[];
in vec4 vs_color[];

flat out vec4 gs_color;

layout(line_strip, max_vertices = 2) out;

uniform mat4 projection;

void main()
{
    if (vs_discard[0] == 0.0) {
        gl_Position = projection * gl_in[0].gl_Position;
        gs_color = vs_color[0];
        EmitVertex();
        gl_Position = projection * vs_next_position[0];
        EmitVertex();
    }
    EndPrimitive();
}
