#version 440

layout(points) in;
in float       vs_atom_radius[];
in vec3        vs_normal[];
in float       vs_discard[];

layout(triangle_strip, max_vertices = 14) out;
flat out vec3  gs_atom_position_cam;
flat out float gs_atom_radius;
flat out vec3  gs_normal;
out vec3       gs_hit_position_cam;


uniform mat4   projection;


void draw_vertex(vec3 offset)
{
    gs_hit_position_cam  = gs_atom_position_cam;
    gs_hit_position_cam += gs_atom_radius * offset;

    gl_Position = projection * vec4(gs_hit_position_cam, 1.0);

    EmitVertex();
}

void main()
{
    gs_atom_position_cam = gl_in[0].gl_Position.xyz;
    gs_atom_radius = vs_atom_radius[0];
    gs_normal = vs_normal[0];

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
