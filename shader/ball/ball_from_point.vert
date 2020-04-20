#version 440

layout(location = 0) in vec4 position;

out float vs_atom_radius;
out vec3  vs_normal;
out float vs_discard;


uniform mat4 view;


void main()
{
    vs_atom_radius = 0.02;
    vec4 new_normal = view * vec4((position.xyz + vec3(0.0, 1.0, 0.0)), 1.0);
    gl_Position = view * vec4(position.xyz, 1.0);
    vs_normal = vec3(new_normal.xyz - gl_Position.xyz);
    if (position.w == 0) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;
    }
}
