#version 440

layout(location = 0) in vec4 position;
layout(location = 1) in float size;

out vec3  vs_normal;
out float vs_discard;
out float vs_size;

uniform mat4 view;

void main()
{
    if (position.w == 0.0 || position.w > 1.0) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;
    }
    gl_Position = view * vec4(position.xyz, 1.0);
    vec4 new_normal = view * vec4((position.xyz + vec3(0.0, 1.0, 0.0)), 1.0);
    vs_normal = normalize(vec3(new_normal.xyz - gl_Position.xyz));
    vs_size = size;
}
