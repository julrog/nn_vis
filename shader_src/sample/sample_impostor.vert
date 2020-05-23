#version 440

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 next_position;
layout(location = 2) in vec4 edge_data_0;
layout(location = 3) in vec4 edge_data_1;

out vec3  vs_normal;
out float vs_discard;
out vec4 vs_next_position;
out float vs_importance;

uniform mat4 view;

void main()
{
    if (position.w < 1.0 || next_position.w > 1.0) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;
    }
    vs_importance = edge_data_0.w;
    vs_next_position = view * vec4(next_position.xyz, 1.0);
    gl_Position = view * vec4(position.xyz, 1.0);
    vec4 new_normal = view * vec4((position.xyz + vec3(0.0, 1.0, 0.0)), 1.0);
    vs_normal = normalize(vec3(new_normal.xyz - gl_Position.xyz));
}
