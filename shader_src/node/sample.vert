#version 440 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 node_data_0;
layout(location = 2) in vec4 node_data_1;
layout(location = 3) in vec4 node_data_2;

flat out float vs_discard;
flat out float vs_size;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    if (position.w == 0.0 || position.w > 1.0) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;
    }
    gl_Position = projection * view * vec4(position.xyz, 1.0);
    vs_size = node_data_2.w;
}