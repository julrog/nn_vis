#version 440

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 node_data_0;
layout(location = 2) in vec4 node_data_1;
layout(location = 3) in vec4 node_data_2;

out vec3  vs_normal;
out float vs_discard;
out float vs_size;
out vec3 vs_color;

uniform mat4 view;

const vec3 color_0 = vec3(0.133, 0.545, 0.133);
const vec3 color_1 = vec3(0, 0, 0.545);
const vec3 color_2 = vec3(0.69, 0.188, 0.376);
const vec3 color_3 = vec3(1, 0.271, 0);
const vec3 color_4 = vec3(1, 1, 0);
const vec3 color_5 = vec3(0.871, 0.722, 0.529);
const vec3 color_6 = vec3(0, 1, 0);
const vec3 color_7 = vec3(0, 1, 1);
const vec3 color_8 = vec3(1, 0, 1);
const vec3 color_9 = vec3(0.392, 0.584, 0.929);


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
    vs_size = node_data_2.w;

    vs_color += color_0 * (node_data_0.x / (node_data_2.z * 10));
    vs_color += color_1 * (node_data_0.y / (node_data_2.z * 10));
    vs_color += color_2 * (node_data_0.z / (node_data_2.z * 10));
    vs_color += color_3 * (node_data_0.w / (node_data_2.z * 10));
    vs_color += color_4 * (node_data_1.x / (node_data_2.z * 10));
    vs_color += color_5 * (node_data_1.y / (node_data_2.z * 10));
    vs_color += color_6 * (node_data_1.z / (node_data_2.z * 10));
    vs_color += color_7 * (node_data_1.w / (node_data_2.z * 10));
    vs_color += color_8 * (node_data_2.x / (node_data_2.z * 10));
    vs_color += color_9 * (node_data_2.y / (node_data_2.z * 10));
}
