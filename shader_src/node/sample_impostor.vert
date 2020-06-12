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
uniform int show_class = 0;
uniform float importance_threshold = 0;

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
    if (position.w == 0.0 || position.w > 1.0 || node_data_2.w == 0.0 || importance_threshold >= node_data_2.w) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;
    }
    gl_Position = view * vec4(position.xyz, 1.0);
    vec4 new_normal = view * vec4((position.xyz + vec3(0.0, 1.0, 0.0)), 1.0);
    vs_normal = normalize(vec3(new_normal.xyz - gl_Position.xyz));
    vs_size = node_data_2.w;

    float importance[10];
    importance[0] = node_data_0.x/(node_data_2.z * 10.0);
    importance[1] = node_data_0.y/(node_data_2.z * 10.0);
    importance[2] = node_data_0.z/(node_data_2.z * 10.0);
    importance[3] = node_data_0.w/(node_data_2.z * 10.0);
    importance[4] = node_data_1.x/(node_data_2.z * 10.0);
    importance[5] = node_data_1.y/(node_data_2.z * 10.0);
    importance[6] = node_data_1.z/(node_data_2.z * 10.0);
    importance[7] = node_data_1.w/(node_data_2.z * 10.0);
    importance[8] = node_data_2.x/(node_data_2.z * 10.0);
    importance[9] = node_data_2.y/(node_data_2.z * 10.0);

    vec3 color_list[10];
    color_list[0] = color_0;
    color_list[1] = color_1;
    color_list[2] = color_2;
    color_list[3] = color_3;
    color_list[4] = color_4;
    color_list[5] = color_5;
    color_list[6] = color_6;
    color_list[7] = color_7;
    color_list[8] = color_8;
    color_list[9] = color_9;

    vs_color = vec3(0.0, 0.0, 0.0);
    if (show_class == 0) {
        for (uint i = 0; i < 10; i++)
        {
            vs_color += color_list[i] * importance[i];
        }
    } else {
        vs_color += color_list[show_class - 1] * importance[show_class - 1];
    }
}
