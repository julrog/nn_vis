#version 440 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 next_position;
layout(location = 2) in vec4 edge_data_0;
layout(location = 3) in vec4 edge_data_1;
layout(location = 4) in vec4 edge_data_2;
layout(location = 5) in vec4 edge_data_3;
layout(location = 6) in vec4 edge_data_4;
layout(location = 7) in vec4 edge_data_5;
layout(location = 8) in vec4 edge_data_6;

flat out float vs_discard;
flat out vec3 vs_color;

uniform mat4 projection;
uniform mat4 view;
uniform int max_sample_points;

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
    if (position.w == 1.0) {
        vs_discard = 0.0;
    } else {
        vs_discard = 1.0;
    }
    gl_Position = projection * view * vec4(position.xyz, 1.0);

    float t = mod(gl_VertexID, max_sample_points)/edge_data_0.x;
    float importance[10];
    importance[0] = (1.0 - t) * (edge_data_2.x / edge_data_1.x) + t * (edge_data_4.z / edge_data_1.y);
    importance[1] = (1.0 - t) * (edge_data_2.y / edge_data_1.x) + t * (edge_data_4.w / edge_data_1.y);
    importance[2] = (1.0 - t) * (edge_data_2.z / edge_data_1.x) + t * (edge_data_5.x / edge_data_1.y);
    importance[3] = (1.0 - t) * (edge_data_2.w / edge_data_1.x) + t * (edge_data_5.y / edge_data_1.y);
    importance[4] = (1.0 - t) * (edge_data_3.x / edge_data_1.x) + t * (edge_data_5.z / edge_data_1.y);
    importance[5] = (1.0 - t) * (edge_data_3.y / edge_data_1.x) + t * (edge_data_5.w / edge_data_1.y);
    importance[6] = (1.0 - t) * (edge_data_3.z / edge_data_1.x) + t * (edge_data_6.x / edge_data_1.y);
    importance[7] = (1.0 - t) * (edge_data_3.w / edge_data_1.x) + t * (edge_data_6.y / edge_data_1.y);
    importance[8] = (1.0 - t) * (edge_data_4.x / edge_data_1.x) + t * (edge_data_6.z / edge_data_1.y);
    importance[9] = (1.0 - t) * (edge_data_4.y / edge_data_1.x) + t * (edge_data_6.w / edge_data_1.y);

    vs_color += color_0 * importance[0];
    vs_color += color_1 * importance[1];
    vs_color += color_2 * importance[2];
    vs_color += color_3 * importance[3];
    vs_color += color_4 * importance[4];
    vs_color += color_5 * importance[5];
    vs_color += color_6 * importance[6];
    vs_color += color_7 * importance[7];
    vs_color += color_8 * importance[8];
    vs_color += color_9 * importance[9];
}