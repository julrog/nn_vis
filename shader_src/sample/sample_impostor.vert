#version 440

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 next_position;
layout(location = 2) in vec4 edge_data_0;
layout(location = 3) in vec4 edge_data_1;
layout(location = 4) in vec4 edge_data_2;
layout(location = 5) in vec4 edge_data_3;
layout(location = 6) in vec4 edge_data_4;
layout(location = 7) in vec4 edge_data_5;
layout(location = 8) in vec4 edge_data_6;

out vec3  vs_normal;
out float vs_discard;
out vec4 vs_next_position;
out float vs_importance;
out vec3 vs_color;

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
    if (position.w < 1.0 || next_position.w > 1.0) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;
        vs_importance = edge_data_0.w * edge_data_1.z;
        vs_next_position = view * vec4(next_position.xyz, 1.0);
        gl_Position = view * vec4(position.xyz, 1.0);
        vec4 new_normal = view * vec4((position.xyz + vec3(0.0, 1.0, 0.0)), 1.0);
        vs_normal = normalize(vec3(new_normal.xyz - gl_Position.xyz));

        float t = clamp(mod(gl_InstanceID, max_sample_points)/edge_data_0.x, 0.0, 1.0);
        float importance[10] = float[10](
        (1.0 - t) * (edge_data_2.x / (edge_data_1.z * 10.0)) + t * (edge_data_4.z / (edge_data_1.w * 10.0)),
        (1.0 - t) * (edge_data_2.y / (edge_data_1.z * 10.0)) + t * (edge_data_4.w / (edge_data_1.w * 10.0)),
        (1.0 - t) * (edge_data_2.z / (edge_data_1.z * 10.0)) + t * (edge_data_5.x / (edge_data_1.w * 10.0)),
        (1.0 - t) * (edge_data_2.w / (edge_data_1.z * 10.0)) + t * (edge_data_5.y / (edge_data_1.w * 10.0)),
        (1.0 - t) * (edge_data_3.x / (edge_data_1.z * 10.0)) + t * (edge_data_5.z / (edge_data_1.w * 10.0)),
        (1.0 - t) * (edge_data_3.y / (edge_data_1.z * 10.0)) + t * (edge_data_5.w / (edge_data_1.w * 10.0)),
        (1.0 - t) * (edge_data_3.z / (edge_data_1.z * 10.0)) + t * (edge_data_6.x / (edge_data_1.w * 10.0)),
        (1.0 - t) * (edge_data_3.w / (edge_data_1.z * 10.0)) + t * (edge_data_6.y / (edge_data_1.w * 10.0)),
        (1.0 - t) * (edge_data_4.x / (edge_data_1.z * 10.0)) + t * (edge_data_6.z / (edge_data_1.w * 10.0)),
        (1.0 - t) * (edge_data_4.y / (edge_data_1.z * 10.0)) + t * (edge_data_6.w / (edge_data_1.w * 10.0)));

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

        /*uint max_importance_index = 0;
        float current_max = importance[max_importance_index];
        for (uint i = 1; i < 10; i++)
        {
            float current_value = importance[i];
            if (current_value >= current_max) {
                current_max = current_value;
                max_importance_index = i;
            }
        }
        vs_color += color_list[max_importance_index];*/

        for (uint i = 0; i < 10; i++)
        {
            vs_color += color_list[i] * importance[i];
        }
    }
}
