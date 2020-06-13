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
out vec4 vs_color;

uniform mat4 view;
uniform int max_sample_points;
uniform float importance_threshold = 0;
uniform int edge_importance_type = 0;
uniform int show_class = -1;

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
    if (position.w == 0.0 || position.w == -1.0 || importance_threshold >= edge_data_0.w * edge_data_1.z) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;
        vs_next_position = view * vec4(next_position.xyz, 1.0);
        gl_Position = view * vec4(position.xyz, 1.0);
        vec4 new_normal = view * vec4((position.xyz + vec3(0.0, 1.0, 0.0)), 1.0);
        vs_normal = normalize(vec3(new_normal.xyz - gl_Position.xyz));

        float importance[10];
        if (edge_importance_type == 0) {
            float t = clamp(mod(gl_InstanceID + 1, max_sample_points)/edge_data_0.x, 0.0, 1.0);
            importance[0] = (1.0 - t) * edge_data_2.x/(edge_data_1.z * 10.0) + t * edge_data_4.z/(edge_data_1.w * 10.0);
            importance[1] = (1.0 - t) * edge_data_2.y/(edge_data_1.z * 10.0) + t * edge_data_4.w/(edge_data_1.w * 10.0);
            importance[2] = (1.0 - t) * edge_data_2.z/(edge_data_1.z * 10.0) + t * edge_data_5.x/(edge_data_1.w * 10.0);
            importance[3] = (1.0 - t) * edge_data_2.w/(edge_data_1.z * 10.0) + t * edge_data_5.y/(edge_data_1.w * 10.0);
            importance[4] = (1.0 - t) * edge_data_3.x/(edge_data_1.z * 10.0) + t * edge_data_5.z/(edge_data_1.w * 10.0);
            importance[5] = (1.0 - t) * edge_data_3.y/(edge_data_1.z * 10.0) + t * edge_data_5.w/(edge_data_1.w * 10.0);
            importance[6] = (1.0 - t) * edge_data_3.z/(edge_data_1.z * 10.0) + t * edge_data_6.x/(edge_data_1.w * 10.0);
            importance[7] = (1.0 - t) * edge_data_3.w/(edge_data_1.z * 10.0) + t * edge_data_6.y/(edge_data_1.w * 10.0);
            importance[8] = (1.0 - t) * edge_data_4.x/(edge_data_1.z * 10.0) + t * edge_data_6.z/(edge_data_1.w * 10.0);
            importance[9] = (1.0 - t) * edge_data_4.y/(edge_data_1.z * 10.0) + t * edge_data_6.w/(edge_data_1.w * 10.0);
            vs_importance =((1.0 - t) * edge_data_1.z + t * edge_data_1.w) * edge_data_0.w;
        }
        if (edge_importance_type == 1) {
            importance[0] = edge_data_2.x/(edge_data_1.z * 10.0);
            importance[1] = edge_data_2.y/(edge_data_1.z * 10.0);
            importance[2] = edge_data_2.z/(edge_data_1.z * 10.0);
            importance[3] = edge_data_2.w/(edge_data_1.z * 10.0);
            importance[4] = edge_data_3.x/(edge_data_1.z * 10.0);
            importance[5] = edge_data_3.y/(edge_data_1.z * 10.0);
            importance[6] = edge_data_3.z/(edge_data_1.z * 10.0);
            importance[7] = edge_data_3.w/(edge_data_1.z * 10.0);
            importance[8] = edge_data_4.x/(edge_data_1.z * 10.0);
            importance[9] = edge_data_4.y/(edge_data_1.z * 10.0);
            vs_importance = edge_data_1.z * edge_data_0.w;
        }
        if (edge_importance_type == 2) {
            highp float divisor = (edge_data_1.z * 10.0 + edge_data_1.w * 10.0);
            importance[0] = (edge_data_2.x + edge_data_4.z)/divisor;
            importance[1] = (edge_data_2.y + edge_data_4.w)/divisor;
            importance[2] = (edge_data_2.z + edge_data_5.x)/divisor;
            importance[3] = (edge_data_2.w + edge_data_5.y)/divisor;
            importance[4] = (edge_data_3.x + edge_data_5.z)/divisor;
            importance[5] = (edge_data_3.y + edge_data_5.w)/divisor;
            importance[6] = (edge_data_3.z + edge_data_6.x)/divisor;
            importance[7] = (edge_data_3.w + edge_data_6.y)/divisor;
            importance[8] = (edge_data_4.x + edge_data_6.z)/divisor;
            importance[9] = (edge_data_4.y + edge_data_6.w)/divisor;
            vs_importance = (edge_data_1.z + edge_data_1.w) * edge_data_0.w;
        }
        if (edge_importance_type == 3) {
            importance[0] = edge_data_4.z/(edge_data_1.w * 10.0);
            importance[1] = edge_data_4.w/(edge_data_1.w * 10.0);
            importance[2] = edge_data_5.x/(edge_data_1.w * 10.0);
            importance[3] = edge_data_5.y/(edge_data_1.w * 10.0);
            importance[4] = edge_data_5.z/(edge_data_1.w * 10.0);
            importance[5] = edge_data_5.w/(edge_data_1.w * 10.0);
            importance[6] = edge_data_6.x/(edge_data_1.w * 10.0);
            importance[7] = edge_data_6.y/(edge_data_1.w * 10.0);
            importance[8] = edge_data_6.z/(edge_data_1.w * 10.0);
            importance[9] = edge_data_6.w/(edge_data_1.w * 10.0);
            vs_importance = edge_data_1.w * edge_data_0.w;
        }

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

        vs_color = vec4(0.0, 0.0, 0.0, 0.0);
        if (show_class == 1) {
            vec3 combined_color = vec3(0.0, 0.0, 0.0);
            for (uint i = 0; i < 10; i++)
            {
                combined_color += color_list[i] * importance[i];
            }
            vs_color = vec4(combined_color, vs_importance);
        } else {
            if (show_class == 0) {
                vs_color = vec4(0.0, 0.0, 0.0, vs_importance);
            } else {
                vs_color = vec4(color_list[show_class - 2] * importance[show_class - 2], importance[show_class - 2]);
            }
        }
    }
}
