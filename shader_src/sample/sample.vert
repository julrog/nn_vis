#version 440 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 next_position;
$$layout(location = $r_edgebuffer_group_location$) in vec4 edge_data_$r_edgebuffer_group_id$;$$

flat out float vs_discard;
flat out vec4 vs_color;
out float vs_importance;

uniform mat4 projection;
uniform mat4 view;
uniform int max_sample_points;
uniform float importance_threshold = 0;
uniform int edge_importance_type = 0;
uniform int show_class = -1;

$$const vec3 color_$r_class_id$ = $r_class_color$;$$

void main()
{
    if (position.w == 0.0 || position.w == -1.0 || importance_threshold >= edge_data_$edgebuffer_importance$ * edge_data_$edgebuffer_start_average$) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;

        gl_Position = projection * view * vec4(position.xyz, 1.0);

        float importance[$num_classes$];
        if (edge_importance_type == 0) {
            float t = clamp(mod(gl_InstanceID + 1, max_sample_points)/edge_data_$edgebuffer_samples$, 0.0, 1.0);
            $$importance[$r_class_id$] = (1.0 - t) * edge_data_$r_edgebuffer_start_class_importance$/(edge_data_$edgebuffer_start_average$ * $num_classes$.0) + t * edge_data_$r_edgebuffer_end_class_importance$/(edge_data_$edgebuffer_end_average$ * $num_classes$.0);$$
            vs_importance =((1.0 - t) * edge_data_$edgebuffer_start_average$ + t * edge_data_$edgebuffer_end_average$) * edge_data_$edgebuffer_importance$;
        }
        if (edge_importance_type == 1) {
            $$importance[$r_class_id$] = edge_data_$r_edgebuffer_start_class_importance$/(edge_data_$edgebuffer_start_average$ * $num_classes$.0);$$
            vs_importance = edge_data_$edgebuffer_start_average$ * edge_data_$edgebuffer_importance$;
        }
        if (edge_importance_type == 2) {
            highp float divisor = (edge_data_$edgebuffer_start_average$ * $num_classes$.0 + edge_data_$edgebuffer_end_average$ * $num_classes$.0);
            $$importance[$r_class_id$] = (edge_data_$r_edgebuffer_start_class_importance$ + edge_data_$r_edgebuffer_end_class_importance$)/divisor;$$
            vs_importance = edge_data_$edgebuffer_start_average$ * edge_data_$edgebuffer_end_average$ * edge_data_$edgebuffer_importance$;
        }
        if (edge_importance_type == 3) {
            $$importance[$r_class_id$] = edge_data_$r_edgebuffer_end_class_importance$/(edge_data_$edgebuffer_end_average$ * $num_classes$.0);$$
            vs_importance = edge_data_$edgebuffer_end_average$ * edge_data_$edgebuffer_importance$;
        }

        vec3 color_list[$num_classes$];
        $$color_list[$r_class_id$] = color_$r_class_id$;$$

        vs_color = vec4(0.0, 0.0, 0.0, 0.0);
        if (show_class == 1) {
            vec3 combined_color = vec3(0.0, 0.0, 0.0);
            for (uint i = 0; i < $num_classes$; i++)
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
