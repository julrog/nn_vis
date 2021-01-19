#version 440

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 next_position;
$$layout(location = $class_edge_buffer_group_location$) in vec4 edge_data_$class_edge_buffer_group_id$;$$

out vec3  vs_normal;
out float vs_discard;
out vec4 vs_next_position;
out float vs_importance;
out vec4 vs_color;
out float vs_edge;

uniform mat4 view;
uniform int max_sample_points;
uniform float importance_threshold = 0;
uniform float importance_max = 1.0;
uniform int edge_importance_type = 0;
uniform int show_class = -1;

$$const vec3 color_$class_id$ = $class_color$;$$


void main()
{
    if (position.w == 0.0 || position.w == -1.0 || importance_threshold >= edge_data_$class_importance_edge_buffer_id$ * edge_data_$class_start_average_edge_buffer_id$) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;

        vs_edge = 0.0;
        if(next_position.w <= 0.0) vs_edge = 1.0;
        if(position.w > 1.0) vs_edge = 1.0;
        if(mod(gl_InstanceID + 1, max_sample_points) >= edge_data_$class_samples_edge_buffer_id$ - 2.0) vs_edge = 1.0;

        vs_next_position = view * vec4(next_position.xyz, 1.0);
        gl_Position = view * vec4(position.xyz, 1.0);
        vec4 new_normal = view * vec4((position.xyz + vec3(0.0, 1.0, 0.0)), 1.0);
        vs_normal = normalize(vec3(new_normal.xyz - gl_Position.xyz));

        float importance[$num_classes$];
        if (edge_importance_type == 0) {
            float t = clamp(mod(gl_InstanceID + 1, max_sample_points)/edge_data_$class_samples_edge_buffer_id$, 0.0, 1.0);
            $$importance[$class_id$] = (1.0 - t) * edge_data_$class_importance_start_edge_buffer_id$/(edge_data_$class_start_average_edge_buffer_id$ * $num_classes$.0) + t * edge_data_$class_importance_end_edge_buffer_id$/(edge_data_$class_end_average_edge_buffer_id$ * $num_classes$.0);$$
            vs_importance =((1.0 - t) * edge_data_$class_start_average_edge_buffer_id$ + t * edge_data_$class_end_average_edge_buffer_id$) * edge_data_$class_importance_edge_buffer_id$;
        }
        if (edge_importance_type == 1) {
            $$importance[$class_id$] = edge_data_$class_importance_start_edge_buffer_id$/(edge_data_$class_start_average_edge_buffer_id$ * $num_classes$.0);$$
            vs_importance = edge_data_$class_start_average_edge_buffer_id$ * edge_data_$class_importance_edge_buffer_id$;
        }
        if (edge_importance_type == 2) {
            highp float divisor = (edge_data_$class_start_average_edge_buffer_id$ * $num_classes$.0 + edge_data_$class_end_average_edge_buffer_id$ * $num_classes$.0);
            $$importance[$class_id$] = (edge_data_$class_importance_start_edge_buffer_id$ + edge_data_$class_importance_end_edge_buffer_id$)/divisor;$$
            vs_importance = edge_data_$class_start_average_edge_buffer_id$ * edge_data_$class_end_average_edge_buffer_id$ * edge_data_$class_importance_edge_buffer_id$;
        }
        if (edge_importance_type == 3) {
            $$importance[$class_id$] = edge_data_$class_importance_end_edge_buffer_id$/(edge_data_$class_end_average_edge_buffer_id$ * $num_classes$.0);$$
            vs_importance = edge_data_$class_end_average_edge_buffer_id$ * edge_data_$class_importance_edge_buffer_id$;
        }

        vec3 color_list[$num_classes$];
        $$color_list[$class_id$] = color_$class_id$;$$

        vs_color = vec4(0.0, 0.0, 0.0, 0.0);
        if (show_class == 1) {
            vec3 combined_color = vec3(0.0, 0.0, 0.0);
            for (uint i = 0; i < $num_classes$; i++)
            {
                combined_color += color_list[i] * importance[i];
            }
            vs_color = vec4(combined_color, vs_importance/importance_max);
        } else {
            if (show_class == 0) {
                vs_color = vec4(0.0, 0.0, 0.0, vs_importance/importance_max);
            } else {
                vs_color = vec4(color_list[show_class - 2] * importance[show_class - 2], importance[show_class - 2]);
            }
        }
    }
}
