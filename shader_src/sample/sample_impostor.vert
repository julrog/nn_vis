#version 410

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 next_position;
//$$layout(location = $r_edgebuffer_group_location$) in vec4 edge_data_$r_edgebuffer_group_id$;$$

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

//$$const vec3 color_$r_class_id$ = $r_class_color$;$$


void main()
{
    //$if (position.w == 0.0 || position.w == -1.0 || importance_threshold >= edge_data_$edgebuffer_importance$ * edge_data_$edgebuffer_start_average$) {
        vs_discard = 1.0;
    //$} else {
        vs_discard = 0.0;

        vs_edge = 0.0;
        if(next_position.w <= 0.0) vs_edge = 1.0;
        if(position.w > 1.0) vs_edge = 1.0;
        //$if(mod(gl_InstanceID + 1, max_sample_points) >= edge_data_$edgebuffer_samples$ - 2.0) vs_edge = 1.0;

        vs_next_position = view * vec4(next_position.xyz, 1.0);
        gl_Position = view * vec4(position.xyz, 1.0);
        vec4 new_normal = view * vec4((position.xyz + vec3(0.0, 1.0, 0.0)), 1.0);
        vs_normal = normalize(vec3(new_normal.xyz - gl_Position.xyz));

        //$float importance[$num_classes$];
        if (edge_importance_type == 0) {
            //$float t = clamp(mod(gl_InstanceID + 1, max_sample_points)/edge_data_$edgebuffer_samples$, 0.0, 1.0);
            //$$importance[$r_class_id$] = (1.0 - t) * edge_data_$r_edgebuffer_start_class_importance$/(edge_data_$edgebuffer_start_average$ * $num_classes$.0) + t * edge_data_$r_edgebuffer_end_class_importance$/(edge_data_$edgebuffer_end_average$ * $num_classes$.0);$$
            //$vs_importance =((1.0 - t) * edge_data_$edgebuffer_start_length$ + t * edge_data_$edgebuffer_end_length$) * edge_data_$edgebuffer_importance$;
        }
        if (edge_importance_type == 1) {
            //$$importance[$r_class_id$] = edge_data_$r_edgebuffer_start_class_importance$/(edge_data_$edgebuffer_start_average$ * $num_classes$.0);$$
            //$vs_importance = edge_data_$edgebuffer_start_length$ * edge_data_$edgebuffer_importance$;
        }
        if (edge_importance_type == 2) {
            //$highp float divisor = (edge_data_$edgebuffer_start_average$ * $num_classes$.0 + edge_data_$edgebuffer_end_average$ * $num_classes$.0);
            //$$importance[$r_class_id$] = (edge_data_$r_edgebuffer_start_class_importance$ + edge_data_$r_edgebuffer_end_class_importance$)/divisor;$$
            //$vs_importance = edge_data_$edgebuffer_start_length$ * edge_data_$edgebuffer_end_length$ * edge_data_$edgebuffer_importance$;
        }
        if (edge_importance_type == 3) {
            //$$importance[$r_class_id$] = edge_data_$r_edgebuffer_end_class_importance$/(edge_data_$edgebuffer_end_average$ * $num_classes$.0);$$
            //$vs_importance = edge_data_$edgebuffer_end_length$ * edge_data_$edgebuffer_importance$;
        }

        //$vec3 color_list[$num_classes$];
        //$$color_list[$r_class_id$] = color_$r_class_id$;$$

        vs_color = vec4(0.0, 0.0, 0.0, 0.0);
        if (show_class == 1) {
            vec3 combined_color = vec3(0.0, 0.0, 0.0);
            //$for (uint i = 0; i < $num_classes$; i++)
            {
                combined_color += color_list[i] * importance[i];
            }
            //$vs_color = vec4(combined_color, vs_importance/sqrt($num_classes$.0));
        } else {
            if (show_class == 0) {
                //$vs_color = vec4(0.0, 0.0, 0.0, vs_importance/sqrt($num_classes$.0));
            } else {
                vs_color = vec4(color_list[show_class - 2] * importance[show_class - 2], importance[show_class - 2]);
            }
        }
    //$}
}
