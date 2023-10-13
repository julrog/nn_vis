#version 410

layout(location = 0) in vec4 position;
//$$layout(location = $r_nodebuffer_group_location$) in vec4 node_data_$r_nodebuffer_group_id$;$$

out vec3  vs_normal;
out float vs_discard;
out float vs_size;
out vec4 vs_color;
out float vs_importance;

uniform mat4 view;
uniform int show_class = 0;
uniform float importance_threshold = 0;
uniform float importance_max = 1.0;
uniform float scale;

//$$const vec3 color_$r_class_id$ = $r_class_color$;$$

void main()
{
    //$if (position.w == 0.0 || position.w > 1.0 || node_data_$nodebuffer_length$ == 0.0 || importance_threshold >= node_data_$nodebuffer_length$) {
        vs_discard = 1.0;
    //$} else {
        vs_discard = 0.0;
        gl_Position = view * vec4(position.xyz * scale, 1.0);
        vec4 new_normal = view * vec4((position.xyz * scale + vec3(0.0, 1.0, 0.0)), 1.0);
        vs_normal = normalize(vec3(new_normal.xyz - gl_Position.xyz));
        //$vs_size = node_data_$nodebuffer_length$ * scale;
        //$vs_importance = node_data_$nodebuffer_average$;

        //$float importance[$num_classes$];
        //$$importance[$r_class_id$] = node_data_$r_nodebuffer_class_importance$/(node_data_$nodebuffer_average$ * $num_classes$.0);$$

        //$vec3 color_list[$num_classes$];
        //$$color_list[$r_class_id$] = color_$r_class_id$;$$

        vs_color = vec4(0.0, 0.0, 0.0, 0.0);
        if (show_class == 1) {
            vec3 combined_color = vec3(0.0, 0.0, 0.0);
            //$for (uint i = 0; i < $num_classes$; i++)
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
    //$}
}
