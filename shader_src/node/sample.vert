#version 410 core

layout(location = 0) in vec4 position;
//$$layout(location = $r_nodebuffer_group_location$) in vec4 node_data_$r_nodebuffer_group_id$;$$

flat out float vs_discard;
flat out float vs_size;

uniform mat4 projection;
uniform mat4 view;
uniform float scale;

void main()
{
    if (position.w == 0.0 || position.w > 1.0) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;
    }
    gl_Position = projection * view * vec4(position.xyz * scale, 1.0);
    //vs_size = node_data_$nodebuffer_length$;
}
