#version 410

flat in float vs_discard;

uniform float screen_width;
uniform float screen_height;

void main()
{
    if (vs_discard == 1.0) discard;
    vec3 base_color = vec3(1.0, 1.0, 1.0);
    vec3 color = vec3(gl_FragCoord.y/screen_height, 1.0 - ((gl_FragCoord.x/screen_width + gl_FragCoord.y/screen_height)/2.0), gl_FragCoord.x/screen_width);
    gl_FragColor =  vec4(color, 1.0);
}
