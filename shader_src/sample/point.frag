#version 440

flat in float vs_discard;

out vec4 fColor;

void main()
{
    if (vs_discard == 1.0) discard;
    vec3 color = vec3(1.0, gl_FragCoord.x/1920.0, gl_FragCoord.y/1080.0);
    fColor =  vec4(color, 1.0);
}