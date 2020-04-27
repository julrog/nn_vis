#version 440

flat in float vs_density;

out vec4 fColor;

void main()
{
    vec3 base_color = vec3(1.0, 1.0, 1.0);
    vec3 color = vec3(1.0 * vs_density/100.0, gl_FragCoord.x/1920.0 * vs_density/100.0, gl_FragCoord.y/1080.0 * vs_density/100.0);
    fColor =  vec4(color * 0.9 + 0.1 * base_color, 1.0);
}