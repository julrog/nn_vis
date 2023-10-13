#version 410

uniform float screen_width;
uniform float screen_height;

void main() {
    gl_FragColor = vec4(1.0, gl_FragCoord.x/screen_width, gl_FragCoord.y/screen_height, 1.0);
}
