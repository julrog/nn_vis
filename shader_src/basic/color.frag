#version 440

flat in vec4 gs_color;

void main() {
    gl_FragColor = vec4(gs_color);
}
