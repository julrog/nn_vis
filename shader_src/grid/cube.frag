#version 120

void main() {
    gl_FragColor = vec4(1.0, gl_FragCoord.x/1920.0, gl_FragCoord.y/1080.0, 1.0);
}
