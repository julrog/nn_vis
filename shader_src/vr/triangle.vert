#version 410

uniform mat4 cameraToProjection;
uniform mat4 modelToCamera;

void main() {
    float angle = gl_VertexID * (3.14159*2.0/3.0);
    vec4 modelPos = vec4(cos(angle), sin(angle), -2.0, 1.0);
    gl_Position = cameraToProjection * (modelToCamera * modelPos);
}
