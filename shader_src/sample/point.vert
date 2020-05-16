#version 440 core

layout (location=0) in vec4 vPosition;

uniform mat4 projection;
uniform mat4 view;

flat out float vs_discard;

void main()
{
	if (vPosition.w == 0.0 || vPosition.w > 1.0) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;
    }
	gl_Position = projection * view * vec4(vPosition.xyz, 1.0);
}