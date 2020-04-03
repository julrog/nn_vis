#version 440

out vec4 fColor;

uniform vec3 point_color;

void main()
{
	 fColor =  vec4( point_color, 1.0 );
}