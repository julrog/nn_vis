#version 440

out vec4 fColor;

uniform vec3 point_color;

void main()
{
     vec3 color = vec3(gl_FragCoord.x/1280.0, gl_FragCoord.y/900.0, 0.0);
	 fColor =  vec4( color.xyz * 0.5 + point_color.xyz * 0.5, 1.0 );
}