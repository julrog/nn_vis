#version 440

out vec4 fColor;

void
main()
{
	 fColor =  vec4( gl_FragCoord.xyz, 1.0 );
}