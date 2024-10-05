#version 330 core

layout (location=0) in vec3 vertexPos;
layout (location=1) in vec2 vertexTexCoord;
layout (location=2) in mat4 model;
layout (location=6) in vec2 tempclr_radius;

uniform mat4 view;
uniform mat4 projection;
uniform float size_op;

out vec2 fragmentTexCoord;
flat out int instanceTextureIndex;

void main()
{
    // gl_Position = projection * view * model * vec4(vertexPos*vec3(tempclr_radius.y), 1.0);
    vec3 vpos = mix(vertexPos, vertexPos*vec3(0.085), step(0.0, size_op));
    gl_Position = projection * view * model * vec4(vpos, 1.0);
    fragmentTexCoord = vertexTexCoord;
    instanceTextureIndex = int(tempclr_radius.x);
}
