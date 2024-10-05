#version 330 core

in vec2 fragmentTexCoord;
flat in int instanceTextureIndex;

out vec4 color;

uniform sampler2D imageTextures[5];

void main()
{
    color = texture(imageTextures[instanceTextureIndex], vec2(fragmentTexCoord.s, 1-fragmentTexCoord.t));
}
