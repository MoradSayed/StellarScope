#version 330 core

in vec2 fragmentTexCoord;

out vec4 fragmentColor;

uniform sampler2D material;

const float offset = 1.0 / 300.0;
const vec2 offsets[9] = vec2[](
    vec2(-offset,  offset), // top-left
    vec2( 0.0f,    offset), // top-center
    vec2( offset,  offset), // top-right
    vec2(-offset,  0.0f),   // center-left
    vec2( 0.0f,    0.0f),   // center-center
    vec2( offset,  0.0f),   // center-right
    vec2(-offset, -offset), // bottom-left
    vec2( 0.0f,   -offset), // bottom-center
    vec2( offset, -offset)  // bottom-right    
);

const float sobel_dx_kernel[9] = float[](
    1, 0, -1,
    2, 0, -2,
    1, 0, -1
);

const float sobel_dy_kernel[9] = float[](
     1,  2,  1,
     0,  0,  0,
    -1, -2, -1
);

const float gauss_blur_kernel[9] = float[](
    1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
    2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
    1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0
);

vec4 Monochrome(vec3 tintColor) {
    vec4 color = texture(material, fragmentTexCoord);
    float average = 0.299*color.r + 0.587*color.g + 0.114*color.b;
    return vec4(tintColor * vec3(average), 1.0);
}

vec4 Edge(vec3 tintColor) {

    vec3 dx = vec3(0);
    vec3 dy = vec3(0);
    for(int i = 0; i < 9; i++)
    {
        dx += sobel_dx_kernel[i] * vec3(texture(material, fragmentTexCoord.st + offsets[i]));
        dy += sobel_dy_kernel[i] * vec3(texture(material, fragmentTexCoord.st + offsets[i]));
    }

    float r = sqrt(dx.r * dx.r + dy.r * dy.r);
    float g = sqrt(dx.g * dx.g + dy.g * dy.g);
    float b = sqrt(dx.b * dx.b + dy.b * dy.b);
    
    return vec4(tintColor * vec3(r, g, b), 1.0);
}

vec4 Blur(vec3 tintColor) {

    vec3 color = vec3(0);

    for(int i = 0; i < 9; i++)
    {
        color += gauss_blur_kernel[i] * vec3(texture(material, fragmentTexCoord.st + offsets[i]));
    }
    
    return vec4(tintColor * color, 1.0);
}

vec4 Identity() {
    return texture(material, fragmentTexCoord);
}

void main() {
    //fragmentColor = 0.5 * Monochrome(vec3(1,1,1)) + 0.5 * Edge(vec3(0,1,0));
    //fragmentColor = Identity();
    fragmentColor = Identity()*vec4(0.3, 0.788, 0.69, 1.0);
}