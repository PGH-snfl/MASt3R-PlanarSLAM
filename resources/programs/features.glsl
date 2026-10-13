#version 330
#if defined VERTEX_SHADER
in vec3 in_position;
in vec3 in_color;
uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;
uniform float point_size;
out vec3 color;
void main() {
    gl_Position = m_proj * m_camera * m_model * vec4(in_position, 1.0);
    gl_Position.z -= 0.00005 * gl_Position.w;
    gl_PointSize = point_size;
    color = in_color;
}
#elif defined FRAGMENT_SHADER
in vec3 color;
out vec4 fragColor;
void main() {
    float r = length(gl_PointCoord - vec2(0.5));
    if (r > 0.5) discard;
    fragColor = vec4(r > 0.36 ? vec3(0.0) : color, 1.0);
}
#endif
