#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform float width;
uniform float height;
uniform float depth;
uniform vec3 camera; // theta, phi, distance

// Simple camera setup
vec3 getCameraPos() {
    float theta = camera.x;
    float phi = camera.y;
    float r = camera.z;
    float x = r * sin(phi) * cos(theta);
    float y = r * cos(phi);
    float z = r * sin(phi) * sin(theta);
    return vec3(x, y, z);
}

// Ray direction from camera through screen
vec3 getRayDir(vec2 uv, vec3 camPos) {
    vec3 target = vec3(0.0, 0.0, 0.0);
    vec3 forward = normalize(target - camPos);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = cross(forward, right);
    float fov = 1.0;
    vec3 ray = normalize(forward + (uv.x - 0.5) * right * fov + (uv.y - 0.5) * up * fov);
    return ray;
}

// SDF for a box (prism)
float boxSDF(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

// Scene SDF
float sceneSDF(vec3 p) {
    return boxSDF(p, vec3(width, height, depth) * 0.5);
}

// Raymarching
float raymarch(vec3 ro, vec3 rd, out vec3 pHit) {
    float t = 0.0;
    for (int i = 0; i < 128; ++i) {
        pHit = ro + rd * t;
        float d = sceneSDF(pHit);
        if (d < 0.001) return t;
        t += d;
        if (t > 20.0) break;
    }
    return -1.0;
}

// Normal estimation
vec3 estimateNormal(vec3 p) {
    float eps = 0.001;
    return normalize(vec3(
        sceneSDF(p + vec3(eps, 0, 0)) - sceneSDF(p - vec3(eps, 0, 0)),
        sceneSDF(p + vec3(0, eps, 0)) - sceneSDF(p - vec3(0, eps, 0)),
        sceneSDF(p + vec3(0, 0, eps)) - sceneSDF(p - vec3(0, 0, eps))
    ));
}

// Lighting
vec3 getLighting(vec3 p, vec3 n, vec3 ro) {
    vec3 lightDir = normalize(vec3(1, 2, 1));
    float diff = max(dot(n, lightDir), 0.0);
    float ambient = 0.2;
    float spec = pow(max(dot(reflect(-lightDir, n), normalize(ro - p)), 0.0), 32.0);
    return vec3(1.0, 0.8, 0.6) * (ambient + diff) + vec3(1.0) * spec * 0.2;
}

void main() {
    vec3 camPos = getCameraPos();
    vec3 rayDir = getRayDir(uv, camPos);
    vec3 pHit;
    float t = raymarch(camPos, rayDir, pHit);
    if (t > 0.0) {
        vec3 n = estimateNormal(pHit);
        vec3 color = getLighting(pHit, n, camPos);
        fragColor = vec4(color, 1.0);
    } else {
        fragColor = vec4(0.1, 0.1, 0.15, 1.0);
    }
}
