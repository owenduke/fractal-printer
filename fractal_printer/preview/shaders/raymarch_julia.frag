#version 330 core
in vec2 uv;
out vec4 fragColor;

uniform float cx;
uniform float cy;
uniform float cz;
uniform float cw;
uniform float slice;
uniform int power;
uniform int iterations;
uniform int bailout;
uniform float offset;

uniform vec3 camera; // theta, phi, distance


vec4 qmul(vec4 a, vec4 b) {
    return vec4(
        a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
        a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
        a.x*b.z + a.z*b.x + a.w*b.y - a.y*b.w,
        a.x*b.w + a.w*b.x + a.y*b.z - a.z*b.y);
}
vec4 qpow(vec4 a, int n) {
    vec4 b = vec4(a);
    
    for (int i = 1; i < n; i++){
        b = qmul(b, a);
    }
    return b;
}
// SDF for a quaternionic julia set
float juliaSDF(vec3 p, vec4 c, float w, int power, int iterations, int bailout, float offset) {
    vec4 z = vec4(p,w);
    
    float z2 = dot(z, z);
    float zp2 = 1.0;

    for (int i = 0; i < iterations; i++) {
        zp2 = float(power * power) * pow(z2,power-1) * zp2;
        z = qpow(z, power) + c;
        z2 = dot(z,z);
        if (z2 > bailout * bailout) {
            break;
        }
    }

    float dist = sqrt(z2/zp2)*log(z2)*0.25;
    return dist - offset;
}

// Scene SDF
float sceneSDF(vec3 p) {
    float d = juliaSDF(p, vec4(cx,cy,cz,cw), slice, power, iterations, bailout, offset);
    // Alternative simple test shape (uncomment if you need a box):
    //return length(abs(p / vec3(1.0, 2.0, 4.0)) - vec3(1.0));
    //return min(length(p-vec3(0.0,1.0,0.0)) - 1, length(p-vec3(0.0,-1.0,0.0)) - 1.5);// + 0.000000001*sin(d);
    return d;
}

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

// Raymarching
float raymarch(vec3 ro, vec3 rd, out vec3 pHit) {
    float t = 0.0;
    //return sceneSDF(ro + rd * length(ro));
    for (int i = 0; i < 128; ++i) {
        pHit = ro + rd * t;
        float d = sceneSDF(pHit);
        if (d < 0.001) return t;
        if (d > 1.0) d = 1.0;
        t += d;
        if (t > 2*length(ro)) break;
    }
    return -1.0;
}

// Normal estimation
vec3 estimateNormal(vec3 p) {
    float eps = 0.001;
    //return normalize(p);
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
    //camPos = vec3(0.01,5,0);
    vec3 rayDir = getRayDir(uv, camPos);
    vec3 pHit;
    float t = raymarch(camPos, rayDir, pHit);
    //fragColor = vec4(0.5+sin(t)/2,float(t>0.0),0.0,1.0);
    if (t>0.0) {
        vec3 n = estimateNormal(pHit);
        vec3 color = getLighting(pHit, n, camPos);
        fragColor = vec4(color,1.0);
    } else {
        fragColor = vec4(uv.x*0.1, uv.y*0.1, 0.15, 1.0);
    }
}
