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
uniform float aspect; // ratio of width to hight


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

        zp2 *= float(power * power) * pow(z2,power-1); 
        zp2 = max(zp2, 1e-6);
        z = qpow(z, power) + c;
        z2 = dot(z,z);
        if (z2 > bailout * bailout) {
            break;
        }
    }
    
    float dist = sqrt(z2/zp2)*log(z2)*0.25;
    return dist - (offset-0.001);
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
float raymarch(vec3 ro, vec3 rd, out vec3 pHit, float maxDist) {
    float t = 0.0;
    float d = 1.0;
    for (int i = 0; i < 500; ++i) {
        if (d < 0.001) return t;
        if (t > maxDist) break;
        pHit = ro + rd * t;
        //d = min(sceneSDF(pHit),1.0);
        
        d = min(sceneSDF(pHit),1.0);
        // if (d > 5e5){
        //     return 1e6;
        // }
        t += d;
        
        //if (d > 1.0) d = 1.0;
        
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

// Shade a surface point (diffuse + two lights + Blinn-Phong specular)
vec3 shadeSurface(vec3 p, vec3 n, vec3 ro) {
    vec3 lightDir1 = normalize(vec3(1.0, 2.0, 1.0));
    vec3 lightDir2 = normalize(vec3(-1.0, 1.0, 0.5));
    vec3 baseColor = vec3(1.0, 0.8, 0.6);

    float diff1 = max(dot(n, lightDir1), 0.0);
    float diff2 = max(dot(n, lightDir2), 0.0);
    float diffuse = diff1 * 0.9 + diff2 * 0.6;

    vec3 viewDir = normalize(ro - p);
    vec3 half1 = normalize(lightDir1 + viewDir);
    vec3 half2 = normalize(lightDir2 + viewDir);
    float spec1 = pow(max(dot(n, half1), 0.0), 64.0);
    float spec2 = pow(max(dot(n, half2), 0.0), 32.0);
    float specular = spec1 * 0.8 + spec2 * 0.5;

    float ambient = 0.12;
    vec3 col = baseColor * (ambient + diffuse) + vec3(1.0) * specular * 0.35;
    return col;
}

// Trace a reflection ray and return the shaded color (simple single-bounce)
vec3 traceReflection(vec3 p, vec3 rd, vec3 camPos) {
    vec3 origin = p + rd * 0.01;
    vec3 hit;
    float t = raymarch(origin, rd, hit, 4.0 * length(camPos));
    if (t > 0.0) {
        vec3 n2 = estimateNormal(hit);
        return shadeSurface(hit, n2, camPos);
    }
    return vec3(0.02, 0.04, 0.06); // environment color
}

// Simple ambient occlusion by sampling along the normal
float ambientOcclusion(vec3 p, vec3 n) {
    float ao = 0.0;
    float sca = 1.0;
    for (int i = 1; i <= 5; i++) {
        float dist = sceneSDF(p + n * (float(i) * 0.02));
        ao += (float(i) * 0.02 - dist) * sca;
        sca *= 0.5;
    }
    ao = clamp(1.0 - ao, 0.0, 1.0);
    return ao;
}

vec2 correct_aspect(vec2 uv){ 
    if (aspect > 1){ // wide view, expand 
        return vec2(0.5+(uv.x-0.5)*aspect, uv.y);
    }else{
        return vec2(uv.x, 0.5 + (uv.y-0.5)/aspect);
    }
}
void main() {
    vec3 camPos = getCameraPos();
    //camPos = vec3(0.01,5,0);
    vec3 rayDir = getRayDir(correct_aspect(uv), camPos);
    vec3 pHit;
    float maxDist = 4.0 * length(camPos);
    float t = raymarch(camPos, rayDir, pHit, maxDist);
    if (t > 5e5){
        //fragcolor = vec4(1.0, 0.0, 0.0, 1.0); 
        fragColor = vec4(1.0, uv.y*0.1, 0.15, 1.0);
    } else if (t>0.0) {
        vec3 n = estimateNormal(pHit);
        float ao = ambientOcclusion(pHit, n);

        vec3 local = shadeSurface(pHit, n, camPos);

        vec3 reflDir = reflect(rayDir, n);
        vec3 reflColor = traceReflection(pHit, reflDir, camPos);

        vec3 viewDir = normalize(camPos - pHit);
        float fresnel = pow(1.0 - max(dot(n, viewDir), 0.0), 5.0);
        float reflectivity = mix(0.15, 0.4, fresnel);

        vec3 color = mix(local * ao, reflColor, reflectivity);
        fragColor = vec4(color,1.0);
    } else {
        fragColor = vec4(uv.x*0.1, uv.y*0.1, 0.15, 1.0);
    }
}
