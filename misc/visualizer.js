const int max_iterations = 255;

vec2 complex_square(vec2 v) {
    return vec2(
        v.x * v.x - v.y * v.y,
        v.x * v.y * 2.0
    );
}

// Function to smoothly transition between values
float smoothTransition(float time, float duration, float delay) {
    float t = mod(time - delay, duration * 2.0);
    if (t < duration) {
        return smoothstep(0.0, 1.0, t / duration);
    } else {
        return smoothstep(1.0, 0.0, (t - duration) / duration);
    }
}

// Get coordinates of spiral structures in this Julia set
vec2 getSpiralTarget(int index) {
    index = index % 8; // Ensure we stay within bounds
    
    // These coordinates represent the spiral structures in the Julia set
    // based on mathematical analysis
    if (index == 0) return vec2(0.65, 0.0);      // 0°
    if (index == 1) return vec2(0.46, 0.46);     // 45°
    if (index == 2) return vec2(0.0, 0.65);      // 90°
    if (index == 3) return vec2(-0.46, 0.46);    // 135°
    if (index == 4) return vec2(-0.65, 0.0);     // 180°
    if (index == 5) return vec2(-0.46, -0.46);   // 225°
    if (index == 6) return vec2(0.0, -0.65);     // 270°
    return vec2(0.46, -0.46);                    // 315°
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Define cycle timing
    float cycleTime = 10.0; // Total cycle time in seconds
    int cycleIndex = int(floor(mod(iTime / cycleTime, 8.0))); // Cycle through 8 targets
    
    // Compute zoom transition for the current cycle
    float zoomPhase = smoothTransition(mod(iTime, cycleTime), cycleTime * 0.5, 0.0);
    float zoomFactor = mix(1.0, 30.0, zoomPhase);
    
    // Get the target for this cycle
    vec2 currentTarget = getSpiralTarget(cycleIndex);
    
    // Original coordinate setup from the baseline code
    vec2 uv = fragCoord.xy - iResolution.xy * 0.5;
    uv *= 2.5 / min(iResolution.x, iResolution.y);
    
    // Apply zoom with correct formula to keep target centered
    vec2 zoomedUV = (uv - currentTarget) / zoomFactor + currentTarget;

    // Julia set parameters from the baseline code
    vec2 c = vec2(0.285, 0.01);
    vec2 v = zoomedUV;
    float scale = 0.01;
    
    // Fractal computation from the baseline code
    int count = max_iterations;
    for (int i = 0; i < max_iterations; i++) {
        v = c + complex_square(v);
        if (dot(v, v) > 4.0) {
            count = i;
            break;
        }
    }

    // Use the original coloring but enhance it slightly for better visualization
    // The original used: fragColor = vec4(float(count) * scale);
    // We'll keep this basic approach but with more blue tint to match screenshots
    float intensity = float(count) * scale;
    
    // For the main spiral structures (not fully escaped)
    vec3 color = vec3(0.0, intensity * 1.2, intensity * 1.5);
    
    fragColor = vec4(color, 1.0);
}