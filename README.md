# Pianoscope Visualizer

This page uses Shaders to to visualize the input audio of your device microphone. The goal is to play music and generate nice visuals that follow the soundwaves of the music.
const int max_iterations = 255;

vec2 complex_square(vec2 v) {
    return vec2(
        v.x * v.x - v.y * v.y,
        v.x * v.y * 2.0
    );
}

// Smooth transition for zoom
float smoothTransition(float time, float duration) {
    float t = mod(time, duration);
    return smoothstep(0.0, 1.0, t / duration);
}

// Get an actual spiral target based on the index
vec2 getSpiralTarget(int index) {
    // Based on careful observation of the image
    if (index == 0) return vec2(0.7, 0.0);    // Right spiral
    if (index == 1) return vec2(0.5, 0.5);    // Top-right spiral
    if (index == 2) return vec2(0.0, 0.7);    // Top spiral
    if (index == 3) return vec2(-0.5, 0.5);   // Top-left spiral
    if (index == 4) return vec2(-0.7, 0.0);   // Left spiral
    if (index == 5) return vec2(-0.5, -0.5);  // Bottom-left spiral
    if (index == 6) return vec2(0.0, -0.7);   // Bottom spiral
    return vec2(0.5, -0.5);                   // Bottom-right spiral
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Original coordinate setup from the baseline
    vec2 uv = fragCoord.xy - iResolution.xy * 0.5;
    uv *= 2.5 / min(iResolution.x, iResolution.y);
    
    // Select a target spiral based on time
    float cycleTime = 10.0; // Time for one complete zoom cycle
    int targetIndex = int(floor(mod(iTime / cycleTime, 8.0)));
    vec2 target = getSpiralTarget(targetIndex);
    
    // Zoom in during the first 70% of the cycle, then out during the remaining 30%
    float zoomPhase = mod(iTime, cycleTime) / cycleTime;
    float zoomFactor;
    
    if (zoomPhase < 0.7) {
        // Zoom in (first 70% of cycle)
        zoomFactor = mix(1.0, 20.0, smoothstep(0.0, 1.0, zoomPhase / 0.7));
    } else {
        // Zoom out (last 30% of cycle)
        zoomFactor = mix(20.0, 1.0, smoothstep(0.0, 1.0, (zoomPhase - 0.7) / 0.3));
    }
    
    // Apply the zoom, keeping target centered
    vec2 zoomedUV = (uv - target) / zoomFactor + target;

    // Use the exact Julia set calculation from the baseline
    vec2 c = vec2(0.285, 0.01);
    vec2 v = zoomedUV;
    
    int count = max_iterations;
    for (int i = 0; i < max_iterations; i++) {
        v = c + complex_square(v);
        if (dot(v, v) > 4.0) {
            count = i;
            break;
        }
    }

    // Enhanced coloring to better match the image - blue spirals on dark background
    float smooth_iter = float(count);
    
    // Add smooth coloring for better detail
    if (count < max_iterations) {
        smooth_iter = float(count) - log2(log2(dot(v, v))) + 4.0;
    }
    
    float normalized = smooth_iter / float(max_iterations);
    vec3 color = vec3(0.0, normalized * 0.8, normalized * 1.2);
    
    fragColor = vec4(color, 1.0);
}