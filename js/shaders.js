
// Sample Shadertoy shaders to quickly test
export const SAMPLE_SHADERS = {
    "Primordial Soup": `
    // Helper function to apply neon psychedelic color palette
vec3 NeonPsychedelicColor(float t) {
    return vec3(
        0.5 + 0.5 * sin(6.0 * t + 1.0),
        0.4 + 0.5 * sin(66.0 * t + 2.4),
        0.5 + 0.5 * sin(6.0 * t + 1.7)
    );
}

// Modify the fractal function to be audioreactive
vec3 audioreactiveFractal(vec2 c, vec2 c2, float audioInput) {
    vec2 z = c;
    float ci = 0.0;
    float mean = 0.0;

    for (int i = 0; i < 64; i++) {
        vec2 a = vec2(z.x, abs(z.y));
        float b = atan(a.y, a.x);
        
        if (b > 0.0) b -= 6.283185307179586;
        
        // Use audioInput to modulate the fractal equation
        z = vec2(log(length(a + audioInput * 0.05)), b) + c2;
        
        if (i > 5) mean += length(z);
    }

    mean /= float(62);
    ci = 1.0 - log2(.05 * log2(mean / 1.0));

    return NeonPsychedelicColor(ci);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy - 0.5;
    uv.x *= iResolution.x / iResolution.y;
    float time = iTime;
    
    // Sample audio input from iChannel0
    float audioInput = texture(iChannel0, vec2(0.5, 0.5)).r * 2.0 - 1.0;

    uv *= 2.0; // Scale the UV coordinates for symmetry
    float angle = atan(uv.y, uv.x); // Calculate angle
    float radius = length(uv); // Calculate radius
    
    // Apply symmetry to the angle
    angle = mod(angle + time * 0.2, 2.0 * 3.14159265359);
    
    // Convert back to Cartesian coordinates
    uv = radius * vec2(cos(angle), sin(angle));

    // Apply rotation to uv based on time
    float rot = sin(time * 0.025) * 4.7;
    uv = vec2(uv.x * cos(rot) - uv.y * sin(rot), uv.x * sin(rot) + uv.y * cos(rot));

    // Apply julia set parameters
    float juliax = sin(time) * 0.021 + 0.12;
    float juliay = cos(time * 0.23) * 0.2 + 5.7;
    
    // Calculate the final color using the audioreactive fractal function
    fragColor = vec4(audioreactiveFractal(uv, vec2(juliax, juliay), audioInput), 1.0);
}
`,
"Fractured Portal":`precision highp float;

uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D iChannel0;
uniform vec4 iMouse;

// Constants for tweaking visual effects
const float SPEED = 0.5;          // Overall animation speed
const float ZOOM_FACTOR = 1.2;    // Zoom intensity
const float WARP_STRENGTH = 1.8;  // How much the audio warps the patterns
const float COMPLEXITY = 8.0;     // Visual complexity
const float SYMMETRY = 6.0;       // Number of symmetry folds
const int MAX_ITERATIONS = 8;     // Detail level (higher = more detailed but slower)

// Color palette function creating purple to gold transitions
vec3 palette(float t) {
    // Rich purple to gold palette
    vec3 purple = vec3(0.4, 0.0, 0.7);
    vec3 violet = vec3(0.6, 0.2, 0.8);
    vec3 gold = vec3(1.0, 0.8, 0.1);
    vec3 amber = vec3(1.0, 0.6, 0.0);
    
    // Smooth transitions between colors
    if (t < 0.33) {
        return mix(purple, violet, t * 3.0);
    } else if (t < 0.66) {
        return mix(violet, gold, (t - 0.33) * 3.0);
    } else {
        return mix(gold, amber, (t - 0.66) * 3.0);
    }
}

// Helper function to create swirling effects
vec2 rotate(vec2 p, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

// Audio reactivity helper functions
float getAudioLow() {
    // Sample low frequencies (bass)
    float bass = 0.0;
    for (int i = 0; i < 10; i++) {
        bass += texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
    }
    return bass * 0.15; // Normalize
}

float getAudioMid() {
    // Sample mid frequencies
    float mid = 0.0;
    for (int i = 10; i < 40; i++) {
        mid += texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
    }
    return mid * 0.04; // Normalize
}

float getAudioHigh() {
    // Sample high frequencies
    float high = 0.0;
    for (int i = 40; i < 80; i++) {
        high += texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
    }
    return high * 0.03; // Normalize
}

// Main fractal function
float fractalPattern(vec2 uv, float audioReactive) {
    float result = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    
    // Layer several sinusoidal patterns for complexity
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        // Create wave patterns
        result += amplitude * abs(sin(uv.x * frequency) * sin(uv.y * frequency));
        
        // Rotate and scale for each iteration
        uv = rotate(uv, 0.8 + audioReactive * 1.5);
        frequency *= 1.7 + audioReactive * 0.2;
        amplitude *= 0.65;
        
        // Apply audio-reactive zoom
        uv *= 1.1 + audioReactive * 0.1;
    }
    
    return result;
}

void main() {
    // Normalized coordinates
    vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / min(iResolution.x, iResolution.y);
    
    // Get audio values
    float audioLow = getAudioLow();
    float audioMid = getAudioMid();
    float audioHigh = getAudioHigh();
    float audioTotal = audioLow + audioMid + audioHigh;
    
    // Apply mouse interaction for manual control (if used)
    vec2 mouse = iMouse.xy / iResolution.xy - 0.5;
    if (iMouse.z > 0.0) {
        uv += mouse * 0.5;
    }
    
    // Time variables
    float time = iTime * SPEED;
    
    // Apply symmetrical folding (creates kaleidoscope effect)
    float angle = atan(uv.y, uv.x);
    float radius = length(uv);
    
    // Audio-reactive symmetry
    float folds = SYMMETRY + audioMid * 10.0;
    angle = mod(angle, 6.28318 / folds) - 3.14159 / folds;
    
    // Transform back to cartesian
    uv = vec2(cos(angle), sin(angle)) * radius;
    
    // Apply audio-reactive zoom
    float zoom = ZOOM_FACTOR + audioLow * 3.0;
    uv *= zoom * (1.0 + sin(time * 0.2) * 0.1);
    
    // Apply swirl effect
    float swirl = sin(radius * 5.0 - time) * (WARP_STRENGTH + audioTotal * 2.0);
    uv = rotate(uv, swirl);
    
    // Apply time-based motion
    uv += vec2(sin(time * 0.7), cos(time * 0.5)) * 0.2;
    
    // Calculate pattern with audio reactivity
    float pattern = fractalPattern(uv, audioTotal);
    
    // Add pulsing rings based on bass
    pattern += 0.2 * sin(radius * 20.0 - time * 4.0 + audioLow * 10.0);
    
    // Color mapping based on pattern value
    vec3 color = palette(pattern);
    
    // Add highlights based on high frequencies
    color += vec3(0.2, 0.1, 0.4) * audioHigh * 4.0;
    
    // Add gold shimmer
    color += vec3(0.4, 0.3, 0.0) * pow(sin(pattern * 40.0 + time) * 0.5 + 0.5, 5.0);
    
    // Add subtle vignette
    color *= smoothstep(1.8, 0.5, length(uv));
    
    // Output final color
    gl_FragColor = vec4(color, 1.0);
}`
,
"Precision Plasma Flower": `precision highp float;

uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D iChannel0;
uniform vec4 iMouse;

// ========== CONFIGURATION (feel free to tweak) ==========
const float SPEED = 0.7;               // Animation speed
const float KALEIDOSCOPE_SIDES = 7.0;  // Number of kaleidoscope reflections
const int MAX_ITERATIONS = 9;          // Fractal detail level
const float FEEDBACK_STRENGTH = 0.4;   // Visual feedback intensity
const float METALLIC_SHININESS = 3.0;  // Gold metallic effect intensity
const float FREQUENCY_SCALING = 1.5;   // Audio frequency intensity

// ========== AUDIO EXTRACTION FUNCTIONS ==========
// Extract bass frequencies (important for beat detection)
float getAudioBass() {
    float bass = 0.0;
    for (int i = 0; i < 15; i++) {
        float sample = texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        // Apply non-linear scaling for better dynamics
        bass += pow(sample, 1.4);
    }
    return min(bass * 0.12, 1.0); // Normalize with headroom
}

// Extract mid frequencies (vocals and melodic elements)
float getAudioMid() {
    float mid = 0.0;
    for (int i = 15; i < 45; i++) {
        float sample = texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        mid += sample * (1.0 + float(i - 15) * 0.01); // Weight higher mids more
    }
    return min(mid * 0.035, 1.0);
}

// Extract high frequencies (hi-hats, cymbals)
float getAudioHigh() {
    float high = 0.0;
    for (int i = 45; i < 100; i++) {
        high += texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
    }
    return min(high * 0.025, 1.0);
}

// Extract transients (sudden changes - good for percussive elements)
float getAudioTransients() {
    float current = 0.0;
    float previous = 0.0;
    
    for (int i = 5; i < 30; i++) {
        current += texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        previous += texture2D(iChannel0, vec2(float(i+1) / 128.0, 0.0)).x;
    }
    
    return min(max(0.0, (current - previous) * 5.0), 1.0);
}

// ========== VISUAL EFFECT FUNCTIONS ==========
// Create psychedelic color palette with purple and gold focus
vec3 purpleGoldPalette(float t) {
    // Rich purple/gold spectrum
    const vec3 deepPurple = vec3(0.3, 0.0, 0.5);
    const vec3 richPurple = vec3(0.5, 0.0, 0.8);
    const vec3 neonPurple = vec3(0.7, 0.3, 0.9);
    const vec3 magenta = vec3(0.9, 0.4, 0.7);
    const vec3 amber = vec3(1.0, 0.6, 0.0);
    const vec3 gold = vec3(1.0, 0.8, 0.2);
    
    t = fract(t); // Ensure we loop through the palette
    
    // Multi-point gradient
    if (t < 0.2) return mix(deepPurple, richPurple, t * 5.0);
    else if (t < 0.4) return mix(richPurple, neonPurple, (t - 0.2) * 5.0);
    else if (t < 0.6) return mix(neonPurple, magenta, (t - 0.4) * 5.0);
    else if (t < 0.8) return mix(magenta, amber, (t - 0.6) * 5.0);
    else return mix(amber, gold, (t - 0.8) * 5.0);
}

// Advanced rotation with distortion
vec2 rotateDistort(vec2 p, float angle, float distortion) {
    float s = sin(angle);
    float c = cos(angle);
    
    // Apply non-linear distortion based on radius
    float r = length(p);
    float distortionFactor = 1.0 + distortion * sin(r * 3.0);
    
    return vec2(
        p.x * c * distortionFactor - p.y * s,
        p.x * s + p.y * c * distortionFactor
    );
}

// Domain warping function - makes patterns more organic
vec2 warpDomain(vec2 p, float time, float strength) {
    // Primary warping
    p.x += strength * sin(p.y * 1.5 + time * 0.8);
    p.y += strength * 0.8 * sin(p.x * 1.7 + time * 0.7);
    
    // Secondary higher-frequency warping
    p.x += strength * 0.3 * sin(p.y * 5.0 + time * 1.2);
    p.y += strength * 0.2 * sin(p.x * 7.0 + time * 1.1);
    
    return p;
}

// Generate metallic highlights (for gold effect)
float metallicHighlight(float pattern, float time, float shininess) {
    float phase = pattern * 20.0 + time;
    return pow(0.5 + 0.5 * sin(phase), shininess);
}

// Create a fractal feedback pattern
float fractalPattern(vec2 uv, float time, float audioTotal) {
    float pattern = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    vec2 p = uv;
    
    // Layer multiple octaves for fractal effect
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        // Apply audio-reactive rotation
        p = rotateDistort(p, time * 0.1 + float(i) * 0.2 + audioTotal * 2.0, 0.2);
        
        // Create psychedelic pattern
        float wave = sin(p.x * freq) * sin(p.y * freq);
        wave *= sin(length(p) * freq * 0.5 + time);
        
        // Add to the total pattern
        pattern += amp * abs(wave);
        
        // Modify parameters for next iteration
        freq *= 1.8 + audioTotal * 0.5;
        amp *= 0.6;
        
        // Apply domain warping for non-linear complexity
        p = warpDomain(p, time * 0.2, 0.1 + audioTotal * 0.5);
    }
    
    return pattern;
}

// ========== MAIN SHADER FUNCTION ==========
void main() {
    // Screen coordinates normalized to [-1,1]
    vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / min(iResolution.x, iResolution.y);
    
    // Extract audio features
    float bass = getAudioBass() * FREQUENCY_SCALING;
    float mid = getAudioMid() * FREQUENCY_SCALING;
    float high = getAudioHigh() * FREQUENCY_SCALING;
    float transients = getAudioTransients() * FREQUENCY_SCALING;
    float audioTotal = (bass + mid + high) * 0.6;
    
    // Apply motion and dynamics
    float time = iTime * SPEED;
    
    // Create pulsing zoom effect synchronized with bass
    float zoom = 1.0 + 0.3 * sin(time * 0.7) + bass * 1.5;
    uv *= zoom;
    
    // Apply mouse interaction if active
    if (iMouse.z > 0.0) {
        vec2 mouseOffset = (iMouse.xy / iResolution.xy - 0.5) * 2.0;
        uv += mouseOffset * 0.5;
    }
    
    // Apply kaleidoscopic effect (symmetry folding)
    float sides = KALEIDOSCOPE_SIDES + floor(mid * 5.0); // Audio-reactive sides
    float angle = atan(uv.y, uv.x);
    float radius = length(uv);
    
    // Audio-reactive kaleidoscope
    float segment = 3.14159 * 2.0 / sides;
    angle = mod(angle, segment);
    angle = abs(angle - segment * 0.5);
    
    // Transform back to Cartesian coordinates
    uv = vec2(cos(angle), sin(angle)) * radius;
    
    // Apply audio-reactive domain warping
    uv = warpDomain(uv, time, 0.3 + bass * 0.8);
    
    // Apply spiral effect based on bass
    float spiralFactor = 0.5 + bass * 3.0;
    uv = rotateDistort(uv, radius * spiralFactor + time * 0.5, 0.3 + mid * 0.7);
    
    // Create reactive visual feedback loop
    vec2 feedbackUV = uv;
    feedbackUV = rotateDistort(feedbackUV, time * 0.05 + audioTotal, 0.1);
    feedbackUV *= 0.9 + 0.2 * sin(time * 0.1);
    float feedback = fractalPattern(feedbackUV, time * 0.5, audioTotal * 0.5);
    
    // Generate primary pattern
    float pattern = fractalPattern(uv, time, audioTotal);
    
    // Blend with feedback for more psychedelic effect
    pattern = mix(pattern, feedback, FEEDBACK_STRENGTH + bass * 0.3);
    
    // Add concentric rings modulated by audio
    pattern += 0.15 * sin(radius * (10.0 + mid * 20.0) - time * 2.0);
    
    // Add transient flashes
    pattern += transients * 0.5;
    
    // Map pattern to color palette
    vec3 color = purpleGoldPalette(pattern + time * 0.05);
    
    // Add metallic gold highlights
    float shine = metallicHighlight(pattern, time * 1.5, METALLIC_SHININESS);
    color += vec3(1.0, 0.9, 0.3) * shine * (0.5 + bass * 0.5);
    
    // Add purple glow based on high frequencies
    color += vec3(0.5, 0.0, 1.0) * high * 0.7;
    
    // Add subtle vignette for depth
    float vignette = smoothstep(1.5, 0.5, length(uv / zoom));
    color *= vignette;
    
    // Add subtle chromatic aberration for more psychedelic look
    float aberration = 0.01 + high * 0.01;
    vec3 colorShift;
    colorShift.r = purpleGoldPalette(pattern + aberration + time * 0.05).r;
    colorShift.b = purpleGoldPalette(pattern - aberration + time * 0.05).b;
    colorShift.g = color.g;
    color = mix(color, colorShift, 0.3);
    
    // Boost contrast
    color = pow(color, vec3(0.9 + bass * 0.2));
    
    // Final output
    gl_FragColor = vec4(color, 1.0);
}`
,
    "Plasma Ball": `#define SAMPLE_WIDTH 8.0

////////////////////////////////////////////////////
// Simple rotation function (used for swirling UV)
////////////////////////////////////////////////////
mat2 rotate2D(float a) {
    float s = sin(a), c = cos(a);
    return mat2(c, -s, s, c);
}

////////////////////////////////////////////////////
// A function to draw a horizontal "yLine" shape
////////////////////////////////////////////////////
vec3 yLine(vec2 uv, float y, float thickness) {
    // The shape is basically a brightness that spikes
    // near uv.y = -y, with "thickness" controlling how thick the line is
    float colwave = thickness / abs(uv.y + y);
    return vec3(colwave);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Normalize coordinates to [-1..1], preserving aspect ratio
    vec2 uv = (fragCoord.xy * 2.0 - iResolution.xy)
              / min(iResolution.x, iResolution.y);

    // We'll warp (swirl) the coordinate system based on time and distance
    float swirlStrength = 2.0;     // How aggressive the swirl is
    float distFromCenter = length(uv);
    float angle = swirlStrength * distFromCenter
                  * sin(iTime * 1.2 + distFromCenter * 8.0);
    uv = rotate2D(angle) * uv;

    // Base color
    vec3 col = vec3(0.0);

    //////////////////////////////////////////////////////////
    // AVERAGE AMPLITUDE: pulsing factor from audio samples
    //////////////////////////////////////////////////////////
    float avgAmp = 0.0;
    for (float i = 0.0; i < SAMPLE_WIDTH; ++i) {
        avgAmp += texelFetch(iChannel0, ivec2(8 + int(i), 1), 0).x
                  * (1.0 / SAMPLE_WIDTH);
    }
    // Map avgAmp from [0.1..0.9] -> [0..1.0] for stronger pulsing
    avgAmp = (clamp(avgAmp, 0.1, 0.9) - 0.1) * 1.25;

    //////////////////////////////////////////////////////////
    // BASE SHAPES (lines & circle)
    //////////////////////////////////////////////////////////
    // 1) A circular effect that grows/shrinks with avgAmp
    //    We make 'col' brighter near the center.
    col += length(uv) / max(avgAmp, 0.1);

    // 2) Multiple wavy lines that shift over time
    float t = iTime;
    col += yLine(uv, sin(uv.x + t * 1.0) + sin(uv.y + t * 1.0), 0.01);
    col += yLine(uv, sin(uv.x + t * 0.2) + cos(uv.y + t * 2.0), 0.01);
    col += yLine(uv, sin(uv.x + t * 4.0) + sin(uv.y + t * 0.5), 0.01);
    col += yLine(uv, cos(uv.x + t * 0.2) + sin(uv.y + t * 1.5), 0.01);

    // The original ring logic: clamp brightness to keep a ring shape
    col = max(-abs(col - 2.0) + 2.0, 0.0);

    //////////////////////////////////////////////////////////
    // SAMPLE DIFFERENT FREQUENCIES for R/G/B
    //////////////////////////////////////////////////////////
    float r = avgAmp * col.x;
    float g = texelFetch(iChannel0, ivec2(250,0), 0).x * col.y;
    float b = texelFetch(iChannel0, ivec2(500,0), 0).x * col.z;

    //////////////////////////////////////////////////////////
    // BOOST COLOR VARIATION:
    //  - If one channel is the smallest, reduce it further to avoid
    //    constant white
    //////////////////////////////////////////////////////////
    if (r < g && r < b) {
        r *= 1.0 - max(g - r, b - r);
    }
    if (g < r && g < b) {
        g *= 1.0 - max(r - g, b - g);
    }
    if (b < r && b < g) {
        b *= 1.0 - max(r - b, g - b);
    }

    // Combine with our shape
    vec3 finalColor = col * vec3(r, g, b);

    //////////////////////////////////////////////////////////
    // EXTRA "CRAZY" EFFECTS
    //////////////////////////////////////////////////////////

    // 1) Color Cycling: swirl the channels over time
    //    We'll do a wave-based channel swap
    float colorCycleSpeed = 2.0;
    float mixVal = 0.5 + 0.5 * sin(iTime * colorCycleSpeed + distFromCenter * 10.0);

    // newColor = finalColor rotated among channels
    vec3 newColor = vec3(finalColor.g, finalColor.b, finalColor.r);
    finalColor = mix(finalColor, newColor, mixVal);

    // 2) A subtle strobe that pulses brightness
    float strobeSpeed = 6.0;
    float strobe = 0.5 + 0.5 * sin(iTime * strobeSpeed);
    finalColor *= (0.8 + 0.2 * strobe);

    // 3) Slight tinted glow at the edges
    //    We'll push alpha based on distance from center
    //    (just for fun, though we only output finalColor's RGB).
    float glow = smoothstep(0.8, 1.5, distFromCenter) * 0.5;
    finalColor += glow * finalColor;

    // Output final color
    fragColor = vec4(finalColor, 1.0);
}
`,
    "Audio Reactive Fractal": `
    void mainImage(out vec4 fragColor, in vec2 fragCoord)
    {
        // Time and screen-space coordinates
        float T = iTime;
        vec2 r = iResolution.xy;
        vec2 u = (fragCoord * 2.0 - r) / r.y;
    
        // Audio reactivity: Sample audio signal
        float audio = texture2D(iChannel0, vec2(0.01, fragCoord.y / r.y)).r;
    
        // Mouse interaction or dynamic motion when mouse is not active
        vec2 m = iMouse.xy;
        if (iMouse.z < 0.5) {
            m = (vec2(
                     sin(T * 0.3) * sin(T * 0.17) + sin(T * 0.3),
                     (1.0 - cos(T * 0.632)) * sin(T * 0.131) + cos(T * 0.3)) +
                 1.0) *
                r;
        }
    
        // Fractal center position
        vec2 p = (m - r) / r.y;
    
        // Fractal calculation variables
        float f = 3.0, g = f, d;
        for (int i = 0; i < 20; i++) {
            // Fractal symmetry
            u = vec2(u.x, -u.y) / dot(u, u) + p;
            u.x = abs(u.x);
    
            // Max and min accumulations for brightness and glow
            f = max(f, dot(u - p, u - p));
            g = min(g, sin(dot(u + p, u + p)) + 1.0);
        }
    
        // Color palette and glow dynamics
        f = abs(-log(f) / 3.5);
        g = abs(-log(g) / 8.0);
        vec3 col = vec3(g, g * f, f);
    
        // Add pulsation and audio modulation
        col += 0.2 * audio * sin(T * 3.0 + col * 10.0);
        col *= 1.5 + 0.5 * audio; // Amplify brightness with audio
    
        // Final color output
        fragColor = vec4(min(col, 1.0), 1.0);
    }
    `,

    "Plasma": `

#define SAMPLE_WIDTH 8.0

vec3 yLine(vec2 uv,float y, float thickness){
    float colwave = thickness /  abs(uv.y+y);
    return vec3(colwave);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord) { 
    vec2 uv = (fragCoord.xy * 2.0 - iResolution.xy) / min(iResolution.x, iResolution.y);

    vec3 col = vec3(0.0);
    
    // Average together some of the lower samples to get the circle to pulse to the music.
    float avgAmp = 0.0;
    for (float i = 0.0; i < SAMPLE_WIDTH; ++i)
    {
        avgAmp += texelFetch(iChannel0, ivec2(8 + int(i), 1), 0).x * (1.0 / SAMPLE_WIDTH);
    }
    // Most times the average Amplitude is between [0.1, 0.9], so we map that range to 0, 1
    // for a fuller pulsing effect.
    avgAmp = (clamp(avgAmp, 0.1, 0.9) - 0.1) * 1.25;
    
    col += length(uv) / max(avgAmp, 0.1);
    col += yLine(uv, sin(uv.x + iTime * 1.0) + sin(uv.y + iTime * 1.0), 0.01);
    col += yLine(uv, sin(uv.x + iTime * 0.2) + cos(uv.y + iTime * 2.0), 0.01);
    col += yLine(uv, sin(uv.x + iTime * 4.0) + sin(uv.y + iTime * 0.5), 0.01);
    col += yLine(uv, cos(uv.x + iTime * 0.2) + sin(uv.y + iTime * 1.5), 0.01);
    // In the original the color keeps increasing past the edge of the circle so the whole screen is white,
    // this makes the color falloff back to zero the brighter it gets so we get a ring.
    col = max(-abs(col - 2.0) + 2.0, 0.0);

    // Change ivec2 x values to sample different frequencies.
    float r = avgAmp * col.x;
    float g = texelFetch(iChannel0, ivec2(250,0), 0).x * col.y;
    float b = texelFetch(iChannel0, ivec2(500,0), 0).x * col.z;
    
    // Takes the lowest color value and reduces it by the min difference between the other two color channels.
    // This is done to have colors pop more often and only be white when all frequencies are around the same
    // amplitude.
    if (r < g && r < b)
    {
        r *= 1.0 - max(g - r, b - r);
    }
    
    if (g < r && g < b)
    {
        g *= 1.0 - max(r - g, b - g);
    }
    
    if (b < r && b < g)
    {
        b *= 1.0 - max(r - b, g - b);
    }
    
    vec3 finalColor = col * vec3(r,g,b);
    fragColor = vec4(finalColor, 1.0);
}`,

    "Goldee": `// Enhanced Purple & Gold Metallic Psychedelic Shader (GLSL Shadertoy)

#define SAMPLE_WIDTH 8

float getAmplitude(int startBin, int endBin){
    float sumAmp = 0.0;
    for (int i = 0; i < 512; i++){
        if(i >= startBin && i <= endBin){
            sumAmp += texelFetch(iChannel0, ivec2(i, 1), 0).x;
        }
    }
    return sumAmp / float(endBin - startBin + 1);
}

vec2 swirl(vec2 uv, float strength){
    float angle = strength * length(uv);
    float s = sin(angle);
    float c = cos(angle);
    mat2 rot = mat2(c, -s, s, c);
    return uv * rot;
}

vec3 palette(float t){
    vec3 purple = vec3(0.6, 0.2, 0.9);
    vec3 gold   = vec3(1.0, 0.85, 0.3);
    return mix(purple, gold, t);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = (fragCoord.xy - 0.5 * iResolution.xy) / min(iResolution.x, iResolution.y);

    float bassAmp = getAmplitude(2, 12);
    float midAmp  = getAmplitude(30, 100);
    float trebleAmp = getAmplitude(200, 250);

    bassAmp = clamp(bassAmp * 2.5, 0.0, 1.0);
    midAmp = clamp(midAmp * 2.5, 0.0, 1.0);
    trebleAmp = clamp(trebleAmp * 2.5, 0.0, 1.0);

    // Enhanced swirl effect for sharpness
    uv = swirl(uv, bassAmp * 8.0 + iTime * 0.5);

    // Sharper kaleidoscopic reflection
    float segments = 12.0 + floor(midAmp * 12.0);
    float angle = atan(uv.y, uv.x);
    float radius = length(uv);
    angle = mod(angle, 2.0 * 3.141592 / segments);
    uv = vec2(cos(angle), sin(angle)) * radius;

    float pattern = sin(radius * 20.0 - iTime * 3.0 + trebleAmp * 6.0);
    pattern = smoothstep(0.45, 0.55, pattern);

    vec3 color = palette(pattern);

    // Metallic effect by increasing contrast and sharp highlights
    float metallic = pow(pattern, 3.0);
    color *= 0.8 + metallic * 1.5;
    color += trebleAmp * 0.2;

    fragColor = vec4(color, 1.0);
}`,
"Yawning Void!": `precision highp float;

uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D iChannel0;
uniform vec4 iMouse;

// ========== CONFIGURATION ==========
const float FLOW_SPEED = 0.5;           // How fast the fluid moves
const float FLUID_SCALE = 3.0;          // Scale of fluid simulation
const float COLOR_INTENSITY = 0.7;      // Color saturation multiplier
const float PARTICLE_DENSITY = 2.0;     // Density of particle system
const float TURBULENCE = 2.8;           // Turbulence in the fluid
const int OCTAVES = 5;                  // Detail level for noise

// ========== NOISE FUNCTIONS ==========
// Hash function for random values
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// Smooth noise function
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    // Smooth interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    // Sample 4 corners
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal Brownian Motion
float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < OCTAVES; i++) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value;
}

// ========== AUDIO ANALYSIS ==========
// Extract bass for primary flow
float getAudioBass() {
    float bass = 0.0;
    for (int i = 0; i < 20; i++) {
        float sample = texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        bass += sample;
    }
    return bass * 0.08; // Scale to reasonable range
}

// Extract mids for fluid details
float getAudioMid() {
    float mid = 0.0;
    for (int i = 20; i < 50; i++) {
        float sample = texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        mid += sample;
    }
    return mid * 0.04;
}

// Extract highs for particle effects
float getAudioHigh() {
    float high = 0.0;
    for (int i = 50; i < 100; i++) {
        float sample = texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        high += sample;
    }
    return high * 0.025;
}

// Analyze overall spectrum shape for color modulation
float getAudioSpectrum(float freqRange) {
    int startBin = int(freqRange * 100.0);
    int endBin = int(min(freqRange * 100.0 + 10.0, 127.0));
    
    float value = 0.0;
    for (int i = 0; i < 128; i++) {
        if (i >= startBin && i <= endBin) {
            value += texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        }
    }
    
    return value * 0.1;
}

// ========== FLUID DYNAMICS ==========
// Calculate fluid vector field
vec2 fluidField(vec2 uv, float time, float bass) {
    // First layer - large slow movement
    vec2 flow1 = vec2(
        fbm(uv * 0.5 + vec2(0.0, time * 0.1)),
        fbm(uv * 0.5 + vec2(time * 0.1, 0.0))
    );
    
    // Second layer - medium detail
    vec2 flow2 = vec2(
        fbm(uv * 1.0 + vec2(0.0, time * 0.2) + flow1 * TURBULENCE),
        fbm(uv * 1.0 + vec2(time * 0.2, 0.0) + flow1 * TURBULENCE)
    );
    
    // Final layer with bass boost
    vec2 finalFlow = vec2(
        fbm(uv * 1.5 + vec2(0.0, time * 0.3) + flow2 * bass * TURBULENCE),
        fbm(uv * 1.5 + vec2(time * 0.3, 0.0) + flow2 * bass * TURBULENCE)
    );
    
    return finalFlow * 2.0 - 1.0; // Range -1 to 1
}

// ========== PARTICLE SYSTEM ==========
float particleSystem(vec2 uv, float time, vec2 flowField, float audio) {
    float particles = 0.0;
    
    // Grid of potential particle positions
    for (float i = 0.0; i < 3.0; i++) {
        for (float j = 0.0; j < 3.0; j++) {
            // Create particle cell
            vec2 cellUV = floor(uv * PARTICLE_DENSITY + vec2(i, j)) / PARTICLE_DENSITY;
            
            // Random position within cell
            float random = hash(cellUV);
            vec2 particlePos = cellUV + vec2(random, hash(cellUV + 1.234));
            
            // Move particle with flow field
            particlePos += flowField * FLOW_SPEED * (0.1 + audio) * time * (0.5 + random * 0.5);
            
            // Wrap position
            particlePos = fract(particlePos);
            
            // Calculate distance to particle
            float dist = length(uv - particlePos);
            
            // Create star/dot effect
            float brightness = 0.0005 / (dist * dist);
            
            // Vary particle size with audio
            brightness *= 1.0 + audio * 10.0 * hash(cellUV + time);
            
            // Accumulate
            particles += brightness;
        }
    }
    
    return particles;
}

// ========== COLOR FUNCTIONS ==========
// Purple-gold space color mapping
vec3 spaceColors(float value, float time, float audio) {
    // Create a cosmic palette with stars, nebulae, and golden accents
    
    // Base space color (deeper, darker purple)
    vec3 spaceColor = vec3(0.02, 0.0, 0.05);
    
    // Add cosmic purple dust clouds (more subtle)
    vec3 purpleNebula = vec3(0.25, 0.0, 0.4) * smoothstep(0.1, 0.7, value);
    
    // Add golden nebula accents (more muted)
    vec3 goldNebula = vec3(0.7, 0.5, 0.15) * smoothstep(0.7, 0.95, value);
    
    // Audio-reactive color shift
    float colorShift = audio * 2.0 * sin(time * 0.1);
    
    // Combine all elements
    vec3 finalColor = spaceColor;
    finalColor += purpleNebula * (1.0 + colorShift);
    finalColor += goldNebula * (1.0 - colorShift * 0.5);
    
    return finalColor;
}

// ========== MAIN FUNCTION ==========
void main() {
    // Normalized coordinates
    vec2 uv = gl_FragCoord.xy / iResolution.xy;
    vec2 centered = uv * 2.0 - 1.0;
    centered.x *= iResolution.x / iResolution.y; // Correct aspect ratio
    
    // Time variables
    float time = iTime * FLOW_SPEED;
    
    // Get audio values
    float bass = getAudioBass();
    float mid = getAudioMid();
    float high = getAudioHigh();
    float audioTotal = bass + mid + high;
    
    // Frequency-specific audio analysis for color
    float lowMids = getAudioSpectrum(0.2); // ~20Hz range
    float highMids = getAudioSpectrum(0.5); // ~50Hz range
    
    // Apply mouse movement if active
    vec2 mouseOffset = vec2(0.0);
    if (iMouse.z > 0.0) {
        mouseOffset = (iMouse.xy / iResolution.xy - 0.5) * 2.0;
        centered += mouseOffset * 0.2;
    }
    
    // Scale for zoom effect
    float zoom = 1.0 + bass * 0.5;
    centered /= zoom;
    
    // Calculate fluid flow field
    vec2 flowField = fluidField(centered * FLUID_SCALE, time, bass);
    
    // Use flow field to distort coordinates for the fluid effect
    vec2 distortedUV = centered + flowField * (0.1 + bass * 0.2);
    
    // Generate multi-layered fluid patterns
    float fluidLayer1 = fbm(distortedUV * 2.0 + time * 0.1);
    float fluidLayer2 = fbm(distortedUV * 4.0 - flowField * 3.0 + time * 0.2);
    float fluidLayer3 = fbm(distortedUV * 8.0 + flowField * 1.0 - time * 0.3);
    
    // Layer the fluid effects with audio modulation
    float fluidPattern = fluidLayer1 * 0.5;
    fluidPattern += fluidLayer2 * 0.3 * (1.0 + mid * 2.0);
    fluidPattern += fluidLayer3 * 0.2 * (1.0 + high * 4.0);
    
    // Add subtle vortex effect
    float angle = atan(centered.y, centered.x);
    float radius = length(centered);
    fluidPattern += 0.1 * sin(angle * 3.0 + radius * 5.0 - time + bass * 5.0);
    
    // Generate particle star field
    float particles = particleSystem(centered, time, flowField, high);
    
    // Map fluid pattern to cosmic colors
    vec3 fluidColor = spaceColors(fluidPattern, time, lowMids);
    
    // Add audio-reactive golden particles (dimmer)
    vec3 particleColor = vec3(0.8, 0.7, 0.4) * particles * (3.0 + high * 20.0);
    
    // Add subtle purple accents in dark regions
    vec3 purpleAccent = vec3(0.3, 0.0, 0.6) * smoothstep(0.6, 0.0, fluidPattern) * highMids * 1.5;
    
    // Combine all elements
    vec3 finalColor = fluidColor;
    finalColor += particleColor;
    finalColor += purpleAccent;
    
    // Apply subtle pulsing glow
    float pulse = 0.5 + 0.5 * sin(time * 0.5) * bass;
    finalColor *= 1.0 + pulse * 0.2;
    
    // Color correction - darker with more contrast
    finalColor = pow(finalColor, vec3(1.1)); // Gamma correction (higher value = darker)
    finalColor *= COLOR_INTENSITY; // Reduced intensity
    
    // Add subtle vignette
    float vignette = smoothstep(1.8, 0.5, radius / zoom);
    finalColor *= vignette;
    
    // Output
    gl_FragColor = vec4(finalColor, 1.0);
}`,
"Sblatterrr":`
precision highp float;

uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D iChannel0;
uniform vec4 iMouse;

// ========== CONFIGURATION ==========
const int MAX_SPLATTERS = 25;         // Maximum number of blood splatters
const float BASS_THRESHOLD = 0.35;    // Threshold for bass detection
const float SPLATTER_INTENSITY = 1.8; // Intensity of splatter effects
const float DECAY_SPEED = 4.0;        // How quickly blood fades away (higher = faster)
const float MIN_SPLATTER_SIZE = 0.03; // Minimum size of splatters
const float MAX_SPLATTER_SIZE = 0.2;  // Maximum size of splatters
const int SPLATTER_DETAIL = 3;        // Detail level in splatter patterns

// ========== RANDOM FUNCTIONS ==========
// Hash function
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// Noise function for organic patterns
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    // Smoothstep interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    // Four corners
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal noise for detail
float fbm(vec2 p) {
    float sum = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    
    for (int i = 0; i < SPLATTER_DETAIL; i++) {
        sum += amp * noise(p * freq);
        freq *= 2.2;
        amp *= 0.5;
    }
    
    return sum;
}

// ========== AUDIO ANALYSIS ==========
// Get bass power
float getBassPower() {
    float bass = 0.0;
    // Sample low frequencies
    for (int i = 0; i < 15; i++) {
        float sample = texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        // Square to emphasize peaks
        bass += sample * sample;
    }
    return min(bass * 0.25, 1.0);
}

// Detect bass hits (sudden increases)
float detectBassHit(float currentBass) {
    // Simple threshold detection
    return smoothstep(BASS_THRESHOLD, BASS_THRESHOLD + 0.2, currentBass);
}

// Get mid frequencies for variation
float getMids() {
    float mids = 0.0;
    for (int i = 15; i < 40; i++) {
        mids += texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
    }
    return min(mids * 0.06, 1.0);
}

// Get high frequencies for texture details
float getHighs() {
    float highs = 0.0;
    for (int i = 40; i < 80; i++) {
        highs += texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
    }
    return min(highs * 0.05, 1.0);
}

// ========== SPLATTER SYSTEM ==========
// Blood splatter parameters struct
struct Splatter {
    vec2 position;    // Center position
    float size;       // Base size
    float rotation;   // Rotation angle
    float complexity; // Shape complexity
    float birth;      // Creation time
    float seed;       // Random seed
    float active;     // Whether this splatter is active
};

// Calculate a single splatter
float createSplatter(vec2 uv, Splatter s, float time) {
    if (s.active < 0.5) return 0.0;
    
    // Calculate age of splatter
    float age = time - s.birth;
    
    // Early exit if too old (optimization)
    if (age > 3.0) return 0.0;
    
    // Rotate coordinates around center
    float sinR = sin(s.rotation);
    float cosR = cos(s.rotation);
    vec2 rotUV = vec2(
        cosR * (uv.x - s.position.x) + sinR * (uv.y - s.position.y),
        -sinR * (uv.x - s.position.x) + cosR * (uv.y - s.position.y)
    ) + s.position;
    
    // Base shape - start with distance from center
    float dist = distance(rotUV, s.position);
    
    // Create complex rough edge using fbm noise
    float noiseVal = fbm((rotUV - s.position) * (10.0 + s.complexity * 10.0) + vec2(s.seed * 100.0));
    
    // Scale the shape with noise for rough edges
    float scaledDist = dist / (s.size * (1.0 + noiseVal * 0.7));
    
    // Base splatter shape
    float splatter = 1.0 - smoothstep(0.8, 1.0, scaledDist);
    
    // Create splatters with different shapes
    // Shape type based on seed
    float shapeType = fract(s.seed * 17.31);
    
    if (shapeType < 0.33) {
        // Elongated splatter
        vec2 stretchUV = (rotUV - s.position) / vec2(s.size * 1.8, s.size * 0.7);
        float stretchDist = length(stretchUV);
        float stretchSplatter = 1.0 - smoothstep(0.8, 1.0, stretchDist);
        splatter = max(splatter, stretchSplatter * 0.7);
    } else if (shapeType < 0.66) {
        // Multi-droplet splatter
        for (int i = 0; i < 5; i++) {
            float subAngle = s.rotation + hash(vec2(s.seed, float(i))) * 3.14159 * 2.0;
            float subDist = s.size * (0.5 + hash(vec2(float(i), s.seed)) * 0.7);
            vec2 subPos = s.position + vec2(cos(subAngle), sin(subAngle)) * subDist;
            
            float subSize = s.size * (0.2 + hash(vec2(s.seed, float(i) + 10.0)) * 0.3);
            float subSplatter = 1.0 - smoothstep(0.7, 1.0, distance(rotUV, subPos) / subSize);
            
            splatter = max(splatter, subSplatter * 0.9);
        }
    } else {
        // Spiky splatter
        float angle = atan(rotUV.y - s.position.y, rotUV.x - s.position.x);
        float spikes = 3.0 + floor(s.seed * 5.0);
        float spikeFactor = 0.2 + 0.3 * abs(sin(angle * spikes + s.seed * 10.0));
        
        float spikyDist = dist / (s.size * (1.0 + spikeFactor));
        float spikySplatter = 1.0 - smoothstep(0.7, 1.0, spikyDist);
        
        splatter = max(splatter, spikySplatter);
    }
    
    // Add small random droplets around the main splatter
    for (int i = 0; i < 8; i++) {
        float dropAngle = s.rotation + hash(vec2(s.seed, float(i) * 1.23)) * 6.28;
        float dropDist = s.size * (1.0 + hash(vec2(float(i) * 2.34, s.seed)) * 1.0);
        vec2 dropPos = s.position + vec2(cos(dropAngle), sin(dropAngle)) * dropDist;
        
        float dropSize = s.size * (0.05 + hash(vec2(s.seed, float(i) * 3.45)) * 0.15);
        float droplet = 1.0 - smoothstep(0.7, 1.0, distance(uv, dropPos) / dropSize);
        
        splatter = max(splatter, droplet * 0.7);
    }
    
    // Apply fade out over time - quickly appear, slowly fade
    float fadeIn = smoothstep(0.0, 0.1, age); // Quick fade in
    float fadeOut = 1.0 - smoothstep(0.2, 1.0, age / DECAY_SPEED); // Slower fade out
    float alpha = fadeIn * fadeOut;
    
    return splatter * alpha;
}

// ========== COLOR FUNCTIONS ==========
// Blood color function
vec3 bloodColor(float value, float seed, float variation) {
    // Base blood red
    vec3 darkRed = vec3(0.4, 0.0, 0.0);
    // Bright blood red
    vec3 brightRed = vec3(0.9, 0.05, 0.05);
    
    // Add variation based on seed
    float seedVar = (hash(vec2(seed * 7.89, seed * 13.45)) - 0.5) * variation;
    darkRed += vec3(seedVar * 0.1, 0.0, 0.0);
    brightRed += vec3(seedVar * 0.15, seedVar * 0.05, 0.0);
    
    // Mix based on value
    return mix(darkRed, brightRed, value);
}

// ========== MAIN FUNCTION ==========
void main() {
    // Normalized coordinates
    vec2 uv = gl_FragCoord.xy / iResolution.xy;
    
    // Get audio values
    float bass = getBassPower();
    float mids = getMids();
    float highs = getHighs();
    float bassHit = detectBassHit(bass);
    
    // Start with pure black background
    vec3 color = vec3(0.0, 0.0, 0.0);
    
    // Current time
    float time = iTime;
    
    // Use deterministic pseudo-random number generation based on time
    // This creates the illusion of new splatters while keeping a fixed max number
    float timeSlice = floor(time * 0.5); // Change splatters every 2 seconds
    float bassHitFactor = bassHit * 0.3; // Bass influence
    
    // Create array of splatters
    Splatter splatters[MAX_SPLATTERS];
    
    // Initialize/update splatter properties
    for (int i = 0; i < MAX_SPLATTERS; i++) {
        // Base seed for this splatter
        float baseSeed = float(i) + 123.45;
        
        // Consistent position based on seed
        float posX = hash(vec2(baseSeed, baseSeed * 2.0));
        float posY = hash(vec2(baseSeed * 3.0, baseSeed * 4.0));
        
        // Make some splatters more likely to appear at top of screen
        if (hash(vec2(baseSeed * 5.0, 89.45)) > 0.7) {
            posY *= 0.5; // Upper half of screen more likely
        }
        
        vec2 position = vec2(posX, posY);
        
        // Size depends partially on bass energy
        float sizeBase = MIN_SPLATTER_SIZE + 
                       hash(vec2(baseSeed * 6.0, baseSeed * 7.0)) * (MAX_SPLATTER_SIZE - MIN_SPLATTER_SIZE);
        
        // Rotation
        float rotation = hash(vec2(baseSeed * 8.0, baseSeed * 9.0)) * 3.14159 * 2.0;
        
        // Complexity for noise
        float complexity = hash(vec2(baseSeed * 10.0, baseSeed * 11.0));
        
        // Birth time - stagger creation to create continuous new splatters
        float seedForTime = hash(vec2(timeSlice, baseSeed * 12.0));
        
        // Determine if this splatter should be active based on time and bass
        float birthThreshold = 0.85 - bass * 0.3; // More bass = more splatters
        float birthTrigger = hash(vec2(baseSeed * 13.0, timeSlice));
        float bassActivation = step(0.97, bassHit * hash(vec2(time * 10.0, baseSeed)));
        
        // Calculate birth time and active status
        float birth;
        float active;
        
        if (birthTrigger > birthThreshold || bassActivation > 0.5) {
            // New active splatter
            birth = time - seedForTime * 0.1; // Slight variation in birth time
            active = 1.0;
        } else {
            // Inactive or old splatter
            birth = time - 10.0; // Old enough to be faded out
            active = 0.0;
        }
        
        // Create the splatter info
        splatters[i] = Splatter(
            position,
            sizeBase * (1.0 + bassHit * 0.5),
            rotation,
            complexity,
            birth,
            baseSeed,
            active
        );
    }
    
    // Render all splatters
    float totalSplatter = 0.0;
    
    for (int i = 0; i < MAX_SPLATTERS; i++) {
        float splatter = createSplatter(uv, splatters[i], time);
        
        if (splatter > 0.0) {
            // Generate color for this splatter
            float colorValue = hash(vec2(splatters[i].seed * 14.0, splatters[i].seed * 15.0));
            vec3 splatColor = bloodColor(colorValue, splatters[i].seed, mids);
            
            // Add subtle shading
            float age = time - splatters[i].birth;
            float darkening = smoothstep(0.1, 0.8, age / DECAY_SPEED) * 0.3;
            splatColor *= 1.0 - darkening;
            
            // Accumulate
            color = mix(color, splatColor, splatter * (1.0 - totalSplatter));
            totalSplatter = min(1.0, totalSplatter + splatter);
        }
    }
    
    // Add subtle film grain
    float grain = hash(uv + time) * 0.03 - 0.015;
    color += grain;
    
    // Add vignette for atmosphere
    float vignette = 1.0 - smoothstep(0.4, 1.2, length((uv - 0.5) * 1.2));
    color *= vignette;
    
    // Output final color
    gl_FragColor = vec4(color, 1.0);
}`,
"Ankara Test":`
precision highp float;

uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D iChannel0;
uniform vec4 iMouse;

// ========== CONFIGURATION ==========
const float PATTERN_SCALE = 10.0;     // Scale of the base patterns
const float ANIMATION_SPEED = 0.4;    // Speed of pattern animations
const float BASS_INTENSITY = 2.0;     // Bass influence on patterns
const float MID_INFLUENCE = 1.5;      // Mids influence on patterns
const float PATTERN_COMPLEXITY = 2.0;  // Complexity of generated patterns
const int MAX_SHAPES = 25;           // Maximum number of Adinkra-inspired shapes

// ========== COLOR PALETTE ==========
// West African inspired color palette
vec3 getColor(float t) {
    // Rich earth tones, golds, and vibrant accent colors inspired by 
    // traditional West African textiles and art
    const vec3 earthRed = vec3(0.85, 0.2, 0.1);     // Deep red-orange
    const vec3 ochre = vec3(0.8, 0.5, 0.1);         // Golden yellow-brown
    const vec3 mudCloth = vec3(0.3, 0.2, 0.1);      // Dark brown
    const vec3 indigo = vec3(0.1, 0.1, 0.4);        // Deep blue
    const vec3 kente1 = vec3(0.9, 0.7, 0.0);        // Bright gold
    const vec3 kente2 = vec3(0.0, 0.5, 0.2);        // Forest green
    
    t = fract(t); // Wrap to 0-1 range
    
    if (t < 0.2) return mix(earthRed, ochre, t * 5.0);
    else if (t < 0.4) return mix(ochre, kente1, (t - 0.2) * 5.0);
    else if (t < 0.6) return mix(kente1, kente2, (t - 0.4) * 5.0);
    else if (t < 0.8) return mix(kente2, indigo, (t - 0.6) * 5.0);
    else return mix(indigo, mudCloth, (t - 0.8) * 5.0);
}

// ========== RANDOM/NOISE FUNCTIONS ==========
// Hash function
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// 2D noise
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    // Cubic Hermite interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    // Four corners
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal Brownian Motion (multiple octaves of noise)
float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < 5; i++) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value;
}

// ========== AUDIO ANALYSIS ==========
// Get bass frequencies
float getAudioBass() {
    float bass = 0.0;
    for (int i = 0; i < 15; i++) {
        float sample = texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        // Apply non-linear scaling for better dynamics
        bass += sample * sample;
    }
    return min(bass * 0.2, 1.0); // Normalize
}

// Get mid frequencies
float getAudioMid() {
    float mid = 0.0;
    for (int i = 15; i < 50; i++) {
        float sample = texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        mid += sample;
    }
    return min(mid * 0.05, 1.0);
}

// Get high frequencies
float getAudioHigh() {
    float high = 0.0;
    for (int i = 50; i < 100; i++) {
        float sample = texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        high += sample;
    }
    return min(high * 0.04, 1.0);
}

// Detect rhythmic pulses
float detectPulse(float value, float threshold) {
    return smoothstep(threshold, threshold + 0.2, value);
}

// ========== PATTERN FUNCTIONS ==========
// Apply grid-based repetition
vec2 repeatGrid(vec2 p, float size) {
    return mod(p, size) - size * 0.5;
}

// Create a single geometric shape
float geometricShape(vec2 p, float type, float size, float rotation) {
    // Apply rotation
    float s = sin(rotation);
    float c = cos(rotation);
    p = vec2(c * p.x - s * p.y, s * p.x + c * p.y);
    
    float shape = 0.0;
    type = floor(type * 5.0); // 5 different shape types
    
    if (type < 1.0) {
        // Concentric circles (Adinkrahene - greatness, charisma)
        float radius = length(p);
        shape = smoothstep(size * 0.9, size, radius) - 
                smoothstep(size, size * 1.1, radius);
        
        // Add inner circle
        shape += smoothstep(size * 0.4, size * 0.5, radius) - 
                 smoothstep(size * 0.5, size * 0.6, radius);
        
        // Add center dot
        shape += 1.0 - smoothstep(size * 0.1, size * 0.2, radius);
        
    } else if (type < 2.0) {
        // Crosshatch pattern (Nkyinkyim - initiative, dynamism)
        float lineWidth = size * 0.1;
        
        // Horizontal lines
        for (float i = -2.0; i <= 2.0; i += 1.0) {
            shape += smoothstep(lineWidth, 0.0, abs(p.y - i * size * 0.3));
        }
        
        // Vertical lines
        for (float i = -2.0; i <= 2.0; i += 1.0) {
            shape += smoothstep(lineWidth, 0.0, abs(p.x - i * size * 0.3));
        }
        
    } else if (type < 3.0) {
        // Diamond pattern (Nsaa - excellence, genuineness)
        p = abs(p);
        shape = 1.0 - smoothstep(size * 0.8, size * 0.9, p.x + p.y);
        
        // Inner diamond
        shape *= smoothstep(size * 0.3, size * 0.4, p.x + p.y);
        
    } else if (type < 4.0) {
        // Spiral-like shape (Sankofa - learn from the past)
        float angle = atan(p.y, p.x);
        float radius = length(p);
        
        float spiral = mod(angle + radius * 2.0, 3.14159 * 0.5);
        shape = smoothstep(0.5, 0.0, abs(spiral - 0.7));
        
        // Contain within circle
        shape *= smoothstep(size * 1.1, size, radius);
        
    } else {
        // Star pattern (Nyame nti - faith)
        float angle = atan(p.y, p.x);
        float radius = length(p);
        
        // Create 8-pointed star
        float r = size * (0.5 + 0.3 * sin(angle * 8.0));
        shape = 1.0 - smoothstep(r * 0.9, r, radius);
    }
    
    return shape;
}

// Create Adinkra-inspired geometric pattern
float adinkraPattern(vec2 uv, float time, float seed, float audioReactive) {
    // Create grid of shapes
    vec2 p = repeatGrid(uv * PATTERN_SCALE, 2.0);
    
    // Use deterministic randomness based on grid position
    vec2 cellCenter = floor(uv * PATTERN_SCALE * 0.5) * 2.0;
    float cellSeed = hash(cellCenter + seed);
    
    // Shape type
    float shapeType = hash(cellCenter + vec2(12.34, 56.78));
    
    // Size and rotation with audio reactivity
    float size = 0.6 + 0.4 * hash(cellCenter + vec2(90.12, 34.56));
    size *= 1.0 + audioReactive * 0.3;
    
    float baseRotation = hash(cellCenter + vec2(78.90, 12.34)) * 6.28;
    float rotation = baseRotation + time * ANIMATION_SPEED * (hash(cellCenter) - 0.5);
    
    // Create the shape
    return geometricShape(p, shapeType, size, rotation);
}

// Create kente-cloth inspired stripes
float kentePattern(vec2 uv, float time, float audioReactive) {
    // Basic stripe pattern
    float pattern = 0.0;
    
    // Horizontal stripes
    float hStripe = step(0.7, fract(uv.y * 8.0 + sin(uv.x * 3.0) * 0.1));
    
    // Vertical accent stripes
    float vStripe = step(0.85, fract(uv.x * 12.0));
    
    // Combine with audio reactivity
    pattern = mix(hStripe, vStripe, 0.3 + audioReactive * 0.2);
    
    // Add subtle wave motion
    pattern += 0.2 * sin(uv.y * 20.0 + uv.x * 5.0 + time * 2.0);
    
    return pattern;
}

// Create mud cloth (Bogolanfini) inspired pattern
float mudClothPattern(vec2 uv, float time, float audioReactive) {
    // Grid of symbols
    float gridSize = 0.5;
    vec2 gv = fract(uv / gridSize) - 0.5;
    vec2 id = floor(uv / gridSize);
    
    // Generate symbol type based on cell
    float symbolType = hash(id + 1234.5);
    
    float symbol = 0.0;
    
    // Different symbol types
    if (symbolType < 0.25) {
        // Dots
        symbol = 1.0 - smoothstep(0.1, 0.15, length(gv));
    } else if (symbolType < 0.5) {
        // Lines
        symbol = 1.0 - smoothstep(0.03, 0.05, abs(gv.x));
    } else if (symbolType < 0.75) {
        // Crosses
        symbol = 1.0 - smoothstep(0.03, 0.05, abs(gv.x));
        symbol = max(symbol, 1.0 - smoothstep(0.03, 0.05, abs(gv.y)));
    } else {
        // Zigzag
        float zigzag = abs(fract(gv.x * 4.0) - 0.5);
        symbol = 1.0 - smoothstep(0.05, 0.1, abs(gv.y - zigzag * 0.2));
    }
    
    // Add audio reaction - symbols grow/pulse with audio
    float pulse = 1.0 + audioReactive * sin(time * 5.0 + hash(id) * 10.0);
    symbol *= pulse;
    
    return symbol;
}

// Create a djembe drum-inspired circular pattern
float djembePattern(vec2 uv, float time, float bass) {
    float radius = length(uv);
    float angle = atan(uv.y, uv.x);
    
    // Create concentric rings
    float rings = smoothstep(0.01, 0.0, abs(mod(radius * 10.0, 1.0) - 0.5));
    
    // Add radiating lines
    float lines = smoothstep(0.03, 0.0, abs(mod(angle * 10.0 / 3.14159, 1.0) - 0.5));
    
    // Pulse with bass
    float pulse = 1.0 + bass * 0.5 * sin(time * 10.0);
    
    // Combine
    return (rings * 0.7 + lines * 0.3) * pulse * smoothstep(1.0, 0.8, radius);
}

// ========== MAIN FUNCTION ==========
void main() {
    // Normalized coordinates
    vec2 uv = gl_FragCoord.xy / iResolution.xy;
    
    // Center coordinates
    vec2 centered = uv * 2.0 - 1.0;
    centered.x *= iResolution.x / iResolution.y; // Correct aspect ratio
    
    // Get audio values
    float bass = getAudioBass() * BASS_INTENSITY;
    float mid = getAudioMid() * MID_INFLUENCE;
    float high = getAudioHigh();
    float audioTotal = bass + mid * 0.5 + high * 0.3;
    
    // Detect rhythmic pulses for pattern changes
    float bassPulse = detectPulse(bass, 0.5);
    float midPulse = detectPulse(mid, 0.4);
    
    // Time variables
    float time = iTime * ANIMATION_SPEED;
    
    // Apply mouse movement if active
    if (iMouse.z > 0.0) {
        vec2 mouseOffset = 2.0 * (iMouse.xy / iResolution.xy - 0.5);
        centered += mouseOffset * 0.2;
    }
    
    // Create base background pattern
    float baseBg = fbm(centered * 5.0 + time * 0.1) * 0.5 + 0.25;
    baseBg += 0.1 * sin(centered.x * 10.0) * sin(centered.y * 10.0 + time);
    
    // Add subtle pulse to background
    baseBg *= 1.0 + bass * 0.2 * sin(time * 3.0);
    
    // Initialize color
    vec3 color = getColor(baseBg + time * 0.02);
    
    // Add kente-inspired pattern
    float kente = kentePattern(centered, time, mid);
    color = mix(color, getColor(baseBg + 0.3 + time * 0.05), kente * 0.7);
    
    // Add mud cloth inspired pattern
    float mudCloth = mudClothPattern(centered * 1.5, time, audioTotal);
    color = mix(color, getColor(0.7 + time * 0.1), mudCloth * 0.6);
    
    // Add Adinkra symbols that react to bass
    float patternIntensity = 0.0;
    
    // Generate multiple Adinkra-inspired shapes
    for (int i = 0; i < MAX_SHAPES; i++) {
        // Create seed and parameters for this shape
        float seed = float(i) * 123.456;
        
        // Position based on bass pulse and seed
        float radius = 0.2 + hash(vec2(seed, seed * 2.0)) * 0.8;
        float angle = hash(vec2(seed * 3.0, seed * 4.0)) * 3.14159 * 2.0 + time * ANIMATION_SPEED;
        
        // Calculate position with some circular motion
        vec2 pos = vec2(
            radius * cos(angle),
            radius * sin(angle)
        );
        
        // Adjust position based on audio
        pos *= 1.0 + bassPulse * hash(vec2(seed * 5.0, seed * 6.0));
        
        // Scale based on audio energy
        float scale = 0.05 + hash(vec2(seed * 7.0, seed * 8.0)) * 0.1;
        scale *= 1.0 + bass * hash(vec2(seed * 9.0, time));
        
        // Shape type varies with seed
        float shapeType = hash(vec2(seed * 10.0, seed * 11.0));
        
        // Create the shape
        float dist = length(centered - pos) / scale;
        float shape = 0.0;
        
        // Only process if we're close enough (optimization)
        if (dist < 3.0) {
            // Different shape types
            if (shapeType < 0.2) {
                // Circle with cross
                shape = 1.0 - smoothstep(0.8, 1.0, dist);
                shape *= smoothstep(0.3, 0.4, abs(centered.x - pos.x) / scale) * 
                         smoothstep(0.3, 0.4, abs(centered.y - pos.y) / scale);
            } else if (shapeType < 0.4) {
                // Spiral
                float angle = atan(centered.y - pos.y, centered.x - pos.x);
                shape = (1.0 - smoothstep(0.8, 1.0, dist)) * 
                        (0.5 + 0.5 * sin(angle * 8.0 + dist * 10.0 + time * 2.0));
            } else if (shapeType < 0.6) {
                // Diamond
                vec2 p = abs(centered - pos) / scale;
                shape = 1.0 - smoothstep(0.8, 1.0, p.x + p.y);
            } else if (shapeType < 0.8) {
                // Concentric circles
                shape = 1.0 - smoothstep(0.8, 1.0, dist);
                shape *= sin(dist * 10.0 + time) * 0.5 + 0.5;
            } else {
                // Radial lines
                float angle = atan(centered.y - pos.y, centered.x - pos.x);
                shape = (1.0 - smoothstep(0.8, 1.0, dist)) * 
                        step(0.7, sin(angle * 12.0 + time) * 0.5 + 0.5);
            }
            
            // Make symbol pulse with audio
            float pulse = 1.0 + bassPulse * hash(vec2(seed * 12.0, seed * 13.0)) * 0.3;
            shape *= pulse;
            
            // Add to pattern
            patternIntensity += shape * (0.5 + hash(vec2(seed * 14.0, seed * 15.0)) * 0.5);
        }
    }
    
    // Add central djembe-inspired pattern
    float djembe = djembePattern(centered, time, bass);
    
    // Apply dynamic layering based on audio
    float layerMix = 0.5 + 0.5 * sin(time * 0.2 + uv.x * 3.0);
    
    // First layer blend
    color = mix(color, getColor(0.5 + mid + time * 0.1), patternIntensity * 0.7);
    
    // Second layer blend
    color = mix(color, getColor(0.9 + bass + time * 0.05), djembe * 0.8);
    
    // Add subtle rhythmic pulse
    color *= 1.0 + bass * 0.3 * sin(time * 10.0);
    
    // Add warm glow
    float glow = smoothstep(1.0, 0.0, length(centered));
    color += getColor(time * 0.1) * glow * bass * 0.6;
    
    // Output
    gl_FragColor = vec4(color, 1.0);
}
`,
"Ankara 2": `
precision highp float;

uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D iChannel0;
uniform vec4 iMouse;

// ========== CONFIGURATION ==========
const float PATTERN_SCALE = 8.0;      // Scale of the base patterns (reduced for cleaner look)
const float ANIMATION_SPEED = 0.35;   // Slightly slower for more deliberate movement
const float BASS_INTENSITY = 2.5;     // Increased bass influence for stronger reactions
const float MID_INFLUENCE = 1.8;      // Increased mid influence
const float PATTERN_COMPLEXITY = 1.5;  // Reduced complexity for cleaner patterns
const int MAX_SHAPES = 18;           // Fewer shapes for cleaner composition

// ========== COLOR PALETTE ==========
// Enhanced West African inspired color palette
vec3 getColor(float t) {
    // More vibrant, higher contrast palette inspired by 
    // contemporary West African fabrics and traditional art
    const vec3 earthRed = vec3(0.9, 0.15, 0.05);   // Brighter red-orange
    const vec3 ochre = vec3(0.95, 0.6, 0.05);      // Richer golden yellow
    const vec3 mudCloth = vec3(0.25, 0.15, 0.05);  // Rich dark brown
    const vec3 indigo = vec3(0.05, 0.05, 0.35);    // Deeper, richer blue
    const vec3 kente1 = vec3(1.0, 0.8, 0.0);       // Bright vibrant gold
    const vec3 kente2 = vec3(0.0, 0.6, 0.25);      // Brighter green
    const vec3 akwete = vec3(0.95, 0.3, 0.0);      // Bright orange (Nigerian Akwete cloth)
    
    t = fract(t); // Wrap to 0-1 range
    
    // More distinct color transitions for cleaner look
    if (t < 0.16) return mix(earthRed, ochre, t * 6.25);
    else if (t < 0.33) return mix(ochre, kente1, (t - 0.16) * 5.88);
    else if (t < 0.5) return mix(kente1, akwete, (t - 0.33) * 5.88);
    else if (t < 0.66) return mix(akwete, kente2, (t - 0.5) * 6.25);
    else if (t < 0.83) return mix(kente2, indigo, (t - 0.66) * 5.88);
    else return mix(indigo, mudCloth, (t - 0.83) * 5.88);
}

// ========== RANDOM/NOISE FUNCTIONS ==========
// Hash function
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// 2D noise
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    // Cubic Hermite interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    // Four corners
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal Brownian Motion (multiple octaves of noise)
float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < 5; i++) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value;
}

// ========== AUDIO ANALYSIS ==========
// Get bass frequencies
float getAudioBass() {
    float bass = 0.0;
    for (int i = 0; i < 15; i++) {
        float sample = texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        // Apply non-linear scaling for better dynamics
        bass += sample * sample;
    }
    return min(bass * 0.2, 1.0); // Normalize
}

// Get mid frequencies
float getAudioMid() {
    float mid = 0.0;
    for (int i = 15; i < 50; i++) {
        float sample = texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        mid += sample;
    }
    return min(mid * 0.05, 1.0);
}

// Get high frequencies
float getAudioHigh() {
    float high = 0.0;
    for (int i = 50; i < 100; i++) {
        float sample = texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        high += sample;
    }
    return min(high * 0.04, 1.0);
}

// Detect rhythmic pulses
float detectPulse(float value, float threshold) {
    return smoothstep(threshold, threshold + 0.2, value);
}

// ========== PATTERN FUNCTIONS ==========
// Apply grid-based repetition
vec2 repeatGrid(vec2 p, float size) {
    return mod(p, size) - size * 0.5;
}

// Create a single geometric shape
float geometricShape(vec2 p, float type, float size, float rotation) {
    // Apply rotation
    float s = sin(rotation);
    float c = cos(rotation);
    p = vec2(c * p.x - s * p.y, s * p.x + c * p.y);
    
    float shape = 0.0;
    type = floor(type * 5.0); // 5 different shape types
    
    if (type < 1.0) {
        // Concentric circles (Adinkrahene - greatness, charisma)
        float radius = length(p);
        shape = smoothstep(size * 0.9, size, radius) - 
                smoothstep(size, size * 1.1, radius);
        
        // Add inner circle
        shape += smoothstep(size * 0.4, size * 0.5, radius) - 
                 smoothstep(size * 0.5, size * 0.6, radius);
        
        // Add center dot
        shape += 1.0 - smoothstep(size * 0.1, size * 0.2, radius);
        
    } else if (type < 2.0) {
        // Crosshatch pattern (Nkyinkyim - initiative, dynamism)
        float lineWidth = size * 0.1;
        
        // Horizontal lines
        for (float i = -2.0; i <= 2.0; i += 1.0) {
            shape += smoothstep(lineWidth, 0.0, abs(p.y - i * size * 0.3));
        }
        
        // Vertical lines
        for (float i = -2.0; i <= 2.0; i += 1.0) {
            shape += smoothstep(lineWidth, 0.0, abs(p.x - i * size * 0.3));
        }
        
    } else if (type < 3.0) {
        // Diamond pattern (Nsaa - excellence, genuineness)
        p = abs(p);
        shape = 1.0 - smoothstep(size * 0.8, size * 0.9, p.x + p.y);
        
        // Inner diamond
        shape *= smoothstep(size * 0.3, size * 0.4, p.x + p.y);
        
    } else if (type < 4.0) {
        // Spiral-like shape (Sankofa - learn from the past)
        float angle = atan(p.y, p.x);
        float radius = length(p);
        
        float spiral = mod(angle + radius * 2.0, 3.14159 * 0.5);
        shape = smoothstep(0.5, 0.0, abs(spiral - 0.7));
        
        // Contain within circle
        shape *= smoothstep(size * 1.1, size, radius);
        
    } else {
        // Star pattern (Nyame nti - faith)
        float angle = atan(p.y, p.x);
        float radius = length(p);
        
        // Create 8-pointed star
        float r = size * (0.5 + 0.3 * sin(angle * 8.0));
        shape = 1.0 - smoothstep(r * 0.9, r, radius);
    }
    
    return shape;
}

// Create Adinkra-inspired geometric pattern
float adinkraPattern(vec2 uv, float time, float seed, float audioReactive) {
    // Create grid of shapes
    vec2 p = repeatGrid(uv * PATTERN_SCALE, 2.0);
    
    // Use deterministic randomness based on grid position
    vec2 cellCenter = floor(uv * PATTERN_SCALE * 0.5) * 2.0;
    float cellSeed = hash(cellCenter + seed);
    
    // Shape type
    float shapeType = hash(cellCenter + vec2(12.34, 56.78));
    
    // Size and rotation with audio reactivity
    float size = 0.6 + 0.4 * hash(cellCenter + vec2(90.12, 34.56));
    size *= 1.0 + audioReactive * 0.3;
    
    float baseRotation = hash(cellCenter + vec2(78.90, 12.34)) * 6.28;
    float rotation = baseRotation + time * ANIMATION_SPEED * (hash(cellCenter) - 0.5);
    
    // Create the shape
    return geometricShape(p, shapeType, size, rotation);
}

// Create enhanced kente-cloth inspired stripes
float kentePattern(vec2 uv, float time, float audioReactive) {
    // More structured, cleaner pattern
    float pattern = 0.0;
    
    // Primary horizontal bands (thicker, more defined)
    float primaryBand = step(0.75, fract(uv.y * 6.0));
    
    // Secondary horizontal stripes (thinner, for detail)
    float secondaryBand = step(0.85, fract(uv.y * 12.0));
    
    // Vertical accent stripes (more defined)
    float vStripe = step(0.92, fract(uv.x * 10.0));
    
    // Audio-reactive accents
    float dynamicAccent = step(0.7 - audioReactive * 0.3, 
                              fract(uv.x * 15.0 + sin(uv.y * 8.0 + time) * 0.2));
    
    // Combine in phases for cleaner result
    pattern = mix(primaryBand, secondaryBand, 0.4);
    pattern = mix(pattern, vStripe, 0.5);
    pattern = max(pattern, dynamicAccent * audioReactive);
    
    // Add subtle geometric blocks (kente geometric elements)
    vec2 blockUV = fract(uv * vec2(5.0, 3.0)) - 0.5;
    float block = step(0.35, max(abs(blockUV.x), abs(blockUV.y))) * step(0.9, fract(uv.y * 3.0 + uv.x * 1.0));
    
    pattern = max(pattern, block * (0.7 + audioReactive * 0.3));
    
    return pattern;
}

// Create refined mud cloth (Bogolanfini) inspired pattern
float mudClothPattern(vec2 uv, float time, float audioReactive) {
    // More organized, cleaner grid of symbols
    float gridSize = 0.6; // Larger cells for clarity
    vec2 gv = fract(uv / gridSize) - 0.5;
    vec2 id = floor(uv / gridSize);
    
    // Create borders between cells for cleaner separation
    float border = smoothstep(0.48, 0.5, max(abs(gv.x), abs(gv.y)));
    
    // Determine cell type based on position (more organized pattern)
    float rowType = mod(id.y, 3.0); // 3-row repeat pattern
    float colType = mod(id.x, 4.0); // 4-column repeat pattern
    
    // Combine for deterministic pattern (less random, more structured)
    float symbolType = fract(hash(vec2(rowType, colType)) + id.x * 0.05);
    
    float symbol = 0.0;
    
    // More defined, cleaner symbol types
    if (rowType < 1.0) {
        // Row 1: Geometric patterns
        if (colType < 1.0) {
            // Circles
            symbol = 1.0 - smoothstep(0.25, 0.3, length(gv));
            symbol *= smoothstep(0.1, 0.15, length(gv)); // Hollow center
        } else if (colType < 2.0) {
            // Diamond
            symbol = 1.0 - smoothstep(0.28, 0.32, abs(gv.x) + abs(gv.y));
        } else if (colType < 3.0) {
            // Concentric squares
            vec2 absGV = abs(gv);
            float squareSize = max(absGV.x, absGV.y);
            symbol = step(0.19, squareSize) * step(squareSize, 0.28);
            symbol += step(0.05, squareSize) * step(squareSize, 0.12);
        } else {
            // Cross
            float crossWidth = 0.08;
            symbol = step(0.5 - crossWidth, abs(gv.x) + abs(gv.y) - abs(gv.x - gv.y));
        }
    } else if (rowType < 2.0) {
        // Row 2: Linear patterns
        if (symbolType < 0.33) {
            // Horizontal lines
            symbol = step(0.7, fract(gv.y * 8.0 + 0.5));
        } else if (symbolType < 0.66) {
            // Vertical lines
            symbol = step(0.7, fract(gv.x * 8.0 + 0.5));
        } else {
            // Grid
            symbol = step(0.8, max(fract(gv.x * 5.0 + 0.5), fract(gv.y * 5.0 + 0.5)));
        }
    } else {
        // Row 3: Complex symbols
        if (symbolType < 0.25) {
            // Spiral-like
            float angle = atan(gv.y, gv.x);
            float radius = length(gv);
            symbol = step(0.15, radius) * step(radius, 0.3) * 
                    step(0.7, fract(angle * 1.0 / 3.14159 * 4.0 + radius * 5.0));
        } else if (symbolType < 0.5) {
            // Star
            float angle = atan(gv.y, gv.x);
            float radius = length(gv);
            float star = 0.2 + 0.1 * sin(angle * 6.0);
            symbol = 1.0 - smoothstep(star - 0.05, star, radius);
        } else if (symbolType < 0.75) {
            // Dotted circle
            float radius = length(gv);
            float angle = atan(gv.y, gv.x);
            symbol = step(0.25, radius) * step(radius, 0.3) * 
                    step(0.6, fract(angle * 1.0 / 3.14159 * 6.0));
        } else {
            // Zigzag border
            float edge = max(abs(gv.x), abs(gv.y));
            float zigzag = 0.05 * sin(gv.x * 40.0) * sin(gv.y * 40.0);
            symbol = step(0.35 + zigzag, edge) * step(edge, 0.45 + zigzag);
        }
    }
    
    // Audio-reactive pulsing with cleaner animation
    float pulse = 1.0 + audioReactive * 0.5 * (0.5 + 0.5 * sin(time * 4.0 + hash(id) * 8.0));
    
    // Create final pattern with borders
    return symbol * pulse * (1.0 - border * 0.7);
}

// Create an enhanced djembe drum-inspired circular pattern
float djembePattern(vec2 uv, float time, float bass) {
    float radius = length(uv);
    // Fix potential atan2 undefined behavior when both x and y are 0
    float angle = (abs(uv.x) < 0.0001 && abs(uv.y) < 0.0001) ? 0.0 : atan(uv.y, uv.x);
    
    // Create cleaner concentric rings with varying widths
    float ringCount = 8.0; // Fewer, more defined rings
    float ringWidth = 0.5; // Wider rings for clarity
    float ringPattern = smoothstep(ringWidth * 0.8, ringWidth * 0.9, 
                                 abs(mod(radius * ringCount, 1.0) - ringWidth));
    
    // Add primary radiating lines - fewer, more prominent
    // Using a fixed loop count to avoid potential issues with variable loop bounds
    const int FIXED_LINE_COUNT = 12; // Clear division of the circle
    float lineWidth = 0.06; // Thicker lines
    float linePattern = 0.0;
    
    for (int i = 0; i < FIXED_LINE_COUNT; i++) {
        float lineAngle = float(i) / 12.0 * 3.14159 * 2.0;
        // Fix potential numerical precision issues in the angle calculation
        float angleDiff = abs(mod(angle + 3.14159, 3.14159 * 2.0) - lineAngle);
        angleDiff = min(angleDiff, 3.14159 * 2.0 - angleDiff); // Take shortest path
        linePattern = max(linePattern, smoothstep(lineWidth, 0.0, angleDiff));
    }
    
    // Add secondary pattern details (inspired by djembe carvings)
    float detailPattern = 0.0;
    float detailRadius = mod(radius * 15.0 + sin(angle * 6.0) * 0.2, 1.0);
    detailPattern = smoothstep(0.4, 0.6, detailRadius) * smoothstep(0.8, 0.6, detailRadius);
    
    // Dynamic bass-reactive pulsing
    float bassIntensity = bass * 2.0; // Stronger reaction
    float pulse = 1.0 + bassIntensity * (0.3 + 0.2 * sin(time * 8.0));
    
    // Central symbol (inspired by drum head designs)
    float centerSymbol = 0.0;
    if (radius < 0.2) {
        float symbolRadius = radius / 0.2; // Normalize to 0-1 in center area
        float symbolAngle = atan(uv.y, uv.x);
        
        // Create a spiral or similar traditional pattern
        centerSymbol = smoothstep(0.2, 0.0, abs(mod(symbolAngle * 3.0 + symbolRadius * 5.0, 1.0) - 0.5));
        centerSymbol *= smoothstep(0.0, 0.2, symbolRadius) * smoothstep(1.0, 0.8, symbolRadius);
        centerSymbol *= 1.0 + bass * sin(time * 12.0); // Faster pulsing in center
    }
    
    // Combine all elements with cleaner layering
    float pattern = 0.0;
    pattern = max(ringPattern * 0.9, linePattern * 0.95);
    pattern = max(pattern, detailPattern * 0.7);
    pattern = max(pattern, centerSymbol);
    
    // Apply distance falloff and pulse
    return pattern * pulse * smoothstep(1.0, 0.5, radius * (0.8 + bass * 0.4));
}

// ========== MAIN FUNCTION ==========
void main() {
    // Normalized coordinates
    vec2 uv = gl_FragCoord.xy / iResolution.xy;
    
    // Center coordinates
    vec2 centered = uv * 2.0 - 1.0;
    centered.x *= iResolution.x / iResolution.y; // Correct aspect ratio
    
    // Get enhanced audio values
    float bass = getAudioBass() * BASS_INTENSITY;
    float mid = getAudioMid() * MID_INFLUENCE;
    float high = getAudioHigh();
    float audioTotal = bass + mid * 0.5 + high * 0.3;
    
    // Improved rhythmic pulse detection
    float bassPulse = detectPulse(bass, 0.45); // Slightly more sensitive
    float midPulse = detectPulse(mid, 0.35);
    float highPulse = detectPulse(high, 0.6);
    
    // Time variables
    float time = iTime * ANIMATION_SPEED;
    
    // Apply mouse movement if active
    if (iMouse.z > 0.0) {
        vec2 mouseOffset = 2.0 * (iMouse.xy / iResolution.xy - 0.5);
        centered += mouseOffset * 0.3; // Increased influence
    }
    
    // Create cleaner, more defined base pattern
    // Use multiple overlaid patterns instead of fbm for cleaner look
    float basePattern1 = sin(centered.x * 3.0 + time * 0.2) * sin(centered.y * 3.0 + time * 0.15);
    float basePattern2 = sin(centered.x * 5.0 - time * 0.1) * sin(centered.y * 5.0 - time * 0.12);
    float baseBg = mix(basePattern1, basePattern2, 0.5) * 0.5 + 0.5;
    
    // Add clean rhythmic pulse to background
    float bgPulse = 0.8 + 0.2 * sin(time * 5.0);
    bgPulse = mix(bgPulse, 1.0 + bass * 0.5, bassPulse);
    baseBg *= bgPulse;
    
    // Initialize color with base gradient
    vec3 color = getColor(baseBg * 0.8 + time * 0.02);
    
    // Create region-based layout (divides the screen into functional regions)
    float regionMask1 = step(0.0, centered.y); // Top half
    float regionMask2 = step(centered.y, 0.0) * step(-0.7, centered.y); // Middle-bottom
    float regionMask3 = step(centered.y, -0.7); // Very bottom
    
    // Add enhanced kente-inspired pattern in top region
    if (regionMask1 > 0.5) {
        float kente = kentePattern(centered * vec2(1.0, 1.2), time, mid);
        vec3 kenteColor = getColor(fract(baseBg + 0.33 + bass * 0.1 + time * 0.05));
        color = mix(color, kenteColor, kente * 0.8);
    }
    
    // Add enhanced mud cloth inspired pattern in middle-bottom region
    if (regionMask2 > 0.5) {
        float mudCloth = mudClothPattern(centered * 1.2 + vec2(0.0, 0.7), time, audioTotal);
        vec3 mudClothColor = getColor(fract(0.66 + mid * 0.2 + time * 0.1));
        color = mix(color, mudClothColor, mudCloth * 0.7);
    }
    
    // Add organized Adinkra symbols that react to bass
    float patternIntensity = 0.0;
    
    // Generate cleaner, more focused Adinkra-inspired shapes
    // Use a fixed number for loop to avoid any potential issues with MAX_SHAPES constant
    for (int i = 0; i < 18; i++) {  // Using the value from MAX_SHAPES=18
        // Create seed and parameters for this shape
        float seed = float(i) * 123.456;
        
        // Use more organized position distribution
        // Based on golden ratio spiral for better visual balance
        float idx = float(i) / float(MAX_SHAPES);
        float goldenAngle = 2.399; // Close to golden ratio * 2π
        float radius = 0.2 + 0.6 * sqrt(idx); // Spiral radius
        float angle = goldenAngle * float(i) + time * (0.2 + 0.1 * hash(vec2(seed, 456.7)));
        
        // Calculate position with balanced motion
        vec2 pos = vec2(
            radius * cos(angle),
            radius * sin(angle)
        );
        
        // Apply subtle audio reactivity to position
        pos *= 1.0 + bassPulse * 0.2;
        
        // Scale based on position in sequence and audio
        float scale = 0.07 + 0.08 * (1.0 - idx) * (1.0 + bass * 0.5);
        
        // More deliberate shape selection based on position
        float shapeType = floor(mod(float(i), 5.0)) / 5.0; // 5 distinct shape types
        
        // Create the shape
        float dist = length(centered - pos) / scale;
        float shape = 0.0;
        
        // Only process if close enough (optimization)
        if (dist < 2.5) {
            // Different shape types - cleaner geometry
            if (shapeType < 0.2) {
                // Gye Nyame symbol - simplified
                shape = 1.0 - smoothstep(0.8, 1.0, dist);
                // Fix potential division by zero if scale is very small
                float normalizedX = abs(centered.x - pos.x) / max(scale, 0.001);
                float normalizedY = (centered.y - pos.y) / max(scale, 0.001);
                shape *= smoothstep(0.25, 0.35, normalizedX) * step(0.0, normalizedY);
                
                // Add details
                float detailSize = scale * 0.2;
                float detailShape = smoothstep(detailSize, 0.0, 
                                  length(vec2(centered.x - pos.x, centered.y - pos.y - scale * 0.6)));
                shape = max(shape, detailShape);
                
            } else if (shapeType < 0.4) {
                // Adinkrahene (chief) - concentric circles
                shape = smoothstep(0.9, 0.8, dist) - smoothstep(0.7, 0.6, dist);
                shape += smoothstep(0.5, 0.4, dist) - smoothstep(0.3, 0.2, dist);
                shape += smoothstep(0.15, 0.05, dist);
                
            } else if (shapeType < 0.6) {
                // Dwennimmen (ram's horns) - simplified
                vec2 p = (centered - pos) / scale;
                float horns = smoothstep(0.05, 0.0, abs(abs(p.x) - 0.5) - 0.2 * p.y);
                horns *= step(0.0, p.y) * step(p.y, 0.6);
                
                float base = smoothstep(0.05, 0.0, abs(p.x));
                base *= step(-0.6, p.y) * step(p.y, 0.0);
                
                shape = horns + base;
                
            } else if (shapeType < 0.8) {
                // Sankofa (learn from the past) - spiral-like
                float angle = atan(centered.y - pos.y, centered.x - pos.x);
                float spiral = mod(angle + dist * 3.0, 3.14159 * 0.5);
                shape = smoothstep(0.15, 0.0, abs(spiral - 0.7)) * smoothstep(1.0, 0.0, dist);
                
            } else {
                // Nea Onnim (knowledge) - abstract representation
                vec2 p = abs((centered - pos) / scale);
                shape = step(max(p.x, p.y), 0.7) * step(0.3, max(p.x, p.y));
                
                // Add cross detail
                float cross = step(p.x, 0.15) + step(p.y, 0.15);
                cross *= step(max(p.x, p.y), 0.7);
                shape = max(shape, cross);
            }
            
            // Cleaner audio reactivity
            float pulse = 1.0 + bassPulse * 0.4;
            shape *= pulse;
            
            // Add to pattern with improved clarity
            patternIntensity += shape * (0.7 + hash(vec2(seed * 14.0, seed * 15.0)) * 0.3);
        }
    }
    
    // Add central djembe-inspired pattern (avoiding potential numerical issues)
    float djembe = djembePattern(centered * 0.8, time, clamp(bass, 0.0, 1.0));
    
    // Apply layering with cleaner transitions
    vec3 patternColor = getColor(fract(0.2 + mid + time * 0.1));
    color = mix(color, patternColor, min(1.0, patternIntensity * 0.6));
    
    // Apply djembe overlay with clean contrast
    vec3 djembeColor = getColor(fract(0.8 + bass * 0.3 + time * 0.05));
    color = mix(color, djembeColor, djembe * 0.85);
    
    // Add sharper rhythm-driven accents
    vec3 accentColor = getColor(fract(time * 0.2 + 0.5));
    float accentPattern = step(0.7, bassPulse) * smoothstep(1.0, 0.0, length(centered)) * 0.4;
    accentPattern *= sin(time * 20.0) * 0.5 + 0.5; // Fast strobe on bass hits
    color = mix(color, accentColor, accentPattern);
    
    // Add subtle vignette for focus
    float vignette = smoothstep(1.2, 0.5, length(centered));
    color *= vignette * 0.7 + 0.3;
    
    // Enhance contrast
    color = pow(color, vec3(0.9)); // Slight gamma adjustment for better visibility
    
    // Output final color
    gl_FragColor = vec4(color, 1.0);
}
}

`, 
"Test": `
`
};

