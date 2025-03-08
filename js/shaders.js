
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

// ========== CONFIGURATION (TUNED PARAMETERS) ==========
const float SPEED = 0.4;               // Reduced from 0.7 for slower animation
const float KALEIDOSCOPE_SIDES = 7.0;  // Kept the same kaleidoscope reflections
const int MAX_ITERATIONS = 7;          // Reduced from 9 for less detail/complexity
const float FEEDBACK_STRENGTH = 0.3;   // Reduced from 0.4 for less intense feedback
const float METALLIC_SHININESS = 3.0;  // Kept the same gold metallic effect intensity
const float FREQUENCY_SCALING = 0.7;   // Reduced from 1.5 to make audio less dominating

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
// Create psychedelic color palette with purple and gold focus - KEPT EXACTLY AS ORIGINAL
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

// Advanced rotation with reduced distortion
vec2 rotateDistort(vec2 p, float angle, float distortion) {
    float s = sin(angle);
    float c = cos(angle);
    
    // Apply milder non-linear distortion based on radius
    float r = length(p);
    float distortionFactor = 1.0 + distortion * sin(r * 2.5);
    
    return vec2(
        p.x * c * distortionFactor - p.y * s,
        p.x * s + p.y * c * distortionFactor
    );
}

// Domain warping function - makes patterns more organic, but less aggressive
vec2 warpDomain(vec2 p, float time, float strength) {
    // Primary warping with reduced intensity
    p.x += strength * 0.7 * sin(p.y * 1.5 + time * 0.7);
    p.y += strength * 0.6 * sin(p.x * 1.7 + time * 0.6);
    
    // Secondary higher-frequency warping with reduced intensity
    p.x += strength * 0.2 * sin(p.y * 5.0 + time * 1.0);
    p.y += strength * 0.15 * sin(p.x * 7.0 + time * 0.9);
    
    return p;
}

// Generate metallic highlights (for gold effect) - KEPT EXACTLY AS ORIGINAL
float metallicHighlight(float pattern, float time, float shininess) {
    float phase = pattern * 20.0 + time;
    return pow(0.5 + 0.5 * sin(phase), shininess);
}

    // Create a fractal feedback pattern with reduced complexity
float fractalPattern(vec2 uv, float time, float audioTotal) {
    float pattern = 0.0;
    float amp = 1.0;
    float freq = 1.5; // Higher base frequency for sharper detail
    vec2 p = uv;
    
    // Add minimum detail level independent of audio
    float baseDetail = 0.2; 
    float minAudioTotal = max(audioTotal, baseDetail);
    
    // Layer multiple octaves for fractal effect but with fewer iterations
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        // Apply rotation with minimum distortion to maintain sharpness
        p = rotateDistort(p, time * 0.1 + float(i) * 0.2 + minAudioTotal * 1.2, 0.15);
        
        // Create psychedelic pattern with higher frequency components
        float wave = sin(p.x * freq) * sin(p.y * freq);
        wave *= sin(length(p) * freq * 0.5 + time);
        
        // Add edge enhancement for sharper details
        float edge = abs(sin(p.x * freq * 2.0)) * abs(sin(p.y * freq * 2.0));
        wave = mix(wave, edge, 0.2);
        
        // Add to the total pattern
        pattern += amp * abs(wave);
        
        // Modify parameters for next iteration with minimum scaling
        freq *= 1.7 + minAudioTotal * 0.3;
        amp *= 0.65;
        
        // Apply minimal warping to preserve sharpness
        p = warpDomain(p, time * 0.15, 0.08 + minAudioTotal * 0.3);
    }
    
    // Apply contrast enhancement to maintain sharpness
    pattern = pow(pattern, 0.9);
    
    return pattern;
}

// ========== MAIN SHADER FUNCTION ==========
void main() {
    // Screen coordinates normalized to [-1,1]
    vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / min(iResolution.x, iResolution.y);
    
    // Extract audio features with reduced scaling
    float bass = getAudioBass() * FREQUENCY_SCALING;
    float mid = getAudioMid() * FREQUENCY_SCALING;
    float high = getAudioHigh() * FREQUENCY_SCALING;
    float transients = getAudioTransients() * FREQUENCY_SCALING;
    float audioTotal = (bass + mid + high) * 0.6;
    
    // Apply motion and dynamics
    float time = iTime * SPEED;
    
    // Create pulsing zoom effect with minimum zoom level to prevent blurring
    float minZoom = 1.0;
    float audioZoom = 0.2 * sin(time * 0.7) + bass * 1.0;
    float zoom = max(minZoom, minZoom + audioZoom * 0.8);
    uv *= zoom;
    
    // Apply mouse interaction if active
    if (iMouse.z > 0.0) {
        vec2 mouseOffset = (iMouse.xy / iResolution.xy - 0.5) * 2.0;
        uv += mouseOffset * 0.5;
    }
    
    // Apply kaleidoscopic effect (symmetry folding)
    float sides = KALEIDOSCOPE_SIDES + floor(mid * 3.0); // Reduced audio influence
    float angle = atan(uv.y, uv.x);
    float radius = length(uv);
    
    // Audio-reactive kaleidoscope
    float segment = 3.14159 * 2.0 / sides;
    angle = mod(angle, segment);
    angle = abs(angle - segment * 0.5);
    
    // Transform back to Cartesian coordinates
    uv = vec2(cos(angle), sin(angle)) * radius;
    
    // Apply audio-reactive domain warping with reduced intensity
    uv = warpDomain(uv, time, 0.2 + bass * 0.6);
    
    // Apply spiral effect based on bass with reduced intensity
    float spiralFactor = 0.4 + bass * 2.0;
    uv = rotateDistort(uv, radius * spiralFactor + time * 0.4, 0.2 + mid * 0.4);
    
    // Create reactive visual feedback loop with reduced intensity
    vec2 feedbackUV = uv;
    feedbackUV = rotateDistort(feedbackUV, time * 0.04 + audioTotal, 0.08);
    feedbackUV *= 0.92 + 0.15 * sin(time * 0.1);
    float feedback = fractalPattern(feedbackUV, time * 0.4, audioTotal * 0.4);
    
    // Generate primary pattern
    float pattern = fractalPattern(uv, time, audioTotal);
    
    // Add a minimum pattern definition to prevent blurriness
    float minimumPattern = 0.1 * sin(radius * 25.0 - time) + 0.15 * sin(angle * 15.0);
    
    // Blend with feedback but limit the blend at low volumes to prevent blurriness
    float adaptiveFeedback = min(FEEDBACK_STRENGTH + bass * 0.2, 0.35);
    pattern = mix(pattern, feedback, adaptiveFeedback);
    
    // Add constant high-frequency detail that's present even at low volumes
    pattern += 0.05 * sin(radius * 30.0) + 0.05 * sin(angle * 20.0);
    
    // Add concentric rings with minimum modulation
    pattern += 0.1 * sin(radius * (10.0 + mid * 15.0) - time * 1.5);
    
    // Add sharpening effect
    float sharpening = 0.1 * sin(radius * 40.0) * sin(angle * 30.0);
    pattern += sharpening;
    
    // Add reduced transient flashes
    pattern += transients * 0.3;
    
    // Map pattern to color palette - KEEPING ORIGINAL COLOR MAPPING INTACT
    vec3 color = purpleGoldPalette(pattern + time * 0.05);
    
    // Add metallic gold highlights - KEEPING THE ORIGINAL INTENSITY
    float shine = metallicHighlight(pattern, time * 1.5, METALLIC_SHININESS);
    color += vec3(1.0, 0.9, 0.3) * shine * (0.5 + bass * 0.5);
    
    // Add purple glow based on high frequencies - KEEPING ORIGINAL
    color += vec3(0.5, 0.0, 1.0) * high * 0.7;
    
    // Add subtle vignette for depth
    float vignette = smoothstep(1.5, 0.5, length(uv / zoom));
    color *= vignette;
    
    // Add subtle chromatic aberration for more psychedelic look (reduced)
    float aberration = 0.008 + high * 0.008;
    vec3 colorShift;
    colorShift.r = purpleGoldPalette(pattern + aberration + time * 0.05).r;
    colorShift.b = purpleGoldPalette(pattern - aberration + time * 0.05).b;
    colorShift.g = color.g;
    color = mix(color, colorShift, 0.3);
    
    // Enhance contrast and add sharpening without using derivatives
    color = pow(color, vec3(0.9 + bass * 0.2));
    
    // Apply alternative edge enhancement for sharper details
    // This uses pattern variation instead of derivatives
    float highFreqPattern = sin(radius * 50.0) * sin(angle * 40.0);
    float patternVariation = abs(pattern - highFreqPattern);
    float edgeStrength = 0.2;
    color = mix(color, color * (1.0 + patternVariation), edgeStrength);
    
    // Final output
    gl_FragColor = vec4(color, 1.0);
}
`
,
"Black Hole Pulser": `
precision highp float;

uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D iChannel0;

void main() {
    // Sample audio with minimal processing
    float bass = texture2D(iChannel0, vec2(0.05, 0.0)).x;
    float lowMid = texture2D(iChannel0, vec2(0.15, 0.0)).x;
    float mid = texture2D(iChannel0, vec2(0.3, 0.0)).x;
    float high = texture2D(iChannel0, vec2(0.7, 0.0)).x;
    
    // Beat detection - compare current bass to a threshold
    // This creates more defined pulses instead of continuous response
    float beatThreshold = 0.15;
    float beatPulse = smoothstep(beatThreshold, beatThreshold + 0.1, bass) * 0.3;
    
    // Overall audio intensity (weighted toward bass)
    float audioIntensity = bass * 0.6 + lowMid * 0.25 + mid * 0.1 + high * 0.05;
    
    // Keep original time speed
    float timeFactor = 0.2; // Original speed from XorDev's shader
    
    // Small pulse effect for beats
    float beatPulseFactor = 1.0 + beatPulse;
    
    // Black hole size/density responds to audio intensity
    // Inverse relationship - higher sound = smaller hole
    float baseHoleSize = 0.7; // Base size
    float holeSize = baseHoleSize * (1.0 - audioIntensity * 0.35);
    
    // Density factor - increases with audio intensity
    float densityFactor = 1.0 + audioIntensity * 0.9;
    
    // Audio-reactive parameters that respect the original algorithm
    float distortion = 0.1;  // Keep original distortion
    float iterationBase = 9.0; // Original iteration count
    float iterationCount = iterationBase; // Keep consistent iterations
    
    // Begin with the same coordinate setup as the original
    float i;
    vec2 r = iResolution.xy,
         p = (gl_FragCoord.xy + gl_FragCoord.xy - r) / r.y / 0.7,
         d = vec2(-1, 1), 
         q = 5.0 * p - d,
         c = p * mat2(1, 1, d / (distortion + 5.0 / dot(q, q)));
         
    // Create the spiral effect using log-polar coordinates - maintaining original speed
    vec2 v = c * mat2(cos(log(length(c)) + iTime * timeFactor + vec4(0, 33, 11, 0))) * 5.0;
   
    // Initialize output color
    vec4 O = vec4(0.0);
    i = 0.0;
    
    // Create the iterative pattern
    for(int j = 0; j < 15; j++) {
        // Break based on audio-reactive iteration count
        if(float(j) >= iterationCount) break;
        
        // Add audio-reactive variation to the sine pattern
        float audioMod = 1.0 + bass * 0.2 * sin(i * 0.7); // Subtle per-iteration audio effect
        O += (1.0 + sin(v.xyyx)) * audioMod;
        
        // Update v with the original formula plus subtle audio influence
        v += 0.7 * sin(v.yx * i + iTime) / (i + 0.1) + 0.5; // Added 0.1 to prevent division by zero
        
        i += 1.0; // Increment our float counter
    }
    
    // Calculate i using the original formula
    i = length(sin(v / 0.3) * 0.2 + c * vec2(1, 2)) - 1.0;
    
    // Apply pulse effect to the visual elements
    // Ring size changes with the hole size to create expansion/contraction
    float ringSize = holeSize;
    
    // Ring width pulses with beats 
    float ringWidth = 0.03 * beatPulseFactor;
    float ringEffect = ringWidth + abs(length(p) - ringSize);
    
    // Glow intensity increases with audio intensity and pulses with beats
    float glowIntensity = 3.5 * beatPulseFactor * densityFactor;
    float glowEffect = 0.5 + glowIntensity * exp(0.3 * c.y - dot(c, c));
    
    // Apply subtle color shift with mid frequency
    vec4 colorModifier = vec4(0.6, -0.4, -1.0, 0.0);
    
    // Final color calculation - density increases with audio intensity
    O = 1.0 - exp(-exp(c.x * colorModifier) / O / (1.0 + i * i * densityFactor) / glowEffect / ringEffect * beatPulseFactor);
    
    // Output
    gl_FragColor = O;
}`,

    "Black Hole Regular": `
    precision highp float;

uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D iChannel0;

void main() {
    // Sample audio with minimal processing
    float bass = texture2D(iChannel0, vec2(0.1, 0.0)).x;
    float mid = texture2D(iChannel0, vec2(0.5, 0.0)).x;
    float high = texture2D(iChannel0, vec2(0.9, 0.0)).x;
    
    // Keep original time speed, but create pulse effects for beats
    float timeFactor = 0.2; // Original speed from XorDev's shader
    float pulseFactor = 1.0 + bass * 0.4; // Subtle pulse effect
    
    // Audio-reactive parameters that respect the original algorithm
    float distortion = 0.1;  // Keep original distortion
    float iterationBase = 9.0; // Original iteration count
    float iterationCount = iterationBase; // Keep consistent iterations
    
    // Begin with the same coordinate setup as the original
    float i;
    vec2 r = iResolution.xy,
         p = (gl_FragCoord.xy + gl_FragCoord.xy - r) / r.y / 0.7,
         d = vec2(-1, 1), 
         q = 5.0 * p - d,
         c = p * mat2(1, 1, d / (distortion + 5.0 / dot(q, q)));
         
    // Create the spiral effect using log-polar coordinates - maintaining original speed
    vec2 v = c * mat2(cos(log(length(c)) + iTime * timeFactor + vec4(0, 33, 11, 0))) * 5.0;
   
    // Initialize output color
    vec4 O = vec4(0.0);
    i = 0.0;
    
    // Create the iterative pattern
    for(int j = 0; j < 15; j++) {
        // Break based on audio-reactive iteration count
        if(float(j) >= iterationCount) break;
        
        // Add audio-reactive variation to the sine pattern
        float audioMod = 1.0 + bass * 0.2 * sin(i * 0.7); // Subtle per-iteration audio effect
        O += (1.0 + sin(v.xyyx)) * audioMod;
        
        // Update v with the original formula plus subtle audio influence
        v += 0.7 * sin(v.yx * i + iTime) / (i + 0.1) + 0.5; // Added 0.1 to prevent division by zero
        
        i += 1.0; // Increment our float counter
    }
    
    // Calculate i using the original formula
    i = length(sin(v / 0.3) * 0.2 + c * vec2(1, 2)) - 1.0;
    
    // Apply pulse effect to the visual elements rather than animation speed
    float ringSize = 0.7 * (1.0 + bass * 0.15); // Subtle pulse on ring size
    float ringWidth = 0.03 * (1.0 + bass * 0.2); // Ring width pulses with bass
    float ringEffect = ringWidth + abs(length(p) - ringSize);
    
    // Glow intensity pulses with the bass
    float glowIntensity = 3.5 * pulseFactor;
    float glowEffect = 0.5 + glowIntensity * exp(0.3 * c.y - dot(c, c));
    
    // Apply subtle color shift with mid frequency
    vec4 colorModifier = vec4(0.6, -0.4, -1.0, 0.0) * (1.0 + mid * vec4(0.1, 0.05, -0.05, 0.0));
    
    // Final color calculation - with pulsation effect on intensity
    O = 1.0 - exp(-exp(c.x * colorModifier) / O / (1.0 + i * i) / glowEffect / ringEffect * pulseFactor);
    
    // Output
    gl_FragColor = O;
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
"Trippy Cheese Rail":`
// "Fractal Cartoon" - former "DE edge detection" by Kali
// Modified with audio reactivity for jumping on beat

// There are no lights and no AO, only color by normals and dark edges.

//#define SHOWONLYEDGES
#define NYAN 
#define WAVES
#define BORDER

#define RAY_STEPS 300

#define BRIGHTNESS 1.2
#define GAMMA 1.4
#define SATURATION .65

#define detail .001

const vec3 origin=vec3(-1.,.7,0.);
float det=0.0;

// Audio analysis variables
float audioLevel = 0.0;      // Current overall audio level
float prevAudioLevel = 0.0;  // Previous audio level for beat detection
float beatIntensity = 0.0;   // Current beat intensity
float jumpHeight = 0.0;      // Current jump height
float timeFactor = 0.0;      // Animation time factor

// Audio analysis function
void processAudio() {
    // Sample multiple frequency bands for a better overall level
    float bassLevel = texture2D(iChannel0, vec2(0.05, 0.0)).x * 1.5;
    float lowMidLevel = texture2D(iChannel0, vec2(0.1, 0.0)).x * 1.0;
    float midLevel = texture2D(iChannel0, vec2(0.2, 0.0)).x * 0.8;
    
    // Calculate overall audio level (emphasizing bass for beat detection)
    float currentLevel = bassLevel * 0.7 + lowMidLevel * 0.2 + midLevel * 0.1;
    
    // Determine audio presence
    bool hasAudio = currentLevel > 0.01;
    
    // Update time factor - move very slowly when no audio, speed up with audio level
    float baseSpeed = 0.05; // Very slow base speed when no music
    float audioSpeed = 0.5; // Full speed with audio (original pace)
    timeFactor = iTime * (baseSpeed + audioLevel * (audioSpeed - baseSpeed));
    
    // Beat detection by comparing current level to previous level
    float beatThreshold = 0.15;  // Minimum intensity to be considered a beat
    float beatRatio = currentLevel / (prevAudioLevel + 0.01);  // Prevent division by zero
    
    // Detect beat when audio level has a significant jump
    if (beatRatio > 1.5 && currentLevel > beatThreshold) {
        // New beat detected
        beatIntensity = currentLevel * 2.5;  // Scale for more impact
    } else {
        // Gradually reduce beat intensity
        beatIntensity *= 0.9;  // Decay factor
    }
    
    // Update jump height based on beat intensity with a maximum cap
    float gravity = 0.1;
    float maxJumpHeight = 0.7; // Reduced maximum height to prevent showing only sky
    jumpHeight = min(maxJumpHeight, max(0.0, jumpHeight + beatIntensity - gravity));
    
    // Store current level for next frame comparison
    prevAudioLevel = mix(prevAudioLevel, currentLevel, 0.3);  // Smooth transition
    audioLevel = currentLevel;
}

// 2D rotation function
mat2 rot(float a) {
    return mat2(cos(a),sin(a),-sin(a),cos(a));    
}

// "Amazing Surface" fractal - standard, no audio effects
vec4 formula(vec4 p) {
    p.xz = abs(p.xz+1.)-abs(p.xz-1.)-p.xz;
    p.y-=.25;
    p.xy*=rot(radians(35.));
    p=p*2./clamp(dot(p.xyz,p.xyz),.2,1.);
    return p;
}

// Distance function - with jump effect on beat
float de(vec3 pos) {
    // Apply jump effect - offset Y position based on jumpHeight
    // The bridge part of the fractal jumps up and falls back
    // Scale factor reduced to prevent excessive jumping
    float jumpOffset = jumpHeight * 0.35 * max(0.0, 1.0 - abs(pos.z - sin(timeFactor*6.0)*1.5) / 2.0);
    pos.y -= jumpOffset;
    
#ifdef WAVES
    pos.y+=sin(pos.z-timeFactor*6.)*.15; //waves!
#endif
    float hid=0.;
    vec3 tpos=pos;
    tpos.z=abs(3.-mod(tpos.z,6.));
    vec4 p=vec4(tpos,1.);
    for (int i=0; i<4; i++) {p=formula(p);}
    float fr=(length(max(vec2(0.),p.yz-1.5))-1.)/p.w;
    float ro=max(abs(pos.x+1.)-.3,pos.y-.35);
          ro=max(ro,-max(abs(pos.x+1.)-.1,pos.y-.5));
    pos.z=abs(.25-mod(pos.z,.5));
          ro=max(ro,-max(abs(pos.z)-.2,pos.y-.3));
          ro=max(ro,-max(abs(pos.z)-.01,-pos.y+.32));
    float d=min(fr,ro);
    return d;
}

// Camera path - original path but controlled by audio
vec3 path(float ti) {
    ti*=1.5;
    vec3 p=vec3(sin(ti),(1.-sin(ti*2.))*.5,-ti*5.)*.5;
    return p;
}

// Calc normals, and here is edge detection, set to variable "edge"
float edge=0.;
vec3 normal(vec3 p) { 
    vec3 e = vec3(0.0,det*5.,0.0);

    float d1=de(p-e.yxx),d2=de(p+e.yxx);
    float d3=de(p-e.xyx),d4=de(p+e.xyx);
    float d5=de(p-e.xxy),d6=de(p+e.xxy);
    float d=de(p);
    edge=abs(d-0.5*(d2+d1))+abs(d-0.5*(d4+d3))+abs(d-0.5*(d6+d5));//edge finder
    edge=min(1.,pow(edge,.55)*15.);
    return normalize(vec3(d1-d2,d3-d4,d5-d6));
}

// Used Nyan Cat code by mu6k, with some mods
vec4 rainbow(vec2 p)
{
    float q = max(p.x,-0.1);
    float s = sin(p.x*7.0+timeFactor*70.0)*0.08;
    p.y+=s;
    p.y*=1.1;
    
    vec4 c;
    if (p.x>0.0) c=vec4(0,0,0,0); else
    if (0.0/6.0<p.y&&p.y<1.0/6.0) c= vec4(255,43,14,255)/255.0; else
    if (1.0/6.0<p.y&&p.y<2.0/6.0) c= vec4(255,168,6,255)/255.0; else
    if (2.0/6.0<p.y&&p.y<3.0/6.0) c= vec4(255,244,0,255)/255.0; else
    if (3.0/6.0<p.y&&p.y<4.0/6.0) c= vec4(51,234,5,255)/255.0; else
    if (4.0/6.0<p.y&&p.y<5.0/6.0) c= vec4(8,163,255,255)/255.0; else
    if (5.0/6.0<p.y&&p.y<6.0/6.0) c= vec4(122,85,255,255)/255.0; else
    if (abs(p.y)-.05<0.0001) c=vec4(0.,0.,0.,1.); else
    if (abs(p.y-1.)-.05<0.0001) c=vec4(0.,0.,0.,1.); else
        c=vec4(0,0,0,0);
    c.a*=.8-min(.8,abs(p.x*.08));
    c.xyz=mix(c.xyz,vec3(length(c.xyz)),.15);
    return c;
}

vec4 nyan(vec2 p)
{
    vec2 uv = p*vec2(0.4,1.0);
    float ns=3.0;
    float nt = timeFactor*ns; nt-=mod(nt,240.0/256.0/6.0); nt = mod(nt,240.0/256.0);
    float ny = mod(timeFactor*ns,1.0); ny-=mod(ny,0.75); ny*=-0.05;
    vec4 color = texture(iChannel1,vec2(uv.x/3.0+210.0/256.0-nt+0.05,.5-uv.y-ny));
    if (uv.x<-0.3) color.a = 0.0;
    if (uv.x>0.2) color.a=0.0;
    return color;
}

// Raymarching and 2D graphics
vec3 raymarch(in vec3 from, in vec3 dir) 
{
    edge=0.;
    vec3 p, norm;
    float d=100.;
    float totdist=0.;
    for (int i=0; i<RAY_STEPS; i++) {
        if (d>det && totdist<25.0) {
            p=from+totdist*dir;
            d=de(p);
            det=detail*exp(.13*totdist);
            totdist+=d; 
        }
    }
    vec3 col=vec3(0.);
    p-=(det-d)*dir;
    norm=normal(p);
#ifdef SHOWONLYEDGES
    col=1.-vec3(edge); // show wireframe version
#else
    col=(1.-abs(norm))*max(0.,1.-edge*.8); // set normal as color with dark edges
#endif        
    totdist=clamp(totdist,0.,26.);
    dir.y-=.02;
    float sunsize=7.-max(0.,texture(iChannel0,vec2(.6,.2)).x)*5.; // responsive sun size
    float an=atan(dir.x,dir.y)+timeFactor*1.5; // angle for drawing and rotating sun
    float s=pow(clamp(1.0-length(dir.xy)*sunsize-abs(.2-mod(an,.4)),0.,1.),.1); // sun
    float sb=pow(clamp(1.0-length(dir.xy)*(sunsize-.2)-abs(.2-mod(an,.4)),0.,1.),.1); // sun border
    float sg=pow(clamp(1.0-length(dir.xy)*(sunsize-4.5)-.5*abs(.2-mod(an,.4)),0.,1.),3.); // sun rays
    float y=mix(.45,1.2,pow(smoothstep(0.,1.,.75-dir.y),2.))*(1.-sb*.5); // gradient sky
    
    // set up background with sky and sun
    vec3 backg=vec3(0.5,0.,1.)*((1.-s)*(1.-sg)*y+(1.-sb)*sg*vec3(1.,.8,0.15)*3.);
         backg+=vec3(1.,.9,.1)*s;
         backg=max(backg,sg*vec3(1.,.9,.5));
    
    col=mix(vec3(1.,.9,.3),col,exp(-.004*totdist*totdist));// distant fading to sun color
    if (totdist>25.) col=backg; // hit background
    col=pow(col,vec3(GAMMA))*BRIGHTNESS;
    col=mix(vec3(length(col)),col,SATURATION);
#ifdef SHOWONLYEDGES
    col=1.-vec3(length(col));
#else
    col*=vec3(1.,.9,.85);
#ifdef NYAN
    // Make Nyan Cat jump up and down with the beat
    float nyanJump = jumpHeight * 0.7;
    
    dir.yx*=rot(dir.x);
    vec2 ncatpos=(dir.xy+vec2(-3.+mod(-timeFactor,6.),-.27-nyanJump)); // Jump applies to Y position
    vec4 ncat=nyan(ncatpos*5.);
    vec4 rain=rainbow(ncatpos*10.+vec2(.8,.5));
    if (totdist>8.) col=mix(col,max(vec3(.2),rain.xyz),rain.a*.9);
    if (totdist>8.) col=mix(col,max(vec3(.2),ncat.xyz),ncat.a*.9);
#endif
#endif
    return col;
}

// get camera position
vec3 move(inout vec3 dir) {
    vec3 go=path(timeFactor);
    vec3 adv=path(timeFactor+.7);
    float hd=de(adv);
    vec3 advec=normalize(adv-go);
    float an=adv.x-go.x; an*=min(1.,abs(adv.z-go.z))*sign(adv.z-go.z)*.7;
    dir.xy*=mat2(cos(an),sin(an),-sin(an),cos(an));
    
    // Add slight camera jump to reflect the beat intensity
    // This makes the camera jump up and down with the beat
    float cameraJump = jumpHeight * 0.15; // Scale down for more subtle effect
    an=advec.y*1.7 + cameraJump;
    
    dir.yz*=mat2(cos(an),sin(an),-sin(an),cos(an));
    an=atan(advec.x,advec.z);
    dir.xz*=mat2(cos(an),sin(an),-sin(an),cos(an));
    
    // Apply a small vertical offset to camera position based on jump
    vec3 jumpOffset = vec3(0.0, jumpHeight * 0.1, 0.0);
    return go + jumpOffset;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Process audio first to update all the audio-reactive variables
    processAudio();
    
    vec2 uv = fragCoord.xy / iResolution.xy*2.-1.;
    vec2 oriuv=uv;
    uv.y*=iResolution.y/iResolution.x;
    vec2 mouse=(iMouse.xy/iResolution.xy-.5)*3.;
    if (iMouse.z<1.) mouse=vec2(0.,-0.05);
    float fov=.9-max(0.,.7-iTime*.3);
    vec3 dir=normalize(vec3(uv*fov,1.));
    dir.yz*=rot(mouse.y);
    dir.xz*=rot(mouse.x);
    vec3 from=origin+move(dir);
    vec3 color=raymarch(from,dir); 
#ifdef BORDER
    color=mix(vec3(0.),color,pow(max(0.,.95-length(oriuv*oriuv*oriuv*vec2(1.05,1.1))),.3));
#endif
    fragColor = vec4(color,1.);
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
"Ankara 2": `precision highp float;

uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D iChannel0;
uniform vec4 iMouse;

// ========== CONFIGURATION ==========
const float PATTERN_SCALE = 8.0;    // Scale of patterns
const float ANIMATION_SPEED = 0.3;  // Speed of animations
const float COLOR_INTENSITY = 1.2;  // Color vibrancy

// ========== HELPER FUNCTIONS ==========
// Simple hash function
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// Basic noise function
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

// West African inspired color palette
vec3 africanPalette(float t) {
    // Rich gold, earth tones, and indigo colors inspired by
    // Kente cloth, mud cloth, and West African textiles
    const vec3 gold = vec3(0.9, 0.7, 0.1);
    const vec3 earthRed = vec3(0.75, 0.3, 0.1);
    const vec3 indigo = vec3(0.1, 0.15, 0.4);
    const vec3 forestGreen = vec3(0.0, 0.4, 0.2);
    const vec3 black = vec3(0.1, 0.05, 0.05);
    
    t = fract(t);
    
    if (t < 0.2) return mix(black, indigo, t * 5.0);
    else if (t < 0.4) return mix(indigo, forestGreen, (t - 0.2) * 5.0);
    else if (t < 0.6) return mix(forestGreen, earthRed, (t - 0.4) * 5.0);
    else if (t < 0.8) return mix(earthRed, gold, (t - 0.6) * 5.0);
    else return mix(gold, black, (t - 0.8) * 5.0);
}

// ========== AUDIO ANALYSIS ==========
// Get bass power
float getAudioBass() {
    float bass = 0.0;
    for (int i = 0; i < 15; i++) {
        float sample = texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
        bass += sample * sample;  // Square for better dynamics
    }
    return min(bass * 0.2, 1.0);
}

// Get mid frequencies
float getAudioMid() {
    float mid = 0.0;
    for (int i = 15; i < 50; i++) {
        mid += texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
    }
    return min(mid * 0.05, 1.0);
}

// Get high frequencies
float getAudioHigh() {
    float high = 0.0;
    for (int i = 50; i < 100; i++) {
        high += texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
    }
    return min(high * 0.04, 1.0);
}

// ========== PATTERN FUNCTIONS ==========
// Create kente-inspired pattern
float kentePattern(vec2 uv, float time, float audio) {
    // Scale UV for pattern size
    vec2 st = uv * PATTERN_SCALE;
    
    // Create grid cells
    vec2 grid = floor(st);
    float cellX = grid.x;
    float cellY = grid.y;
    
    // Create seed for this cell
    float seed = hash(grid);
    
    // Determine pattern direction (horizontal or vertical stripes)
    bool horizontal = mod(cellY, 2.0) < 1.0;
    
    // Create fractional position within cell
    vec2 pos = fract(st);
    
    // Create stripe pattern
    float pattern;
    
    if (horizontal) {
        // Horizontal stripes
        float stripeWidth = 0.2 + 0.2 * seed;
        pattern = step(stripeWidth, mod(pos.y + time * 0.1, 1.0));
        
        // Add some detail to stripes
        if (mod(cellX, 3.0) < 1.0) {
            float detail = step(0.5, mod(pos.x * 5.0, 1.0));
            pattern = horizontal ? pattern * detail : pattern;
        }
    } else {
        // Vertical stripes
        float stripeWidth = 0.2 + 0.2 * seed;
        pattern = step(stripeWidth, mod(pos.x + time * 0.1, 1.0));
        
        // Add some detail to stripes
        if (mod(cellY, 3.0) < 1.0) {
            float detail = step(0.5, mod(pos.y * 5.0, 1.0));
            pattern = pattern * detail;
        }
    }
    
    // Make pattern react to audio
    pattern *= 0.7 + audio * 0.5;
    
    return pattern;
}

// Create circular symbolic patterns
float symbolPattern(vec2 uv, float time, float audio) {
    // Distance from center
    float dist = length(uv);
    
    // Angle for radial patterns
    float angle = atan(uv.y, uv.x);
    
    // Create ripple pattern
    float ripples = 0.5 + 0.5 * sin(dist * 20.0 - time * 3.0);
    
    // Create radial lines
    float lines = 0.5 + 0.5 * sin(angle * 8.0);
    
    // Mix patterns based on audio
    float pattern = mix(ripples, lines, audio);
    
    // Add center circle
    float circle = 1.0 - smoothstep(0.1, 0.2, dist);
    
    return pattern * (1.0 - circle) + circle;
}

// Create mud cloth inspired geometric patterns
float mudClothPattern(vec2 uv, float time, float audio) {
    // Scale for pattern
    vec2 st = uv * (PATTERN_SCALE * 0.5);
    
    // Grid cells
    vec2 grid = floor(st);
    vec2 pos = fract(st);
    
    // Seed for this cell
    float seed = hash(grid + time * 0.1);
    
    // Choose pattern type based on seed
    float patternType = floor(seed * 4.0);
    
    float pattern = 0.0;
    
    if (patternType < 1.0) {
        // Dots pattern
        vec2 dotPos = pos - 0.5;
        pattern = 1.0 - smoothstep(0.1, 0.2, length(dotPos));
    } 
    else if (patternType < 2.0) {
        // Cross pattern
        float lineX = 1.0 - smoothstep(0.1, 0.15, abs(pos.x - 0.5));
        float lineY = 1.0 - smoothstep(0.1, 0.15, abs(pos.y - 0.5));
        pattern = max(lineX, lineY);
    }
    else if (patternType < 3.0) {
        // Diamond pattern
        vec2 diamondPos = abs(pos - 0.5);
        pattern = 1.0 - smoothstep(0.2, 0.25, diamondPos.x + diamondPos.y);
    }
    else {
        // Zigzag pattern
        float zigzag = abs(pos.y - (0.5 + 0.2 * sin(pos.x * 6.28 + time)));
        pattern = 1.0 - smoothstep(0.1, 0.15, zigzag);
    }
    
    // Make pattern react to audio
    pattern *= 0.7 + audio * 0.5;
    
    return pattern;
}

// ========== MAIN FUNCTION ==========
void main() {
    // Normalized coordinates
    vec2 uv = gl_FragCoord.xy / iResolution.xy;
    
    // Center coordinates
    vec2 centered = (uv - 0.5) * 2.0;
    centered.x *= iResolution.x / iResolution.y; // Correct aspect ratio
    
    // Time and animation
    float time = iTime * ANIMATION_SPEED;
    
    // Get audio values
    float bass = getAudioBass();
    float mid = getAudioMid();
    float high = getAudioHigh();
    float audioTotal = bass + mid + high;
    
    // Apply mouse movement if active
    if (iMouse.z > 0.0) {
        vec2 mouseOffset = 2.0 * (iMouse.xy / iResolution.xy - 0.5);
        centered += mouseOffset * 0.3;
    }
    
    // Background noise texture
    float bgNoise = noise(centered * 3.0 + time * 0.1);
    
    // Create base color
    vec3 color = africanPalette(bgNoise + time * 0.1);
    
    // Add kente cloth inspired pattern
    float kente = kentePattern(centered * 0.5, time, mid);
    color = mix(color, africanPalette(kente + time * 0.2), kente * 0.8);
    
    // Add symbolic patterns that react to bass
    float symbolScale = 0.5 + bass * 0.5; // Pattern gets larger with bass
    float symbol = symbolPattern(centered * symbolScale, time, bass);
    color = mix(color, africanPalette(symbol + bass + time * 0.3), symbol * bass * 0.7);
    
    // Add mud cloth inspired patterns for high frequencies
    float mudCloth = mudClothPattern(centered, time, high);
    color = mix(color, africanPalette(mudCloth + time * 0.15), mudCloth * high * 0.6);
    
    // Add global audio reactivity - make everything pulse with the beat
    color *= 0.8 + audioTotal * 0.4;
    
    // Add subtle vignette
    float vignette = smoothstep(1.5, 0.5, length(centered));
    color *= vignette;
    
    // Boost saturation
    color = mix(color, color * COLOR_INTENSITY, 0.5);
    
    // Output final color
    gl_FragColor = vec4(color, 1.0);
}
`,
"3D Play": `precision highp float;

uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D iChannel0;
uniform vec4 iMouse;

// Constants for 3D visualization
const int MAX_STEPS = 100;       // Maximum raymarching steps
const float MIN_DIST = 0.001;    // Minimum distance to surface
const float MAX_DIST = 50.0;     // Maximum raymarching distance
const float EPSILON = 0.0001;    // Small value for normal calculations

// Visual style parameters
const float SPEED = 0.5;         // Animation speed
const int ITERATIONS = 5;        // Detail level for fractals
const float GLOW_INTENSITY = 1.2; // Intensity of volumetric glow

// Audio reactive parameters
float audioLow = 0.0;
float audioMid = 0.0;
float audioHigh = 0.0;
float audioTotal = 0.0;

// Color functions
vec3 purpleGoldGradient(float t) {
    // Rich purple to gold color palette
    vec3 purple = vec3(0.4, 0.0, 0.7);
    vec3 magenta = vec3(0.7, 0.1, 0.5);
    vec3 gold = vec3(1.0, 0.8, 0.1);
    vec3 amber = vec3(1.0, 0.6, 0.0);
    
    t = fract(t);
    if (t < 0.33) {
        return mix(purple, magenta, t * 3.0);
    } else if (t < 0.66) {
        return mix(magenta, gold, (t - 0.33) * 3.0);
    } else {
        return mix(gold, amber, (t - 0.66) * 3.0);
    }
}

// Rotation matrix for 3D transformations
mat3 rotateY(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat3(
        c, 0.0, s,
        0.0, 1.0, 0.0,
        -s, 0.0, c
    );
}

mat3 rotateX(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat3(
        1.0, 0.0, 0.0,
        0.0, c, -s,
        0.0, s, c
    );
}

// Audio analysis functions
void analyzeAudio() {
    // Sample low frequencies (bass)
    audioLow = 0.0;
    for (int i = 0; i < 10; i++) {
        audioLow += texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
    }
    audioLow *= 0.15; // Normalize
    
    // Sample mid frequencies
    audioMid = 0.0;
    for (int i = 10; i < 40; i++) {
        audioMid += texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
    }
    audioMid *= 0.04; // Normalize
    
    // Sample high frequencies
    audioHigh = 0.0;
    for (int i = 40; i < 80; i++) {
        audioHigh += texture2D(iChannel0, vec2(float(i) / 128.0, 0.0)).x;
    }
    audioHigh *= 0.03; // Normalize
    
    // Total audio intensity
    audioTotal = audioLow + audioMid + audioHigh;
}

// SDF (Signed Distance Function) for a sphere
float sdSphere(vec3 p, float radius) {
    return length(p) - radius;
}

// SDF for a box
float sdBox(vec3 p, vec3 size) {
    vec3 d = abs(p) - size;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

// SDF for a torus
float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

// Twist space transformation
vec3 twist(vec3 p, float strength) {
    float c = cos(strength * p.y);
    float s = sin(strength * p.y);
    mat2 m = mat2(c, -s, s, c);
    return vec3(m * p.xz, p.y);
}

// Main shape function - creates a complex audio-reactive 3D shape
float sceneSDF(vec3 p) {
    // Apply audio-reactive transformations
    float time = iTime * SPEED;
    
    // Create audio-reactive rotations
    mat3 rot = rotateY(time * 0.3 + audioLow * 5.0) * rotateX(time * 0.2 + audioMid * 3.0);
    vec3 p1 = rot * p;
    
    // Apply twist based on high frequencies
    vec3 p2 = twist(p1, 0.3 + audioHigh * 3.0);
    
    // Create base shape - torus
    float torusSize = 2.0 + audioLow * 2.0;
    float torusThickness = 0.5 + audioMid * 0.8;
    float mainShape = sdTorus(p2, vec2(torusSize, torusThickness));
    
    // Add spheres along the torus path
    float sphereDistance = 1000.0;
    int numSpheres = 8 + int(audioMid * 12.0);
    for (int i = 0; i < 12; i++) {
        if (i >= numSpheres) break;
        
        float angle = float(i) * 6.28318 / float(numSpheres);
        vec3 offset = vec3(cos(angle) * torusSize, 0.0, sin(angle) * torusSize);
        float sphereSize = 0.6 + 0.4 * sin(time + float(i) * 0.5) + audioHigh * 0.8;
        
        // Apply audio-reactive oscillation to sphere positions
        offset.y += sin(time * 1.5 + float(i)) * (0.5 + audioMid * 2.0);
        
        float sphere = sdSphere(p2 - offset, sphereSize);
        sphereDistance = min(sphereDistance, sphere);
    }
    
    // Add central geometric structure
    float centralSize = 1.0 + audioLow * 1.5;
    vec3 boxSize = vec3(centralSize, centralSize, centralSize);
    float centralBox = sdBox(p2 * (1.0 + sin(time * 0.3) * 0.1), boxSize);
    
    // Combine shapes with smooth union for organic feel
    float k = 0.2 + audioTotal * 0.5; // Smoothing factor
    mainShape = min(mainShape, sphereDistance);
    float d = mainShape - centralBox * (0.5 + audioMid * 0.5);
    return d;
}

// Calculate surface normal
vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
        sceneSDF(vec3(p.x, p.y, p.z + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

// Raymarching algorithm
float raymarch(vec3 ro, vec3 rd) {
    float depth = 0.0;
    
    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + depth * rd;
        float dist = sceneSDF(p);
        
        if (dist < MIN_DIST) {
            return depth;
        }
        
        depth += dist;
        
        if (depth >= MAX_DIST) {
            return MAX_DIST;
        }
    }
    
    return MAX_DIST;
}

// Calculate ambient occlusion
float calculateAO(vec3 p, vec3 n) {
    float ao = 0.0;
    float weight = 1.0;
    
    for (int i = 0; i < 5; i++) {
        float dist = 0.1 + 0.1 * float(i);
        float sampleDist = sceneSDF(p + n * dist);
        ao += weight * (dist - sampleDist);
        weight *= 0.5;
    }
    
    return 1.0 - clamp(ao * 2.0, 0.0, 1.0);
}

// Volumetric glow effect
float calculateGlow(vec3 ro, vec3 rd, float depth) {
    float glow = 0.0;
    float t = 0.1;
    
    for (int i = 0; i < 16; i++) {
        vec3 p = ro + rd * t;
        float d = abs(sceneSDF(p));
        
        // Add glow based on proximity to surface
        glow += 0.1 / (1.0 + d * d * 40.0);
        
        t += 0.2;
        if (t >= depth) break;
    }
    
    return glow * GLOW_INTENSITY * (1.0 + audioTotal * 2.0);
}

// Main rendering function
vec3 render(vec3 ro, vec3 rd) {
    // Raymarch to find distance
    float d = raymarch(ro, rd);
    
    // Base color for background (space)
    vec3 color = vec3(0.01, 0.0, 0.03);
    
    // If we hit a surface
    if (d < MAX_DIST) {
        vec3 p = ro + rd * d;
        vec3 normal = estimateNormal(p);
        
        // Basic lighting
        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
        float diff = max(dot(normal, lightDir), 0.0);
        
        // Add ambient occlusion
        float ao = calculateAO(p, normal);
        
        // Determine base color based on position and time
        float colorFactor = length(p) * 0.1 + iTime * 0.1;
        colorFactor += audioMid * 2.0; // Audio-reactive color shifting
        vec3 objectColor = purpleGoldGradient(colorFactor);
        
        // Add a second light source that's audio reactive
        vec3 lightDir2 = normalize(vec3(sin(iTime), cos(iTime * 0.5), 0.5));
        float diff2 = max(dot(normal, lightDir2), 0.0) * (0.5 + audioHigh * 2.0);
        
        // Add specular highlight
        vec3 reflection = reflect(rd, normal);
        float spec = pow(max(dot(reflection, lightDir), 0.0), 10.0);
        
        // Combine lighting
        color = objectColor * (diff * 0.5 + 0.5) * ao;
        color += objectColor * diff2 * 0.5;
        color += vec3(1.0, 0.9, 0.5) * spec * 0.5; // Golden specular highlight
        
        // Add rim lighting in purple
        float rim = 1.0 - max(dot(-rd, normal), 0.0);
        rim = pow(rim, 3.0);
        color += vec3(0.5, 0.0, 1.0) * rim * 0.3 * (1.0 + audioHigh * 3.0);
    }
    
    // Add glow
    float glow = calculateGlow(ro, rd, d);
    color += purpleGoldGradient(iTime * 0.05 + audioLow) * glow;
    
    // Add subtle fog/depth effect
    color = mix(color, vec3(0.01, 0.0, 0.03), 1.0 - exp(-0.01 * d * d));
    
    // Add subtle vignette
    vec2 uv = gl_FragCoord.xy / iResolution.xy;
    color *= 0.5 + 0.5 * pow(16.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.2);
    
    return color;
}

void main() {
    // Analyze audio first
    analyzeAudio();
    
    // Setup camera
    vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / min(iResolution.x, iResolution.y);
    
    // Camera setup with audio-reactive movement
    float camDist = 8.0 + sin(iTime * 0.3) * 2.0 - audioLow * 3.0;
    float camAngle = iTime * 0.2;
    if (iMouse.z > 0.0) {
        // Allow user interaction if mouse is pressed
        camAngle = 10.0 * iMouse.x / iResolution.x;
        camDist = 5.0 + 10.0 * iMouse.y / iResolution.y;
    }
    
    vec3 ro = vec3(camDist * sin(camAngle), 1.0 + sin(iTime * 0.5) * 0.5, camDist * cos(camAngle));
    vec3 target = vec3(0.0, 0.0, 0.0);
    
    // Add subtle camera shake on beat
    if (audioLow > 0.5) {
        ro += vec3(sin(iTime * 20.0), cos(iTime * 15.0), sin(iTime * 17.0)) * audioLow * 0.1;
    }
    
    // Camera orientation
    vec3 forward = normalize(target - ro);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(right, forward));
    
    // Ray direction
    vec3 rd = normalize(forward + uv.x * right + uv.y * up);
    
    // Render the scene
    vec3 color = render(ro, rd);
    
    // Tone mapping and gamma correction
    color = pow(color, vec3(0.4545)); // Gamma correction
    
    // Add subtle film grain
    float grain = fract(sin(gl_FragCoord.x * 12.9898 + gl_FragCoord.y * 78.233 + iTime) * 43758.5453);
    color += (grain - 0.5) * 0.03;
    
    gl_FragColor = vec4(color, 1.0);
}`
};

