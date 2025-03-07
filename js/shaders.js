
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
    "falling":``
};

