
// Sample Shadertoy shaders to quickly test
export const SAMPLE_SHADERS = {
    "Need Space": `//CBS
//Parallax scrolling fractal galaxy.
//Inspired by JoshP's Simplicity shader: https://www.shadertoy.com/view/lslGWr

// http://www.fractalforums.com/new-theories-and-research/very-simple-formula-for-fractal-patterns/
float field(in vec3 p,float s) {
	float strength = 7. + .03 * log(1.e-6 + fract(sin(iTime) * 4373.11));
	float accum = s/4.;
	float prev = 0.;
	float tw = 0.;
	for (int i = 0; i < 26; ++i) {
		float mag = dot(p, p);
		p = abs(p) / mag + vec3(-.5, -.4, -1.5);
		float w = exp(-float(i) / 7.);
		accum += w * exp(-strength * pow(abs(mag - prev), 2.2));
		tw += w;
		prev = mag;
	}
	return max(0., 5. * accum / tw - .7);
}

// Less iterations for second layer
float field2(in vec3 p, float s) {
	float strength = 7. + .03 * log(1.e-6 + fract(sin(iTime) * 4373.11));
	float accum = s/4.;
	float prev = 0.;
	float tw = 0.;
	for (int i = 0; i < 18; ++i) {
		float mag = dot(p, p);
		p = abs(p) / mag + vec3(-.5, -.4, -1.5);
		float w = exp(-float(i) / 7.);
		accum += w * exp(-strength * pow(abs(mag - prev), 2.2));
		tw += w;
		prev = mag;
	}
	return max(0., 5. * accum / tw - .7);
}

vec3 nrand3( vec2 co )
{
	vec3 a = fract( cos( co.x*8.3e-3 + co.y )*vec3(1.3e5, 4.7e5, 2.9e5) );
	vec3 b = fract( sin( co.x*0.3e-3 + co.y )*vec3(8.1e5, 1.0e5, 0.1e5) );
	vec3 c = mix(a, b, 0.5);
	return c;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 uv = 2. * fragCoord.xy / iResolution.xy - 1.;
	vec2 uvs = uv * iResolution.xy / max(iResolution.x, iResolution.y);
	vec3 p = vec3(uvs / 4., 0) + vec3(1., -1.3, 0.);
	p += .2 * vec3(sin(iTime / 16.), sin(iTime / 12.),  sin(iTime / 128.));
	
	float freqs[4];
	//Sound
	freqs[0] = texture( iChannel0, vec2( 0.01, 0.25 ) ).x;
	freqs[1] = texture( iChannel0, vec2( 0.07, 0.25 ) ).x;
	freqs[2] = texture( iChannel0, vec2( 0.15, 0.25 ) ).x;
	freqs[3] = texture( iChannel0, vec2( 0.30, 0.25 ) ).x;

	float t = field(p,freqs[2]);
	float v = (1. - exp((abs(uv.x) - 1.) * 6.)) * (1. - exp((abs(uv.y) - 1.) * 6.));
	
    //Second Layer
	vec3 p2 = vec3(uvs / (4.+sin(iTime*0.11)*0.2+0.2+sin(iTime*0.15)*0.3+0.4), 1.5) + vec3(2., -1.3, -1.);
	p2 += 0.25 * vec3(sin(iTime / 16.), sin(iTime / 12.),  sin(iTime / 128.));
	float t2 = field2(p2,freqs[3]);
	vec4 c2 = mix(.4, 1., v) * vec4(1.3 * t2 * t2 * t2 ,1.8  * t2 * t2 , t2* freqs[0], t2);
	
	
	//Let's add some stars
	//Thanks to http://glsl.heroku.com/e#6904.0
	vec2 seed = p.xy * 2.0;	
	seed = floor(seed * iResolution.x);
	vec3 rnd = nrand3( seed );
	vec4 starcolor = vec4(pow(rnd.y,40.0));
	
	//Second Layer
	vec2 seed2 = p2.xy * 2.0;
	seed2 = floor(seed2 * iResolution.x);
	vec3 rnd2 = nrand3( seed2 );
	starcolor += vec4(pow(rnd2.y,40.0));
	
	fragColor = mix(freqs[3]-.3, 1., v) * vec4(1.5*freqs[2] * t * t* t , 1.2*freqs[1] * t * t, freqs[3]*t, 1.0)+c2+starcolor;
}`,
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
`
,

"Condense Lava Lamp": `#define T iTime

#define PSD (abs(texture(iChannel0, vec2(.5)).r)*abs(texture(iChannel0, vec2(.5)).r))

// HG_SDF rotate function
#define r(p, a) {p = cos(a)*p + sin(a)*vec2(p.y,-p.x);}

// Cabbibo's HSV
vec3 hsv(float h, float s, float v) {return mix( vec3( 1.0 ), clamp( ( abs( fract(h + vec3( 3.0, 2.0, 1.0 ) / 3.0 ) * 6.0 - 3.0 ) - 1.0 ), 0.0, 1.0 ), s ) * v;}

void mainImage( out vec4 c, in vec2 w )
{
	vec2 u = (-iResolution.xy+2.*w.xy) / iResolution.y;
    vec3 ro = vec3(u, 1.), rd = normalize(vec3(u, -1.)), p; // Camera and ray dir
    float d = 0., m; // Distance for march
    for (float i = 1.; i > 0.; i-=0.02)
    {
        p = ro + rd * d;
        r(p.zy, T);
        r(p.zx, T);
        m = length(cos(abs(p)+sin(abs(p))+T))-(PSD + .5); // Distance function
        d += m;
        c = vec4(hsv(T, 1.,1.)*i*i, 1.);
        if (m < 0.02) break;
    }
    
}`,
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

"Ankara Test": `precision highp float;

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

#define GAMMA 1.4
#define SATURATION 0.85
#define BRIGHTNESS 0.9  // Reduced from 1.2

// Audio analysis zones
const float BASS_START = 0.0;
const float BASS_END = 0.2;
const float MIDS_START = 0.2;
const float MIDS_END = 0.6;
const float HIGHS_START = 0.6;
const float HIGHS_END = 1.0;
const float AUDIO_STEP = 0.01;

// Fractal parameters
const int MAX_ITERATIONS = 12;
const float BAILOUT = 2.0;
const float POWER = 2.0;

// Color palette function with darker base
vec3 palette(float t) {
    // Reduced base brightness for richer colors
    return vec3(
        0.4 + 0.4 * sin(6.28318 * (t + 0.0)),
        0.4 + 0.4 * sin(6.28318 * (t + 0.333)),
        0.4 + 0.4 * sin(6.28318 * (t + 0.666))
    );
}

// Audio reactive parameters
float getAudioLevel(float start, float end) {
    float level = 0.0;
    float count = 0.0;
    
    for(float i = 0.0; i < 1.0; i += AUDIO_STEP) {
        if(i >= start && i <= end) {
            level += texture2D(iChannel0, vec2(i, 0.0)).x;
            count += 1.0;
        }
    }
    return count > 0.0 ? level / count : 0.0;
}

vec2 rot(vec2 p, float a) {
    float c = cos(a), s = sin(a);
    return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

float fractal(vec3 p) {
    float bass = getAudioLevel(BASS_START, BASS_END);
    float mids = getAudioLevel(MIDS_START, MIDS_END);
    float highs = getAudioLevel(HIGHS_START, HIGHS_END);
    
    float scale = 1.0 + bass * 0.4;  // Reduced bass influence
    
    vec3 z = p;
    float dr = 1.0;
    float r = 0.0;
    
    for(int i = 0; i < MAX_ITERATIONS; i++) {
        r = length(z);
        if(r > BAILOUT) break;
        
        float theta = acos(z.z / r);
        float phi = atan(z.y, z.x);
        
        dr = pow(r, POWER - 1.0) * POWER * dr + 1.0;
        
        float zr = pow(r, POWER);
        theta = theta * POWER + mids * 0.4;  // Reduced mid influence
        phi = phi * POWER + highs * 0.2;     // Reduced high influence
        
        z = zr * vec3(
            sin(theta) * cos(phi),
            sin(theta) * sin(phi),
            cos(theta)
        );
        
        z += p * scale;
    }
    return 0.5 * log(r) * r / dr;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / min(iResolution.x, iResolution.y);
    
    float bass = getAudioLevel(BASS_START, BASS_END);
    float mids = getAudioLevel(MIDS_START, MIDS_END);
    float highs = getAudioLevel(HIGHS_START, HIGHS_END);
    
    float time = iTime * 0.3;
    vec3 camera = vec3(
        3.0 * sin(time + bass),
        2.0 * cos(time * 0.5 + mids),
        4.0 * cos(time + highs)
    );
    
    vec3 lookat = vec3(0.0);
    vec3 forward = normalize(lookat - camera);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(right, forward));
    
    vec3 rd = normalize(forward + right * uv.x + up * uv.y);
    
    float t = 0.0;
    float detail = 0.001 + bass * 0.01;
    
    vec3 color = vec3(0.0);
    
    // Adjusted raymarching loop for better depth
    for(int i = 0; i < 100; i++) {
        vec3 pos = camera + rd * t;
        float dist = fractal(pos);
        
        if(dist < detail || t > 20.0) break;
        t += dist * 0.5;
        
        // Reduced color intensity and adjusted layering
        vec3 col = palette(t * 0.15 + iTime * 0.1);
        col *= 0.7 + bass * 1.5;     // Reduced bass boost
        col += mids * 0.3 * palette(t * 0.2);   // Reduced mids
        col += highs * 0.2 * palette(t * 0.3);  // Reduced highs
        
        // Adjusted color accumulation with stronger distance falloff
        color += col * 0.08 * exp(-t * 0.1);
    }
    
    // Enhanced contrast in post-processing
    color = pow(color, vec3(GAMMA));
    color *= BRIGHTNESS;
    color = mix(vec3(length(color)), color, SATURATION);
    
    // Stronger vignette effect
    float vignette = 1.0 - dot(uv, uv) * 0.7;  // Increased vignette intensity
    vignette = pow(vignette, 1.5 + bass * 1.5);
    color *= vignette;
    
    // Add subtle dark edges
    color *= 0.8 + 0.2 * smoothstep(0.0, 0.1, length(uv));
    
    fragColor = vec4(color, 1.0);
}`,
"Chaossss": `void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec2 p = (-3.5 + 7.0 * uv) * vec2(iResolution.x/iResolution.y, 1.0);
    
    // Sound texture sampling
    int soundTx = int(uv.x * 512.0);
    float wave = texelFetch(iChannel0, ivec2(soundTx, 1), 0).x;
    
    // Calculate base distance
    float d = length(p/1.0 - vec2(sin(iTime*0.1), cos(iTime*0.4)));
    p = p/1.0 - vec2(sin(iTime*0.1), cos(iTime*0.4));
    
    // Avoid potential division by zero in sin(wave + uv.y)
    float waveOffset = wave + uv.y + 0.001;
    d = smoothstep(0.4, 0.5, d * d * (-2.0 * sin((wave/1.5) - uv.y)) + 2.0/sin(waveOffset)) * d * d;
    
    // Calculate phi, handling potential division by zero
    float phi = p.x != 0.0 ? atan(p.y, p.x) - iTime * 0.8 : sign(p.y) * 3.14159/2.0 - iTime * 0.8;
    
    d *= sin(phi * 46.0);
    
    // Fix potential undefined behavior in color calculations
    vec3 col = vec3(atan(max(d, -100.0)), 0.1, -sqrt(max(0.0, d))) + d;
    col *= sin(iTime/(d + 0.001) * (d + 0.001));
    col -= d/1.2;
    
    // Avoid division by zero and undefined behavior
    float denominator = max(p.x * p.x + p.y * p.y, 0.001);
    float timeEffect = tan(mod(iTime, 6.28318530718)) * 0.5;
    col *= 1.0 * sin((-(denominator) * 2.0 - iTime * 6.0) / -denominator * timeEffect);
    
    // Clamp final color to avoid undefined values
    col = clamp(col, -1.0, 1.0);
    
    fragColor = vec4(col, 1.0);
}`,
"House": `precision mediump float;

uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D iChannel0;
uniform vec4 iMouse;

// House music specific frequency ranges
#define KICK_FREQ 0.07     // 20-60Hz for kick drums
#define BASS_FREQ 0.15     // 60-150Hz for bass lines
#define MID_FREQ 0.4       // Mids for synths and vocals
#define HIGH_FREQ 0.7      // Highs for hi-hats and cymbals

// House-inspired color palette
#define NEON_BLUE vec3(0.0, 0.8, 1.0)
#define NEON_PINK vec3(1.0, 0.1, 0.8)
#define NEON_GREEN vec3(0.1, 1.0, 0.4)
#define NEON_PURPLE vec3(0.6, 0.0, 1.0)

// Beat detection vars
float prevKick = 0.0;
float kickTrigger = 0.0;
float beatCount = 0.0;
float lastBeatTime = 0.0;

// Get frequency response from a specific range
float getFrequencyResponse(float lowFreq, float highFreq) {
    float sum = 0.0;
    int steps = int((highFreq - lowFreq) / 0.01);
    
    for(int j = 0; j < 100; j++) {
        if(j >= steps) break; // Ensure we don't exceed the calculated steps
        float i = lowFreq + float(j) * 0.01;
        sum += texture2D(iChannel0, vec2(i, 0.0)).x;
    }
    
    return sum / max(1.0, float(steps));
}

// Detect kick drum for the 4/4 house beat
float detectKick(float time) {
    float currentKick = getFrequencyResponse(0.01, KICK_FREQ);
    
    // Detect sudden increase in kick frequency energy
    float kickHit = max(0.0, currentKick - prevKick * 1.2);
    
    // If we detect a significant kick
    if(kickHit > 0.1) {
        // Calculate BPM and track beat count
        float timeSinceLastBeat = time - lastBeatTime;
        if(timeSinceLastBeat > 0.2) {  // Avoid false triggers
            beatCount = mod(beatCount + 1.0, 4.0);
            lastBeatTime = time;
        }
    }
    
    // Keep track of the previous frame's value
    prevKick = mix(prevKick, currentKick, 0.4); // Fast reaction
    
    // Trigger that decays for visual impact
    kickTrigger = max(kickTrigger * 0.9, kickHit * 8.0);
    
    return kickTrigger;
}

// 2D rotation function
mat2 rotate2D(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
}

// Grid function with audio reactivity
vec3 audioReactiveGrid(vec2 uv, float time, float kick, float bass, float high) {
    // Rotate and scale UV based on kick and bass
    uv = rotate2D(time * 0.1 + kick * 0.2) * uv;
    uv *= 1.0 + bass * 0.3 - kick * 0.2;
    
    // Create grid lines that pulse with the beat
    vec2 grid = abs(fract(uv * (5.0 + high * 5.0)) - 0.5);
    float gridLines = smoothstep(0.05 + kick * 0.05, 0.0, min(grid.x, grid.y));
    
    // Create concentric circles that pulse with kicks
    float circles = abs(fract(length(uv) * (3.0 + bass * 3.0)) - 0.5);
    float circleLines = smoothstep(0.05 + kick * 0.1, 0.0, circles);
    
    // Calculate angle for color variation
    float angle = atan(uv.y, uv.x) / (3.14159 * 2.0) + 0.5;
    
    // Create color based on position and beat
    vec3 color = mix(NEON_BLUE, NEON_PINK, angle + bass * 0.5);
    color = mix(color, NEON_GREEN, fract(length(uv) * 2.0 - time * 0.1));
    
    // Apply grid and circles
    color = mix(color * 0.2, color, gridLines);
    color = mix(color, NEON_PURPLE, circleLines * kick);
    
    // Add kick flash
    color += NEON_PINK * kick * 0.5;
    
    // Pulse intensity based on beat count (4/4 rhythm)
    float beatIntensity = beatCount == 0.0 ? 1.0 : 
                         (beatCount == 2.0 ? 0.8 : 0.6);
    
    color *= 0.8 + beatIntensity * kick * 0.5;
    
    return color;
}

// Tunnel effect that responds to bass
vec3 audioReactiveTunnel(vec2 uv, float time, float kick, float bass, float high) {
    // Calculate polar coordinates
    float angle = atan(uv.y, uv.x);
    float radius = length(uv);
    
    // Distort based on beat
    angle += sin(radius * 10.0 - time * 2.0) * 0.2 * bass;
    radius += sin(angle * 8.0 + time) * 0.1;
    
    // Create tunnel effect
    float tunnel = fract(1.0 / radius * (0.5 + bass * 0.5) - time * 0.5);
    tunnel = smoothstep(0.0, kick * 0.5 + 0.5, tunnel) * 
             smoothstep(1.0, 0.7 - kick * 0.3, tunnel);
    
    // Add radial lines
    float lines = fract(angle * (8.0 + high * 8.0) / 3.14159);
    lines = smoothstep(0.5, 0.0, abs(lines - 0.5)) * tunnel;
    
    // Create color based on distance and angle
    vec3 color = mix(NEON_BLUE, NEON_PINK, fract(angle / 3.14159 * 2.0 + time * 0.1));
    color = mix(color, NEON_GREEN, fract(radius * 2.0 - time * 0.2));
    
    // Apply tunnel and lines
    color *= tunnel;
    color = mix(color, NEON_PURPLE, lines * 0.7);
    
    // Add kick flash
    color += NEON_BLUE * kick * 0.3 * (1.0 - radius);
    
    return color;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Normalized coordinates
    vec2 uv = (fragCoord.xy - 0.5 * iResolution.xy) / min(iResolution.x, iResolution.y);
    
    // Audio analysis
    float kick = detectKick(iTime);
    float bass = getFrequencyResponse(KICK_FREQ, BASS_FREQ);
    float mid = getFrequencyResponse(BASS_FREQ, MID_FREQ);
    float high = getFrequencyResponse(MID_FREQ, HIGH_FREQ);
    
    // Mouse interaction
    float mixFactor = 0.5;
    if (iMouse.z > 0.0) {
        mixFactor = iMouse.x / iResolution.x;
    }
    
    // Create two different visual styles
    vec3 gridColor = audioReactiveGrid(uv, iTime, kick, bass, high);
    vec3 tunnelColor = audioReactiveTunnel(uv, iTime, kick, bass, high);
    
    // Mix between styles based on mouse or mid frequencies
    vec3 finalColor = mix(gridColor, tunnelColor, mixFactor + mid * 0.3);
    
    // Add vignette effect
    float vignette = smoothstep(1.2, 0.5, length(uv * 1.2));
    finalColor *= vignette;
    
    // Add beat-synchronized flash effect
    finalColor += NEON_PINK * kick * 0.2 * (1.0 - length(uv));
    
    // Add subtle strobe effect on certain beats
    if (beatCount == 0.0 || beatCount == 2.0) {
        finalColor *= 1.0 + kick * 0.3;
    }
    
    // Output final color
    fragColor = vec4(finalColor, 1.0);
}`,

"Temporal Fractal": `
// CC0: Appolloian with a twist II
//  Playing around with shadows in 2D
//  Needed a somewhat more complex distance field than boxes
//  The appolloian fractal turned out quite nice so while
//  similar to an earlier shader of mine I think it's
//  distrinctive enough to share
#define RESOLUTION  iResolution
#define TIME        iTime
#define MAX_MARCHES 30
#define TOLERANCE   0.0001
#define ROT(a)      mat2(cos(a), sin(a), -sin(a), cos(a))
#define PI          3.141592654
#define TAU         (2.0*PI)

const mat2 rot0 = ROT(0.0);
mat2 g_rot0 = rot0;
mat2 g_rot1 = rot0;

// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
float sRGB(float t) { return mix(1.055*pow(t, 1./2.4) - 0.055, 12.92*t, step(t, 0.0031308)); }
// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
vec3 sRGB(in vec3 c) { return vec3 (sRGB(c.x), sRGB(c.y), sRGB(c.z)); }

// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
const vec4 hsv2rgb_K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www);
  return c.z * mix(hsv2rgb_K.xxx, clamp(p - hsv2rgb_K.xxx, 0.0, 1.0), c.y);
}

float apolloian(vec3 p, float s, out float h) {
  float scale = 1.0;
  for(int i=0; i < 5; ++i) {
    p = -1.0 + 2.0*fract(0.5*p+0.5);
    float r2 = dot(p,p);
    float k  = s/r2;
    p       *= k;
    scale   *= k;
  }
  
  vec3 ap = abs(p/scale);  
  float d = length(ap.xy);
  d = min(d, ap.z);

  float hh = 0.0;
  if (d == ap.z){
    hh += 0.5;
  }
  h = hh;
  return d;
}

float df(vec2 p, out float h) {
  const float fz = 1.0-0.0;
  float z = 1.55*fz;
  p /= z;
  vec3 p3 = vec3(p,0.1);
  p3.xz*=g_rot0;
  p3.yz*=g_rot1;
  float d = apolloian(p3, 1.0/fz, h);
  d *= z;
  return d;
}

float shadow(vec2 lp, vec2 ld, float mint, float maxt) {
  const float ds = 1.0-0.4;
  float t = mint;
  float nd = 1E6;
  float h;
  const float soff = 0.05;
  const float smul = 1.5;
  for (int i=0; i < MAX_MARCHES; ++i) {
    vec2 p = lp + ld*t;
    float d = df(p, h);
    if (d < TOLERANCE || t >= maxt) {
      float sd = 1.0-exp(-smul*max(t/maxt-soff, 0.0));
      return t >= maxt ? mix(sd, 1.0, smoothstep(0.0, 0.025, nd)) : sd;
    }
    nd = min(nd, d);
    t += ds*d;
  }
  float sd = 1.0-exp(-smul*max(t/maxt-soff, 0.0));
  return sd;
}

vec3 effect(vec2 p, vec2 q) {
  float aa = 2.0/RESOLUTION.y;
  float a = 0.1*TIME;
  g_rot0 = ROT(0.5*a); 
  g_rot1 = ROT(sqrt(0.5)*a);

  vec2  lightPos  = vec2(0.0, 1.0);
  lightPos        *= (g_rot1);
  vec2  lightDiff = lightPos - p;
  float lightD2   = dot(lightDiff,lightDiff);
  float lightLen  = sqrt(lightD2);
  vec2  lightDir  = lightDiff / lightLen;
  vec3  lightPos3 = vec3(lightPos, 0.0);
  vec3  p3        = vec3(p, -1.0);
  float lightLen3 = distance(lightPos3, p3);
  vec3  lightDir3 = normalize(lightPos3-p3);
  vec3  n3        = vec3(0.0, 0.0, 1.0);
  float diff      = max(dot(lightDir3, n3), 0.0);

  float h;
  float d   = df(p, h);
  float ss  = shadow(p,lightDir, 0.005, lightLen);
  vec3 bcol = hsv2rgb(vec3(fract(h-0.2*length(p)+0.25*TIME), 0.666, 1.0));

  vec3 col = vec3(0.0);
  col += mix(0., 1.0, diff)*0.5*mix(0.1, 1.0, ss)/(lightLen3*lightLen3);
  col += exp(-300.0*abs(d))*sqrt(bcol);
  col += exp(-40.0*max(lightLen-0.02, 0.0));
 
  return col;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 q = fragCoord/RESOLUTION.xy;
  vec2 p = -1. + 2. * q;
  p.x *= RESOLUTION.x/RESOLUTION.y;

  vec3 col = effect(p, q);
  col *= mix(0.0, 1.0, smoothstep(0.0, 4.0, TIME));
  col = sRGB(col);
  
  fragColor = vec4(col, 1.0);
}
`,
"Golden Vow": `// The MIT License
// Copyright © 2016 Zhirnov Andrey
// Fixed and enhanced for audio visualization

#define ANIMATE 0.5
#define COLORED

vec2 Hash22(vec2 p) {
    vec2 q = vec2(dot(p, vec2(127.1, 311.7)), 
                  dot(p, vec2(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

float Hash21(vec2 p) {
    return fract(sin(p.x + p.y * 64.0) * 104003.9);
}

vec2 Hash12(float f) {
    return fract(cos(f) * vec2(10003.579, 37049.7));
}

float Hash11(float a) {
    return Hash21(vec2(fract(a * 2.0), fract(a * 4.0)));
}

vec4 voronoi(in vec2 x) {
    // from https://www.shadertoy.com/view/ldl3W8
    // The MIT License
    // Copyright © 2013 Inigo Quilez
    
    vec2 n = floor(x);
    vec2 f = fract(x);
    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
    vec2 mg, mr;
    float md = 8.0;
    
    for(int j = -1; j <= 1; j++)
    for(int i = -1; i <= 1; i++) {
        vec2 g = vec2(float(i), float(j));
        vec2 o = Hash22(n + g);
        #ifdef ANIMATE
        o = 0.5 + 0.5 * sin(iTime * ANIMATE + 6.2831 * o);
        #endif
        vec2 r = g + o - f;
        float d = dot(r, r);
        if(d < md) {
            md = d;
            mr = r;
            mg = g;
        }
    }
    
    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for(int j = -2; j <= 2; j++)
    for(int i = -2; i <= 2; i++) {
        vec2 g = mg + vec2(float(i), float(j));
        vec2 o = Hash22(n + g);
        #ifdef ANIMATE
        o = 0.5 + 0.5 * sin(iTime * ANIMATE + 6.2831 * o);
        #endif
        vec2 r = g + o - f;
        if(dot(mr - r, mr - r) > 0.00001)
            md = min(md, dot(0.5 * (mr + r), normalize(r - mr)));
    }
    
    return vec4(x - (n + mr + f), md, Hash21(mg + n));
}

vec3 HSVtoRGB(vec3 hsv) {
    // from http://chilliant.blogspot.ru/2014/04/rgbhsv-in-hlsl-5.html
    vec3 col = vec3(abs(hsv.x * 6.0 - 3.0) - 1.0,
                     2.0 - abs(hsv.x * 6.0 - 2.0),
                     2.0 - abs(hsv.x * 6.0 - 4.0));
    return ((clamp(col, vec3(0.0), vec3(1.0)) - 1.0) * hsv.y + 1.0) * hsv.z;
}

// Modified for copper-orange color scheme
vec3 Rainbow(float color, float dist) {
    dist = pow(dist, 8.0);
    // Original rainbow
    // vec3 rainbow = HSVtoRGB(vec3(color, 1.0, 1.0));
    
    // Copper-orange color scheme
    float hue = 0.05 + color * 0.05; // Limited orange-copper range
    float sat = 0.8 + color * 0.2;
    vec3 copper = HSVtoRGB(vec3(hue, sat, 1.0));
    
    return mix(vec3(1.0), copper, 1.0 - dist);
}

vec3 VoronoiFactal(in vec2 coord, float time) {
    const float freq = 4.0;
    const float freq2 = 6.0;
    const int iterations = 4;
    
    vec2 uv = coord * freq;
    
    vec3 color = vec3(0.0);
    float alpha = 0.0;
    float value = 0.0;
    
    for(int i = 0; i < iterations; ++i) {
        vec4 v = voronoi(uv);
        
        uv = (v.xy * 0.5 + 0.5) * freq2 + Hash12(v.w);
        
        float f = pow(0.01 * float(iterations - i), 3.0);
        float a = 1.0 - smoothstep(0.0, 0.08 + f, v.z);
        
        vec3 c = Rainbow(Hash11(float(i + 1) / float(iterations) + value * 1.341), i > 1 ? 0.0 : a);
        
        color = color * alpha + c * a;
        alpha = max(alpha, a);
        value = v.w;
    }
    
    #ifdef COLORED
        return color;
    #else
        return vec3(alpha) * Rainbow(0.06, alpha);
    #endif
}

// Fixed function definition - removed the space before parenthesis
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Adjust coordinates to make it look more like your reference image
    vec2 uv = fragCoord.xy / iResolution.xx;
    
    // Create darker background
    vec3 color = VoronoiFactal(uv, iTime);
    
    // Adjust for copper tones
    color = mix(vec3(0.1, 0.05, 0.02), color, color);
    
    // Boost orange-copper tones
    color.r = min(1.0, color.r * 1.2);
    color.g = min(1.0, color.g * 0.8);
    color.b = min(1.0, color.b * 0.5);
    
    fragColor = vec4(color, 1.0);
}`
};



export const RAPHAEL_SHADERS = [
    "Black Hole",
    "Plasma",
    "Yawning Void"
]

export const CHISTIAN_SHADERS = [
    "Black Hole",
    "Plasma",
    "Yawning Void"
]

export const INI_SHADERS = [
    "Black Hole",
    "Plasma",
    "Yawning Void"
]

