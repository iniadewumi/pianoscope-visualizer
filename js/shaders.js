
// Sample Shadertoy shaders to quickly test
export const SAMPLE_SHADERS = {
"Procedural Circuitry": `// This content is under the MIT License.

#define time iTime*.02
#define width .005
float zoom = .18;

float shape = 0.;
vec3 color = vec3(0.), randcol;

vec3 palette(float t) {
    return 0.5 + 0.5*cos(6.28318*t + vec3(0.0, 0.33, 0.67));
}

void formula(vec2 z, float c) {
    float minit = 0.;
    float o, ot2, ot = ot2 = 1000.;
    for (int i = 0; i < 9; i++) {
        z = abs(z) / clamp(dot(z, z), .1, .5) - c;
        float l = length(z);
        o = min(max(abs(min(z.x, z.y)),-l+.25), abs(l-.25));
        ot = min(ot, o);
        ot2 = min(l * .1, ot2);
        minit = max(minit, float(i) * (1. - abs(sign(ot - o))));
    }
    minit += 1.;
    float w = width * minit * 2.;
    float circ = pow(max(0., w - ot2) / w, 6.);
    shape += max(pow(max(0., w - ot) / w, .25), circ);
    vec3 col = normalize(.1 + texture(iChannel1, vec2(minit * .1)).rgb);
    color += col * (.4 + mod(minit / 9. - time * 10. + ot2 * 2., 1.) * 1.6);
    color += vec3(1., .7, .3) * circ * (10. - minit) * 3. * 
             smoothstep(0., .5, .15 + texture(iChannel0, vec2(.0, 1.)).x - .5);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 pos = fragCoord.xy / iResolution.xy - .5;
    pos.x *= iResolution.x / iResolution.y;
    vec2 uv = pos;
    float sph = length(uv); 
    sph = sqrt(1. - sph * sph) * 1.5; 
    uv = normalize(vec3(uv, sph)).xy;
    
    float a = time + mod(time, 1.) * .5;
    float b = a * 5.48535;
    uv *= mat2(cos(b), sin(b), -sin(b), cos(b));
    uv += vec2(sin(a), cos(a * .5)) * 8.;
    uv *= zoom;
    
    float pix = .5 / iResolution.x * zoom / sph;
    float dof = max(1., (10. - mod(time, 1.) / .01));
    float c = 1.5 + mod(floor(time), 6.) * .125;
    for (int aa = 0; aa < 36; aa++) {
        vec2 aauv = floor(vec2(float(aa) / 6., mod(float(aa), 6.)));
        formula(uv + aauv * pix * dof, c);
    }
    shape /= 36.;
    color /= 36.;
    
    vec3 pal = palette(time * 0.3 + shape);
    vec3 base = mix(vec3(.15), color, shape);
    vec3 colo = base * pal * (1. - length(pos)) * min(1., abs(.5 - mod(time + .5, 1.)) * 10.);
    
    // Brighten the output by increasing the multipliers
    colo *= vec3(1.4, 1.3, 1.2);
    fragColor = vec4(colo, 1.0);
}
`,
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
"Temporal Fractal Mic'd": `// CC0: Appolloian with a twist II - Subtle Audio Reactive Version
// Original shader with very gentle audio reactivity

#define RESOLUTION  iResolution
#define TIME        iTime
#define MAX_MARCHES 30
#define TOLERANCE   0.0001
#define ROT(a)      mat2(cos(a), sin(a), -sin(a), cos(a))
#define PI          3.141592654
#define TAU         (2.0*PI)

// Audio sampling constants
#define BASS_FREQ   0.05  // Low frequencies
#define MID_FREQ    0.3   // Mid-range frequencies
#define HIGH_FREQ   0.8   // High frequencies

const mat2 rot0 = ROT(0.0);
mat2 g_rot0 = rot0;
mat2 g_rot1 = rot0;

// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
float sRGB(float t) { return mix(1.055*pow(t, 1./2.4) - 0.055, 12.92*t, step(t, 0.0031308)); }
// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
vec3 sRGB(in vec3 c) { return vec3(sRGB(c.x), sRGB(c.y), sRGB(c.z)); }

// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
const vec4 hsv2rgb_K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www);
  return c.z * mix(hsv2rgb_K.xxx, clamp(p - hsv2rgb_K.xxx, 0.0, 1.0), c.y);
}

// Audio reactive helper functions
float getBassIntensity() {
    return texture2D(iChannel0, vec2(BASS_FREQ, 0.0)).x;
}

float getMidIntensity() {
    return texture2D(iChannel0, vec2(MID_FREQ, 0.0)).x;
}

float getHighIntensity() {
    return texture2D(iChannel0, vec2(HIGH_FREQ, 0.0)).x;
}

// Get spectrum-wide intensity with slight bias toward bass
float getOverallIntensity() {
    float bass = getBassIntensity();
    float mid = getMidIntensity();
    float high = getHighIntensity();
    
    return bass * 0.5 + mid * 0.3 + high * 0.2;
}

float apolloian(vec3 p, float s, out float h) {
  // Very subtle bass influence on scale
  float bassBoost = 1.0 + getBassIntensity() * 0.05;
  float scale = 1.0;
  
  for(int i=0; i < 5; ++i) {
    // Extremely subtle mid-frequency distortion
    float midMod = getMidIntensity() * 0.01;
    p = -1.0 + 2.0*fract(0.5*p + 0.5 + vec3(midMod, -midMod, midMod));
    
    float r2 = dot(p,p);
    float k = (s * bassBoost)/r2;
    p *= k;
    scale *= k;
  }
  
  vec3 ap = abs(p/scale);  
  float d = length(ap.xy);
  d = min(d, ap.z);
  
  float hh = 0.0;
  if (d == ap.z){
    // Very subtle high frequency influence
    hh += 0.5 + getHighIntensity() * 0.02;
  }
  h = hh;
  return d;
}

float df(vec2 p, out float h) {
  // Subtle overall intensity modulation
  float intensityMod = getOverallIntensity() * 0.03;
  const float fz = 1.0 - 0.0;
  float z = 1.55 * fz * (1.0 + intensityMod);
  
  p /= z;
  
  // Very subtle mid-frequency modulation
  float midMod = getMidIntensity() * 0.01;
  
  vec3 p3 = vec3(p, 0.1 + midMod);
  p3.xz *= g_rot0;
  p3.yz *= g_rot1;
  
  float d = apolloian(p3, 1.0/fz, h);
  d *= z;
  return d;
}

float shadow(vec2 lp, vec2 ld, float mint, float maxt) {
  // Subtle bass influence on shadows
  float bassImpact = 1.0 - getBassIntensity() * 0.05; 
  float ds = 1.0 - 0.4 * bassImpact;
  
  float t = mint;
  float nd = 1E6;
  float h;
  
  // Very subtle high-frequency modulation
  float highMod = getHighIntensity();
  float soff = 0.05 + highMod * 0.005;
  float smul = 1.5 + highMod * 0.1;
  
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
  
  // Keep original time-based rotation
  float a = 0.1*TIME;
  
  g_rot0 = ROT(0.5*a); 
  g_rot1 = ROT(sqrt(0.5)*a);
  
  // Very subtle light position influence
  float midMod = getMidIntensity() * 0.05;
  vec2 lightPos = vec2(midMod * 0.05, 1.0 + midMod * 0.05);
  
  lightPos *= (g_rot1);
  vec2 lightDiff = lightPos - p;
  float lightD2 = dot(lightDiff, lightDiff);
  float lightLen = sqrt(lightD2);
  vec2 lightDir = lightDiff / lightLen;
  vec3 lightPos3 = vec3(lightPos, 0.0);
  vec3 p3 = vec3(p, -1.0);
  float lightLen3 = distance(lightPos3, p3);
  vec3 lightDir3 = normalize(lightPos3-p3);
  vec3 n3 = vec3(0.0, 0.0, 1.0);
  float diff = max(dot(lightDir3, n3), 0.0);
  
  float h;
  float d = df(p, h);
  float ss = shadow(p, lightDir, 0.005, lightLen);
  
  // Subtle color modulation with audio
  float bassHueShift = getBassIntensity() * 0.03;
  float highSatMod = getHighIntensity() * 0.05;
  vec3 bcol = hsv2rgb(vec3(
      fract(h - 0.2 * length(p) + 0.25 * TIME + bassHueShift), 
      0.666 + highSatMod, 
      1.0
  ));
  
  vec3 col = vec3(0.0);
  
  // Subtle lighting intensity modulation
  float lightIntensity = 0.5 * (1.0 + getOverallIntensity() * 0.1);
  col += mix(0., 1.0, diff) * lightIntensity * mix(0.1, 1.0, ss)/(lightLen3 * lightLen3);
  
  // Subtle glow modulation with bass
  float glowIntensity = 300.0 + getBassIntensity() * 20.0;
  col += exp(-glowIntensity * abs(d)) * sqrt(bcol);
  
  // Subtle bloom modulation with high frequencies
  float bloomIntensity = 40.0 + getHighIntensity() * 5.0;
  col += exp(-bloomIntensity * max(lightLen-0.02, 0.0));
 
  return col;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 q = fragCoord/RESOLUTION.xy;
  vec2 p = -1. + 2. * q;
  p.x *= RESOLUTION.x/RESOLUTION.y;
  
  vec3 col = effect(p, q);
  
  // Very subtle bass pulse
  float bassPulse = 1.0 + getBassIntensity() * 0.05;
  col *= bassPulse * mix(0.0, 1.0, smoothstep(0.0, 4.0, TIME));
  
  col = sRGB(col);
  
  fragColor = vec4(col, 1.0);
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
    
    // Enhanced sound analysis
    int soundTx = int(uv.x * 512.0);
    float wave = texelFetch(iChannel0, ivec2(soundTx, 1), 0).x;
    
    // Smooth out the wave response
    float smoothWave = wave * 0.5 + 0.5; // Normalize to 0-1 range
    smoothWave = pow(smoothWave, 1.5); // Add some curve to the response
    
    // Base movement speed
    float baseSpeed = 0.4;
    vec2 baseOffset = vec2(sin(iTime*baseSpeed), cos(iTime*baseSpeed));
    
    // Calculate distance with smooth falloff
    float d = length(p/1.0 - baseOffset);
    p = p/1.0 - baseOffset;
    
    // Improved center visualization
    float centerSoftness = 0.8;
    float centerSize = 0.4;
    float smoothD = smoothstep(centerSize, centerSize + centerSoftness, d);
    
    // Create dynamic wave rings
    float ringCount = 5.0;
    float ringSpeed = 2.0;
    float rings = sin(d * ringCount - iTime * ringSpeed);
    rings = smoothstep(-0.2, 0.2, rings);
    
    // Wave effect calculation with improved visual integration
    float waveStrength = smoothWave;
    float waveFrequency = 1.5 + waveStrength * 2.0;
    float waveOffset = sin(uv.y * waveFrequency + iTime) * waveStrength;
    
    // Create a more dynamic wave pattern
    float wavePattern = sin((d + waveOffset) * 10.0) * 0.5 + 0.5;
    wavePattern *= smoothstep(1.0, 0.0, d); // Fade out with distance
    
    // Calculate phi with rotation
    float rotationSpeed = 2.0;
    float phi = p.x != 0.0 ? atan(p.y, p.x) - iTime * rotationSpeed : 
                            sign(p.y) * 3.14159/2.0 - iTime * rotationSpeed;
    
    // Enhanced radial pattern
    float patternDetail = 36.0;
    float radialPattern = sin(phi * patternDetail + d * 2.0);
    
    // Combine patterns with wave influence
    float combinedPattern = mix(radialPattern, rings, waveStrength * 0.3);
    d *= combinedPattern;
    
    // Color calculation with wave influence
    vec3 baseColor = vec3(atan(max(d, -100.0)), 0.1, -sqrt(max(0.0, d))) + d;
    
    // Add wave-reactive color shifts
    vec3 waveColor = vec3(0.4, 0.6, 1.0); // Blue-ish tone for wave
    float colorIntensity = sin(iTime/(d + 0.001) * (d + 0.001));
    baseColor *= colorIntensity;
    baseColor -= d/1.2;
    
    // Add wave-reactive glow
    vec3 glowColor = vec3(0.2, 0.4, 1.0) * wavePattern * waveStrength;
    
    // Spatial color variation
    float denominator = max(p.x * p.x + p.y * p.y, 0.001);
    float timeEffect = tan(mod(iTime, 6.28318530718)) * 0.5;
    float colorWave = sin((-(denominator) * 2.0 - iTime * 6.0) / -denominator * timeEffect);
    
    // Final color composition
    vec3 col = baseColor;
    col *= mix(1.0, colorWave, smoothD);
    col += glowColor * smoothstep(0.5, 0.0, d); // Add glow in center
    
    // Wave-reactive color enhancement
    col = mix(col, waveColor, wavePattern * waveStrength * 0.3);
    
    // Add subtle pulse on strong beats
    float pulse = smoothstep(0.7, 1.0, waveStrength) * sin(iTime * 10.0) * 0.2;
    col += pulse * vec3(0.2, 0.3, 0.5);
    
    // Final color correction
    col = mix(col, vec3(0.5), smoothstep(0.0, 0.1, d) * 0.2);
    col = clamp(col, -1.0, 1.0);
    
    fragColor = vec4(col, 1.0);
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
}`,
"Solar": `
// Audio reactive sun shader with granite-like texture
precision highp float;

// Improved noise function for more granite-like patterns
float hash(float n) {
    return fract(sin(n) * 43758.5453);
}

float noise(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    
    float n = p.x + p.y * 157.0 + 113.0 * p.z;
    return mix(
        mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
            mix(hash(n + 157.0), hash(n + 158.0), f.x), f.y),
        mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
            mix(hash(n + 270.0), hash(n + 271.0), f.x), f.y), f.z
    );
}

// Fractional Brownian Motion for more detailed texture
float fbm(vec3 x) {
    float v = 0.0;
    float a = 0.5;
    vec3 shift = vec3(100.0);
    
    for (int i = 0; i < 5; ++i) {
        v += a * noise(x);
        x = x * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

float getAudioData(float x) {
    return texture2D(iChannel0, vec2(x, 0.0)).x;
}

// Enhanced granite-like texture
vec3 graniteTexture(vec2 uv, float time) {
    float audio = getAudioData(0.1) * 2.0;
    float audioMid = getAudioData(0.5) * 1.5;
    float audioHigh = getAudioData(0.8);
    
    // Multiple layers of noise for granite effect
    vec3 p = vec3(uv * 8.0, time * 0.2);
    float pattern = fbm(p);
    
    // Add swirling motion
    p.xy += vec2(
        sin(time * 0.5 + audio * 2.0) * 0.2,
        cos(time * 0.3 + audioMid * 2.0) * 0.2
    );
    
    // Second layer of noise
    pattern += fbm(p * 2.0) * 0.5;
    
    // Add plasma-like effect
    float plasma = sin(uv.x * 10.0 + time) * cos(uv.y * 8.0 - time) * 0.25;
    pattern += plasma * audioHigh;
    
    // Create color variations
    vec3 color1 = vec3(1.0, 0.9, 0.6); // Warm light color
    vec3 color2 = vec3(1.0, 0.7, 0.3); // Darker warm color
    vec3 color3 = vec3(0.9, 0.6, 0.2); // Orange accent
    
    vec3 finalColor = mix(color1, color2, pattern);
    finalColor = mix(finalColor, color3, fbm(p * 4.0 + audio));
    
    // Add audio reactive highlights
    finalColor += vec3(1.0, 0.9, 0.7) * audioHigh * 0.5;
    
    return finalColor;
}

float snoise(vec3 uv, float res) {
    const vec3 s = vec3(1e0, 1e2, 1e4);
    
    uv *= res;
    vec3 uv0 = floor(mod(uv, res))*s;
    vec3 uv1 = floor(mod(uv+vec3(1.), res))*s;
    
    vec3 f = fract(uv); f = f*f*(3.0-2.0*f);
    
    vec4 v = vec4(uv0.x+uv0.y+uv0.z, uv1.x+uv0.y+uv0.z,
                  uv0.x+uv1.y+uv0.z, uv1.x+uv1.y+uv0.z);
    
    vec4 r = fract(sin(v*1e-3)*1e5);
    float r0 = mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y);
    
    r = fract(sin((v + uv1.z - uv0.z)*1e-3)*1e5);
    float r1 = mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y);
    
    return mix(r0, r1, f.z)*2.-1.;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float freqs[4];
    freqs[0] = getAudioData(0.01);
    freqs[1] = getAudioData(0.07);
    freqs[2] = getAudioData(0.15);
    freqs[3] = getAudioData(0.30);
    
    float brightness = freqs[1] * 0.25 + freqs[2] * 0.25;
    float radius = 0.24 + brightness * 0.2;
    float invRadius = 1.0/radius;
    
    vec3 orange = vec3(0.8, 0.65, 0.3);
    vec3 orangeRed = vec3(0.8, 0.35, 0.1);
    float time = iTime * 0.1;
    float aspect = iResolution.x/iResolution.y;
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec2 p = -0.5 + uv;
    p.x *= aspect;
    
    float fade = pow(length(2.0 * p), 0.5);
    float fVal1 = 1.0 - fade;
    float fVal2 = 1.0 - fade;
    
    float angle = atan(p.x, p.y)/6.2832;
    float dist = length(p);
    vec3 coord = vec3(angle, dist, time * 0.1);
    
    float newTime1 = abs(snoise(coord + vec3(0.0, -time * (0.35 + brightness * 0.001), time * 0.015), 15.0));
    float newTime2 = abs(snoise(coord + vec3(0.0, -time * (0.15 + brightness * 0.001), time * 0.015), 45.0));
    
    for(int i=1; i<=7; i++) {
        float power = pow(2.0, float(i + 1));
        fVal1 += (0.5 / power) * snoise(coord + vec3(0.0, -time, time * 0.2), (power * (10.0) * (newTime1 + 1.0)));
        fVal2 += (0.5 / power) * snoise(coord + vec3(0.0, -time, time * 0.2), (power * (25.0) * (newTime2 + 1.0)));
    }
    
    float corona = pow(fVal1 * max(1.1 - fade, 0.0), 2.0) * 50.0;
    corona += pow(fVal2 * max(1.1 - fade, 0.0), 2.0) * 50.0;
    corona *= 1.2 - newTime1;
    
    vec3 starSphere = vec3(0.0);
    
    vec2 sp = -1.0 + 2.0 * uv;
    sp.x *= aspect;
    sp *= (2.0 - brightness);
    float r = dot(sp,sp);
    float f = (1.0-sqrt(abs(1.0-r)))/(r) + brightness * 0.5;
    
    if(dist < radius) {
        corona *= pow(dist * invRadius, 24.0);
        vec2 newUv;
        newUv.x = sp.x*f;
        newUv.y = sp.y*f;
        newUv += vec2(time, 0.0);
        
        // Use the new granite texture
        vec3 granite = graniteTexture(newUv, time);
        float uOff = (granite.g * brightness * 4.5 + time);
        vec2 starUV = newUv + vec2(uOff, 0.0);
        starSphere = graniteTexture(starUV, time * 1.5);
        
        // Make it more reactive to audio
        float audioHigh = getAudioData(0.8);
        starSphere *= 1.0 + audioHigh * 2.0;
    }
    
    float starGlow = min(max(1.0 - dist * (1.0 - brightness), 0.0), 1.0);
    float audioInfluence = getAudioData(0.2) * 0.5;
    
    orange = mix(orange, orange * 1.2, audioInfluence);
    orangeRed = mix(orangeRed, orangeRed * 1.3, audioInfluence);
    
    fragColor.rgb = vec3(f * (0.75 + brightness * 0.3) * orange) + 
                    starSphere + 
                    corona * orange + 
                    starGlow * orangeRed;
    fragColor.a = 1.0;
}`,
"Test": `// Misty Lake. Created by Reinder Nijhoff 2013
// Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
// @reindernijhoff
//
// https://www.shadertoy.com/view/MsB3WR
//

#define BUMPFACTOR 0.1
#define EPSILON 0.1
#define BUMPDISTANCE 60.

#define time (iTime+285.)

// Noise functions by inigo quilez 

float noise( const in vec2 x ) {
    vec2 p = floor(x);
    vec2 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy) + f.xy;
	return textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).x;
}

float noise( const in vec3 x ) {
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

mat2 rot(const in float a) {
	return mat2(cos(a),sin(a),-sin(a),cos(a));	
}

const mat2 m2 = mat2( 0.60, -0.80, 0.80, 0.60 );

const mat3 m3 = mat3( 0.00,  0.80,  0.60,
                     -0.80,  0.36, -0.48,
                     -0.60, -0.48,  0.64 );

float fbm( in vec3 p ) {
    float f = 0.0;
    f += 0.5000*noise( p ); p = m3*p*2.02;
    f += 0.2500*noise( p ); p = m3*p*2.03;
    f += 0.1250*noise( p ); p = m3*p*2.01;
    f += 0.0625*noise( p );
    return f/0.9375;
}

float hash( in float n ) {
    return fract(sin(n)*43758.5453);
}

// intersection functions

bool intersectPlane(const in vec3 ro, const in vec3 rd, const in float height, inout float dist) {	
	if (rd.y==0.0) {
		return false;
	}
		
	float d = -(ro.y - height)/rd.y;
	d = min(100000.0, d);
	if( d > 0. && d < dist ) {
		dist = d;
		return true;
    } else {
		return false;
	}
}

// light direction

vec3 lig = normalize(vec3( 0.3,0.5, 0.6));

vec3 bgColor( const in vec3 rd ) {
	float sun = clamp( dot(lig,rd), 0.0, 1.0 );
	vec3 col = vec3(0.5, 0.52, 0.55) - rd.y*0.2*vec3(1.0,0.8,1.0) + 0.15*0.75;
	col += vec3(1.0,.6,0.1)*pow( sun, 8.0 );
	col *= 0.95;
	return col;
}

// coulds functions by inigo quilez

#define CLOUDSCALE (500./(64.*0.03))

float cloudMap( const in vec3 p, const in float ani ) {
	vec3 r = p/CLOUDSCALE;

	float den = -1.8+cos(r.y*5.-4.3);
		
	float f;
	vec3 q = 2.5*r*vec3(0.75,1.0,0.75)  + vec3(1.0,2.0,1.0)*ani*0.15;
    f  = 0.50000*noise( q ); q = q*2.02 - vec3(-1.0,1.0,-1.0)*ani*0.15;
    f += 0.25000*noise( q ); q = q*2.03 + vec3(1.0,-1.0,1.0)*ani*0.15;
    f += 0.12500*noise( q ); q = q*2.01 - vec3(1.0,1.0,-1.0)*ani*0.15;
    f += 0.06250*noise( q ); q = q*2.02 + vec3(1.0,1.0,1.0)*ani*0.15;
    f += 0.03125*noise( q );
	
	return 0.065*clamp( den + 4.4*f, 0.0, 1.0 );
}

vec3 raymarchClouds( const in vec3 ro, const in vec3 rd, const in vec3 bgc, const in vec3 fgc, const in float startdist, const in float maxdist, const in float ani ) {
    // dithering	
	float t = startdist+CLOUDSCALE*0.02*hash(rd.x+35.6987221*rd.y+time);//0.1*texture( iChannel0, fragCoord.xy/iChannelResolution[0].x ).x;
	
    // raymarch	
	vec4 sum = vec4( 0.0 );
	for( int i=0; i<64; i++ ) {
		if( sum.a > 0.99 || t > maxdist ) continue;
		
		vec3 pos = ro + t*rd;
		float a = cloudMap( pos, ani );

        // lighting	
		float dif = clamp(0.1 + 0.8*(a - cloudMap( pos + lig*0.15*CLOUDSCALE, ani )), 0., 0.5);
		vec4 col = vec4( (1.+dif)*fgc, a );
		// fog		
	//	col.xyz = mix( col.xyz, fgc, 1.0-exp(-0.0000005*t*t) );
		
		col.rgb *= col.a;
		sum = sum + col*(1.0 - sum.a);	

        // advance ray with LOD
		t += (0.03*CLOUDSCALE)+t*0.012;
	}

    // blend with background	
	sum.xyz = mix( bgc, sum.xyz/(sum.w+0.0001), sum.w );
	
	return clamp( sum.xyz, 0.0, 1.0 );
}

// terrain functions
float terrainMap( const in vec3 p ) {
	return (textureLod( iChannel1, (-p.zx*m2)*0.000046, 0. ).x*600.) * smoothstep( 820., 1000., length(p.xz) ) - 2. + noise(p.xz*0.5)*15.;
}

vec3 raymarchTerrain( const in vec3 ro, const in vec3 rd, const in vec3 bgc, const in float startdist, inout float dist ) {
	float t = startdist;

    // raymarch	
	vec4 sum = vec4( 0.0 );
	bool hit = false;
	vec3 col = bgc;
	
	for( int i=0; i<80; i++ ) {
		if( hit ) break;
		
		t += 8. + t/300.;
		vec3 pos = ro + t*rd;
		
		if( pos.y < terrainMap(pos) ) {
			hit = true;
		}		
	}
	if( hit ) {
		// binary search for hit		
		float dt = 4.+t/400.;
		t -= dt;
		
		vec3 pos = ro + t*rd;	
		t += (0.5 - step( pos.y , terrainMap(pos) )) * dt;		
		for( int j=0; j<2; j++ ) {
			pos = ro + t*rd;
			dt *= 0.5;
			t += (0.5 - step( pos.y , terrainMap(pos) )) * dt;
		}
		pos = ro + t*rd;
		
		vec3 dx = vec3( 100.*EPSILON, 0., 0. );
		vec3 dz = vec3( 0., 0., 100.*EPSILON );
		
		vec3 normal = vec3( 0., 0., 0. );
		normal.x = (terrainMap(pos + dx) - terrainMap(pos-dx) ) / (200. * EPSILON);
		normal.z = (terrainMap(pos + dz) - terrainMap(pos-dz) ) / (200. * EPSILON);
		normal.y = 1.;
		normal = normalize( normal );		

		col = vec3(0.2) + 0.7*texture( iChannel2, pos.xz * 0.01 ).xyz * 
				   vec3(1.,.9,0.6);
		
		float veg = 0.3*fbm(pos*0.2)+normal.y;
					
		if( veg > 0.75 ) {
			col = vec3( 0.45, 0.6, 0.3 )*(0.5+0.5*fbm(pos*0.5))*0.6;
		} else 
		if( veg > 0.66 ) {
			col = col*0.6+vec3( 0.4, 0.5, 0.3 )*(0.5+0.5*fbm(pos*0.25))*0.3;
		}
		col *= vec3(0.5, 0.52, 0.65)*vec3(1.,.9,0.8);
		
		vec3 brdf = col;
		
		float diff = clamp( dot( normal, -lig ), 0., 1.);
		
		col = brdf*diff*vec3(1.0,.6,0.1);
		col += brdf*clamp( dot( normal, lig ), 0., 1.)*vec3(0.8,.6,0.5)*0.8;
		col += brdf*clamp( dot( normal, vec3(0.,1.,0.) ), 0., 1.)*vec3(0.8,.8,1.)*0.2;
		
		dist = t;
		t -= pos.y*3.5;
		col = mix( col, bgc, 1.0-exp(-0.0000005*t*t) );
		
	}
	return col;
}

float waterMap( vec2 pos ) {
	vec2 posm = pos * m2;
	
	return abs( fbm( vec3( 8.*posm, time ))-0.5 )* 0.1;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
	vec2 q = fragCoord.xy / iResolution.xy;
    vec2 p = -1.0 + 2.0*q;
    p.x *= iResolution.x/ iResolution.y;
	
	// camera parameters
	vec3 ro = vec3(0.0, 0.5, 0.0);
	vec3 ta = vec3(0.0, 0.45,1.0);
	if (iMouse.z>=1.) {
		ta.xz *= rot( (iMouse.x/iResolution.x-.5)*7. );
	}
		
	ta.xz *= rot( mod(iTime * 0.05, 6.2831852) );
    
	// build ray
    vec3 ww = normalize( ta - ro);
    vec3 uu = normalize(cross( vec3(0.0,1.0,0.0), ww ));
    vec3 vv = normalize(cross(ww,uu));
    vec3 rd = normalize( p.x*uu + p.y*vv + 2.5*ww );

	float fresnel, refldist = 5000., maxdist = 5000.;
	bool reflected = false;
	vec3 normal, col = bgColor( rd );
	vec3 roo = ro, rdo = rd, bgc = col;
	
	if( intersectPlane( ro, rd, 0., refldist ) && refldist < 200. ) {
		ro += refldist*rd;	
		vec2 coord = ro.xz;
		float bumpfactor = BUMPFACTOR * (1. - smoothstep( 0., BUMPDISTANCE, refldist) );
				
		vec2 dx = vec2( EPSILON, 0. );
		vec2 dz = vec2( 0., EPSILON );
		
		normal = vec3( 0., 1., 0. );
		normal.x = -bumpfactor * (waterMap(coord + dx) - waterMap(coord-dx) ) / (2. * EPSILON);
		normal.z = -bumpfactor * (waterMap(coord + dz) - waterMap(coord-dz) ) / (2. * EPSILON);
		normal = normalize( normal );		
		
		float ndotr = dot(normal,rd);
		fresnel = pow(1.0-abs(ndotr),5.);

		rd = reflect( rd, normal);

		reflected = true;
		bgc = col = bgColor( rd );
	}

	col = raymarchTerrain( ro, rd, col, reflected?(800.-refldist):800., maxdist );
    col = raymarchClouds( ro, rd, col, bgc, reflected?max(0.,min(150.,(150.-refldist))):150., maxdist, time*0.05 );
	
	if( reflected ) {
		col = mix( col.xyz, bgc, 1.0-exp(-0.0000005*refldist*refldist) );
		col *= fresnel*0.9;		
		vec3 refr = refract( rdo, normal, 1./1.3330 );
		intersectPlane( ro, refr, -2., refldist );
		col += mix( texture( iChannel2, (roo+refldist*refr).xz*1.3 ).xyz * 
				   vec3(1.,.9,0.6), vec3(1.,.9,0.8)*0.5, clamp( refldist / 3., 0., 1.) ) 
			   * (1.-fresnel)*0.125;
	}
	
	col = pow( col, vec3(0.7) );
	
	// contrast, saturation and vignetting	
	col = col*col*(3.0-2.0*col);
    col = mix( col, vec3(dot(col,vec3(0.33))), -0.5 );
 	col *= 0.25 + 0.75*pow( 16.0*q.x*q.y*(1.0-q.x)*(1.0-q.y), 0.1 );
	
    fragColor = vec4( col, 1.0 );
}
`,
"Graviton Glider": `// FM-2030's messenger - Result of an improvised live code session on Twitch
// Thankx to crundle for the help and haptix for suggestions
// LIVE SHADER CODING, SHADER SHOWDOWN STYLE, EVERY TUESDAYS 21:00 Uk time:
// https://www.twitch.tv/evvvvil_

// "I have a deep nostalgia for the future." - FM-2030

vec2 z,v,e=vec2(.00035,-.00035); float t,tt,b,bb,g,gg;vec3 np,bp,pp,cp,dp,po,no,al,ld;//global vars. About as exciting as vegans talking about sausages.
float bo(vec3 p,vec3 r){p=abs(p)-r;return max(max(p.x,p.y),p.z);} //box primitive function. Box is the only primitve I hang out with, I find the others have too many angles and seem to have a multi-faced agenda.
mat2 r2(float r){return mat2(cos(r),sin(r),-sin(r),cos(r));} //rotate function. Short and sweet, just like a midget wrestler covered in Mapple syrup.
float smin(float a,float b,float h){ float k=clamp((a-b)/h*.5+.5,0.,1.);return mix(a,b,k)-k*(1.-k)*h;} //Smooth min function, because sometimes brutality isn't the answer. Put that in your pipe and smoke it, Mr Officer.
float noi(vec3 p){ //Noise function stolen from Virgill who, like me, doesn't understand it. But, unlike me, Virgill can play the tuba.
  vec3 f=floor(p),s=vec3(7,157,113);
  p-=f; vec4 h=vec4(0,s.yz,s.y+s.z)+dot(f,s);;
  p=p*p*(3.-2.*p);
  h=mix(fract(sin(h)*43758.5),fract(sin(h+s.x)*43758.5),p.x);
  h.xy=mix(h.xz,h.yw,p.y);
  return mix(h.x,h.y,p.z);  
}
vec2 fb( vec3 p, float s ) // fb "fucking bit" function make a base geometry which we use to make spaceship and central structures using more complex positions defined in mp
{ //fb just does a bunch blue hollow boxes inside eachother with a white edge on top + a couple of black bars going through to symbolise the lack of middle class cyclists incarcerated for crimes against fun. Stay boring, Dave, I'm watching you.
  vec2 h,t=vec2(bo(p,vec3(5,5,2)),s); //Dumb fucking blue boxes (could also be said about Chelsea Football Club's fans)
  t.x=max(t.x,-bo(p,vec3(3.5,3.5,2))); //Dig a hole in them blue boxes, and just like with Chelsea Football Club - less is more
  t.x=abs(t.x)-.3; //Onion skin blue boxes for more geom
  t.x=max(t.x,bo(p,vec3(10,10,1)));//Cut front & back of box to reveal onion edges. In reality onions are boring and have no edges, I suspect they listen to Coldplay or Muse. Yeah, Dave, I'm still watching you!
  h=vec2(bo(p,vec3(5,5,2)),6); //Dumb fucking white boxes (could also be said about Tottenham Football Club's fans)
  h.x=max(h.x,-bo(p,vec3(3.5,3.5,2))); //Dig hole in white boxes, make it hollow, just like Tottenham FC's trophy cabinet.
  h.x=abs(h.x)-.1; //Onion skin the fucking white boxes for more geom
  h.x=max(h.x,bo(p,vec3(10,10,1.4))); //Cut front & back of box to reveal onion edges. Onions are like Tottenham FC's style of football: they make people cry.
  t=t.x<h.x?t:h; //Merge blue and white geom while retaining material ID
  h=vec2(length(abs(p.xz)-vec2(2,0))-.2,3); //Black prison bars, to symbolise the meta-physical struggle of half eaten sausages.
  t=t.x<h.x?t:h; return t; //Pack into a colourful sausage and hand it over to the feds...
}
vec2 mp( vec3 p )
{ 
  bp=p+vec3(0,0,tt*10.);
  np=p+noi(bp*.05)*15.+noi(bp*.5)*1.+noi(bp*4.)*.1+noi(bp*0.01)*20.; 
  vec2 h,t=vec2(np.y+20.,5); //TERRAIN
  t.x=smin(t.x,0.75*(length(abs(np.xy-vec2(0,10.+sin(p.x*.1)*10.))-vec2(65,0))-(18.+sin(np.z*.1+tt)*10.)),15.); //LEFT TERRAIN CYLINDER
  t.x*=0.5;  
  pp=p+vec3(10,15,0);
  pp.x+=sin(p.z*.02+tt/5.)*7.+sin(p.z*.001+20.+tt/100.)*4.; //ROAD POSITON
  bp=abs(pp);bp.xy*=r2(-.785);
  h=vec2(bo(bp-vec3(0,6,0),vec3(2,0.5,1000)),6); //ROAd WHITE
  t=t.x<h.x?t:h;
  h=vec2(bo(bp-vec3(0,6.2,0),vec3(1.,.8,1000)),3); //ROAd BLACK
  t=t.x<h.x?t:h;  
  cp=pp-dp; //SPACESHIP POSITON
  cp.xy*=r2(sin(tt*.4)*.5);  
  h=vec2(length(cp.xy)-(max(-1.,.3+cp.z*.03)),6); 
  h.x=max(h.x,bo(cp+vec3(0,0,25),vec3(10,10,30)));
  g+=0.1/(0.1*h.x*h.x*(20.-abs(sin(abs(cp.z*.1)-tt*3.))*19.7));
  t=t.x<h.x?t:h;
  cp*=1.3;
  for(int i=0;i<3;i++){ //SPACESHIP KIFS
    cp=abs(cp)-vec3(-2,0.5,4); 
    cp.xy*=r2(2.0);     
    cp.xz*=r2(.8+sin(cp.z*.1)*.2);     
    cp.yz*=r2(-.8+sin(cp.z*.1)*.2);     
  } 
  h=fb(cp,8.); h.x*=0.5;  t=t.x<h.x?t:h; //SPACESHIP  
  pp.z=mod(pp.z+tt*10.,40.)-20.; //CENTRAL STRUCTURE POSITION  
  pp=abs(pp)-vec3(0,20,0);  
  for(int i=0;i<3;i++){ //CENTRAL STRUCTURE KIFS
    pp=abs(pp)-vec3(4.2,3,0); 
    pp.xy*=r2(.785); 
    pp.x-=2.;
  }  
  h=fb(pp.zyx,7.); t=t.x<h.x?t:h; //CENTRAL STRUCTURE
  h=vec2(0.5*bo(abs(pp.zxy)-vec3(7,0,0),vec3(0.1,0.1,1000)),6); //GLOWY LINES CENTRAL STRUCTURE
  g+=0.2/(0.1*h.x*h.x*(50.+sin(np.y*np.z*.001+tt*3.)*48.)); t=t.x<h.x?t:h;
  t=t.x<h.x?t:h; return t; // Add central structure and return the whole shit
}
vec2 tr( vec3 ro, vec3 rd ) // main trace / raycast / raymarching loop function 
{
  vec2 h,t= vec2(.1); //Near plane because when it all started the hipsters still lived in Norwich and they only wore tweed.
  for(int i=0;i<128;i++){ //Main loop de loop 
    h=mp(ro+rd*t.x); //Marching forward like any good fascist army: without any care for culture theft. (get distance to geom)
    if(h.x<.0001||t.x>250.) break; //Conditional break we hit something or gone too far. Don't let the bastards break you down!
    t.x+=h.x;t.y=h.y; //Huge step forward and remember material id. Let me hold the bottle of gin while you count the colours.
  }
  if(t.x>250.) t.y=0.;//If we've gone too far then we stop, you know, like Alexander The Great did when he realised his wife was sexting some Turkish bloke. (10 points whoever gets the reference)
  return t;
}
#define a(d) clamp(mp(po+no*d).x/d,0.,1.)
#define s(d) smoothstep(0.,1.,mp(po+ld*d).x/d)
void mainImage( out vec4 fragColor, in vec2 fragCoord )//2 lines above are a = ambient occlusion and s = sub surface scattering
{
  vec2 uv=(fragCoord.xy/iResolution.xy-0.5)/vec2(iResolution.y/iResolution.x,1); //get UVs, nothing fancy, 
  tt=mod(iTime+3.,62.82);  //Time variable, modulo'ed to avoid ugly artifact. Imagine moduloing your timeline, you would become a cry baby straight after dying a bitter old man. Christ, that's some fucking life you've lived, Steve.
  dp=vec3(sin(tt*.4)*4.,20.+sin(tt*.4)*2.,-200.+mod(tt*30.,471.2388));
  vec3 ro=mix(dp-vec3(10,20.+sin(tt*.4)*5.,40),vec3(17,-5,0),ceil(sin(tt*.4))),//Ro=ray origin=camera position We build camera right here broski. Gotta be able to see, to peep through the keyhole.
  cw=normalize(dp-vec3(10,15,0)-ro), cu=normalize(cross(cw,normalize(vec3(0,1,0)))),cv=normalize(cross(cu,cw)),
  rd=mat3(cu,cv,cw)*normalize(vec3(uv,.5)),co,fo;//rd=ray direction (where the camera is pointing), co=final color, fo=fog color
  ld=normalize(vec3(.2,.4,-.3)); //ld=light direction
  co=fo=vec3(.1,.1,.15)-length(uv)*.1-rd.y*.1;//background is dark blueish with vignette and subtle vertical gradient based on ray direction y axis. 
  z=tr(ro,rd);t=z.x; //Trace the trace in the loop de loop. Sow those fucking ray seeds and reap them fucking pixels.
  if(z.y>0.){ //Yeah we hit something, unlike you at your best man speech.
    po=ro+rd*t; //Get ray pos, know where you at, be where you is.
    no=normalize(e.xyy*mp(po+e.xyy).x+e.yyx*mp(po+e.yyx).x+e.yxy*mp(po+e.yxy).x+e.xxx*mp(po+e.xxx).x); //Make some fucking normals. You do the maths while I count how many instances of Holly Willoughby there really is.
    al=mix(vec3(.4,.0,.1),vec3(.7,.1,.1),cos(bp.y*.08)*.5+.5); //al=albedo=base color, by default it's a gradient between red and darker red. 
    if(z.y<5.) al=vec3(0); //material ID < 5 makes it black
    if(z.y>5.) al=vec3(1); //material ID > 5 makes it white
    if(z.y>6.) al=clamp(mix(vec3(.0,.1,.4),vec3(.4,.0,.1),sin(np.y*.1+2.)*.5+.5)+(z.y>7.?0.:abs(ceil(cos(pp.x*1.6-1.1))-ceil(cos(pp.x*1.6-1.3)))),0.,1.);
    float dif=max(0.,dot(no,ld)), //Dumb as fuck diffuse lighting
    fr=pow(1.+dot(no,rd),4.), //Fr=fresnel which adds background reflections on edges to composite geometry better
    sp=pow(max(dot(reflect(-ld,no),-rd),0.),30.); //Sp=specular, stolen from Shane
    co=mix(sp+mix(vec3(.8),vec3(1),abs(rd))*al*(a(.1)*a(.4)+.2)*(dif),fo,min(fr,.3)); //Building the final lighting result, compressing the fuck outta everything above into an RGB shit sandwich
    co=mix(fo,co,exp(-.0000007*t*t*t)); //Fog soften things, but it won't stop your mother from being unimpressed by your current girlfriend
  }
  fo=mix(vec3(.1,.2,.4),vec3(.1,.1,.5),0.5+0.5*sin(np.y*.1-tt*2.));//Glow colour is actual a grdient to make it more intresting
  fragColor = vec4(pow(co+g*0.15*mix(fo.xyz,fo.zyx,clamp(sin(tt*.5),-.5,.5)+.5),vec3(.55)),1);// Naive gamma correction and glow applied at the end. Glow switches from blue to red hues - nice idea by Haptix - cheers broski
}`,
"Julia Colors": `/****************************************************
 * Combined Julia set with time-based zoom
 ****************************************************/

#define RECURSION_LIMIT 10000
#define PI 3.141592653589793238

// ---------------------------------
// Uniforms used by ShaderToy/WebGL:
// ---------------------------------
// uniform vec3 iResolution; // (width, height, 1.0)
// uniform float iTime;      // Time in seconds

// Method for the mathematical construction of the julia set
int juliaSet(vec2 z, vec2 constant) {
    int recursionCount;
    for (recursionCount = 0; recursionCount < RECURSION_LIMIT; recursionCount++) {
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + constant;
        if (length(z) > 2.0) {
            break;
        }
    }
    return recursionCount;
}

// Smooth transition helper (if you want it for some effect)
float smoothTransition(float time, float duration) {
    float t = mod(time, duration);
    return smoothstep(0.0, 1.0, t / duration);
}

// Choose a target point for focusing/zooming
vec2 getSpiralTarget(int index) {
    // Feel free to adjust these to match interesting areas
    if (index == 0) return vec2( 0.5,  0.10);
    if (index == 1) return vec2( 0.7,  0.10);
    if (index == 2) return vec2( -0.30,  0.4);
     if (index == 3) return vec2(-0.90,  0.40);
    if (index == 4) return vec2( 0.32, 0.42);
    if (index == 5) return vec2(-0.35, -0.30);
    if (index == 6) return vec2( 0.2, -0.70);
    // default (index == 7):
    return vec2( 0.6, -0.75);
}

// Main method of the shader
void mainImage(out vec4 fragColor, in vec2 fragCoord) 
{
    // ------------------------------------------------------
    // 1) Handle time-based zoom and target selection
    // ------------------------------------------------------
    float cycleTime = 10.0;  // total duration of one zoom in/out cycle
    float zoomPhase = mod(iTime, cycleTime) / cycleTime;
    
    // pick which target to center on based on time
    int targetIndex = int(floor(mod(iTime / cycleTime, 8.0)));
    vec2 target     = getSpiralTarget(targetIndex);
    
    // zoom in for first 70% of the cycle, out for last 30%
    float zoomFactor;
    if (zoomPhase < 0.7) {
        // zoom in
        zoomFactor = mix(1.0, 10.0, smoothstep(0.0, 1.0, zoomPhase / 0.7));
    } else {
        // zoom out
        zoomFactor = mix(10.0, 1.0, smoothstep(0.0, 1.0, (zoomPhase - 0.7) / 0.3));
    }
    
    // ------------------------------------------------------
    // 2) Original UV setup (but we apply zoom shift first)
    // ------------------------------------------------------
    // Normalized pixel coordinates from -1..+1 or so:
    // (Equivalent to: 2.0 * (fragCoord - 0.5*iResolution.xy) / iResolution.y)
    vec2 uv = fragCoord - 0.5 * iResolution.xy;
    uv /= iResolution.y;
    uv *= 2.0;
    
    // Apply the zoom, translating around our chosen target
    uv = (uv - target) / zoomFactor + target;

    // ------------------------------------------------------
    // 3) Keep the original rotation and scale
    // ------------------------------------------------------
    // Make a copy for color calculations after fractal
    vec2 uv2 = uv;
    
    // rotation angle [rad]
    float a = PI / 3.0;
    vec2 U  = vec2(cos(a), sin(a));  // new x-axis
    vec2 V  = vec2(-U.y, U.x);       // new y-axis
    // rotate
    uv = vec2(dot(uv, U), dot(uv, V));
    // optional scale
    uv *= 0.9;

    // ------------------------------------------------------
    // 4) Compute the Julia set
    // ------------------------------------------------------
    const vec2[6] constants = vec2[](
        vec2(-0.7176, -0.3842),
        vec2(-0.4,    -0.59),
        vec2( 0.34,   -0.05),
        vec2( 0.355,   0.355),
        vec2(-0.54,    0.54),
        vec2( 0.355534,-0.337292)
    );
    
    // pick whichever constant from above (e.g. #3 is classic)
    vec2 c = uv;
    int recursionCount = juliaSet(c, constants[3]);
    
    // ------------------------------------------------------
    // 5) Preserve the original coloring
    // ------------------------------------------------------
    float f   = float(recursionCount) / float(RECURSION_LIMIT);
    float ff  = pow(f, 1.0 - (f * 1.0));
    vec3 col  = vec3(1.0);
    float offset = 0.5;
    vec3 saturation = vec3(1.0, 1.0, 1.0);
    float totalSaturation = 1.0;
    
    // Original color approach
    col.r = smoothstep(0.0, 1.0, ff) * (uv2.x * 0.5 + 0.3);
    col.b = smoothstep(0.0, 1.0, ff) * (uv2.y * 0.5 + 0.3);
    col.g = smoothstep(0.0, 1.0, ff) * (-uv2.x * 0.5 + 0.3);
    
    // Scale to make the colors "pop" as in the original
    col.rgb *= 5000.0 * saturation * totalSaturation;
    
    // Output the final color to the screen
    fragColor = vec4(col, 1.0);
}
`,
"Julia Blue": `
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
    if (index == 0) return vec2(0.5, 0.10);  
    if (index == 1) return vec2(0.5, 0.5);    
    if (index == 2) return vec2(0.5, 0.85);   
    if (index == 3) return vec2(-0.5, 0.15);   
    if (index == 4) return vec2(0.52, 0.52);; 
    if (index == 5) return vec2(-0.5, -0.52); 
    if (index == 6) return vec2(0.2, -0.7);   
    return vec2(0.6, -0.75);                   
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
        zoomFactor = mix(1.0, 200.0, smoothstep(0.0, 1.0, zoomPhase / 0.7));
    } else {
        // Zoom out (last 30% of cycle)
        zoomFactor = mix(200.0, 1.0, smoothstep(0.0, 1.0, (zoomPhase - 0.7) / 0.3));
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
}`,
"Neural Net": `// variant of https://shadertoy.com/view/msGXW1

  #define S(v)     smoothstep(0.,1., v)                   // for AAdraw AND for arc shape 
  #define H(p)     fract(sin((p+.17) * 12.9898) * 43.5453)// hash
  #define N(s,x)  (s)* floor( 4.*H(d+T-7.7*(X+x)+x)/(s) ) // random node(time#, ball#, stage#)
  #define R       (floor(iResolution.xy/8.)*8.)           // to avoid fwidth artifact
//#define arc(t) mix(i,j, S(fract(t)) ) - U.y             // arc trajectory
  #define arc(t) mix( i+sin(i*iTime)*.2*s, j+sin(j*iTime)*.4*s, S(fract(t)) ) - U.y // animated variant
           
void mainImage( out vec4 O, vec2 U )
{
    U = 4.*U/R - 2.;                                    // normalize window in [-2,2]² = 4 stages
    float s = exp2(-floor(1.-U.x)), S=s, i,j,v,n;       // 2 / number of nodes at each stage
    
    O = vec4(1);                                        // start with a white page
    for( i = .5*s-2.; i < 2.; i += s )                  // === drawing the net (same as ref)
        for( j = s-2.; j < 2.; j += s+s )               // on each stage, loop on in/out pairs
            v = arc(U.x),      
            O *= .5+.5* S( abs(v)/fwidth(v) );          // blacken-draw curve arc()=0

    for ( n=0.; n<1.; n+=.1 )  {                        // === drawing the balls                                  
        float d = H(n),                                 // lauch 10 per second, at random time d
              X = floor(U.x), x = U.x-X,                // X = stage#, T = time#
              t = 2.-iTime+d, T = floor(t); t-=T;       // t = x coords ( fract(t) do each stage in // )                                                      
        s = S;
        if (t<.1 && x>.9 ) s*=2., X++;                  // manage ball at stage transition
        if (t>.9 && x<.1 ) s/=2., X--;
        i = .5*s-2. + N(s   ,0.);                       // select in/out nodes ( mimic draw curve above )
        j =    s-2. + N(2.*s,1.);                       // 1: offset for the input nodes 
        v = arc(t);                                         
        O = mix( vec4(1,0,0,1), O,                      // blend-draw ball
                 S( length( vec2( fract(U.x-t+.5)-.5, v )*R/4. ) -5. ) );
    }                                                   // fract: to draw all stages in parallel
  
    O = sqrt(O);                                        // to sRGB
}`,
"Sunset on river":`// Sunset over the ocean.
// Minimalistic three color 2D shader
// inspired by this wonderful GIF: https://i.gifer.com/4Cb2.gif
//
// Features automatic anti aliasing by using smooth gradients
// removing the need for multi sampling.
//
// Copyright (c) srvstr 2024
// Licensed under MIT
//
#define hh(uv) fract(sin(dot(vec2(12.956, 68.6459), uv)) * 59687.6705)

/* Computes and returns the value noise of
 * the specified position.
 */
float vnoise(in vec2 uv)
{
    vec2 fi = floor(uv);
    vec2 ff = smoothstep(0.0, 1.0, fract(uv));
    float v00 = hh(fi + vec2(0,0));
    float v01 = hh(fi + vec2(0,1));
    float v10 = hh(fi + vec2(1,0));
    float v11 = hh(fi + vec2(1,1));
    return mix(mix(v00, v10, ff.x),
               mix(v01, v11, ff.x),
               ff.y);
}

#define remap(x,w) ((x)*(w)-((w)*0.5))

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    
    // Bias for smoothstep function to simulate anti aliasing with gradients
    float dy = (smoothstep(0.0, -1.0, uv.y) * 40.0 + 1.5) / iResolution.y;
    
    // Delta time transformed to fake irregular wave motion
    vec2 delta = vec2(iTime * 1.5, cos(iTime * 3.141) * 0.5 + iTime);
    
    // Wave displacement factors
    // XY: scale the UV coordinates for the noise
    // Z: scales the noise's strength
    vec3 dispValues[4];
    dispValues[0] = vec3(1.0, 40.0, 8.0);
    dispValues[1] = vec3(5.0, 80.0, 4.0);
    dispValues[2] = vec3(10.0, 120.0, 2.0);
    dispValues[3] = vec3(20.0, 40.0, 2.0);
    
    float avg = 0.0;
    // Compute average of noise displacements
    for (int i = 0; i < 4; i++)
    {
        vec3 disp = dispValues[i];
        avg += remap(vnoise(uv * disp.xy + delta), disp.z);
    }
    avg /= 4.0;
    
    // Displace vertically
    vec2 st = vec2(uv.x, uv.y + clamp(avg * smoothstep(0.1, -1.0, uv.y), -0.1, 0.1));
    
    // Compose output gradients
    fragColor.rgb = mix(vec3(0.85, 0.55, 0),
                        vec3(0.90, 0.40, 0),
                        sqrt(abs(st.y * st.y * st.y)) * 28.0)
                    /* Mask sun */
                    * smoothstep(0.25 + dy, 0.25, length(st))
                    /* Vignette + Background tint */
                    + smoothstep(2.0, 0.5, length(uv)) * 0.1;
    
    // Set alpha channel
    fragColor.a = 1.0;
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

