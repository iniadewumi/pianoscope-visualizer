
// Sample Shadertoy shaders to quickly test
export const SHADERS = {
    "Striped Water": `#define DRAG_MULT 0.38
#define WATER_DEPTH 2.8
#define CAMERA_HEIGHT 3.5
#define ITERATIONS_RAYMARCH 12
#define ITERATIONS_NORMAL 36
#define NormalizedMouse (iMouse.xy / iResolution.xy)

// ── Strip color lookup ────────────────────────────────────────────────────────
// 20 strips across world X axis, 5 colors cycling, 2px black separators
vec3 getStripColor(float worldX, float worldZ) {
  // Use world X position to determine strip
  // Scale so ~20 strips are visible across the view
  float stripScale = 0.4; // tune to match strip width in view
  float pos = worldX * stripScale;
  float stripF = pos - floor(pos);        // 0..1 within strip
  int   stripI = int(abs(floor(pos)));    // strip index

  // 2px black line — line width in strip-fraction units
  float lineW = 0.008;
  if (stripF < lineW || stripF > 1.0 - lineW) {
    return vec3(0.05); // near-black separator
  }

  int c = int(mod(float(stripI), 5.0));
  if (c == 0) return vec3(0.176, 0.220, 0.212); // dark teal
  if (c == 1) return vec3(0.325, 0.463, 0.298); // forest green
  if (c == 2) return vec3(0.902, 0.380, 0.341); // coral/red
  if (c == 3) return vec3(0.969, 0.714, 0.533); // peach
  return             vec3(0.929, 0.659, 0.718); // light pink
}

// ── Wave functions (unchanged from afl_ext) ──────────────────────────────────
vec2 wavedx(vec2 position, vec2 direction, float frequency, float timeshift) {
  float x = dot(direction, position) * frequency + timeshift;
  float wave = exp(sin(x) - 1.0);
  float dx = wave * cos(x);
  return vec2(wave, -dx);
}

float getwaves(vec2 position, int iterations) {
  float wavePhaseShift = length(position) * 0.1;
  float iter = 0.0;
  float frequency = 1.0;
  float timeMultiplier = 2.0;
  float weight = 1.0;
  float sumOfValues = 0.0;
  float sumOfWeights = 0.0;
  for (int i = 0; i < iterations; i++) {
    vec2 p = vec2(sin(iter), cos(iter));
    vec2 res = wavedx(position, p, frequency, iTime * timeMultiplier + wavePhaseShift);
    position += p * res.y * weight * DRAG_MULT;
    sumOfValues  += res.x * weight;
    sumOfWeights += weight;
    weight *= 0.8;
    frequency    *= 1.18;
    timeMultiplier *= 1.07;
    iter += 1232.399963;
  }
  return sumOfValues / sumOfWeights;
}

float raymarchwater(vec3 camera, vec3 start, vec3 end, float depth) {
  vec3 pos = start;
  vec3 dir = normalize(end - start);
  for (int i = 0; i < 64; i++) {
    float height = getwaves(pos.xz, ITERATIONS_RAYMARCH) * depth - depth;
    if (height + 0.01 > pos.y) {
      return distance(pos, camera);
    }
    pos += dir * (pos.y - height);
  }
  return distance(start, camera);
}

vec3 normal(vec2 pos, float e, float depth) {
  vec2 ex = vec2(e, 0);
  float H = getwaves(pos.xy, ITERATIONS_NORMAL) * depth;
  vec3 a = vec3(pos.x, H, pos.y);
  return normalize(cross(
    a - vec3(pos.x - e, getwaves(pos.xy - ex.xy, ITERATIONS_NORMAL) * depth, pos.y),
    a - vec3(pos.x, getwaves(pos.xy + ex.yx, ITERATIONS_NORMAL) * depth, pos.y + e)
  ));
}

mat3 createRotationMatrixAxisAngle(vec3 axis, float angle) {
  float s = sin(angle), c = cos(angle), oc = 1.0 - c;
  return mat3(
    oc*axis.x*axis.x+c,        oc*axis.x*axis.y-axis.z*s, oc*axis.z*axis.x+axis.y*s,
    oc*axis.x*axis.y+axis.z*s, oc*axis.y*axis.y+c,        oc*axis.y*axis.z-axis.x*s,
    oc*axis.z*axis.x-axis.y*s, oc*axis.y*axis.z+axis.x*s, oc*axis.z*axis.z+c
  );
}

vec3 getRay(vec2 fragCoord) {
  vec2 uv = ((fragCoord.xy / iResolution.xy) * 2.0 - 1.0)
            * vec2(iResolution.x / iResolution.y, 1.0);
  vec3 proj = normalize(vec3(uv.x, uv.y, 1.5));
  if (iResolution.x < 600.0) return proj;
  return createRotationMatrixAxisAngle(vec3(0.0, -1.0, 0.0),
           3.0 * ((NormalizedMouse.x + 0.5) * 2.0 - 1.0))
       * createRotationMatrixAxisAngle(vec3(1.0, 0.0, 0.0),
           0.5 + 1.5 * (((NormalizedMouse.y == 0.0 ? 0.27 : NormalizedMouse.y)) * 2.0 - 1.0))
       * proj;
}

float intersectPlane(vec3 origin, vec3 direction, vec3 point, vec3 normal) {
  return clamp(dot(point - origin, normal) / dot(direction, normal), -1.0, 9991999.0);
}

// ── Desert storm sky (ported from afl_ext desert shader) ─────────────────────

vec2 hash22_s(vec2 p) {
  float n = sin(dot(p, vec2(113.0, 1.0)));
  return fract(vec2(2097152.0, 262144.0) * n) * 2.0 - 1.0;
}
float gradN2D_s(in vec2 f) {
  const vec2 e = vec2(0.0, 1.0);
  vec2 p = floor(f); f -= p;
  vec2 w = f * f * (3.0 - 2.0 * f);
  float c = mix(
    mix(dot(hash22_s(p+e.xx), f-e.xx), dot(hash22_s(p+e.yx), f-e.yx), w.x),
    mix(dot(hash22_s(p+e.xy), f-e.xy), dot(hash22_s(p+e.yy), f-e.yy), w.x),
    w.y);
  return c * 0.5 + 0.5;
}
float fBm_s(in vec2 p) {
  return gradN2D_s(p)*0.57 + gradN2D_s(p*2.0)*0.28 + gradN2D_s(p*4.0)*0.15;
}
vec3 getSunDir() {
  return normalize(vec3(0.4, 0.35 + sin(iTime * 0.04 + 1.0) * 0.15, 0.7));
}
vec3 getSky(vec3 dir) {
  vec3 ld = getSunDir();
  // Base: warm sandy horizon blending to blue zenith
  vec3 sky = mix(vec3(0.80, 0.70, 0.50), vec3(0.40, 0.60, 0.90),
                 pow(max(dir.y + 0.15, 0.0), 0.5));
  sky *= vec3(0.84, 1.0, 1.17);
  // Sun glow + disc
  float sun = clamp(dot(ld, dir), 0.0, 1.0);
  sky += vec3(1.0, 0.70, 0.40) * pow(sun, 16.0) * 0.20;
  sky += vec3(1.0, 0.90, 0.60) * pow(sun, 64.0) * 0.35;
  sky += vec3(1.0, 0.95, 0.80) * pow(sun, 720.0) * 1.80;
  // Drifting clouds projected onto sky dome
  vec3 rd2 = normalize(dir * vec3(1.0, 1.0, 1.0 + length(dir.xy)*0.15));
  const float SC = 1e5;
  float ct = SC * 0.15 / (rd2.y + 0.15);
  if (ct > 0.0) {
    vec2 uv = vec2(iTime * 0.8, 0.0) + rd2.xz * ct / SC;
    float cloud = fBm_s(uv * 1.5);
    float cloudMask = smoothstep(0.45, 1.0, cloud)
                    * smoothstep(0.45, 0.55, rd2.y * 0.5 + 0.5);
    sky = mix(sky, vec3(1.8), cloudMask * 0.45);
  }
  return sky;
}

// ── Main ─────────────────────────────────────────────────────────────────────
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec3 ray = getRay(fragCoord);

  if (ray.y >= 0.0) {
    fragColor = vec4(getSky(ray), 1.0);
    return;
  }

  vec3 waterPlaneHigh = vec3(0.0,  0.0,          0.0);
  vec3 waterPlaneLow  = vec3(0.0, -WATER_DEPTH,  0.0);
  vec3 origin = vec3(iTime * 0.2, CAMERA_HEIGHT, 1.0);

  float highPlaneHit = intersectPlane(origin, ray, waterPlaneHigh, vec3(0,1,0));
  float lowPlaneHit  = intersectPlane(origin, ray, waterPlaneLow,  vec3(0,1,0));
  vec3  highHitPos   = origin + ray * highPlaneHit;
  vec3  lowHitPos    = origin + ray * lowPlaneHit;

  float dist       = raymarchwater(origin, highHitPos, lowHitPos, WATER_DEPTH);
  vec3  waterHitPos = origin + ray * dist;

  // Normal + distance smoothing (unchanged)
  vec3 N = normal(waterHitPos.xz, 0.01, WATER_DEPTH);
  N = mix(N, vec3(0.0, 1.0, 0.0), 0.8 * min(1.0, sqrt(dist * 0.01) * 1.1));

  // Fresnel
  float fresnel = 0.04 + (1.0 - 0.04) * pow(1.0 - max(0.0, dot(-N, ray)), 5.0);

  // Reflection ray
  vec3 R = normalize(reflect(ray, N));
  R.y = abs(R.y);
  vec3 reflection = getSky(R);

  // Strip color replaces subsurface scattering
  vec3 stripCol = getStripColor(waterHitPos.x, waterHitPos.z);

  // Blend: fresnel controls reflection vs strip color
  vec3 C = mix(stripCol, reflection, fresnel * 0.55);

  // Subtle depth darkening
  float depthFade = clamp((waterHitPos.y + WATER_DEPTH) / WATER_DEPTH, 0.0, 1.0);
  C *= 0.75 + 0.25 * depthFade;

  fragColor = vec4(C, 1.0);
}`,
    "Murakami Galaxy": `mat4 rotationX( in float angle ) {
    
    float c = cos(angle);
    float s = sin(angle);
    
	return mat4(1.0, 0,	 0,	0,
			 	0, 	 c,	-s,	0,
				0, 	 s,	 c,	0,
				0, 	 0,  0,	1);
}

mat4 rotationY( in float angle ) {
    
    float c = cos(angle);
    float s = sin(angle);
    
	return mat4( c, 0,	 s,	0,
			 	 0,	1.0, 0,	0,
				-s,	0,	 c,	0,
				 0, 0,	 0,	1);
}

mat4 rotationZ( in float angle ) {
    float c = cos(angle);
    float s = sin(angle);
    
	return mat4(c, -s,	0,	0,
			 	s,	c,	0,	0,
				0,	0,	1,	0,
				0,	0,	0,	1);
}



// Murakami Galaxy by Philippe Desgranges
// Email: Philippe.desgranges@gmail.com
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
//
//
// To follow up on my current obsession with Takashi Murakami (see: Infinite Murakami)
// I wanted to give a tribute to his spherical flower patterns (put exemple here)
// but instead of single compositions give it a infinite galaxy scale with a central flower/sun.
// This idea ended up being quite challenging in many aspects and I learned a lot in the process of
// bringing it to a reality, especially :
//
// Sphere parametrization : The first thing I did was to modify the fower pattern I did for Infinite
// Murakami to map it onto a sphere. Of course, some pretty bad distortion around the poles due to
// spherical texture mapping so I had to find a way to compensate for that by reparametrizing polar
// coordinates. I wrapped my head around the problem for a couple of days (no pun intended) and ended
// up finding a tiling scheme consising of mappin the sphere with meridian bands with variying number
// of flowers to compensate for the horizontal stretch and some taper compensation.
//
// Fast Ray casting : The approach I first used was a classic SDF ray casting. My SDF was evaluating
// 27 (3*3*3) adjacent cells (containing zero or one planet each). Empty space had to be traversed with
// a lot of caution and it was full of hooks and crannies so it ended up being super slow, especially on
// my laptop (6fps tops). I realized that because my geometry was qhite simple (a bunch of sphere in a grid)
// I could just traverse the grid using a bresenham-like traching and just evaluate ray/sphere intersection
// analytically along the way in crossed cells. It gave me 10X speedup which brough me an immense satisfaction.
// 
// Anti-aliasing was also a challenge and, although the preview looks decent it is much better looking
// in fullscreen.
//
// I think I'll move on from the Murakami theme for my next entries. I'm done for now :D
//

//#define MSAA // WANING: on some architecture this leads to long compile times

#define MAX_DST 50.0
#define sat(a) clamp(a,0.0,1.0)

const float pi = 3.1415926;
const float halfPi = pi * 0.5;
const float pi2 = pi * 2.0;

const float quadrant = pi / 6.0;

const float blackLevel = 0.3; // True black is too aggressive


#define S(a,b,t) smoothstep(a,b,t)

// Some hash function 2->1
float N2(vec2 p)
{	// Dave Hoskins - https://www.shadertoy.com/view/4djSRW
    p = mod(p, vec2(500.0));
	vec3 p3  = fract(vec3(p.xyx) * vec3(443.897, 441.423, 437.195));
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

// A 2d Noise used for the sun rays
float Noise(vec2 uv)
{
    vec2 corner = floor(uv);
	float c00 = N2(corner + vec2(0.0, 0.0));
	float c01 = N2(corner + vec2(0.0, 1.0));
	float c11 = N2(corner + vec2(1.0, 1.0));
	float c10 = N2(corner + vec2(1.0, 0.0));
    
    vec2 diff = fract(uv);
    
    diff = diff * diff * (vec2(3.0) - 2.0 * diff);
    
    return mix(mix(c00, c10, diff.x), mix(c01, c11, diff.x), diff.y);
}

// An ellipse signed distance function by iq
// https://iquilezles.org/articles/ellipsedist
float sdEllipse( in vec2 z, in vec2 ab )
{
    vec2 p = vec2(abs(z));
    
    if( p.x > p.y ){ p=p.yx; ab=ab.yx; }
	
    float l = ab.y*ab.y - ab.x*ab.x;
    float m = ab.x*p.x/l; float m2 = m*m;
    float n = ab.y*p.y/l; float n2 = n*n;
    float c = (m2 + n2 - 1.0)/3.0; float c3 = c*c*c;
    float q = c3 + m2*n2*2.0;
    float d = c3 + m2*n2;
    float g = m + m*n2;
    

    float co;

    if( d<0.0 )
    {
        float p = acos(q/c3)/3.0;
        float s = cos(p);
        float t = sin(p)*sqrt(3.0);
        float rx = sqrt( -c*(s + t + 2.0) + m2 );
        float ry = sqrt( -c*(s - t + 2.0) + m2 );
        co = ( ry + sign(l)*rx + abs(g)/(rx*ry) - m)/2.0;
    }
    else
    {
        float h = 2.0*m*n*sqrt( d );
        float s = sign(q+h)*pow( abs(q+h), 1.0/3.0 );
        float u = sign(q-h)*pow( abs(q-h), 1.0/3.0 );
        float rx = -s - u - c*4.0 + 2.0*m2;
        float ry = (s - u)*sqrt(3.0);
        float rm = sqrt( rx*rx + ry*ry );
        float p = ry/sqrt(rm-rx);
        co = (p + 2.0*g/rm - m)/2.0;
    }

    float si = sqrt( 1.0 - co*co );
 
    vec2 closestPoint = vec2( ab.x*co, ab.y*si );
	
    return length(closestPoint - p ) * sign(p.y-closestPoint.y);
}


// rotates pos to align the up vector towards up
vec2 rotUp(vec2 pos, vec2 up)
{
    vec2 left = vec2(-up.y, up.x);
    return left * pos.x + up * pos.y;
}

// The mouth is the intersection of two ellipses, I traced them in photoshop to
// compute the right radii and offsets
float mouthDst(vec2 uv)
{
    return max(sdEllipse(uv - vec2(0.0, -0.17), vec2(0.10, 0.2055)),
               sdEllipse(uv - vec2(0.0,  0.07), vec2(0.14, 0.1055)));
}

// For the eye, I use simpler circle distance maths in a scales and rotated space
// as I don't need an accurate distance function to create an outline
vec4 eye(vec2 uv, vec2 up, vec2 spot1, vec2 spot2, float aa)
{
    uv = rotUp(uv, up);
    uv.x *= 1.5;
    
    float len = length(uv);
    float len2 = length(uv + spot1);// vec2(0.010, 0.025));
    float len3 = length(uv + spot2);// vec2(-0.005, -0.017));
    
    vec4 eye;
    
    eye.a = S(0.04 + aa, 0.04 - aa, len);
    
    eye.rgb = vec3(S(0.014 + aa, 0.014 - aa, len2) + S(0.02 + aa, 0.02 - aa, len3) + blackLevel);
    
    return eye;
}

const float cRatio = 1.0 / 255.0;

// I wanted the color palette to be true to the 16 hue rainbow used
// by Murakami but I didn't manage to reproduce the orange-yellow-green part
// using simple maths so I defaulted to a palette. Then I realized I couldn't target
// Webgl < 3.0 (Wich was one of my objectives) with array constructor so I decided
// to build a function selecting the right color with a dichotomic approch in hope
// that the compiler will make a decent job of optimizing all those branches.
vec3 palette(float id)
{
	if (id < 6.0)
    {
        //[0 - 5]
        if (id < 3.0)
        {   //[0 - 2]
            if (id < 1.0) return vec3(181.0, 23.0, 118.0) * cRatio;
            else if (id < 2.0) return vec3(225.0, 27.0, 104.0) * cRatio;
            else return vec3(230.0, 40.0, 24.0) * cRatio;
        }
        else
        {   //[3 - 5]
            if (id < 4.0) return vec3(240.0, 110.0, 14.0) * cRatio;
            else if (id < 5.0) return vec3(253.0, 195.0, 2.0) * cRatio;
            else return vec3(253.0, 241.0, 121.0) * cRatio;
        }
    }
    else
    {   //[6 - 11]
        if (id < 9.0)
        {   //[6 - 8]
            if (id < 7.0) return vec3(167.0, 202.0, 56.0) * cRatio;
            else if (id < 8.0) return  vec3(0.0, 152.0, 69.0) * cRatio;
            else return vec3(2.0, 170.0, 179.0) * cRatio;
        }
        else
        {   //[9 - 11] The darker color are at the end to be avoided by mod
            if (id < 10.0) return vec3(25.0, 186.0, 240.0) * cRatio;
            else if (id < 11.0) return  vec3(0.0, 98.0, 171.0) * cRatio;
            else return vec3(40.0, 49.0, 118.0) * cRatio;
        }
    }
}


// Adapted from BigWIngs
vec4 N24(vec2 t) {
    float n = mod(t.x * 458.0 + t.y * 127.3, 100.0);
	return fract(sin(n*vec4(123., 1024., 1456., 264.))*vec4(6547., 345., 8799., 1564.));
}

// Drawing a Murakami flower from a random seed (how poetic)
vec4 flower(vec2 uv, vec4 rnd, float scale, float aaScale, float petalAngle, out vec3 col, float eyesAA)
{
    
    float rdScale = 1.0;
    
    scale *= rdScale; // The border thickness & AA is scale-independant
    
    uv.xy *= rdScale;
    
    float aa2 = aaScale * 5.0 / iResolution.x; // increase AA over disatnce and facing ratio
    
    float centerDst = length(uv);
        
    float edge; // Mask for the outline edge
    
    vec4 color = vec4(1.0, 1.0, 1.0, 1.0); // Underlying color
   
    
    float thick = 0.002 * scale;
    
    float col1Id = mod((rnd.x + rnd.y) * 345.456, 10.0);
    col = palette(col1Id); // return the 'main' color of the petals
 
    
    if (centerDst < 0.2)
    {
        //Face part
        
        float thres = 0.2 - thick;
        
        // inner part of edge circle surrounding the head
        edge =  S(thres + aa2, thres - aa2, centerDst);
        
        float mouth = mouthDst(uv);
        
        // edge of the mouth
        edge *= S(thick - aa2, thick + aa2, abs(mouth));
        
        // face color
        float faceRnd = fract(rnd.x * 45.0 + rnd.y * 23.45);
        if (faceRnd < 0.5) 
        {
            // Flowers with classic yellow / red faces
        	color.rgb = (mouth < 0.0) ? vec3(1.0, 0.0, 0.0) : vec3(1.0, 1.0, 0.0); 
        }
        else
        {
            // Flowers with white face / random color mouth
            float colId = mod(faceRnd * 545.456, 11.0);
            color.rgb = (mouth < 0.0) ? palette(colId) : vec3(1.0); 
        }
        
        // Eyes
        vec4 eyeImg;
        if (uv.x > 0.0)
        {
           eyeImg = eye(uv - vec2(0.075, 0.095), vec2(-0.7, 1.2),
                       vec2(0.007, 0.025), vec2(-0.004, -0.019), aa2 * eyesAA);
        }
        else   
        {
           eyeImg = eye(uv - vec2(-0.075, 0.095), vec2(0.7, 1.2),
                       vec2(0.024, 0.010), vec2(-0.016, -0.009), aa2 * eyesAA);
        }

        color.rgb = mix(color.rgb, eyeImg.rgb, eyeImg.a);
        
    }
    else
    {
        float rot = petalAngle;
        float angle = fract((atan(uv.x, uv.y) + rot) / pi2);
    
        float section = angle * 12.0;
        float sectionId = floor(section);
        
        if (rnd.z < 0.1 && rnd.w < 0.1)
        {
           // Rainbow flower
           color.rgb = palette(sectionId);//mod(sectionId + (rnd.x + rnd.y) * 345.456, 12.0));
        }
        else if (rnd.y > 0.05)
        {
           
            //Alternating flower
            if (mod(sectionId, 2.0) == 0.0)
            {
                // Color 1
                color.rgb = col;
            }
            else if (rnd.x > 0.75)
            {
                // Color 2
                float colId = mod((rnd.w + rnd.z) * 545.456, 11.0);
                color.rgb = palette(colId);
            }
            // else, Color2 is white by default
        }
		// else, fully white petals
        
        if (centerDst < 0.36)
        {
            //intermediate part, concentric bars
            
            float sectionX = fract(section);
            float edgeDist = 0.5 - abs(sectionX - 0.5);
            
            edgeDist *= centerDst; // Untaper bar space so bars have constant thickness
            
            float aa = aaScale * 10.0 / iResolution.x;
            float bar = thick * 1.7;
            edge = S(bar - aa, bar + aa, edgeDist);

            // outer part of edge circle surrounding the head
            float thres = 0.2 + thick;
            float head = S(thres - aa2, thres + aa2, centerDst);
            edge *= head;
        }
        else
        {
            // Petal tips are actually ellipses, they could have been approximated them with
            // circles but I didn't because I have OCD and I needed the ellipse SDF 
            // for the mouth anyways ;)
            
            // Angle to the center of the quadrant
            float quadAngle = (sectionId + 0.5) * quadrant - rot + pi; 

            // Center of the ellipse
            vec2 petalUp = vec2(-sin(quadAngle), -cos(quadAngle));
            vec2 petalCenter = petalUp * 0.36;

            // Rotation of the ellipse basis
            vec2 petalSpace = rotUp(uv - petalCenter, petalUp);

            // Signed distance function of the ellipse
            float petalDst = sdEllipse(petalSpace, vec2(0.0944, 0.09));

            //border edge and alpha mask
            float borderIn = S(thick + aa2, thick - aa2, petalDst);
            float borderOut = S(-thick + aa2, -thick - aa2, petalDst);

            edge = (borderOut);
            
            color.a = borderIn;
        }
    }
    
    color.rgb = mix(vec3(blackLevel), color.rgb,edge);
    
    return color;
}

struct planet
{
    vec3 center;
    float radius;
};

// randomizes planet position & radius for a sector
void GetPlanet(vec3 sector, out planet res)
{
   	vec4 rnd = N24(vec2(sector.x + sector.z * 1.35, sector.y));
    float rad = mix(0.0, 0.4, rnd.x * rnd.w);
    res.radius = rad;
    res.center = vec3(rad) + rnd.yzw * vec3(1.0 - 2.0 * rad); // the smaller the planet is, the more off center it can get without crossing border
}


float remap(float val, float min, float max)
{
    return sat((val - min) / (max - min));
}

// breaks down a band of UV coordinates on a sphere to a repetition of square-ish cells with minimal distortion
vec2 ringUv(vec2 latLon, float angle, float centerLat)
{
    // latlon : latitude / longitude
    // angle: horizontal angle covered by one rep of the pattern over the equator / angular height of the band
    // centerLat : center latitude of the band
    
    
    // Compute y coords by remapping latitude 
    float halfAngle = angle * 0.5;
    float y = remap(latLon.y, centerLat - halfAngle,  centerLat + halfAngle);
    
    float centerRatio = cos(centerLat); // stretch of the horizontal arc of the pattern at the center of the 
   										// band relative to the equator
    
    float centerAngle = angle / centerRatio; // local longitudianl angle to compensate for stretching at the center of the band. 
    
    float nbSpots = floor(pi2 / centerAngle); // with new angle, how many pattern can we fit in the band?
    float spotWidth = pi2 / nbSpots;          // and what angle would they cover (including spacing padding)?
    
    float cellX = fract(latLon.x / spotWidth); // what would be the u in the current cell then?
                  
                  
    float x = (0.5 - cellX) * (spotWidth / centerAngle); // compensate for taper
    x *= (cos(latLon.y) / centerRatio) * 0.5 + 0.5;
    
    vec2 uvs = vec2(x + 0.5, y);
    return uvs;
}


// Computes the texture of the planet
vec3 sphereColor(vec3 worldPos, float nDotV, float dist, float worldAngle)
{    
    // which planet are we talnikg about already?
    // This is done way to much for final rendering, could be optimized out
    planet p;
	vec3 sector = floor(worldPos);
    GetPlanet(sector, p);

    // Scale AA accourding to disatnce and facing ratio
   	float aaScale = 4.0 - nDotV * 3.8 + min(4.0, dist * dist * 0.025);
    
    // Find local position on the sphere
    vec3 localPos = worldPos - (sector + p.center);
    
    // Random seed that will be used for the two flower layers
    vec4 rnd = N24(vec2(sector.x, sector.y + sector.z * 23.4));
    vec4 rnd2 = N24(rnd.xy * 5.0);
    
    // compensate for the world Z rotation so planets stay upright
    localPos = (rotationZ(-worldAngle) * vec4(localPos, 0.0)).xyz;
    // Planet rotation at random speed
    localPos = (rotationY(iTime * (rnd.w - 0.5)) * vec4(localPos, 0.0)).xyz;
   
    
    // Compute polar coordinates on the sphere
    float lon = (atan(localPos.z, localPos.x)) + pi;  // 0.0 - 2 * pi
    float lat  = (atan(length(localPos.xz), localPos.y)) - halfPi; //-halfPi <-> halfPi
    
    // Compute the number of flowers at the equator according to the size of the planet
    float numAtEquator = floor(3.0 + p.radius * 15.0);
    float angle = pi2 / numAtEquator; // an the angle they cover ath the equator
    
    vec3 col1;
    vec3 col2;
    
    float petalAngle = rnd.w * 45.35 + iTime * 0.1;
    
    // Compute on layer of flower by dividing the sphere in horizontal bands of 'angle' height 
    float eq = (floor(lat / angle + 0.5)) * angle;
    vec2 uvs = ringUv(vec2(lon + eq * rnd.y * 45.0, lat), angle, eq);
    vec4 flPattern1 = flower((vec2(0.5) - uvs) * 0.95, rnd, 2.0, aaScale, petalAngle, col1, 0.8);
    
    
    // Compute a second layer of flowers with bands offset by half angle
    float eq2 = (floor(lat / angle) + 0.5) * angle;
    vec2 uvs2 = ringUv(vec2(lon + eq2 * rnd.x * 33.0, lat), angle, eq2);
    vec4 flPattern2 = flower((vec2(0.5) - uvs2) * 0.95, rnd2, 2.0, aaScale, petalAngle, col2, 0.8);
    

    // Compute flower with planar mapping on xz to cover the poles. 
    vec4 flPattern3 = flower(localPos.xz / p.radius, rnd2, 2.0, aaScale, petalAngle, col2, 0.8);
    
    float bg = (1.0 - nDotV);
    vec3 bgCol = rnd2.y > 0.5 ? col1 : col2; // sphere background is the color of one of the layers
    
    vec3 col = bgCol; 
    
    // mix the 3 layers of flowers together
    col = mix(col, flPattern1.rgb, flPattern1.a);
    col = mix(col, flPattern2.rgb, flPattern2.a);
    col = mix(col, flPattern3.rgb, flPattern3.a);
    
    // add some bogus colored shading
    
    //Front lighting
    //col *= mix(vec3(1.0), bgCol * 0.3, (bg * bg) * 0.8);

    return col;
}


// Analytical nomral compoutation
// Much faster and acuurate than SDF in my situation
vec3 calcNormal( vec3 pos )
{
    // computes planet in sector
    planet p;
    vec3 sector = floor(pos);
    GetPlanet(sector, p);
    
    // return vector 
    return normalize(pos - (sector + p.center));
}


// Lifted from Rye Terrell at https://gist.github.com/wwwtyro/beecc31d65d1004f5a9d
// modified to compute coverage
float raySphereIntersect(vec3 r0, vec3 rd, vec3 s0, float sr, out float coverage) {
    // - r0: ray origin
    // - rd: normalized ray direction
    // - s0: sphere center
    // - sr: sphere radius
    // - Returns distance from r0 to first intersecion with sphere,
    //   or MAX_DST if no intersection.
    float a = dot(rd, rd);
    vec3 s0_r0 = r0 - s0;
    float b = 2.0 * dot(rd, s0_r0);
    float c = dot(s0_r0, s0_r0) - (sr * sr);
    
    float inside = b*b - 4.0*a*c;
    
    if (inside < 0.0) {
        return MAX_DST;
    }
    
    float dst = (-b - sqrt((b*b) - 4.0*a*c))/(2.0*a);
    
    // This is a fallof around the edge used for AO
    // chnage the magic value for a smoother border
    coverage = S(inside, 0.0, 0.65 * sr * dst / iResolution.x);
    
    return dst;
}

// Computes the RGBA of a planet according to intersection result
vec4 RenderPlanet(vec3 pos, float d, vec3 rayDir, float worldAngle, float coverage)
{
	vec3 n = calcNormal(pos);
        
    float nDotV = abs(dot(n, rayDir));
 
    float fog = sat((MAX_DST - d) * 0.1);
    
 
    // compute some rim lighting to kind of blend everything together
    vec3 burn  = sat(mix(vec3(2.0, 2.0, 1.5), vec3(1.0, 0.4, 0.2), sat((MAX_DST - d) * 0.05) + nDotV) * 0.5);
    
    // Compute the flowery 'texture' on the planet
    vec3 flowers = sphereColor(pos, nDotV, d, worldAngle);
    
    
    // bogus lighting from the sun
    vec3 lightPos = pos + vec3(-15.0, -20.0, 60.0);
    float nDotL = sat(dot(n, normalize(lightPos - pos)) * 0.5 + 0.5);
    flowers *= nDotL * 0.8 + 0.5;
    
    // fades the planets at the horizon
    vec4 col;
    col.rgb = flowers + burn;
    
    
    col.a = fog * coverage;
    
    // Uncomment to debug coverage AA
    //col.rgb = mix(vec3(0,1,0), col.rgb, coverage);
    //col.a = fog;
    
    return col;
}

// Blends two colors front to back
vec4 BlendFTB(vec4 frontPremul, vec4 backRGBA)
{
    vec4 res;
    
    res.rgb = backRGBA.rgb * (backRGBA.a * (1.0 - frontPremul.a)) + frontPremul.rgb;
    res.a = 1.0 - ((1.0 - backRGBA.a) * (1.0 - frontPremul.a));
    
    return res;
}

// Finds the intersection of a ray with a planet in a given sector
// The coverage is an small alpha falllof at the edge for AA
// thanks iq for the recommendation
float castPlanet(vec3 cell, vec3 pos, vec3 dir, out float coverage)
{
	vec2 pp = cell.xy + cell.xy;
    if (dot(pp.xy, pp.xy) < 1.5) return MAX_DST; // we leave a 'tunnel' empty along the z axis 
                
 	planet p;
    
    GetPlanet(cell, p);            
    if (p.radius < 0.06) return MAX_DST; // cull planets that are too small
    
    // ray sphere intersection from the start position
    
    return raySphereIntersect(pos, dir,  cell + p.center, p.radius, coverage); 
}

// Traverses the cells grid in a bresenham fashion and test ray/sphere intersection along the way
// This appoach ended up being much faster than SDF for that 'simple' yet dense geometry
//
// Edit: now, this function also performs the accumulation of planet colors according to coverage
// The colors are coputed with the RenderPlanet function, the ray is stopped when full opacity is
// reached
vec4 castRay(vec3 pos, vec3 dir, float maxDst, float worldAngle)
{
    // we assume we are traversing space facing Z
    
    vec3 dirZ = dir / dir.z; // direction vector that adavance a full cell along Z
    
    vec3 cell = floor(pos); // starting cell
    
    vec3 start = pos; // saves the start of the ray
    pos -= fract(pos.z) * dirZ; // pulls back pos on the closes cell boundary behind
   

    float d = 0.0;
    float dst;
    
    vec2 layers[20];
    int num = 0;

    float coverage;
    float opacity = 1.0;

    while (d < MAX_DST)
    {
		// Check current cell
        dst = castPlanet(cell, start, dir, coverage);
        if (dst < MAX_DST)
        {
            // Blends the hit planet behind the previous ones according to coverage
            //ColorFTB = BlendFTB(ColorFTB, RenderPlanet(start + dst * dir, dst, dir, worldAngle, coverage));
            layers[num++] = vec2(dst, coverage);
            opacity *= (1.0 - coverage);
            if (opacity < 0.01) break;
        }
        
        // Advances a step
        pos += dirZ;
        
        //Compute next cell on y
        vec3 newCell = floor(pos);
        
        bool a = false;
        bool b = false;
        float cornerDst = MAX_DST;
        
 		
        if (cell.x != newCell.x) // have we crossed a cell diagonally on X ?
        {
            vec3 stepCell = vec3(newCell.x, cell.yz);

            dst = castPlanet(stepCell, start, dir, coverage);
        	if (dst < cornerDst) cornerDst = dst;
            a == true;
        }
        
        if (cell.y != newCell.y)  // have we crossed a cell diagonally on Y ?
        {
            vec3 stepCell = vec3(cell.x, newCell.y, cell.z);

            dst = castPlanet(stepCell, start, dir, coverage);
        	if (dst < cornerDst) cornerDst = dst;
            b == true;
        }
        
        if (a && b)  // have we crossed a cell diagonally on both X & Y?
        {
            vec3 stepCell = vec3(cell.xy, cell.z);

            dst = castPlanet(stepCell, start, dir, coverage);
        	if (dst < cornerDst) cornerDst = dst;
        }
        
        if (cornerDst < MAX_DST) // We have hit a planet in a corner intersection
        {
            // Blends the hit planet behind the previous ones according to coverage
            //ColorFTB = BlendFTB(ColorFTB, RenderPlanet(start + cornerDst * dir, cornerDst, dir, worldAngle, coverage));
            //if (ColorFTB.a > 0.99) return ColorFTB;
            
            layers[num++] = vec2(cornerDst, coverage);
            opacity *= (1.0 - coverage);
            if (opacity < 0.01) break;
        }
        
        
       	// rinse / repeat
        cell = newCell;
        d += 1.0;
    }
    
        
    vec4 ColorFTB = vec4(0.0);
    
    for (int i = 0; i < num; i++)
    {
        vec2 layer = layers[i];
        ColorFTB = BlendFTB(ColorFTB, RenderPlanet(start + layer.x * dir, layer.x, dir, worldAngle, layer.y));
    }
    
    return ColorFTB;
}


vec3 render(vec3 camPos, vec3 rayDir, vec2 uv)
{
    vec3 col;
    
    // rotates the galaxy around the Z axis, 
    // this rotation will be compensated for when computing planet color so they stay upright
    float worldAngle = iTime * 0.1;
    rayDir = normalize((rotationZ(worldAngle) * vec4(rayDir, 0.0)).xyz);
   
    float coverage;
    
    // cast a ray in the planet field
    vec4 planetCol = castRay(camPos, rayDir, MAX_DST, worldAngle);
    

    // Compute the central rainbow flower and solar god rays by samplin a 2D noise in polar coordinates
	vec3 dummyCol;
    vec4 fl = flower(uv * 1.5, vec4(0.0, 0.0, 0.0, 0.0), 2.0, 0.5, iTime * 0.1, dummyCol, 2.0);
   	col = fl.rgb;
    
    float a = atan(uv.x, uv.y);
    float cdist = length(uv);
    vec2 raysUvs = vec2(a * 20.0 + iTime * 0.5, cdist * 5.0 - iTime + a * 3.0);
    vec3 rays = mix(vec3(2.0, 2.0, 1.5), vec3(1.0, 0.4, 0.2), cdist + Noise(raysUvs) * 0.3);
 	
    col = mix(rays, col, fl.a);
    
    col = col * (1.0 - planetCol.a) + planetCol.rgb;
  
    return col;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv =(fragCoord - .5 * iResolution.xy) / iResolution.y;
    uv *= 1.2;

    // compute camera ray
    vec3 camPos = vec3(0.5, 0.5, iTime * 0.5);
    vec3 camDir = vec3(0.0, 0.0,  1.0);    
    vec3 rayDir = camDir + vec3(uv * 0.13, 0.0);

	//vec3 nrmDir = normalize(rayDir);
    
    vec3 res = render(camPos, rayDir, uv).rgb;
	
    #ifdef MSAA
    if (iResolution.x < 850.0) // Added AA for the thumbnail
    {
        vec3 offset = vec3(0.05, 0.12, 0.0)  / iResolution.x;
         
        for (int i = 0; i < 4 + min(0,iFrame); i++)
        {
            res += render(camPos, rayDir + offset, uv).rgb;
            offset.xy = vec2(-offset.y, offset.x);
        }
        res /= 5.0;
    }
    #endif

    // Output to screen
    fragColor = vec4(res.rgb,1.0);
}`,
    "Sine March": `void mainImage(out vec4 o, vec2 u) {
    float i,a,d,s,t=.4*iTime;
    vec3  p = iResolution;
    u = (u+u-p.xy)/p.y;
    for(o*=i; i++<64.;
        d += s = .01 + abs(s) * .4,
        o += s*d, o.r+=d/s)
        for (p = vec3(u * d, d + t),
            s = min(cos(p.z), 6. - length(p.xy * sin(p.y*.6))),
            a = .8; a < 16.; a += a)
            p += cos(t+p.yzx)*.1,
            s += abs(dot(sin(t+.2*p.z+p * a), .6+p-p)) / a;
    o = tanh(o / 2e4 * length(u));
}`,
    "Hyperloop": `/* Hyperloop by @kishimisu (2024) - https://www.shadertoy.com/view/4XVGWh
   [442 chars -> 439 chars by Snoopeth]

   Playing with a different kind of space repetition using logarithmic scaling

   This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0
   International License (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)
*/

// If you're getting a black image, try uncommenting this line:
// #define tanh(x) smoothstep(0., 1., x)

#define r *= mat2(cos(vec4(0,33,11,0) +//
#define l length(P

void mainImage(out vec4 O, vec2 F) {

    vec3    H   = iResolution,
            Y   ,
            P   ;
    float   E   = iTime,
            R   ,
            L   = .1, o;
    for (   O   *= o; o++ < 50.;
            O   += .02*(tanh(R*1e3)+2.-2.*tanh(l.xy*7.)))*(.8+cos(log(L*L)+o*.07-E+vec4(0,1,2,0)))/++R)
            P   *= L / l=vec3(F+F-H.xy, H.y)),

            P.z--,
            P.xz r .3)),
            P.zy r 1.)),
            P.yx r round((atan(P.y, P.x) + E*.2) * 1.91) / 1.91 - E*.2)),
            Y.x = pow(.67, floor(E - log(P.x)/.4) - E),
            L += R = min(min(
                       l.xy),
                       l-Y    ) - Y.x*.2  ),
                       l-Y*.67) - Y.x*.134) * .8;
}`,
    "Twisted Icosahedron": `// --------------------------------------------------------
// OPTIONS
// --------------------------------------------------------

// Disable to see more colour variety
#define SEAMLESS_LOOP
#define COLOUR_CYCLE
#define HIGH_QUALITY

// --------------------------------------------------------
// http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
// --------------------------------------------------------

mat3 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat3(
        oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
        oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
        oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c
    );
}


// --------------------------------------------------------
// HG_SDF
// https://www.shadertoy.com/view/Xs3GRB
// --------------------------------------------------------

#define PI 3.14159265359
#define PHI (1.618033988749895)


float t;


float vmax(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

float sgn(float x) {
    return (x<0.)?-1.:1.;
}

// Rotate around a coordinate axis (i.e. in a plane perpendicular to that axis) by angle <a>.
// Read like this: R(p.xz, a) rotates "x towards z".
// This is fast if <a> is a compile-time constant and slower (but still practical) if not.
void pR(inout vec2 p, float a) {
    p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}

// Reflect space at a plane
float pReflect(inout vec3 p, vec3 planeNormal, float offset) {
    float t = dot(p, planeNormal)+offset;
    if (t < 0.) {
        p = p - (2.*t)*planeNormal;
    }
    return sign(t);
}

// Repeat around the origin by a fixed angle.
// For easier use, num of repetitions is use to specify the angle.
float pModPolar(inout vec2 p, float repetitions) {
    float angle = 2.*PI/repetitions;
    float a = atan(p.y, p.x) + angle/2.;
    float r = length(p);
    float c = floor(a/angle);
    a = mod(a,angle) - angle/2.;
    p = vec2(cos(a), sin(a))*r;
    // For an odd number of repetitions, fix cell index of the cell in -x direction
    // (cell index would be e.g. -5 and 5 in the two halves of the cell):
    if (abs(c) >= (repetitions/2.)) c = abs(c);
    return c;
}


// --------------------------------------------------------
// IQ
// https://www.shadertoy.com/view/ll2GD3
// --------------------------------------------------------

vec3 pal( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d ) {
    return a + b*cos( 6.28318*(c*t+d) );
}

vec3 spectrum(float n) {
    return pal( n, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.33,0.67) );
}


// --------------------------------------------------------
// knighty
// https://www.shadertoy.com/view/MsKGzw
// --------------------------------------------------------

int Type=5;
vec3 nc;
vec3 pbc;
vec3 pca;
void initIcosahedron() {//setup folding planes and vertex
    float cospin=cos(PI/float(Type)), scospin=sqrt(0.75-cospin*cospin);
    nc=vec3(-0.5,-cospin,scospin);//3rd folding plane. The two others are xz and yz planes
    pbc=vec3(scospin,0.,0.5);//No normalization in order to have 'barycentric' coordinates work evenly
    pca=vec3(0.,scospin,cospin);
    pbc=normalize(pbc); pca=normalize(pca);//for slightly better DE. In reality it's not necesary to apply normalization :)

}

void pModIcosahedron(inout vec3 p) {
    p = abs(p);
    pReflect(p, nc, 0.);
    p.xy = abs(p.xy);
    pReflect(p, nc, 0.);
    p.xy = abs(p.xy);
    pReflect(p, nc, 0.);
}

float splitPlane(float a, float b, vec3 p, vec3 plane) {
    float split = max(sign(dot(p, plane)), 0.);
    return mix(a, b, split);
}

float icosahedronIndex(inout vec3 p) {
    vec3 sp, plane;
    float x, y, z, idx;

    sp = sign(p);
    x = sp.x * .5 + .5;
    y = sp.y * .5 + .5;
    z = sp.z * .5 + .5;

    plane = vec3(-1. - PHI, -1, PHI);

    idx = x + y * 2. + z * 4.;
    idx = splitPlane(idx, 8. + y + z * 2., p, plane * sp);
    idx = splitPlane(idx, 12. + x + y * 2., p, plane.yzx * sp);
    idx = splitPlane(idx, 16. + z + x * 2., p, plane.zxy * sp);

    return idx;
}

vec3 icosahedronVertex(vec3 p) {
    vec3 sp, v, v1, v2, v3, result, plane;
    float split;
    v = vec3(PHI, 1, 0);
    sp = sign(p);
    v1 = v.xyz * sp;
    v2 = v.yzx * sp;
    v3 = v.zxy * sp;

    plane = vec3(1, PHI, -PHI - 1.);

    split = max(sign(dot(p, plane.xyz * sp)), 0.);
    result = mix(v2, v1, split);
    plane = mix(plane.yzx * -sp, plane.zxy * sp, split);
    split = max(sign(dot(p, plane)), 0.);
    result = mix(result, v3, split);

    return normalize(result);
}

// Nearest vertex and distance.
// Distance is roughly to the boundry between the nearest and next
// nearest icosahedron vertices, ensuring there is always a smooth
// join at the edges, and normalised from 0 to 1
vec4 icosahedronAxisDistance(vec3 p) {
    vec3 iv = icosahedronVertex(p);
    vec3 originalIv = iv;

    vec3 pn = normalize(p);
    pModIcosahedron(pn);
    pModIcosahedron(iv);

    float boundryDist = dot(pn, vec3(1, 0, 0));
    float boundryMax = dot(iv, vec3(1, 0, 0));
    boundryDist /= boundryMax;

    float roundDist = length(iv - pn);
    float roundMax = length(iv - vec3(0, 0, 1.));
    roundDist /= roundMax;
    roundDist = -roundDist + 1.;

    float blend = 1. - boundryDist;
    blend = pow(blend, 6.);

    float dist = mix(roundDist, boundryDist, blend);

    return vec4(originalIv, dist);
}

// Twists p around the nearest icosahedron vertex
void pTwistIcosahedron(inout vec3 p, float amount) {
    vec4 a = icosahedronAxisDistance(p);
    vec3 axis = a.xyz;
    float dist = a.a;
    mat3 m = rotationMatrix(axis, dist * amount);
    p *= m;
}


// --------------------------------------------------------
// MAIN
// --------------------------------------------------------

struct Model {
    float dist;
    vec3 colour;
    float id;
};


Model fInflatedIcosahedron(vec3 p) {
    float d = 1000.;

    // Slightly inflated icosahedron
    float idx = icosahedronIndex(p);
    d = min(d, dot(p, pca) - .9);
    d = mix(d, length(p) - .9, 1.0);

    // Colour each icosahedron face differently
    #ifdef SEAMLESS_LOOP
        if (idx == 3.) {
            idx = 2.;
        }
        idx /= 10.;
    #else
        idx /= 20.;
    #endif
    #ifdef COLOUR_CYCLE
        idx = mod(idx + t*1., 1.);
    #endif
    vec3 colour = spectrum(idx);

    d *= .6;
    return Model(d, colour, 1.);
}

Model model(vec3 p) {
    float rate = PI/6.;

    float a = atan(1., PHI + 1.);
    pR(p.yz, a);

    pR(p.yx, t * 2.1 + rate);
    pR(p.yz, a);

    vec3 twistCenter = vec3(.7, 0, 0);
    pR(twistCenter.yx, t * 2.1 + rate);
    pR(twistCenter.yz, a);

    p += twistCenter;
    pTwistIcosahedron(p, 10.5);
    p -= twistCenter;

    #ifdef SEAMLESS_LOOP
        pR(p.yz, -a);
        pR(p.xy, -PI/2.);
        pModPolar(p.xy, 3.);
        pR(p.xy, -PI/2.);
        pR(p.yz, -a);
    #endif

    return fInflatedIcosahedron(p);
}


Model map(vec3 p) {
    return model(p);
}


mat3 calcLookAtMatrix(in vec3 ro, in vec3 ta, in float roll) {
    vec3 ww = normalize( ta - ro );
    vec3 uu = normalize( cross(ww,vec3(sin(roll),cos(roll),0.0) ) );
    vec3 vv = normalize( cross(uu,ww));
    return mat3( uu, vv, ww );
}

const float MAX_TRACE_DISTANCE = 6.0;
const float INTERSECTION_PRECISION = 0.001;
#ifdef HIGH_QUALITY
    const float FUDGE_FACTOR = .2;
#else
    const float FUDGE_FACTOR = .4;
#endif

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    initIcosahedron();
    t = iTime - .25;
    t /= 2.;

    vec2 p = (-iResolution.xy + 2.0*fragCoord.xy)/iResolution.y;

    vec3 camPos = vec3(-1.5,1.6,0);
    vec3 camTar = -camPos + vec3(0,.1,0);
    float camRoll = 0.;

    // camera matrix
    mat3 camMat = calcLookAtMatrix( camPos, camTar, camRoll );  // 0.0 is the camera roll

    // create view ray
    vec3 rd = normalize( camMat * vec3(p.xy,1.0) ); // 2.0 is the lens length

    vec3 color = pow(vec3(.15,0,.2), vec3(2.2));

    vec3 ro = camPos;
    float traceT = 0.0;
    float h = INTERSECTION_PRECISION * 2.0;
    vec3 colour;

    int iter = int(20. / FUDGE_FACTOR);

    for( int i=0; i < iter; i++ ){

        if( traceT > MAX_TRACE_DISTANCE ) break;
        Model m = map( ro+rd*traceT );
        h = abs(m.dist);
        traceT += max(INTERSECTION_PRECISION, h * FUDGE_FACTOR);
        color += m.colour * pow(max(0., (.02 - h)) * 19.5, 10.) * 150.;
        color += m.colour * .001 * FUDGE_FACTOR;
    }

    color = pow(color, vec3(1./1.8)) * 1.5;
    color = pow(color, vec3(1.5));
    color *= 3.5;

    fragColor = vec4(color,1.0);
}`,
    "Elevator to Infinity": `/* Elevator to infinity by @kishimisu (2023)  -  https://www.shadertoy.com/view/mddfW8
   This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)
   *****************************************

     Move the camera with the mouse!

   Alternative audio versions:

   I couldn't decide which audio and camera movement was the best for this scene,
   so I preferred to keep this shader simple and fork the other ideas I liked:

   - Audio-reactive lights:                    https://www.shadertoy.com/view/DddBWM
   - Longer camera anim + dark ambient music:  https://www.shadertoy.com/view/csdBD7
   - Speed increase synced with music buildup: https://www.shadertoy.com/view/dddBWM


   This is my first successful attempt at raymarching infinite buildings.
   In my previous attempts, I was adding details using domain repetition for
   nearly all raymarching operators and it was too hard to maintain.

   In this version, I started with a simpler task, which is to generate only one
   floor using regular raymarching, and then use domain repetition at the very
   beginning to repeat the floor infinitely, thus creating infinite buildings.

   Do you have tips to reduce flickering in the distance ?
*/

// Comment out to disable all lights except elevators
#define LIGHTS_ON

float acc = 0.; // Neon light accumulation
float occ = 1.; // Ambient occlusion (Fake)

// 2D rotation
#define rot(a) mat2(cos(a), -sin(a), sin(a), cos(a))

// Domain rep.
#define rep(p, r) mod(p+r, r+r)-r

// Domain rep. ID
#define rid(p, r) floor((p+r)/(r+r))

// Finite domain rep.
#define lrep(p, r, l) p-r*clamp(round(p/r), -l, l)

// Fast random noise 2 -> 3
vec3 hash(vec2 p) {
    vec2 r = fract(sin(p*mat2(137.1, 12.7, 74.7, 269.5)) * 43478.5453);
    return vec3(r, fract(r.x*r.y*1121.67));
}
// Random noise 3 -> 3 - https://shadertoyunofficial.wordpress.com/2019/01/02/
#define hash33(p) fract(sin(p*mat3(127.1,311.7,74.7,269.5,183.3,246.1,113.5,271.9,124.6))*43758.5453123)

// Distance functions - https://iquilezles.org/articles/distfunctions/
float box(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.)) + min(max(q.x, max(q.y, q.z)), 0.);
}
float rect(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.)) + min(max(d.x, d.y), 0.);
}

#define ext 2.
float opElevatorWindows(vec3 p, float b) {
    float e  = box(p, vec3(ext*.8, 2.7, .3));
    float lv = length(p.xz) - .1;   p.y += 1.;
    float lh = length(p.yz) - .1;
    lh = max(b, lh);
    b  = max(b, -e);
    b  = min(b, min(lv, lh));
    return b;
}

float building(vec3 p0, vec3 p, float L) {
    float B = rect(p.xz, vec2(L, 10)); // Main building
    float B2 = rect(vec2(abs(p.x)-L-ext, p.z), vec2(ext, 10)); // Elevator building

    // (Optim) Skip building calculations
    if (min(B, B2) > .2) return min(B, B2);

    vec3 q = p;
    float var = step(1., mod(rid(p.y, 3.), 6.)); // Railing variation
    p.y = rep(p.y, 3.); // Infinite floor y-repetition
    vec3 pb = vec3(abs(p.x), p.yz);

#ifdef LIGHTS_ON
    // Building lights
    vec3  id = rid(vec3(q.xy, p0.z), vec3(21, 18, 48));
    vec3  rn = hash33(id);
    float rw = fract(rn.x*rn.z*1021.67);

    q.x += 14. * (rn.x*3.-1.);
    q.y += 12. * (floor(rn.y*3.)-1.);
    q.xy = rep(q.xy, vec2(21, 18));

    float l = box(q, vec3(mix(3., 15., rw), rn.z*1.5+.5, 7));
    acc += .5 / (1. + pow(abs(l)*20., 1.5))
                * smoothstep(0., .4, iTime - rw * 20.)
                * step(p0.x, 10. + 2e2*step(20., abs(p0.z)));
#endif

    // Occlusion
    occ = min(occ, smoothstep(3.5, 0., -rect(p.xz, vec2(L+2.,10))));
    occ = min(occ, smoothstep(0.6, 0., -rect(pb.xz-vec2(L+ext,0), vec2(ext,10))));

    // Front hole
    q = p;
    q.x = rep(q.x, 7.);
    q.y -= (1. - var)*1.01;

    float f = box(q + vec3(0,0,10), vec3(6.6, 2. + var, 3));
    B = max(B, -f);
    B = max(B, -rect(q.xz + vec2(0,10), vec2(6.6, .7)*var));

    // Railing
    q = p;
    q.x = rep(q.x, .8);

    float r  = length(p.yz + vec2(1, 9.5-var*.5)) - .2;
    float rv = length(q.xz + vec2(0, 9.5-var*.5)) - .16;
    r = min(r, rv);
    r = max(r, p.y + 1.);

    // Back bars
    q = p;
    q.x = rep(q.x, 1.75);

    float b = length(q.xz + vec2(0, 7.3)) - .2;
    r = min(r, b);

    B = min(B, r);
    B = max(B, abs(p.x) - L);

    // (Optim) Skip elevator calculations
    if (B2 > .04) return min(B, B2);

    // Elevator
    B2 = opElevatorWindows(pb - vec3(L+ext,0,-9.9), B2);
    B2 = opElevatorWindows(vec3(pb.z+8., pb.y, pb.x-L-ext-1.9), B2);

    // Side windows
    q = vec3(pb.xy, pb.z - 1.8);
    q.z = lrep(q.z, 2.5, 2.);

    float w = box(q - vec3(L+ext*2.,1.2,0), vec3(.5, 1.6, 1.2));
    B2 = max(B2, -w);

    return min(B, B2);
}

float map(vec3 p) {
    vec2 id = vec2(step(40., p.x), rid(p.z, 140.));
    vec3 rn = mix(vec3(1, -.5, 0), hash(id), step(.5, id.x+id.y));

    // Buildings
    vec3 p0 = p;
    p.x = abs(abs(p.x - 40.) - 80.);
    p.z = rep(p.z - id.x*200., 200.);

    float bL = 21.4 + id.y*3.;
    float b1 = building(p0, p - vec3(30,0,0), bL);
    float b2 = building(p0, vec3(p.z,p.y,-p.x), 185.);

    // Elevator lights
    float rpy = 80. + 150. * rn.x;;
    p.y = rep(p.y - iTime * 40. * (rn.y*.5+.5), rpy);
    p -= vec3(30.+bL+ext, rn.z*rpy*.5, ext-10.);

    float l = box(p, vec3(ext*.8, 2.7, ext*.8));
    acc += .5 / (1. + pow(abs(l)*18., 1.17));

    // Fix broken distance before 20s
    b2 = min(b2, abs(p0.x + p0.z - 30.) + 6.);

    return min(b1, b2);
}

// https://iquilezles.org/articles/normalsSDF/
vec3 normal(vec3 p) {
    const vec2 k = vec2(1,-1)*.0001;
    return normalize(k.xyy*map(p + k.xyy) + k.yyx*map(p + k.yyx) +
                     k.yxy*map(p + k.yxy) + k.xxx*map(p + k.xxx));
}

void mainImage(out vec4 O, vec2 F) {
    vec2  R = iResolution.xy,
          u = (F+F-R)/R.y,
          M = iMouse.xy/R * 2. - 1.;
          M *= step(1., iMouse.z);

    // Camera animation
    float T  = 1. - pow(1. - clamp(iTime*.025, 0., 1.), 3.);
    float ax = mix(-.8, .36, T);
    float az = mix(-40., -140., T);
    float rx = M.x*.45 - (cos(iTime*.1)*.5+.5)*.4;
    rx = clamp(ax + rx - .55, min(iTime*.05 - 1.6, -.9), .1);

    // Ray origin & direction
    vec3 ro = vec3(0, iTime*10., az);
    vec3 rd = normalize(vec3(u, 3));

    rd.zy *= rot(M.y*1.3);
    rd.zx *= rot(rx);
    ro.zx *= rot(rx);

    // Raymarching
    vec3 p; float d, t = 0.;
    for (int i = 0; i < 60; i++) {
        p = ro + t * rd;
        t += d = map(p);
        if (d < .01 || t > 2200.) break;
    }

    // Base color
    vec3 col = vec3(.13,.11,.26) - vec3(1,1,0)*abs(p.x-40.)*.001;
    col *= clamp(1. + dot(normal(p), normalize(vec3(0,0,1))), .5, 1.);

    // Texture
    col *= 1. - texture(iChannel0, vec2(p.x+p.z, p.y+p.z)*.05).rgb*.7;

    // Occlusion
    col *= occ;

    // Exponential fog
    col = mix(vec3(.002,.005,.015), col, exp(-t*.0025*vec3(.8,1,1.2) - length(u)*.5));

    // Light accumulation
    col += acc * mix(vec3(1,.97,.76), vec3(1,.57,.36), t*.0006);

    // Color correction
    col = pow(col, .46*vec3(.98,.96,1));

    // Vignette
    u = F/R; u *= 1. - u.yx;
    col *= pow(clamp(u.x * u.y * 80., 0., 1.), .2);

    O = vec4(col, 1);
}`,
    "Zozuar Flower": `// Zozuar Flower, mla, 2023. Original by @zozuar/@yonatan.
// Degolfed version of https://twitter.com/zozuar/status/1612919479582728232
/*
for(float i,g,e,R,S;i++<1e2;o.rgb+=hsv(.4-.02/R,
(e=max(e*R*1e4,.7)),.03/exp(e))){S=1.;vec3 p=vec3
((FC.xy/r-.5)*g,g-.3)-i/2e5;p.yz*=rotate2D(.3);
for(p=vec3(log(R=length(p))-t,e=asin(-p.z/R)-.1/
R,atan(p.x,p.y)*3.);S<1e2;S+=S)e+=pow(abs(dot(sin
(p.yxz*S),cos(p*S))),.2)/S;g+=e*R*.1;}
*/

const float PI = 3.14159265;

vec3 hsv(float h, float s, float v) {
    vec3 rgb = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    rgb = rgb * rgb * (3.0 - 2.0 * rgb);
    return v * mix(vec3(1.0), rgb, s);
}

mat2 rotate2D(float t) {
    return mat2(cos(t), sin(t), -sin(t), cos(t));
}

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    float time = iTime;
    fragColor = vec4(0);
    vec2 uv = 0.5 * (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0, 0, -0.6);
    vec3 rd = vec3(uv, 1);
    float t = 0.0;

    for (float i = 0.0; i < 1e2; i++) {
        vec3 p = ro + t * rd - i / 2e5;
        // Top-down default: original only applies mouse rotation when iMouse.x > 0
        p.yz *= rotate2D(0.2);

        float r = length(p);
        float e = asin(-p.z / r) - 0.1 / r;
        float rot = 3.0;
        vec3 q = vec3(log(r) - time, e, rot * atan(p.x, p.y));
        for (float scale = 1.0; scale < 1e2; scale += scale) {
            e += pow(abs(dot(sin(q.yxz * scale), cos(q * scale))), 0.2) / scale;
        }
        t += e * r * 0.1;
        if (t > 50.0) break;
        float k = max(e * r * 1e4, 0.7);
        k = pow(k, 0.4);
        fragColor.rgb += hsv(0.4 - 0.02 / r, k, 0.02 / exp(k));
    }

    fragColor *= 2.0 / (1.0 + fragColor);
    fragColor = pow(fragColor, vec4(0.4545));
    fragColor.a = 1.0;
}`,
    "3D Fire Star": `// Inspired by @XorDev's "3D Fire".

const float PI = radians(180.);

float sdStar( in vec2 p, in float r, in int n, in float m) {
    float an = PI/float(n);
    float en = PI/m;  // m is between 2 and n
    vec2  acs = vec2(cos(an),sin(an));
    vec2  ecs = vec2(cos(en),sin(en)); // ecs=vec2(0,1) for regular polygon

    float bn = mod(atan(p.x,p.y),2.0*an) - an;
    p = length(p)*vec2(cos(bn),abs(sin(bn)));
    p -= r*acs;
    p += ecs*clamp( -dot(p,ecs), 0.0, r*acs.y/ecs.y);
    return length(p); // *sign(p.x);
}

float sdStar3D(vec3 p) {
    float d = sdStar(p.xy, 6.0, 5, 0.6);
    return abs(d + p.z*0.5 - 0.5);
}

mat2 rot2(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, s, -s, c);
}

void mainImage(out vec4 o, vec2 sp) {
    float z = 0.; // Raymarched depth
    vec3 vp = normalize(vec3(sp*2.,0.)-iResolution.xyy);
    o = vec4(0);

    for (int i = 0; i < 50; i++) {
        vec3 p = z * vp; // Raymarch sample point
        p.z += 5.5 - 12.5*sin(iTime*0.3);

        //p.xy *= rot2(-0.1*iTime);
        p.yz *= rot2(PI/2.0 + 0.0*iTime);
        p.xz /= max(p.y * (0.2 - 0.1*cos(iTime*0.3)) + 1.0, 0.3); // Expand upward
        p.xz *= rot2(-p.y*0.05 + 0.2*iTime);
        
        for (float s = 0.8; s < 15.; s /= 0.6)
            p += 0.5*cos(s*(p.yzx - vec3(iTime/.1, iTime, s)))/s;
        
        float d = 0.01 + sdStar3D(p.zxy)/7.; // SDF
        o += 1.0/d * (sin(sin(iTime*0.02)*5.0 + p.y/5.5 + length(p.xz)/4.5 + vec4(9,2,1,0)) + 1.1); // RGB
        z += d; // raymarching step
    }

    o = tanh(o / 1e3); // Tone mapping
    o.a = 1.0;
}`,
    "Apollonian": `// apollonian
float fractal(vec3 p) {
    // weight
    float w = 2.;

    // 6 - 8 iterations is usually the sweet spot
    for (float l, i; i++ < 4.; p *= l, w *= l )
        // sin(p), abs(sin(p))-1., also work,
        // but need to adjust weight(w) and scale(l=2.)
        p  = cos(p-.5),
        // low scale for this fractal type, so we just get snowflake-like shape
        // adjust 2. for scaling
        l = 2.1/dot(p,p);
    return length(p)/w;
}

void mainImage(out vec4 o, vec2 u) {
    float i, // iterator

          // init total dist to a good random value (blue noise)
          // to hide some of the noise flickering
          d = .2*texelFetch(iChannel0, ivec2(u)%1024, 0).a,
          s, // signed distance
          n, // noise iterator
          t = iTime;
    // p is temporarily resolution,
    // then raymarch position
    vec3 p = iResolution;

    // scale coords
    u = (u-p.xy/2.)/p.y;
    u += vec2(cos(t*.3)*.2, sin(t*.2)*.15);

    // clear o, up to 100, accumulate distance, grayscale color
    for(o*=i; i++<1e2;d += s = .001+abs(min(fractal(p), s))*.7,

        // can try below for color
         o += (1.+1.+cos(.3*p.z+vec4(6,1,3,0)))/s)
        //o += 1./s)

        // march, equivalent to p = ro + rd * d, p.z += d+t+t
        for (p = vec3(u * d, d+t),
             // spin by t, twist by p.z, equivalent to p.xy *= rot(.05*t+p.z*.2)
             p.xy *= mat2(cos(.02*t+p.z*.2+vec4(0,33,11,0))),
             // dist to our spiral'ish thing that will be distorted by noise
             s = sin(2.+p.y+p.x),
             // start noise at 6., until 32, grow by n*=1.41
             n = 6.; n < 32.; n *= 1.41 )
                 // subtract noise from s
                 s += abs(dot(cos(p*n), vec3(.3))) / n;
    // tanh tone mapping, divide down brightness
    o = tanh(o*o/6e8);
}`,
    "Fabrice Inversion": `// @FabriceNeyret2 → 490 chars ( from 511 )
// ty :D

#define N(a) a* abs(dot( sin( .1*p.z + iTime*3. + .3*( p+cos(p.yzx) ) /a ) , vec3(2)) )

void mainImage(out vec4 o, vec2 u) {
    float i = 0., s = 0., d = 0.;
    
    vec3 p = vec3(iResolution.xy, 0.0),
         D = normalize(vec3(u = ( u+u - p.xy ) / p.y, 1.2));
    
    for( o*=i; i++ < 1e2 && d < 1e2; )
        p = D*d,
        p.z -= 6e1,
        p.xz *= mat2(cos(iTime/4.+ vec4(0,33,11,0))),
        p *= 1e3/max(dot(p,p), 2e2), // <-- transform
        d += s = .15 + .3* abs( tanh(sin(iTime/4.)*16.)/.1 +2e1 -abs(p.y) 
                               - N(.1) - N(.2) - N(.6) ),
        o += ( vec4(5,2,1,0)/s*2e1 +1e2/length(u) ) / d;
       
    o *= mix( vec4(.3,2,6,0), vec4(3,1.3,.3,0),
              smoothstep(.57, -.7, u.y));
             
    o = sqrt(1.-exp(-o*o/1e6));
    o.a = 1.0;
}`,
    "Apollonian Path": `#define T (iTime*1.5 + 5. + 1.5*sin(iTime*.5))
#define P(z) (vec3(cos((z) * .07) * 16., \
                   0, (z)))
#define R(a) mat2(cos(a+vec4(0,33,11,0)))
#define N normalize

#define HUE_BASS         0.50
#define HUE_MID          0.40
#define PALE_CRUSH       0.42
#define CHROMA_BOOST     0.65
#define PUNCH_LO         0.03
#define PUNCH_HI         0.14

float fft(float x) {
    return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x;
}

float getBass() { return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25; }
float getMid()  { return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.25; }
float getHigh() { return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.25; }

float audioPresence() {
    float s = fft(0.005) + fft(0.05) + fft(0.15) + fft(0.30) + fft(0.60);
    return smoothstep(0.001, 0.008, s);
}

float fallbackBass() { return 0.35 + 0.25 * sin(iTime * 1.1); }
float fallbackMid()  { return 0.28 + 0.18 * sin(iTime * 0.65 + 1.0); }
float fallbackHigh() { return 0.18 + 0.12 * sin(iTime * 2.6 + 2.0); }

float audioBoost(float v) {
    return clamp(pow(v, 0.55) * 2.0, 0.0, 1.0);
}

float apollonian(vec3 p) {
    float i, s, w = 1.;
    vec3 b = vec3(.5, 1., 1.5);
    p.y -= 2.;
    p.yz = p.zy;
    p /= 16.;
    for(; i++ < 6.;) {
        p = mod(p + b, 2. * b) - b;
        s =2. / dot(p, p);
        p *= s;
        w *= s;
    }
    return length(p) / w * 6.;
}

float map(vec3 p) {
    return max(2. - length((p - P(p.z)).xy), apollonian(p));
}

void mainImage(out vec4 o, in vec2 u) {
    float s, d, i, a;
    vec3 r = iResolution;
    u = (u - r.xy / 2.) / r.y;

    float presence = audioPresence();
    float micBass = getBass();
    float micMid  = getMid();
    float micHigh = getHigh();
    float bass = audioBoost(mix(fallbackBass(), micBass, presence));
    float mid  = audioBoost(mix(fallbackMid(),  micMid,  presence));
    float high = audioBoost(mix(fallbackHigh(), micHigh, presence));
    float punch = smoothstep(PUNCH_LO, PUNCH_HI, mix(fallbackBass(), micBass, presence));

    vec4 marchHue = vec4(6., 4., 2., 0.)
        + vec4(-1.8, -0.5, 1.4, 0.) * bass * HUE_BASS
        + vec4(0.6, -1.2, -1.6, 0.) * mid  * HUE_MID
        + vec4(-0.3, 0.5, -0.4, 0.) * high * 0.25;

    vec3 p = P(T * 2.),
         Z = N(P(T * 2. + 7.) - p),
         X = N(vec3(Z.z, 0., -Z)),
         D = N(vec3(R(sin(iTime * .15) * .3) * u, 1.) * mat3(-X, cross(X, Z), Z));

    for (o = vec4(0.); i++ < 128.;)
        p += D * s,
        d += s = map(p) * .8,
        o += (1. + cos(.05 * i + .5 * p.z + marchHue)) / max(s, .0003);

    o = tanh(o / d / 7e4 * exp(vec4(3., 2., 1., 0.) * d / 4e1));

    vec3 col = o.rgb;
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    float pale = smoothstep(0.48, 0.82, lum);
    float detail = smoothstep(0.08, 0.35, lum) * (1.0 - smoothstep(0.72, 0.95, lum));
    float energy = bass + punch * 0.85 + mid * 0.45;

    col = mix(col, col * (1.0 - punch * PALE_CRUSH), pale);
    col = mix(vec3(lum), col, 1.0 + detail * energy * CHROMA_BOOST);
    col = mix(col, col * vec3(0.88, 1.10, 1.14), mid * detail * 0.35);
    col = mix(col, col * vec3(1.08, 0.90, 0.82), high * detail * 0.22);
    o = vec4(clamp(col, 0.0, 1.0), 1.0);
}`,
"PIANOSCOPE Adrinka 2":`
/*
PIANOSCOPE Shader
Name: Adinkra Symbol Field Abstract — v3
Mode: adinkra_symbol_field_abstract
Inspired by: Mandala (polar fold + iterative mod2 scale) + Adinkra stamp geometry
Cultural caution: Procedural circles, arcs, crosses — not exact sacred Adinkra symbols

CHANGES v3 (reactivity restoration):
v2 fixed the jitter by pulling audio out of geometry, but the intensity
gains replacing it were too timid, and audioBoost's pow(0.75) was
COMPRESSING dynamics — lifting quiet signal toward loud, so hits and
breakdowns looked similar. v3 fixes the dynamics, not the geometry:

- PUNCH signal: an expander (smoothstep gate on raw bass) that sits near
  zero between hits and slams to 1 on kick/log-drum transients. This is
  the opposite curve of audioBoost and is what makes hits READ.
- Punch drives: global exposure flash (pre-shoulder, so it can't clip
  flat), reveal radius expansion, and mark swell (gain doubled).
- High shimmer gain raised 0.6 -> 1.0, plus highs now brighten fill.
- Mid palette swing raised.
- Optional BASS_ZOOM camera pump (radial, pre-fold, ~3.5%): the one
  audio->geometry path, kept tiny and gated by punch so it reads as a
  camera kick, not lattice popping. Set to 0.0 to disable if it
  flutters on your projector.

All gains exposed as defines at the top — tune at the venue.

Audio: Bass punch = flash + reveal + swell + zoom kick; Mid = warmth/smoke;
       High = edge shimmer + fill sparkle
Projection: Dark violet field, gold/rust marks, breathes with the mix
*/

#define PI  3.141592654
#define TAU (2.0 * PI)

// ---- reactivity tuning ----
#define BASS_FILL_GAIN   0.10   // mark swell on bass level
#define PUNCH_SWELL      0.06   // extra swell on transients
#define BASS_FLASH       0.55   // global exposure pump on transients
#define BASS_ZOOM        0.035  // camera kick on transients; 0.0 disables
#define REVEAL_PUMP      0.30   // central reveal radius expansion on hits
#define MID_HUE_SWING    0.22   // palette warmth travel with mids
#define HIGH_SHIMMER     1.00   // edge glow on highs
#define PUNCH_LO         0.22   // expander threshold (raw bass)
#define PUNCH_HI         0.70   // expander full-on point
// ---------------------------

#define SECTOR_COUNT     12.0

vec2 centeredUV(vec2 fragCoord) {
    return (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
}

// ---------- audio ----------

float fft(float x) {
    return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x;
}

float bandAvg(float lo, float hi) {
    float s = 0.0;
    for (int i = 0; i < 8; i++) {
        s += fft(mix(lo, hi, (float(i) + 0.5) / 8.0));
    }
    return s * 0.125;
}

float getBass() { return bandAvg(0.001, 0.010); }
float getMid()  { return bandAvg(0.020, 0.090); }
float getHigh() { return bandAvg(0.200, 0.450); }

float audioPresence() {
    float s = fft(0.005) + fft(0.05) + fft(0.15) + fft(0.30) + fft(0.60);
    return smoothstep(0.006, 0.025, s);
}

float fallbackBass() { return 0.35 + 0.25 * sin(iTime * 1.1); }
float fallbackMid()  { return 0.28 + 0.18 * sin(iTime * 0.65 + 1.0); }
float fallbackHigh() { return 0.18 + 0.12 * sin(iTime * 2.6 + 2.0); }

// LEVEL: compressed, smooth — for ambient modulation (palette, smoke)
float audioBoost(float v) {
    return clamp(pow(v, 0.75) * 1.35, 0.0, 1.0);
}

// PUNCH: expanded, gate-like — near zero between hits, 1.0 on transients.
// This is the signal that makes the kick visible.
float punchShape(float raw) {
    return smoothstep(PUNCH_LO, PUNCH_HI, raw);
}

// ---------- helpers ----------

void rot(inout vec2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    p = vec2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

vec2 mod2(inout vec2 p, vec2 size) {
    vec2 c = floor((p + size * 0.5) / size);
    p = mod(p + size * 0.5, size) - size * 0.5;
    return c;
}

vec2 modMirror2(inout vec2 p, vec2 size) {
    vec2 halfsize = size * 0.5;
    vec2 c = floor((p + halfsize) / size);
    p = mod(p + halfsize, size) - halfsize;
    p *= mod(c, vec2(2.0)) * 2.0 - vec2(1.0);
    return c;
}

vec2 toSmith(vec2 p) {
    float d = (1.0 - p.x) * (1.0 - p.x) + p.y * p.y;
    float x = (1.0 + p.x) * (1.0 - p.x) - p.y * p.y;
    float y = 2.0 * p.y;
    return vec2(x, y) / d;
}

vec2 fromSmith(vec2 p) {
    float d = (p.x + 1.0) * (p.x + 1.0) + p.y * p.y;
    float x = (p.x + 1.0) * (p.x - 1.0) + p.y * p.y;
    float y = 2.0 * p.y;
    return vec2(x, y) / d;
}

vec2 toRect(vec2 p)  { return vec2(p.x * cos(p.y), p.x * sin(p.y)); }
vec2 toPolar(vec2 p) { return vec2(length(p), atan(p.y, p.x)); }

float circle(vec2 p, float r) { return length(p) - r; }

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * noise(p);
        p = p * 2.03 + vec2(1.7, 9.2);
        a *= 0.5;
    }
    return v;
}

// ---------- palette ----------

vec3 adinkraPalette(float t, float layer) {
    vec3 midnightViolet = vec3(0.13, 0.06, 0.25);
    vec3 deepBlack      = vec3(0.015, 0.010, 0.018);
    vec3 rustOrange     = vec3(0.70, 0.27, 0.10);
    vec3 mutedGold      = vec3(0.83, 0.57, 0.18);
    vec3 cream          = vec3(0.92, 0.80, 0.56);

    vec3 c = mix(deepBlack, midnightViolet, smoothstep(0.0, 0.35, t));
    c = mix(c, rustOrange, smoothstep(0.25, 0.65, t + layer * 0.12));
    c = mix(c, mutedGold,  smoothstep(0.45, 0.85, t + layer * 0.08));
    c = mix(c, cream,      smoothstep(0.75, 1.00, t) * 0.35);
    return c;
}

// ---------- geometry ----------

float abstractMark(vec2 p, float phase) {
    float d = 10000.0;

    float ring = abs(circle(p, 0.36)) - 0.035;
    d = min(d, ring);

    float hub = circle(p, 0.11);
    d = min(d, hub);

    for (int i = 0; i < 3; i++) {
        float ang = float(i) * TAU / 3.0 + phase;
        vec2 c = vec2(cos(ang), sin(ang)) * 0.24;
        float arc = abs(length(p - c) - 0.075) - 0.022;
        d = min(d, arc);
    }

    float cross = max(abs(p.x), abs(p.y)) - 0.018;
    cross = max(cross, circle(p, 0.16));
    d = min(d, cross);

    float diamond = abs(p.x) + abs(p.y) - 0.30;
    diamond = max(diamond, -circle(p, 0.28));
    d = min(d, abs(diamond) - 0.015);

    return d;
}

// recursion still time-only; bass/punch inflate the field (continuous)
float adinkra_df(float localTime, vec2 p, float bass, float punch) {
    vec2 pp = toPolar(p);
    float a = TAU / SECTOR_COUNT;
    float np = pp.y / a;
    pp.y = mod(pp.y, a);
    if (mod(np, 2.0) > 1.0) {
        pp.y = a - pp.y;
    }
    pp.y += localTime * 0.025;
    p = toRect(pp);
    p = abs(p);
    p -= vec2(0.48);

    float d = 10000.0;
    float scalePulse = 0.5 + 0.5 * sin(localTime * 0.35);
    float swell = bass * BASS_FILL_GAIN + punch * PUNCH_SWELL;

    for (int i = 0; i < 4; i++) {
        float fi = float(i);
        mod2(p, vec2(1.0));
        float wobble = -0.12 * cos(localTime * 0.25 + fi);
        float mark = abstractMark(p, localTime * 0.15 + fi * 0.7) + wobble - swell;
        d = min(d, mark);

        float grow = 1.42 + 0.06 * scalePulse;
        p *= grow;
        rot(p, 0.55 + fi * 0.12);
    }

    return d;
}

vec2 fieldDistort(float localTime, vec2 uv) {
    float lt = 0.08 * localTime;
    vec2 suv = toSmith(uv);
    suv += 0.65 * vec2(cos(lt), sin(sqrt(2.0) * lt));
    uv = fromSmith(suv);
    modMirror2(uv, vec2(1.8 + 0.25 * sin(lt * 0.7)));
    return uv;
}

// ---------- shading ----------

vec3 shadeField(float d, float layerMix, float mid, float high, float reveal) {
    float fill = smoothstep(0.018, -0.012, d);
    float edge = smoothstep(0.006, 0.0, abs(d));
    float band = 0.5 + 0.5 * sin(d * 80.0);
    vec3 base = adinkraPalette(band * 0.5 + layerMix * 0.35 + mid * MID_HUE_SWING, layerMix);
    vec3 col = mix(vec3(0.015, 0.010, 0.018), base,
                   fill * (0.50 + reveal * 0.45 + high * 0.10));
    col += vec3(0.92, 0.80, 0.56) * edge * (0.22 + high * HIGH_SHIMMER);
    col += vec3(0.83, 0.57, 0.18) * edge * edge * high * 0.50;
    return col;
}

vec3 adinkra_post(vec3 col, float localTime, float r, float punch) {
    col = max(col, 0.0);

    col = pow(col, vec3(0.62, 0.75, 1.05));

    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = max(mix(vec3(luma), col, 1.20), 0.0);

    float pulse = sqrt(max(1.0 - 0.65 * sin(localTime * 0.4 + r * 6.0), 0.0));
    col *= mix(0.85, 1.0, pulse);
    col *= 0.55 * sqrt(max(1.05 - r * r, 0.0));

    // exposure flash on transients, BEFORE the shoulder:
    // the shoulder soaks the peak, so the flash brightens without clipping flat
    col *= 1.0 + punch * BASS_FLASH;

    col = 0.85 * col / (col + 0.25);

    return clamp(col, 0.0, 1.0);
}

vec3 sampleField(float localTime, vec2 p, float bass, float mid, float high, float punch) {
    // camera kick: tiny radial zoom on transients. The only audio->geometry
    // path; reads as a camera pump. Set BASS_ZOOM 0.0 to disable.
    p /= 1.0 + punch * BASS_ZOOM;

    vec2 uv = p * 6.5;
    rot(uv, localTime * 0.04);

    vec2 nuv = fieldDistort(localTime, uv);

    vec2 ndx = dFdx(nuv), ndy = dFdy(nuv);
    vec2 udx = dFdx(uv),  udy = dFdy(uv);
    float stretch = (length(ndx) + length(ndy)) /
                    max(length(udx) + length(udy), 1e-6);
    float warpGlow = 1.0 - smoothstep(0.0, 1.2, stretch);

    float d = adinkra_df(localTime, nuv, bass, punch);
    float r = length(p);

    // reveal radius expands outward on hits — the bloom travels
    float reveal = smoothstep(0.50 + punch * REVEAL_PUMP, 0.06, r)
                 * (0.35 + bass * 0.40 + punch * 0.40);
    vec3 col = shadeField(d, r * 0.8 + bass * 0.2, mid, high, reveal);

    float smoke = fbm(p * 2.2 + vec2(localTime * 0.03, -localTime * 0.02));
    col = mix(col, adinkraPalette(smoke, 0.2),
              smoke * (0.05 + mid * 0.06) * (1.0 - reveal * 0.5));

    col += vec3(0.13, 0.06, 0.25) * warpGlow * 0.12;
    col += vec3(0.92, 0.80, 0.56) * warpGlow * warpGlow * high * 0.10;

    col = adinkra_post(col, localTime, r, punch);
    return clamp(col, 0.0, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = centeredUV(fragCoord);
    float localTime = iTime + 20.0;

    float presence = audioPresence();
    float rawBass = mix(fallbackBass(), getBass(), presence);
    float bass = audioBoost(rawBass);
    float mid  = audioBoost(mix(fallbackMid(),  getMid(),  presence));
    float high = audioBoost(mix(fallbackHigh(), getHigh(), presence));
    float punch = punchShape(rawBass);

    vec3 col = sampleField(localTime, uv, bass, mid, high, punch);

    float r = length(uv);
    col = mix(col, vec3(0.015, 0.010, 0.018), smoothstep(0.95, 1.25, r));

    fragColor = vec4(col, 1.0);
}
`,
  "PIANOSCOPE Adinkra Symbol Field Abstract": `
/*
PIANOSCOPE Shader
Name: Adinkra Symbol Field Abstract
Mode: adinkra_symbol_field_abstract
Inspired by: Mandala (polar fold + iterative mod2 scale) + Adinkra stamp geometry
Cultural caution: Procedural circles, arcs, crosses — not exact sacred Adinkra symbols
Audio: Bass = recursion bloom / central reveal; Mid = field rotation; High = edge shimmer
Projection: Dark violet field, gold/rust marks, slow organic drift, no strobe
*/

#define PI  3.141592654
#define TAU (2.0 * PI)
#define BASS_RECUR_GAIN  0.28
#define MID_ROT_GAIN     0.50
#define HIGH_SHIMMER     0.60
#define SECTOR_COUNT     12.0

vec2 centeredUV(vec2 fragCoord) {
    return (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
}

float fft(float x) {
    return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x;
}

float getBass() {
    return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25;
}

float getMid() {
    return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.25;
}

float getHigh() {
    return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.25;
}

float fallbackBass() { return 0.35 + 0.25 * sin(iTime * 1.1); }
float fallbackMid()  { return 0.28 + 0.18 * sin(iTime * 0.65 + 1.0); }
float fallbackHigh() { return 0.18 + 0.12 * sin(iTime * 2.6 + 2.0); }

float safeAudio(float value, float fallback) {
    return max(value, fallback * 0.35);
}

float audioBoost(float v) {
    return clamp(pow(v, 0.75) * 1.35, 0.0, 1.0);
}

void rot(inout vec2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    p = vec2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

vec2 mod2(inout vec2 p, vec2 size) {
    vec2 c = floor((p + size * 0.5) / size);
    p = mod(p + size * 0.5, size) - size * 0.5;
    return c;
}

vec2 modMirror2(inout vec2 p, vec2 size) {
    vec2 halfsize = size * 0.5;
    vec2 c = floor((p + halfsize) / size);
    p = mod(p + halfsize, size) - halfsize;
    p *= mod(c, vec2(2.0)) * 2.0 - vec2(1.0);
    return c;
}

vec2 toSmith(vec2 p) {
    float d = (1.0 - p.x) * (1.0 - p.x) + p.y * p.y;
    float x = (1.0 + p.x) * (1.0 - p.x) - p.y * p.y;
    float y = 2.0 * p.y;
    return vec2(x, y) / d;
}

vec2 fromSmith(vec2 p) {
    float d = (p.x + 1.0) * (p.x + 1.0) + p.y * p.y;
    float x = (p.x + 1.0) * (p.x - 1.0) + p.y * p.y;
    float y = 2.0 * p.y;
    return vec2(x, y) / d;
}

vec2 toRect(vec2 p) {
    return vec2(p.x * cos(p.y), p.x * sin(p.y));
}

vec2 toPolar(vec2 p) {
    return vec2(length(p), atan(p.y, p.x));
}

float circle(vec2 p, float r) {
    return length(p) - r;
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * noise(p);
        p = p * 2.03 + vec2(1.7, 9.2);
        a *= 0.5;
    }
    return v;
}

vec3 adinkraPalette(float t, float layer) {
    vec3 midnightViolet = vec3(0.13, 0.06, 0.25);
    vec3 deepBlack      = vec3(0.015, 0.010, 0.018);
    vec3 rustOrange     = vec3(0.70, 0.27, 0.10);
    vec3 mutedGold      = vec3(0.83, 0.57, 0.18);
    vec3 cream          = vec3(0.92, 0.80, 0.56);

    vec3 c = mix(deepBlack, midnightViolet, smoothstep(0.0, 0.35, t));
    c = mix(c, rustOrange, smoothstep(0.25, 0.65, t + layer * 0.12));
    c = mix(c, mutedGold, smoothstep(0.45, 0.85, t + layer * 0.08));
    c = mix(c, cream, smoothstep(0.75, 1.0, t) * 0.35);
    return c;
}

// Abstract stamp mark: ring, arcs, cross — not a specific Adinkra symbol
float abstractMark(vec2 p, float phase) {
    float d = 10000.0;

    float ring = abs(circle(p, 0.36)) - 0.035;
    d = min(d, ring);

    float hub = circle(p, 0.11);
    d = min(d, hub);

    for (int i = 0; i < 3; i++) {
        float ang = float(i) * TAU / 3.0 + phase;
        vec2 c = vec2(cos(ang), sin(ang)) * 0.24;
        float arc = abs(length(p - c) - 0.075) - 0.022;
        d = min(d, arc);
    }

    float cross = max(abs(p.x), abs(p.y)) - 0.018;
    cross = max(cross, circle(p, 0.16));
    d = min(d, cross);

    float diamond = abs(p.x) + abs(p.y) - 0.30;
    diamond = max(diamond, -circle(p, 0.28));
    d = min(d, abs(diamond) - 0.015);

    return d;
}

float adinkra_df(float localTime, vec2 p, float bass, float mid) {
    vec2 pp = toPolar(p);
    float a = TAU / SECTOR_COUNT;
    float np = pp.y / a;
    pp.y = mod(pp.y, a);
    if (mod(np, 2.0) > 1.0) {
        pp.y = a - pp.y;
    }
    pp.y += localTime * 0.025 + mid * MID_ROT_GAIN * 0.04;
    p = toRect(pp);
    p = abs(p);
    p -= vec2(0.48);

    float d = 10000.0;
    float scalePulse = 0.5 + 0.5 * sin(localTime * 0.35);

    for (int i = 0; i < 4; i++) {
        float fi = float(i);
        mod2(p, vec2(1.0));
        float wobble = -0.12 * cos(localTime * 0.25 + fi);
        float mark = abstractMark(p, localTime * 0.15 + fi * 0.7) + wobble;
        d = min(d, mark);

        float grow = 1.42 + bass * BASS_RECUR_GAIN + 0.06 * scalePulse;
        p *= grow;
        rot(p, 0.55 + mid * MID_ROT_GAIN * 0.08 + fi * 0.12);
    }

    return d;
}

vec2 fieldDistort(float localTime, vec2 uv, float mid) {
    float lt = 0.08 * localTime + mid * MID_ROT_GAIN * 0.05;
    vec2 suv = toSmith(uv);
    suv += 0.65 * vec2(cos(lt), sin(sqrt(2.0) * lt));
    uv = fromSmith(suv);
    modMirror2(uv, vec2(1.8 + 0.25 * sin(lt * 0.7)));
    return uv;
}

vec3 shadeField(float d, float layerMix, float high, float reveal) {
    float fill = smoothstep(0.018, -0.012, d);
    float edge = smoothstep(0.006, 0.0, abs(d));
    float band = 0.5 + 0.5 * sin(d * 80.0);
    vec3 base = adinkraPalette(band * 0.5 + layerMix * 0.35, layerMix);
    vec3 col = mix(vec3(0.015, 0.010, 0.018), base, fill * (0.55 + reveal * 0.45));
    col += vec3(0.92, 0.80, 0.56) * edge * (0.25 + high * HIGH_SHIMMER);
    col += vec3(0.83, 0.57, 0.18) * edge * edge * high * 0.35;
    return col;
}

vec3 adinkra_post(vec3 col, vec2 uv, float localTime, float r) {
    col = clamp(col, 0.0, 1.0);
    col = pow(col, mix(vec3(0.55, 0.72, 1.15), vec3(0.48), r));
    col = col * 0.62 + 0.38 * col * col * (3.0 - 2.0 * col);
    col = mix(col, vec3(dot(col, vec3(0.33))), -0.25);
    float pulse = sqrt(max(1.0 - 0.65 * sin(localTime * 0.4 + r * 6.0), 0.0));
    col *= mix(0.85, 1.0, pulse);
    col *= 0.55 * sqrt(max(1.05 - r * r, 0.0));
    return clamp(col, 0.0, 1.0);
}

vec3 sampleField(float localTime, vec2 p, float bass, float mid, float high) {
    vec2 uv = p * 6.5;
    rot(uv, localTime * 0.04 + mid * MID_ROT_GAIN * 0.12);

    vec2 nuv = fieldDistort(localTime, uv, mid);
    vec2 nuv2 = fieldDistort(localTime, uv + vec2(0.0008), mid);
    float warpGlow = 1.0 - smoothstep(0.0, 0.003, length(nuv - nuv2));

    float d = adinkra_df(localTime, nuv, bass, mid);
    float r = length(p);

    float reveal = smoothstep(0.55, 0.08, r) * (0.45 + bass * 0.55);
    vec3 col = shadeField(d, r * 0.8 + bass * 0.2, high, reveal);

    float smoke = fbm(p * 2.2 + vec2(localTime * 0.03, -localTime * 0.02));
    col = mix(col, adinkraPalette(smoke, 0.2), smoke * 0.08 * (1.0 - reveal * 0.5));

    col += vec3(0.13, 0.06, 0.25) * warpGlow * 0.12;
    col += vec3(0.92, 0.80, 0.56) * warpGlow * warpGlow * high * 0.08;

    col = adinkra_post(col, nuv, localTime, r);
    return clamp(col, 0.0, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = centeredUV(fragCoord);
    float localTime = iTime + 20.0;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(), fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec3 col = sampleField(localTime, uv, bass, mid, high);

    float r = length(uv);
    col = mix(col, vec3(0.015, 0.010, 0.018), smoothstep(0.95, 1.25, r));
    col = sqrt(max(col, 0.0));
    col = min(col, vec3(0.82));

    fragColor = vec4(col, 1.0);
}
`,
        "Sunflower Fields": `
// Sunflower Fields
// By Bitless, Ircss, & Noztol

#define p(t, a, b, c, d) ( a + b*cos( 6.28318*(c*t+d) ) )
#define sp(t) p(t,vec3(.26,.76,.77),vec3(1,.3,1),vec3(.8,.4,.7),vec3(0,.12,.54))
#define hue(v) ( .6 + .76 * cos(6.3*(v) + vec4(0,23,21,0) ) )

#define smoothing      0.006
#define TWO_PI         6.28318530718
#define lineSize       0.01

#define MountainLayerThreecol vec3(26., 65., 74.)/255.
#define MountainLayerFourCol vec3(14., 49., 55.)/255.
#define SunflowerInsideOne    vec3(203., 77., 23.)/255.
#define SunflowerInsideTwo    vec3(134., 71., 48.)/255.
#define SunflowerInsideThree  vec3(158., 159., 33.)/255.
#define SunflowerLeavesOne    vec3(247., 214., 0.)/255.
#define SunflowerHighlight    vec3(247., 218., 63.)/255.
#define SunflowerLeavesTwo    vec3(236., 168., 3.)/255.
#define SunflowerStem         vec3(97., 128., 52.)/255.
#define SunflowerStemBright   vec3(176., 186., 53.)/255.
#define FieldDark             vec3(44., 62., 40.)/255.
#define FieldMid              vec3(94., 121., 62.)/255.

float hash12(vec2 p) {
    vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

vec2 rotate2D (vec2 st, float a){
    return  mat2(cos(a),-sin(a),sin(a),cos(a))*st;
}

float st(float a, float b, float s) { return smoothstep (a-s, a+s, b); }

float noise( in vec2 p ) {
    vec2 i = floor( p );
    vec2 f = fract( p );
    vec2 u = f*f*(3.-2.*f);
    return mix( mix( dot( hash22( i+vec2(0,0) ), f-vec2(0,0) ),
                     dot( hash22( i+vec2(1,0) ), f-vec2(1,0) ), u.x),
                mix( dot( hash22( i+vec2(0,1) ), f-vec2(0,1) ),
                     dot( hash22( i+vec2(1,1) ), f-vec2(1,1) ), u.x), u.y);
}

float s_noise(vec2 p) { return noise(p)*0.5 + 0.5; }

float fbm(vec2 p) {
    float v = 0.0; float a = 0.5; vec2 shift = vec2(100.0);
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.50));
    for (int i = 0; i < 5; ++i) { v += a * s_noise(p); p = rot * p * 2.0 + shift; a *= 0.5; }
    return v;
}

float randOneD(float x) { return fract(sin(x *52.163)*268.2156); }

void DrawHalfVectorWithLength(vec2 origin, vec2 vector, float len, vec2 uv, float size, vec3 lineColor, inout vec3 sceneColor){
    uv  -= origin;
    float v2   = dot(vector, vector);
    float vUv  = dot(vector, uv);
    vec2  p    = vector * vUv/v2;
    float d    = distance(p, uv);
    float m    = 1. - step(0.,vUv/v2);
          m   += step(len, vUv/v2);
    sceneColor = mix(lineColor, sceneColor, clamp(smoothstep(size, size + 0.01, d)+ m, 0. ,1.));
}

void DrawStemLeave(vec2 origin, vec2 vector, float len, vec2 uv, float size, vec3 lineColor, inout vec3 sceneColor){
    uv  -= origin;
    float v2   = dot(vector, vector);
    float vUv  = dot(vector, uv);
    uv.y += pow(vUv/len, 2.)*4.;
    vec2  p    = vector * vUv/v2;
    float d    = distance(p, uv);
    float m    = 1. - step(0.,vUv/v2);
          m   += step(len, vUv/v2);
          size *= smoothstep(0.5, 0.0, abs(vUv - 0.5)/len) *0.5;
    sceneColor = mix(lineColor, sceneColor, clamp(smoothstep(size, size + 0.01, d)+ m, 0. ,1.));
}

void DrawPetals(vec2 uv, inout vec3 col, float seed, float offset, vec3 petalColor) {
     float leavesSpread = 0.35;
     vec2  petalSpace = vec2(fract((offset+ uv.x)* TWO_PI *leavesSpread), uv.y);
     float petalSpaceID = floor((uv.x+offset)* TWO_PI * leavesSpread);
     float petalLength = 1.;
     float petalThickness = smoothstep(-0.01, 1., pow( (1.-(petalSpace.y / petalLength)), 0.85)) *0.9 - smoothstep(0.7,0., petalSpace.y / petalLength);
     petalSpace.x += sin((petalSpace.y + randOneD(petalSpaceID + seed) * TWO_PI )*4. )*0.3 * smoothstep(0.5, 1., (petalSpace.y / petalLength));
     DrawHalfVectorWithLength(vec2(0.5, 0.), vec2(0.,1.), 1., petalSpace, petalThickness,  petalColor, col);
}

void DrawSunFlower(vec2 uv, float seed, inout vec3 col, float mask) {
    vec2 stroke_warp = vec2(s_noise(uv*20.0), s_noise(uv*20.0 + 10.0)) * 0.08;
    vec2 w_uv = uv + stroke_warp;
    float paintStroke = fbm(uv * 25.0) * 0.25;

    vec2 stem_uv = w_uv;

    float flowAmount = sin(stem_uv.y * 1.5 - iTime * 2.0 + seed * TWO_PI) * 0.2;
    float bendFactor = smoothstep(-7.0, 0.0, stem_uv.y);
    stem_uv.x -= flowAmount * bendFactor;

    DrawHalfVectorWithLength(vec2(0.), vec2(0.,-1.), 7., stem_uv, 0.15, SunflowerStem + paintStroke, col);
    DrawStemLeave(vec2(0.,-2.+ randOneD(seed+5.125) *-2.), normalize(vec2(max(0.2,randOneD(seed+712.125)),(randOneD(seed+81.215) -0.3) * 0.3)), 5., stem_uv, 0.3 +randOneD(seed+12.125) *0.4 , SunflowerStemBright + paintStroke, col);
    DrawStemLeave(vec2(0.,-3.+ randOneD(seed+61.125) *-2.), normalize(vec2(-1.0 * max(0.2,randOneD(seed+4.25)),(randOneD(seed+73.25) -0.3) * 0.3)), 5., stem_uv, 0.3 + randOneD(seed+0.125) * 0.4, SunflowerStem + paintStroke, col);

    vec2 head_uv = w_uv;

    float headFlow = sin(0.0 * 1.5 - iTime * 2.0 + seed * TWO_PI) * 0.2;
    float headBend = smoothstep(-7.0, 0.0, 0.0);
    head_uv.x -= headFlow * headBend;

    head_uv = vec2(atan(head_uv.y, head_uv.x), length(head_uv) * 0.55);

    float spinSpeed = 0.15 + (randOneD(seed) * 0.1);
    head_uv.x -= iTime * spinSpeed;

    vec3 DrawnFlower= col;

    DrawPetals(head_uv, DrawnFlower, 53.126 + seed, +0.4 + randOneD(seed), SunflowerLeavesTwo + paintStroke);
    DrawPetals(head_uv, DrawnFlower, 0. + seed, randOneD(seed+7.125)*-0.5, SunflowerLeavesOne + paintStroke);

    float centerMask = smoothstep(lineSize, lineSize + smoothing, 0.45 - head_uv.y);

    if (centerMask > 0.0) {
        float yd = 20.0;
        float r_id = floor((head_uv.y + 0.01) * yd);
        float xd = max(4.0, floor(r_id * TWO_PI * 0.8));
        float spiralAngle = head_uv.x + head_uv.y * 5.0 + seed * 10.0;
        float seg_x = (spiralAngle / TWO_PI) * xd;
        float seg_y = (head_uv.y + 0.01) * yd;

        vec2 cell_id = vec2(floor(seg_x), floor(seg_y));
        vec2 lc = vec2(fract(seg_x), fract(seg_y));
        float n = s_noise(cell_id * 5.0 + seed);
        lc += (vec2(s_noise(cell_id*3.0), s_noise(cell_id*7.0)) - 0.5) * 0.4;

        float dashMask = st(abs(lc.x - 0.5), 0.4, 0.08) * st(abs(lc.y - 0.5), 0.4, 0.08);

        vec3 midRing = mix(SunflowerInsideTwo, SunflowerInsideThree, n);
        vec3 centerBgCol = mix(SunflowerInsideOne, midRing, smoothstep(0.45, 0.20, head_uv.y));
        centerBgCol = mix(centerBgCol, SunflowerInsideThree, smoothstep(0.12, 0.0, head_uv.y));

        vec3 strokeCol = centerBgCol * (1.1 + n * 0.4);
        vec3 gapCol = centerBgCol * 0.4;

        vec3 finalCenterCol = mix(gapCol, strokeCol, dashMask);
        DrawnFlower = mix(DrawnFlower, finalCenterCol, centerMask);
    }

    col = mix(col, DrawnFlower, mask);
}

void DrawSunFlowerField(vec2 OG_UV, float seed, vec2 offset, float fieldMask, float totalMovementSpeed, inout vec3 col,float tiling) {
     OG_UV += offset; OG_UV.x += iTime *totalMovementSpeed;
     vec2  flowerRepeatedSpace = vec2(fract(OG_UV.x*tiling), OG_UV.y*tiling);
     vec2  idFlowerCoord = vec2(floor(OG_UV.x*tiling), seed*21.215);
     flowerRepeatedSpace -= vec2(0.5) + vec2(0.15,0.5) *(randOneD (dot(idFlowerCoord , vec2(1.126, 26.6))) - 0.5) ;
     flowerRepeatedSpace *= 4. + 0.2 *(randOneD (dot(idFlowerCoord , vec2(8.136, 5.316))) - 0.5);
     DrawSunFlower(flowerRepeatedSpace, randOneD (dot(idFlowerCoord , vec2(21.126, 8.3216))), col,fieldMask );
}

void mainImage( out vec4 O, in vec2 g) {
    vec2 r = iResolution.xy;
    vec2 uv = (g+g-r)/r.y;

    vec2 sun_pos = vec2(r.x/r.y * 0.45, -.45);
    vec2 tree_pos = vec2(-r.x/r.y * 0.2, -.2);

    vec2 sh, u, id, lc, t;
    vec3 f = vec3(0), c;
    float xd, yd, h, a=0.0, l;
    vec4 C;
    float sm = 3./r.y;

    sh = rotate2D(sun_pos, noise(uv+iTime*.25)*.3);
    if (uv.y > -.8)
    {
        u = uv + sh;
        yd = 60.;
        id =  vec2((length(u)+.01)*yd,0);
        xd = floor(id.x)*.09;
        h = (hash12(floor(id.xx))*.5+.25)*(iTime+10.)*.25;
        t = rotate2D (u,h);

        id.y = atan(t.y,t.x)*xd;
        lc = fract(id);
        id -= lc;

        t = vec2(cos((id.y+.5)/xd)*(id.x+.5)/yd,sin((id.y+.5)/xd)*(id.x+.5)/yd);
        t = rotate2D(t,-h) - sh;

        h = noise(t*vec2(.5,1)-vec2(iTime*.2,0)) * step(-.25,t.y);
        h = smoothstep (.052,.055, h);

        lc += (noise(lc*vec2(1,4)+id))*vec2(.7,.2);

        f = mix (sp(sin(length(u)-.1))*.35,
                 mix(sp(sin(length(u)-.1)+(hash12(id)-.5)*.15),vec3(1),h),
                 st(abs(lc.x-.5),.4,sm*yd)*st(abs(lc.y-.5),.48,sm*xd));

        vec2 sun_uv = uv - sun_pos;
        vec2 spun_uv = rotate2D(sun_uv, -iTime * 0.5);

        float dToSun = length(sun_uv);
        float angle = atan(spun_uv.y, spun_uv.x);

        float rays = sin(angle * 12.0 + s_noise(spun_uv*15.0)*2.0) * 0.03;
        float sunCore = smoothstep(0.12, 0.02, dToSun);
        float sunCorona = smoothstep(0.35, 0.1, dToSun + rays) * s_noise(spun_uv * 20.0);

        vec3 sunColor = mix(vec3(0.9, 0.4, 0.0), vec3(1.0, 1.0, 0.2), smoothstep(0.15, 0.0, dToSun));

        float swirl = smoothstep(0.0, 0.8, sin(dToSun * 40.0 + angle * 4.0));
        sunColor += swirl * 0.3 * sunCore;

        f = mix(f, sunColor, clamp(sunCore + sunCorona, 0.0, 1.0));
    };

    if (uv.y < 0.25)
    {
        float cld = noise(-sh*vec2(.5,1)  - vec2(iTime*.2,0));
        cld = 1.- smoothstep(.0,.15,cld)*.5;

        u = (uv - vec2(0.0, 0.25)) * vec2(1, 15);
        id = floor(u);

        for (float i = 1.; i > -1.; i--)
        {
            if (id.y+i < 0.0)
            {
                lc = fract(u)-.5;
                lc.y = (lc.y+(sin(uv.x*8.-iTime*0.8+id.y+i))*.3-i)*4.;
                h = hash12(vec2(id.y+i,floor(lc.y)));

                xd = 6.+h*4.;
                yd = 30.;
                lc.x = uv.x*xd+sh.x*9.;
                lc.x += sin(iTime * (.2 + h*1.5))*.5;
                h = .8*smoothstep(5.,.0,abs(floor(lc.x)))*cld+.1;

                vec3 mtnBase = MountainLayerFourCol;
                vec3 mtnHigh = MountainLayerThreecol;
                f = mix(f,mix(mtnBase,mtnHigh,h),st(lc.y,0.,sm*yd));
                lc += noise(lc*vec2(3,.5))*vec2(.1,.6);

                vec3 strokeCol = hue(hash12(floor(lc))*.1+.35).rgb*(1.2+floor(lc.y)*.17);
                f = mix(f,
                    mix(strokeCol,FieldMid,h),
                    st(lc.y,0.,sm*xd)
                    *st(abs(fract(lc.x)-.5),.48,sm*xd)*st(abs(fract(lc.y)-.5),.3,sm*yd)
                );
            }
        }
    }

    vec2 uv_sun = g/r; uv_sun -= 0.5; uv_sun.x *= r.x/r.y; uv_sun *= 5.;
    float totalMovementSpeed = 0.05; float movement = iTime * totalMovementSpeed;

    float fieldMask = smoothstep(0.05, 0.15, 0.1 - uv.y);
    vec3 fieldBaseColor = mix(FieldMid + pow(s_noise((uv_sun + vec2(movement * 4.7,0.))*20.0), 2.)*0.05,
                              SunflowerLeavesOne + pow(s_noise((uv_sun + vec2(movement*1.,0.))*20.0), 2.)*0.05, smoothstep(-0.2, 0.0, uv_sun.y));
    fieldBaseColor = mix(fieldBaseColor, FieldDark + pow(s_noise((uv_sun + vec2(movement*6.2,0.))*20.0), 2.)*0.05, smoothstep(-0.6, -1.4, uv_sun.y));
    f = mix(f, fieldBaseColor, fieldMask);

    O = vec4(f, 1.0);

    vec2 OG_UV = g/r.xy; OG_UV.x *= r.x/r.y;
    float fm = smoothstep(0.01, 0.05, 0.1 - uv_sun.y);

    if (uv.y < 0.1) {
        vec3 tCol = O.rgb;
        DrawSunFlowerField(OG_UV, 0., vec2(0.51,-0.48),    fm, totalMovementSpeed, tCol, 90.);
        totalMovementSpeed *= 1.1; DrawSunFlowerField(OG_UV, 6.621, vec2(0.51,-0.46), fm,  totalMovementSpeed, tCol, 50.);
        totalMovementSpeed *= 1.1; DrawSunFlowerField(OG_UV, 7.23, vec2(0.51,-0.43),  fm,  totalMovementSpeed, tCol, 29.);
        totalMovementSpeed *= 1.1; DrawSunFlowerField(OG_UV, 12.6, vec2(0.51,-0.4),   fm,  totalMovementSpeed, tCol, 22.);
        totalMovementSpeed *= 1.1; DrawSunFlowerField(OG_UV, -7.21, vec2(0.51,-0.35), fm,  totalMovementSpeed, tCol, 15.);
        totalMovementSpeed *= 1.1; DrawSunFlowerField(OG_UV, 2.125, vec2(0.51,-0.3),  fm,  totalMovementSpeed, tCol, 12.);
        O.rgb = tCol;
    }

    float T = sin(iTime*.5);

    if (abs(uv.x+tree_pos.x-.1-T*.1) < .6) {
        u = uv + tree_pos; u.x -= sin(u.y+1.)*.2*(T+.75); u += noise(u*4.5-7.)*.25;
        xd = 10., yd = 60.; t = u * vec2(1,yd); h = hash12(floor(t.yy)); t.x += h*.01; t.x *= xd; lc = fract(t);
        float m = st(abs(t.x-.5),.5,sm*xd)*step(abs(t.y+20.),45.);
        C = mix(vec4(.07), vec4(.5,.3,0,1)*(.4+h*.4), st(abs(lc.y-.5),.4,sm*yd)*st(abs(lc.x-.5),.45,sm*xd)); C.a = m;
        xd = 30., yd = 15.;
        for (float xs =0.;xs<4.;xs++) {
            u = uv + tree_pos + vec2 (xs/xd*.5 -(T +.75)*.15,-.7);
            u += noise(u*vec2(2,1)+vec2(-iTime+xs*.05,0))*vec2(-.25,.1)*smoothstep (.5,-1.,u.y+.7)*.75;
            t = u * vec2(xd,1.); h = hash12(floor(t.xx)+xs*1.4); yd = 5.+ h*7.; t.y *= yd; sh = t; lc = fract(t); h = hash12(t-lc);
            t = (t-lc)/vec2(xd,yd)+vec2(0,.7);
            m = (step(0.,t.y)*step (length(t),.45) + step (t.y,0.)*step (-0.7+sin((floor(u.x)+xs*.5)*15.)*.2,t.y)) *step (abs(t.x),.5) *st(abs(lc.x-.5),.35,sm*xd*.5);
            lc += noise((sh)*vec2(1.,3.))*vec2(.3,.3); f = hue((h+(sin(iTime*.2)*.5+.5))*.2).rgb-t.x;
            C = mix(C, vec4(mix(f*.15,f*.6*(.7+xs*.2), st(abs(lc.y-.5),.47,sm*yd)*st(abs(lc.x-.5),.2,sm*xd)),m), m);
        }
        O = mix (O,C,C.a);
    }

    if (uv.y < 0.1) {
        vec3 tCol = O.rgb;
        totalMovementSpeed *= 1.2; DrawSunFlowerField(OG_UV, 1., vec2(1.126,-0.2),    fm,  totalMovementSpeed, tCol, 8.);
        totalMovementSpeed *= 1.2; DrawSunFlowerField(OG_UV, 5., vec2(0.,-0.05),      fm,  totalMovementSpeed, tCol, 4.);
        totalMovementSpeed *= 1.3; DrawSunFlowerField(OG_UV, 71.612, vec2(0.,0.1),    fm,  totalMovementSpeed, tCol, 3.);
        O.rgb = tCol;
    }

    float global_canvas = fbm(uv * 100.0) * 0.05;
    O.rgb -= global_canvas;
}
`,
"Rainbow Fractal": `
#define l 120
void mainImage(out vec4 FragColor,vec2 FragCoord){
	vec2 v = (FragCoord.xy - iResolution.xy/2.) / min(iResolution.y,iResolution.x) * 30.;
	vec2 vv = v;// vec2 vvv = v;
	float ft = iTime+360.1;
	float tm = ft*0.1;
	float tm2 = ft*0.3;
	vec2 mspt = (vec2(
			sin(tm)+cos(tm*0.2)+sin(tm*0.5)+cos(tm*-0.4)+sin(tm*1.3),
			cos(tm)+sin(tm*0.1)+cos(tm*0.8)+sin(tm*-1.1)+cos(tm*1.5)
			)+1.0)*0.35; //5x harmonics, scale back to [0,1]
	float R = 0.0;
	float RR = 0.0;
	float RRR = 0.0;
	float a = (1.-mspt.x)*0.5;
	float C = cos(tm2*0.03+a*0.01)*1.1;
	float S = sin(tm2*0.033+a*0.23)*1.1;
	float C2 = cos(tm2*0.024+a*0.23)*3.1;
	float S2 = sin(tm2*0.03+a*0.01)*3.3;
	vec2 xa=vec2(C, -S);
	vec2 ya=vec2(S, C);
	vec2 xa2=vec2(C2, -S2);
	vec2 ya2=vec2(S2, C2);
	vec2 shift = vec2( 0.033, 0.14);
	vec2 shift2 = vec2( -0.023, -0.22);
	float Z = 0.4 + mspt.y*0.3;
	float m = 0.99+sin(iTime*0.03)*0.003;
	for ( int i = 0; i < l; i++ ){
		float r = dot(v,v);
		float r2 = dot(vv,vv);
		if ( r > 1.0 )
		{
			r = (1.0)/r ;
			v.x = v.x * r;
			v.y = v.y * r;
		}
		if ( r2 > 1.0 )
		{
			r2 = (1.0)/r2 ;
			vv.x = vv.x * r2;
			vv.y = vv.y * r2;
		}
		R *= m;
		R += r;
		R *= m;
		R += r2;
		if(i < l-1){
			RR *= m;
			RR += r;
			RR *= m;
			RR += r2;
			if(i < l-2){
				RRR *= m;
				RRR += r;
				RRR *= m;
				RRR += r2;
			}
		}
		
		v = vec2( dot(v, xa), dot(v, ya)) * Z + shift;
		vv = vec2( dot(vv, xa2), dot(vv, ya2)) * Z + shift2;
	}
	
	float c = ((mod(R,2.0)>1.0)?1.0-fract(R):fract(R));
	float cc = ((mod(RR,2.0)>1.0)?1.0-fract(RR):fract(RR));
	float ccc = ((mod(RRR,2.0)>1.0)?1.0-fract(RRR):fract(RRR));
	FragColor = vec4(ccc, cc, c, 1.0); 
}`,
    "Modular Synth": `#define lofi(i,j) (floor((i)/(j))*(j))
#define lofir(i,j) (round((i)/(j))*(j))

const float PI=acos(-1.);
const float TAU=PI*2.;

mat2 r2d(float t){
  float c=cos(t),s=sin(t);
  return mat2(c,s,-s,c);
}

mat3 orthbas(vec3 z){
  z=normalize(z);
  vec3 up=abs(z.y)>.999?vec3(0,0,1):vec3(0,1,0);
  vec3 x=normalize(cross(up,z));
  return mat3(x,cross(z,x),z);
}

uvec3 pcg3d(uvec3 s){
  s=s*1145141919u+1919810u;
  s.x+=s.y*s.z;
  s.y+=s.z*s.x;
  s.z+=s.x*s.y;
  s^=s>>16;
  s.x+=s.y*s.z;
  s.y+=s.z*s.x;
  s.z+=s.x*s.y;
  return s;
}

vec3 pcg3df(vec3 s){
  uvec3 r=pcg3d(floatBitsToUint(s));
  return vec3(r)/float(0xffffffffu);
}

struct Grid{
  vec3 s;
  vec3 c;
  vec3 h;
  int i;
  float d;
};

Grid dogrid(vec3 ro,vec3 rd){
  Grid r;
  r.s=vec3(2,2,100);
  for(int i=0;i<3;i++){
    r.c=(floor(ro/r.s)+.5)*r.s;
    r.h=pcg3df(r.c);
    r.i=i;

    if(r.h.x<.4){
      break;
    }else if(i==0){
      r.s=vec3(2,1,100);
    }else if(i==1){
      r.s=vec3(1,1,100);
    }
  }
  
  vec3 src=-(ro-r.c)/rd;
  vec3 dst=abs(.501*r.s/rd);
  vec3 bv=src+dst;
  float b=min(min(bv.x,bv.y),bv.z);
  r.d=b;
  
  return r;
}

float sdbox(vec3 p,vec3 s){
  vec3 d=abs(p)-s;
  return length(max(d,0.))+min(0.,max(max(d.x,d.y),d.z));
}

float sdbox(vec2 p,vec2 s){
  vec2 d=abs(p)-s;
  return length(max(d,0.))+min(0.,max(d.x,d.y));
}

vec4 map(vec3 p,Grid grid){
  p-=grid.c;
  p.z+=.4*sin(2.*iTime+1.*fract(grid.h.z*28.)+.3*(grid.c.x+grid.c.y));
  
  vec3 psize=grid.s/2.;
  psize.z=1.;
  psize-=.02;
  float d=sdbox(p+vec3(0,0,1),psize)-.02;
  
  float pcol=1.;

  vec3 pt=p;
  
  if(grid.i==0){//2x2
    if(grid.h.y<.3){//speaker
      vec3 c=vec3(0);
      pt.xy*=r2d(PI/4.);
      c.xy=lofir(pt.xy,.1);
      pt=pt-c;
      pt.xy*=r2d(-PI/4.);
      
      float r=.02*smoothstep(.9,.7,abs(p.x))*smoothstep(.9,.7,abs(p.y));
      float hole=length(pt.xy)-r;
      d=max(d,-hole);
    }else if(grid.h.y<.5){//eq
      vec3 c=vec3(0);
      c.x=clamp(lofir(pt.x,.2),-.6,.6);
      pt-=c;
      float hole=sdbox(pt.xy,vec2(0.,.7))-.03;
      d=max(d,-hole);
      
      pt.y-=.5-smoothstep(-.5,.5,sin(iTime+c.x+grid.h.z*100.));
      float d2=sdbox(pt,vec3(.02,.07,.07))-.03;
      
      if(d2<d){
        float l=step(abs(pt.y),.02);
        return vec4(d2,2.*l,l,0);
      }
      
      pt=p;
      c.y=clamp(lofir(pt.y,.2),-.6,.6);
      pt-=c;
      pcol*=smoothstep(.0,.01,sdbox(pt.xy,vec2(.07,.0))-.005);

      pt=p;
      c.y=clamp(lofir(pt.y,.6),-.6,.6);
      pt-=c;
      pcol*=smoothstep(.0,.01,sdbox(pt.xy,vec2(.1,.0))-.01);
      
      pcol=mix(1.,pcol,smoothstep(.0,.01,sdbox(pt.xy,vec2(.03,1.))-.01));

    }else if(grid.h.y<.6){//kaosspad
      float hole=sdbox(p.xy,vec2(.9,.9)+.02);
      d=max(d,-hole);

      float d2=sdbox(p,vec3(.9,.9,.05));

      if(d2<d){
        float l=step(abs(p.x),.7)*step(abs(p.y),.7);
        return vec4(d2,4.*l,0,0);
      }
    }else if(grid.h.y<1.){//bigass knob
      float ani=smoothstep(-.5,.5,sin(iTime+grid.h.z*100.));
      pt.xy*=r2d(PI/6.*5.*mix(-1.,1.,ani));

      float metal=step(length(pt.xy),.45);
      float wave=metal*sin(length(pt.xy)*500.)/1000.;
      float d2=length(pt.xy)-.63+.05*pt.z-.02*cos(8.*atan(pt.y,pt.x));
      d2=max(d2,abs(pt.z)-.4-wave);

      float d2b=length(pt.xy)-.67+.05*pt.z;
      d2b=max(d2b,abs(pt.z)-.04);
      d2=min(d2,d2b);
      
      if(d2<d){
        float l=smoothstep(.01,.0,length(pt.xy-vec2(0,.53))-.03);
        return vec4(d2,3.*metal,l,0);
      }
      
      pt=p;
      float a=clamp(lofir(atan(-pt.x,pt.y),PI/12.),-PI/6.*5.,PI/6.*5.);
      pt.xy*=r2d(a);
      pcol*=smoothstep(.0,.01,length(pt.xy-vec2(0,.74))-.015);

      pt=p;
      a=clamp(lofir(atan(-pt.x,pt.y),PI/6.*5.),-PI/6.*5.,PI/6.*5.);
      pt.xy*=r2d(a);
      pcol*=smoothstep(.0,.01,length(pt.xy-vec2(0,.74))-.03);
      
      float d3=length(p-vec3(.7,-.7,0))-.05;
      
      if(d3<d){
        float led=1.-ani;
        led*=.5+.5*sin(iTime*exp2(3.+3.*grid.h.z));
        return vec4(d3,2,led,0);
      }
    }
  }else if(grid.i==1){//2x1
    if(grid.h.y<.4){//fader
      float hole=sdbox(p.xy,vec2(.9,.05));
      d=max(d,-hole);
      
      float ani=smoothstep(-.2,.2,sin(iTime+grid.h.z*100.));
      pt.x-=mix(-.8,.8,ani);
      
      float d2=sdbox(pt,vec3(.07,.25,.4))+.05*p.z;
      d2=max(d2,-p.z);

      if(d2<d){
        float l=smoothstep(.01,.0,abs(p.y)-.02);
        return vec4(d2,0,l,0);
      }
      
      pt=p;
      vec3 c=vec3(0);
      c.x=clamp(lofir(pt.x,.2),-.8,.8);
      pt-=c;
      pcol*=smoothstep(.0,.01,sdbox(pt.xy,vec2(.0,.15))-.005);

      pt=p;
      c=vec3(0);
      c.x=clamp(lofir(pt.x,.8),-.8,.8);
      pt-=c;
      pcol*=smoothstep(.0,.01,sdbox(pt.xy,vec2(.0,.18))-.01);
      
      pcol=mix(1.,pcol,smoothstep(.0,.01,sdbox(p.xy,vec2(1.,.08))));
    }else if(grid.h.y<.5){//button
      vec3 c=vec3(0);
      c.x=clamp(lofi(pt.x,.44)+.44/2.,-.44*1.5,.44*1.5);
      pt-=c;

      float hole=sdbox(pt.xy,vec2(.19,.33))-.01;
      d=max(d,-hole);
      
      float ani=smoothstep(.8,.9,sin(10.*iTime-c.x*2.2+grid.h.z*100.));

      vec4 fuck=vec4(d,0,0,0);
      float d3=length(pt-vec3(0,.22,.04))-.05;
      
      if(d3<fuck.x){
        float led=ani;
        fuck=vec4(d3,2,led,0);
      }

      float d2=sdbox(pt,vec3(.17,.3,.05))-.01;
      d2=min(d2,sdbox(pt-vec3(0,-.1,0),vec3(.17,.2,.08))-.01)+.5*pt.z;

      if(d2<fuck.x){
        fuck=vec4(d2,5,fract(grid.h.z*8.89),0);
      }
      
      if(fuck.x<d){
        return fuck;
      }
      
    }else if(grid.h.y<1.){//meter
      float hole=sdbox(p.xy,vec2(.9,.3)+.02);
      d=max(d,-hole);

      float d2=sdbox(p,vec3(.9,.3,.1));

      if(d2<d){
        float l=step(abs(p.x),.8)*step(abs(p.y),.2);
        return vec4(d2,l,0,0);
      }
    }
  }else{//1x1
    if(grid.h.y<.5){//knob
      float hole=length(p.xy)-.25;
      d=max(d,-hole);
      
      float ani=smoothstep(-.5,.5,sin(2.*iTime+grid.h.z*100.));
      pt.xy*=r2d(PI/6.*5.*mix(-1.,1.,ani));
      
      float d2=length(pt.xy)-.23+.05*pt.z;
      d2=max(d2,abs(pt.z)-.4);
      
      if(d2<d){
        float l=smoothstep(.01,.0,abs(pt.x)-.015);
        l*=smoothstep(.01,.0,-pt.y+.05);
        return vec4(d2,0,l,0);
      }
      
      pt=p;
      float a=clamp(lofir(atan(-pt.x,pt.y),PI/6.),-PI/6.*5.,PI/6.*5.);
      pt.xy*=r2d(a);
      pcol*=smoothstep(.0,.01,sdbox(pt.xy-vec2(0,.34),vec2(.0,.02))-.005);

      pt=p;
      a=clamp(lofir(atan(-pt.x,pt.y),PI/6.*5.),-PI/6.*5.,PI/6.*5.);
      pt.xy*=r2d(a);
      pcol*=smoothstep(.0,.01,sdbox(pt.xy-vec2(0,.34),vec2(.0,.03))-.01);
    }else if(grid.h.y<.8){//jack
      float hole=length(p.xy)-.1;
      d=max(d,-hole);
      
      float d2=length(p.xy)-.15;
      d2=max(d2,abs(p.z)-.12);
      
      pt.xy*=r2d(100.*grid.h.z);
      float d3=abs(pt.y)-.2;
      pt.xy*=r2d(PI/3.*2.);
      d3=max(d3,abs(pt.y)-.2);
      pt.xy*=r2d(PI/3.*2.);
      d3=max(d3,abs(pt.y)-.2);
      d3=max(d3,abs(p.z)-.03);

      d2=min(d2,d3);
      d2=max(d2,-hole);
      
      if(d2<d){
        return vec4(d2,3,0,0);
      }
    }else if(grid.h.y<.99){//button
      pt.y+=.08;
      
      float hole=sdbox(pt.xy,vec2(.22))-.05;
      d=max(d,-hole);
      
      float ani=sin(2.*iTime+grid.h.z*100.);
      float push=smoothstep(.3,.0,abs(ani));
      ani=smoothstep(-.1,.1,ani);
      pt.z+=.06*push;

      float d2=sdbox(pt,vec3(.2,.2,.05))-.05;

      if(d2<d){
        return vec4(d2,0,0,0);
      }
      
      float d3=length(p-vec3(0,.3,0))-.05;
      
      if(d3<d){
        float led=ani;
        return vec4(d3,2,led,0);
      }
    }else if(grid.h.y<1.){//0b5vr
      pt=abs(pt);
      pt.xy=pt.x<pt.y?pt.yx:pt.xy;
      pcol*=smoothstep(.0,.01,sdbox(pt.xy,vec2(.05)));
      pcol*=smoothstep(.0,.01,sdbox(pt.xy-vec2(.2,0),vec2(.05,.15)));
      pcol=1.-pcol;
    }
  }
  
  return vec4(d,0,pcol,0);
}

vec3 nmap(vec3 p,Grid grid,float dd){
  vec2 d=vec2(0,dd);
  return normalize(vec3(
    map(p+d.yxx,grid).x-map(p-d.yxx,grid).x,
    map(p+d.xyx,grid).x-map(p-d.xyx,grid).x,
    map(p+d.xxy,grid).x-map(p-d.xxy,grid).x
  ));
}

struct March{
  vec4 isect;
  vec3 rp;
  float rl;
  Grid grid;
};

March domarch(vec3 ro,vec3 rd,int iter){
  float rl=1E-2;
  vec3 rp=ro+rd*rl;
  vec4 isect;
  Grid grid;
  float gridlen=rl;
  
  for(int i=0;i<iter;i++){
    if(gridlen<=rl){
      grid=dogrid(rp,rd);
      gridlen+=grid.d;
    }
    
    isect=map(rp,grid);
    rl=min(rl+isect.x*.8,gridlen);
    rp=ro+rd*rl;
    
    if(abs(isect.x)<1E-4){break;}
    if(rl>50.){break;}
  }
  
  March r;
  r.isect=isect;
  r.rp=rp;
  r.rl=rl;
  r.grid=grid;
  
  return r;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
  vec2 uv = vec2(fragCoord.x / iResolution.x, fragCoord.y / iResolution.y);
  vec2 p=uv*2.-1.;
  p.x*=iResolution.x/iResolution.y;

  vec3 col=vec3(0);

  float canim=smoothstep(-.2,.2,sin(iTime));
  vec3 co=mix(vec3(-6,-8,-40),vec3(0,-2,-40),canim);
  vec3 ct=vec3(0,0,-50);
  float cr=mix(.5,.0,canim);
  co.xy+=iTime;
  ct.xy+=iTime;
  mat3 cb=orthbas(co-ct);
  vec3 ro=co+cb*vec3(4.*p*r2d(cr),0);
  vec3 rd=cb*normalize(vec3(0,0,-2));
  
  March march=domarch(ro,rd,100);
  
  if(march.isect.x<1E-2){
    vec3 basecol=vec3(.5);
    vec3 speccol=vec3(.2);
    float specpow=30.;
    float ndelta=1E-4;
    
    float mtl=march.isect.y;
    float mtlp=march.isect.z;
    if(mtl==0.){
      mtlp=mix(mtlp,1.-mtlp,step(fract(march.grid.h.z*66.),.1));
      vec3 c=.9+.0*sin(.1*(march.grid.c.x+march.grid.c.y)+march.grid.h.z+vec3(0,2,3));
      basecol=mix(vec3(.04),c,mtlp);
    }else if(mtl==1.){
      basecol=vec3(0);
      speccol=vec3(.5);
      specpow=60.;
      
      vec2 size=vec2(.05,.2);
      vec2 pp=(march.rp-march.grid.c).xy;
      vec2 c=lofi(pp.xy,size)+size/2.;
      vec2 cc=pp-c;
      vec3 led=vec3(1);
      led*=exp(-60.*sdbox(cc,vec2(0.,.08)));
      led*=c.x>.5?vec3(5,1,2):vec3(1,5,2);
      float lv=texture(iChannel0,vec2(march.grid.h.z,0)).x*1.;
      col+=led*step(c.x,-.8+1.6*lv);
      basecol=.04*led;
    }else if(mtl==2.){//led
      basecol=vec3(0);
      speccol=vec3(1.);
      specpow=100.;
      
      col+=mtlp*vec3(2,.5,.5);
    }else if(mtl==3.){//metal
      basecol=vec3(.2);
      speccol=vec3(1.8);
      specpow=100.;
      ndelta=3E-2;
    }else if(mtl==4.){//kaoss
      basecol=vec3(0);
      speccol=vec3(.5);
      specpow=60.;
      
      vec2 size=vec2(.1);
      vec2 pp=(march.rp-march.grid.c).xy;
      vec2 c=lofi(pp.xy,size)+size/2.;
      vec2 cc=pp-c;
      vec3 led=vec3(1);
      led*=exp(-60.*sdbox(cc,vec2(0.,.0)));
      led*=vec3(2,1,2);
      float plasma=sin(length(c)*10.-10.*iTime+march.grid.h.z*.7);
      plasma+=sin(c.y*10.-7.*iTime);
      led*=.5+.5*sin(plasma);
      col+=2.*led;
      basecol=.04*led;
    }else if(mtl==5.){//808
      basecol=vec3(.9,mtlp,.02);
    }
    
    vec3 n=nmap(march.rp,march.grid,ndelta);
    vec3 v=-rd;
    
    {
      vec3 l=normalize(vec3(1,3,5));
      vec3 h=normalize(l+v);
      float dotnl=max(0.,dot(n,l));
      float dotnh=max(0.,dot(n,h));
      float shadow=step(1E-1,domarch(march.rp,l,30).isect.x);
      vec3 diff=basecol/PI;
      vec3 spec=speccol*pow(dotnh,specpow);
      col+=vec3(.5,.6,.7)*shadow*dotnl*(diff+spec);
    }
    {
      vec3 l=normalize(vec3(-1,-1,5));
      vec3 h=normalize(l+v);
      float dotnl=max(0.,dot(n,l));
      float dotnh=max(0.,dot(n,h));
      float shadow=step(1E-1,domarch(march.rp,l,30).isect.x);
      vec3 diff=basecol/PI;
      vec3 spec=speccol*pow(dotnh,specpow);
      col+=shadow*dotnl*(diff+spec);
    }
  }
  
  col=pow(col,vec3(.4545));
  col=smoothstep(vec3(0,-.1,-.2),vec3(1,1.1,1.2),col);
  fragColor = vec4(col,1.0);
}`,

    "Apollonian Gasket": `// Author: Rigel rui@gil.com
// licence: https://creativecommons.org/licenses/by/4.0/
// link: https://www.shadertoy.com/view/lljfRD

/*
This was a study on circles, inspired by this artwork
http://www.dailymail.co.uk/news/article-1236380/Worlds-largest-artwork-etched-desert-sand.html

and implemented with the help of this article
http://www.ams.org/samplings/feature-column/fcarc-kissing

The structure is called an apollonian packing (or gasket)
https://en.m.wikipedia.org/wiki/Apollonian_gasket

There is a lot of apollonians in shadertoy, but not many quite like the image above.
This one by klems is really cool. He uses a technique called a soddy circle. 
https://www.shadertoy.com/view/4s2czK

This shader uses another technique called a Descartes Configuration. 
The only thing that makes this technique interesting is that it can be generalized to higher dimensions.
*/


// a few utility functions
// a signed distance function for a rectangle
float sdfRect(vec2 uv, vec2 s) {vec2 auv = abs(uv); return max(auv.x-s.x,auv.y-s.y); }
// a signed distance function for a circle
float sdfCircle(vec2 uv, vec2 c, float r) { return length(uv-c)-r; }
// fills an sdf in 2d
float fill(float d, float s, float i) { return abs(smoothstep(0.,s,d) - i); }
// makes a stroke of an sdf at the zero boundary
float stroke(float d, float w, float s, float i) { return abs(smoothstep(0.,s,abs(d)-(w*.5)) - i); }
// a simple palette
vec3 pal(float d) { return .5*(cos(6.283*d*vec3(2.,2.,1.)+vec3(.0,1.4,.0))+1.);}
// 2d rotation matrix
mat2 uvRotate(float a) { return mat2(cos(a),sin(a),-sin(a),cos(a)); }
// seeded random number
float hash(vec2 s) { return fract(sin(dot(s,vec2(12.9898,78.2333)))*43758.5453123); }

// this is an algorithm to construct an apollonian packing with a descartes configuration
// remaps the plane to a circle at the origin and a specific radius. vec3(x,y,radius)
vec3 apollonian(vec2 uv) {
    // the algorithm is recursive and must start with a initial descartes configuration
    // each vec3 represents a circle with the form vec3(centerx, centery, 1./radius)
    // the signed inverse radius is also called the bend (refer to the article above)
    vec3 dec[4];
    // a DEC is a configuration of 4 circles tangent to each other
    // the easiest way to build the initial one it to construct a symetric Steiner Chain.
    // http://mathworld.wolfram.com/SteinerChain.html
	float a = 6.283/3.;
	float ra = 1.+sin(a*.5);
	float rb = 1.-sin(a*.5);
	dec[0] = vec3(0.,0.,-1./ra);
    float radius = .5*(ra-rb);
	float bend = 1./radius;
    for (int i=1; i<4; i++) {
        dec[i] = vec3(cos(float(i)*a),sin(float(i)*a),bend);
        // if the point is in one of the starting circles we have already found our solution
        if (length(uv-dec[i].xy) < radius) return vec3(uv-dec[i].xy,radius);
    }
    
    // Now that we have a starting DEC we are going to try to 
    // find the solution for the current point
    for(int i=0; i<7; i++) {
        // find the circle that is further away from the point uv, using euclidean distance
        int fi = 0;
        float d = distance(uv,dec[0].xy)-abs(1./dec[0].z);
        // for some reason, the euclidean distance doesn't work for the circle with negative bend
        // can anyone with proper math skills, explain me why? 
        d *= dec[0].z < 0. ? -.5 : 1.; // just scale it to make it work...
        for(int i=1; i<4; i++) {
            float fd = distance(uv,dec[i].xy)-abs(1./dec[i].z);
            fd *= dec[i].z < 0. ? -.5: 1.;
            if (fd>d) {fi = i;d=fd;}
        }
        // put the cicle found in the last slot, to generate a solution
        // in the "direction" of the point
        vec3 c = dec[3];
        dec[3] = dec[fi];
        dec[fi] = c;
        // generate a new solution
        float bend = (2.*(dec[0].z+dec[1].z+dec[2].z))-dec[3].z;
        vec2 center = vec2((2.*(dec[0].z*dec[0].xy
                               +dec[1].z*dec[1].xy
                               +dec[2].z*dec[2].xy)
                               -dec[3].z*dec[3].xy)/bend);

		vec3 solution = vec3(center,bend);
		// is the solution radius is to small, quit
		if (abs(1./bend) < 0.01) break;
		// if the solution contains the point return the circle
    	if (length(uv-solution.xy) < 1./bend) return vec3(uv-solution.xy,1./bend);
    	// else update the descartes configuration,
    	dec[3] = solution;
    	// and repeat...
	}
	// if nothing is found we return by default the inner circle of the Steiner chain
	return vec3(uv,rb);
}


vec3 scene(vec2 uv) {
    // remap uv to appolonian packing
    vec3 uvApo = apollonian(uv);
    
    float d = 6.2830/360.;
    float a = atan(uvApo.y,uvApo.x);
    float r = length(uvApo.xy);

    float circle = sdfCircle(uv,uv-uvApo.xy,uvApo.z);
	
    // background
	vec3 c = length(uv)*pal(.7)*.2;
    
    // drawing the clocks
    if (uvApo.z > .3) {
    	c = mix(c,pal(.75-r*.1)*.8,fill(circle+.02,.01,1.)); // clock 
    	c = mix(c,pal(.4+r*.1),stroke(circle+(uvApo.z*.03),uvApo.z*.01,.005,1.));// dial

        float h = stroke(mod(a+d*15.,d*30.)-d*15.,.02,0.01,1.);
    	c = mix(c,pal(.4+r*.1),h*stroke(circle+(uvApo.z*.16),uvApo.z*.25,.005,1.0));// hours

        float m = stroke(mod(a+d*15.,d*6.)-d*3.,.005,0.01,1.);
    	c = mix(c,pal(.45+r*.1),(1.-h)*m*stroke(circle+(uvApo.z*.15),uvApo.z*.1,.005,1.0));// minutes, 
    	
    	// needles rotation
    	vec2 uvrh = uvApo.xy*uvRotate(sign(cos(hash(vec2(uvApo.z))*d*180.))*d*iTime*(1./uvApo.z*10.)-d*90.);
    	vec2 uvrm = uvApo.xy*uvRotate(sign(cos(hash(vec2(uvApo.z)*4.)*d*180.))*d*iTime*(1./uvApo.z*120.)-d*90.);
    	// draw needles 
    	c = mix(c,pal(.85),stroke(sdfRect(uvrh+vec2(uvApo.z-(uvApo.z*.8),.0),uvApo.z*vec2(.4,.03)),uvApo.z*.01,0.005,1.));
    	c = mix(c,pal(.9),fill(sdfRect(uvrm+vec2(uvApo.z-(uvApo.z*.65),.0),uvApo.z*vec2(.5,.002)),0.005,1.));
    	c = mix(c,pal(.5+r*10.),fill(circle+uvApo.z-.02,0.005,1.)); // center
    // drawing the gears
    } else if (uvApo.z > .05) {
    	vec2 uvrg = uvApo.xy*uvRotate(sign(cos(hash(vec2(uvApo.z+2.))*d*180.))*d*iTime*(1./uvApo.z*20.));
        float g = stroke(mod(atan(uvrg.y,uvrg.x)+d*22.5,d*45.)-d*22.5,.3,.05,1.0);
        vec2 size = uvApo.z*vec2(.45,.08);
        c = mix(c,pal(.55-r*.6),fill(circle+g*(uvApo.z*.2)+.01,.001,1.)*fill(circle+(uvApo.z*.6),.005,.0));
        c = mix(c,pal(.55-r*.6),fill(min(sdfRect(uvrg,size.xy),sdfRect(uvrg,size.yx)),.005,1.));
    // drawing the screws
    } else { 
 	    vec2 size = uvApo.z * vec2(.5,.1);
 	    c = mix(c, pal(.85-(uvApo.z*2.)), fill(circle + 0.01,.007,1.));
 	    c = mix(c, pal(.8-(uvApo.z*3.)), fill(min(sdfRect(uvApo.xy,size.xy),sdfRect(uvApo.xy,size.yx)), .002, 1.));
    }
	return c;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
	vec2 uv = (fragCoord.xy - iResolution.xy * .5) / iResolution.y;
	fragColor = vec4(scene(uv*4.),1.0);
}`,
    "Loopy Noise": `// Another tuto by Etienne Jacob https://necessarydisorder.wordpress.com/2017/11/15/drawing-from-noise-and-then-making-animated-loopy-gifs-from-there/
// accurate & more faithful but slow variant of https://shadertoy.com/view/MsGyWK

// --- pseudo perlin noise 3D

int MOD = 1;  // type of Perlin noise
#define rot(a) mat2(cos(a),-sin(a),sin(a),cos(a))

#define hash31(p) fract(sin(dot(p,vec3(127.1,311.7, 74.7)))*43758.5453123)
float noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p); f = f*f*(3.-2.*f); // smoothstep

    float v= mix( mix( mix(hash31(i+vec3(0,0,0)),hash31(i+vec3(1,0,0)),f.x),
                       mix(hash31(i+vec3(0,1,0)),hash31(i+vec3(1,1,0)),f.x), f.y), 
                  mix( mix(hash31(i+vec3(0,0,1)),hash31(i+vec3(1,0,1)),f.x),
                       mix(hash31(i+vec3(0,1,1)),hash31(i+vec3(1,1,1)),f.x), f.y), f.z);
	return   MOD==0 ? v
	       : MOD==1 ? 2.*v-1.
           : MOD==2 ? abs(2.*v-1.)
                    : 1.-abs(2.*v-1.);
}

float fbm3(vec3 p) {
    p.xy *= rot(.37);
    p.yz *= rot(.37);
    return 2. * noise3(p);
}
// -------------------------------------

void mainImage( out vec4 O, vec2 U )
{
    vec2 R = iResolution.xy;
    U = ( U+U - R ) / R.y;
    O -= O;
    
    float t = 3.*iTime, K = 1.5, S = 2.;
    float edge = 11. / R.y;   // particle radius in px (was 3)
    float gridStep = 0.038;     // tighter spacing (was 0.05)
    
    for (float j = -1.; j < 1.; j += gridStep)
        for (float i = -1.; i < 1.; i += gridStep) {
            vec2 P = vec2(i,j);
            float k = K * max(1.-length(P),0.);          // displ amplitude
            vec2 d = U - P;
            if (k > 0. && dot(d, d) < k * k) {
                P += k * vec2( fbm3(vec3(S*P, t)), fbm3(vec3(S*P+15., t)) ) / S;
                P = smoothstep( edge, 0., abs(U-P) );
                O += P.x*P.y; 
            }
        }
    
}`,
    "Smaller Waterfall": `// CC0: Smaller Waterfall
//  Trying to minimize the previous waterfall shader a bit.
//  I probably missed something obvious as usual.

// This file is released under CC0 1.0 Universal (Public Domain Dedication).
// To the extent possible under law, mrange has waived all copyright
// and related or neighboring rights to this work.
// See <https://creativecommons.org/publicdomain/zero/1.0/> for details.

// Suggested by: moonlightoctopus
#define L length

void mainImage(out vec4 o, vec2 C) {
  float 
    i
  , z
  , T=.1*iTime+9.
  , d=T
  , j
  ;
  
  vec2
    r=iResolution.xy
  , P=(C+C-r)/r.x
  , Y=vec2(5e-3,1)
  ;
  
  vec4
    U=vec4(0,1,2,4)
  , O
  ;
  
  for(
    ;++i<39.&&d>1e-4
    ;z+=d=1.-sqrt(L(O*O))
  )
    O=z*normalize(vec4(P,2,0))-U.xwyx/4.5
    ;
  
  C=vec2(O.x,atan(O.z,O.y));
  O=vec4(4,16,99,0)/(1e3*dot(P=U.zy*P-r/r.x*U.xy,P)+6.);
  z=5e-4;
  for(
     r=L(fwidth(C))*U.yy
    ;++j<9.
    ;C.x+=Y.x/8.
  )
    i=fract(sin(dot(vec2(j,round(C/Y)),7.+U.xw)*73.))
  , P=C-(T+T*i)*U.xy
  , P-=round(P/Y)*Y
  , o=1.+sin(T+7.*fract(8663.*i)+U)
  , O+=dot(smoothstep(r,-r,vec2(L(max(P,-U.yx)),L(P)-z)-z),vec2(exp(19.*P.y),3))*o*o.w
  ;
  
  o=sqrt(tanh(O-.02*U.zwyy));
}`,
    "Warping Boxes": `// CC0: Warping boxes
//  I tinkered with the multi-level metaballs of yesterday
//  and used boxes instead of circles. This + random tinkering
//  turned out quite nice in the end IMHO.

#define TIME        iTime
#define RESOLUTION  iResolution
#define PI          3.141592654
#define TAU         (2.0*PI)
#define ROT(a)      mat2(cos(a), sin(a), -sin(a), cos(a))

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float box(vec2 p, vec2 b) {
  vec2 d = abs(p)-b;
  return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
vec2 mod2(inout vec2 p, vec2 size) {
  vec2 c = floor((p + size*0.5)/size);
  p = mod(p + size*0.5,size) - size*0.5;
  return c;
}

// License: Unknown, author: Hexler, found: Kodelife example Grid
float hash(vec2 uv) {
  return fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
}

// License: Unknown, author: Unknown, found: don't remember
float tanh_approx(float x) {
  //  Found this somewhere on the interwebs
  //  return tanh(x);
  float x2 = x*x;
  return clamp(x*(27.0 + x2)/(27.0+9.0*x2), -1.0, 1.0);
}

float dot2(vec2 p) {
  return dot(p, p);
}

vec2 df(vec2 p, float aa, out float h, out float sc) {
  vec2 pp = p;
  
  float sz = 2.0;
  
  float r = 0.0;
  
  for (int i = 0; i < 5; ++i) {
    vec2 nn = mod2(pp, vec2(sz));
    sz /= 3.0;
    float rr = hash(nn+123.4);
    r += rr;
    if (rr < 0.5) break;
  }
  
  float d0 = box(pp, vec2(1.45*sz-0.75*aa))-0.05*sz;
  float d1 = sqrt(sqrt(dot2(pp*pp)));
  h = fract(r);
  sc = sz;
  return vec2(d0, d1);
}

vec2 toSmith(vec2 p)  {
  // z = (p + 1)/(-p + 1)
  // (x,y) = ((1+x)*(1-x)-y*y,2y)/((1-x)*(1-x) + y*y)
  float d = (1.0 - p.x)*(1.0 - p.x) + p.y*p.y;
  float x = (1.0 + p.x)*(1.0 - p.x) - p.y*p.y;
  float y = 2.0*p.y;
  return vec2(x,y)/d;
}

vec2 fromSmith(vec2 p)  {
  // z = (p - 1)/(p + 1)
  // (x,y) = ((x+1)*(x-1)+y*y,2y)/((x+1)*(x+1) + y*y)
  float d = (p.x + 1.0)*(p.x + 1.0) + p.y*p.y;
  float x = (p.x + 1.0)*(p.x - 1.0) + p.y*p.y;
  float y = 2.0*p.y;
  return vec2(x,y)/d;
}

vec2 transform(vec2 p) {
  p *= 2.0;
  const mat2 rot0 = ROT(1.0);
  const mat2 rot1 = ROT(-2.0);
  vec2 off0 = 4.0*cos(vec2(1.0, sqrt(0.5))*0.23*TIME);
  vec2 off1 = 3.0*cos(vec2(1.0, sqrt(0.5))*0.13*TIME);
  vec2 sp0 = toSmith(p);
  vec2 sp1 = toSmith((p+off0)*rot0);
  vec2 sp2 = toSmith((p-off1)*rot1);
  vec2 pp = fromSmith(sp0+sp1-sp2);
  p = pp;
  p += 0.25*TIME;
  
  return p;
}

vec3 effect(vec2 p, vec2 np, vec2 pp) {
  p = transform(p);
  np = transform(np);
  float aa = distance(p, np)*sqrt(2.0); 

  float h = 0.0;
  float sc = 0.0;
  vec2 d2 = df(p, aa, h, sc);

  vec3 col = vec3(0.0);

  vec3 rgb = ((2.0/3.0)*(cos(TAU*h+vec3(0.0, 1.0, 2.0))+vec3(1.0))-d2.y/(3.0*sc));
  col = mix(col, rgb, smoothstep(aa, -aa, d2.x));
  
  const vec3 gcol1 = vec3(.5, 2.0, 3.0);
  col += gcol1*tanh_approx(0.025*aa);

  col = clamp(col, 0.0, 1.0);
  col = sqrt(col);
  
  return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
  vec2 q = fragCoord/RESOLUTION.xy;
  vec2 p = -1. + 2. * q;
  vec2 pp = p;
  p.x *= RESOLUTION.x/RESOLUTION.y;
  vec2 np = p+1.0/RESOLUTION.y;
  vec3 col = effect(p, np, pp);
  fragColor = vec4(col, 1.0);
}`,
    "Logarithmic Spirals": `// CC0: Logarithmic spirals tweaked
//  Been travelling and was messing around with this before travel
//  Looked better than I remembered so publishing it now.

#define TIME        iTime
#define RESOLUTION  iResolution
#define PI          3.141592654
#define TAU         (2.0*PI)
#define ROT(a)      mat2(cos(a), sin(a), -sin(a), cos(a))

const float ExpBy = log2(1.2);

float modPolar(inout vec2 p, float repetitions) {
  float angle = TAU/repetitions;
  float a = atan(p.y, p.x) + angle/2.;
  float r = length(p);
  float c = floor(a/angle);
  a = mod(a,angle) - angle/2.;
  p = vec2(cos(a), sin(a))*r;
  // For an odd number of repetitions, fix cell index of the cell in -x direction
  // (cell index would be e.g. -5 and 5 in the two halves of the cell):
  if (abs(c) >= (repetitions/2.0)) c = abs(c);
  return c;
}

float forward(float l) {
  return exp2(ExpBy*l);
}

float reverse(float l) {
  return log2(l)/ExpBy;
}

vec3 sphere(vec3 col, mat2 rot, vec3 bcol, vec2 p, float r, float aa) {
  vec3 lightDir = normalize(vec3(1.0, 1.5, 2.0));
  lightDir.xy *= rot;
  float z2 = (r*r-dot(p, p));
  vec3 rd = -normalize(vec3(p, 0.1));
  if (z2 > 0.0) {
    float z = sqrt(z2);
    vec3 cp = vec3(p, z);
    vec3 cn = normalize(cp);
    vec3 cr = reflect(rd, cn);
    float cd= max(dot(lightDir, cn), 0.0);
    vec3 cspe = pow(max(dot(lightDir, cr), 0.0), 10.0)*tanh(8.0*(bcol))*0.5;
    vec3 ccol = mix(0.2, 1.0, cd*cd)*bcol;
    ccol += cspe;
    float d = length(p)-r;
    col = mix(col, ccol, smoothstep(0.0, -aa, d));
  }
  return col;
}

vec2 toSmith(vec2 p)  {
  // z = (p + 1)/(-p + 1)
  // (x,y) = ((1+x)*(1-x)-y*y,2y)/((1-x)*(1-x) + y*y)
  float d = (1.0 - p.x)*(1.0 - p.x) + p.y*p.y;
  float x = (1.0 + p.x)*(1.0 - p.x) - p.y*p.y;
  float y = 2.0*p.y;
  return vec2(x,y)/d;
}

vec2 fromSmith(vec2 p)  {
  // z = (p - 1)/(p + 1)
  // (x,y) = ((x+1)*(x-1)+y*y,2y)/((x+1)*(x+1) + y*y)
  float d = (p.x + 1.0)*(p.x + 1.0) + p.y*p.y;
  float x = (p.x + 1.0)*(p.x - 1.0) + p.y*p.y;
  float y = 2.0*p.y;
  return vec2(x,y)/d;
}


vec2 transform(vec2 p) {
  vec2 sp0 = toSmith(p-0.);
  vec2 sp1 = toSmith(p+vec2(1.0)*ROT(0.12*TIME));
  vec2 sp2 = toSmith(p-vec2(1.0)*ROT(0.23*TIME));
  p = fromSmith(sp0+sp1-sp2);
  return p;
}

// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
vec3 sRGB(vec3 t) {
  return mix(1.055*pow(t, vec3(1./2.4)) - 0.055, 12.92*t, step(t, vec3(0.0031308)));
}

// License: Unknown, author: Matt Taylor (https://github.com/64), found: https://64.github.io/tonemapping/
vec3 aces_approx(vec3 v) {
  v = max(v, 0.0);
  v *= 0.6;
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp((v*(a*v+b))/(v*(c*v+d)+e), 0.0, 1.0);
}

vec3 effect(vec2 p, vec2 pp) {
//  float aa = 4.0/RESOLUTION.y;
  vec2 np = p + 1.0/RESOLUTION.y;
  vec2 tp = transform(p);
  vec2 ntp = transform(np);
  float aa = 2.0*distance(tp, ntp);
  p = tp;

  float ltm = 0.75*TIME;
  mat2 rot0 = ROT(-0.125*ltm); 
  p *= rot0;
  float mtm = fract(ltm);
  float ntm = floor(ltm);
  float gd = dot(p, p);
  float zz = forward(mtm);

  vec2 p0 = p;
  p0 /= zz;

  float l0 = length(p0);
  
  float n0 = ceil(reverse(l0));
  float r0 = forward(n0);
  float r1 = forward(n0-1.0);
  float r = (r0+r1)/2.0;
  float w = r0-r1;
  float nn = n0;
  n0 -= ntm;

  vec2 p1 = p0;
  float reps = floor(TAU*r/(w));
  mat2 rot1 = ROT(0.66*n0); 
  p1 *= rot1;
  float m1 = modPolar(p1, reps)/reps;
  p1.x -= r;
  
  vec3 ccol = (1.0+cos(0.85*vec3(0.0, 1.0, 2.0)+TAU*(m1)+0.5*n0))*0.5;
  vec3 gcol = (1.+cos(vec3(0.0, 1.0, 2.0) + 0.125*ltm))*0.01;
  mat2 rot2 = ROT(TAU*m1);

  vec3 col = vec3(0.0);
  float fade = 0.5+0.5*cos(TAU*m1+0.33*ltm);
  col = sphere(col, rot0*rot1*rot2, ccol*mix(0.25, 1.0, sqrt(fade)), p1, mix(0.125, 0.5, fade)*w, aa/zz);
  col *= 1.5;
  col += gcol/max(gd, 0.001);
  col += gcol*aa*10.0;
//  col -= 0.05*vec3(0.0, 1.0, 2.0).zyx*(length(pp)+0.25);
  col = aces_approx(col);
  col = sRGB(col);

  return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
  vec2 q = fragCoord/RESOLUTION.xy;
  vec2 p = -1. + 2. * q;
  vec2 pp = p;
  p.x *= RESOLUTION.x/RESOLUTION.y;
  vec3 col = effect(p, p);
  fragColor = vec4(col, 1.0);
}`,
    "Moving without Travelling": `// License CC0: Moving without travelling
#define PI              3.141592654
#define TAU             (2.0*PI)
#define TIME            iTime
#define TTIME           (TAU*TIME)
#define RESOLUTION      iResolution
#define ROT(a)          mat2(cos(a), sin(a), -sin(a), cos(a))
#define BPERIOD         5.6
#define PCOS(x)         (0.5+ 0.5*cos(x))
#define BPM             150.0

const vec4 hsv2rgb_K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www);
  return c.z * mix(hsv2rgb_K.xxx, clamp(p - hsv2rgb_K.xxx, 0.0, 1.0), c.y);
}
// Macro version of above to enable compile-time constants
#define HSV2RGB(c)  (c.z * mix(hsv2rgb_K.xxx, clamp(abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www) - hsv2rgb_K.xxx, 0.0, 1.0), c.y))

const vec3 std_gamma        = vec3(2.2);

float g_th = 0.0;
float g_hf = 0.0;

vec2 g_vx = vec2(0.0);
vec2 g_vy = vec2(0.0);

vec2 g_wx = vec2(0.0);
vec2 g_wy = vec2(0.0);

// https://iquilezles.org/articles/smin
float pmin(float a, float b, float k) {
  float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
  return mix( b, a, h ) - k*h*(1.0-h);
}

float pmax(float a, float b, float k) {
  return -pmin(-a, -b, k);
}

float pabs(float a, float k) {
  return -pmin(-a, a, k);
}

float hash(float co) {
  return fract(sin(co*12.9898) * 13758.5453);
}

vec4 alphaBlend(vec4 back, vec4 front) {
  float w = front.w + back.w*(1.0-front.w);
  vec3 xyz = (front.xyz*front.w + back.xyz*back.w*(1.0-front.w))/w;
  return w > 0.0 ? vec4(xyz, w) : vec4(0.0);
}

vec3 alphaBlend(vec3 back, vec4 front) {
  return mix(back, front.xyz, front.w);
}

float tanh_approx(float x) {
//  return tanh(x);
  float x2 = x*x;
  return clamp(x*(27.0 + x2)/(27.0+9.0*x2), -1.0, 1.0);
}

// https://mercury.sexy/hg_sdf/
float mod1(inout float p, float size) {
  float halfsize = size*0.5;
  float c = floor((p + halfsize)/size);
  p = mod(p + halfsize, size) - halfsize;
  return c;
}

vec2 toPolar(vec2 p) {
  return vec2(length(p), atan(p.y, p.x));
}

vec2 toRect(vec2 p) {
  return vec2(p.x*cos(p.y), p.x*sin(p.y));
}

float modMirror1(inout float p, float size) {
  float halfsize = size*0.5;
  float c = floor((p + halfsize)/size);
  p = mod(p + halfsize,size) - halfsize;
  p *= mod(c, 2.0)*2.0 - 1.0;
  return c;
}

float smoothKaleidoscope(inout vec2 p, float sm, float rep) {
  vec2 hp = p;

  vec2 hpp = toPolar(hp);
  float rn = modMirror1(hpp.y, TAU/rep);

  float sa = PI/rep - pabs(PI/rep - abs(hpp.y), sm);
  hpp.y = sign(hpp.y)*(sa);

  hp = toRect(hpp);

  p = hp;

  return rn;
}

float circle(vec2 p, float r) {
  return length(p) - r;
}

// Based on: https://iquilezles.org/articles/distfunctions2d
float vesica(vec2 p, vec2 sz) {
  if (sz.x < sz.y) {
    sz = sz.yx;
  } else {
    p  = p.yx; 
  }
  vec2 sz2 = sz*sz;
  float d  = (sz2.x-sz2.y)/(2.0*sz.y);
  float r  = sqrt(sz2.x+d*d);
  float b  = sz.x;
  p = abs(p);
  return ((p.y-b)*d>p.x*b) ? length(p-vec2(0.0,b))
                           : length(p-vec2(-d,0.0))-r;
}

float outerEye(vec2 p, float th) {
  float a  = mix(0.0, 1.0, smoothstep(0.995, 1.0, cos(th+TTIME/BPERIOD)));
  const float w = 1.14;
  float h = mix(0.48, 0.05, a);
  float d0 =  vesica(p, vec2(w, h));
  return d0;
}

const vec2 iris_center = vec2(0.0, 0.28);
vec4 completeEye(vec2 p, float th) {
  const float iris_outer = 0.622;
  const float iris_inner = 0.285;
  
  float t0 = abs(0.9*p.x);
  t0 *= t0;
  t0 *= t0;
  t0 *= t0;
  t0 = clamp(t0, 0.0, 1.0);
  float dt0 = mix(0.0125, -0.0025, t0);

  vec2 p0 = p;
  float d0 = outerEye(p, th);
  float d5 = d0;

  vec2 p1 = p;
  p1 -= iris_center;
  float d1 = circle(p1, iris_outer);
  d1 = max(d1,d0+dt0);
  float d6 = d1;

  vec2 p2 = p;
  p2 -= vec2(0.155, 0.35);
  float d2 = circle(p2, 0.065);

  vec2 p3 = p;
  p3 -= iris_center;
  p3 = toPolar(p3);
  float n3 = mod1(p3.x, 0.05);
  float d3 = abs(p3.x)-0.0125*(1.0-1.0*length(p1));

  vec2 p4 = p;
  p4 -= iris_center;
  float d4 = circle(p4, iris_inner);

  d3 = max(d3,-d4);

  d1 = pmax(d1,-d2, 0.0125);
  d1 = max(d1,-d3);

  d0 = abs(d0)-dt0;

  float d = d0;
  d = pmin(d, d1, 0.0125);
  return vec4(d, d6, d5, max(d4, d6));
}


// The path function
vec3 offset(float z) {
  float a = z;
  vec2 p = -0.1*(vec2(cos(a), sin(a*sqrt(2.0))) + vec2(cos(a*sqrt(0.75)), sin(a*sqrt(0.5))));
  return vec3(p, z);
}

// The derivate of the path function
//  Used to generate where we are looking
vec3 doffset(float z) {
  float eps = 0.1;
  return 0.5*(offset(z + eps) - offset(z - eps))/eps;
}

// The second derivate of the path function
//  Used to generate tilt
vec3 ddoffset(float z) {
  float eps = 0.1;
  return 0.125*(doffset(z + eps) - doffset(z - eps))/eps;
}

float noise(vec2 p) {
  float a = sin(p.x);
  float b = sin(p.y);
  float c = 0.5 + 0.5*cos(p.x + p.y);
  float d = mix(a, b, c);
  return d;
}

// https://iquilezles.org/articles/fbm
float fbm(vec2 p, float aa) {
  const mat2 frot = mat2(0.80, 0.60, -0.60, 0.80);

  float f = 0.0;
  float a = 1.0;
  float s = 0.0;
  float m = 2.0;
  for (int x = 0; x < 4; ++x) {
    f += a*noise(p); 
    p = frot*p*m;
    m += 0.01;
    s += a;
    a *= aa;
  }
  return f/s;
}

// https://iquilezles.org/articles/warp
float warp(vec2 p, out vec2 v, out vec2 w) {
  const float r  = 0.5;
  const float rr = 0.25;
  float l2 = length(p);
  float f  = 1.0;

  f = smoothstep(-0.1, 0.15, completeEye(p, g_th).x);
  const float rep = 50.0;
  const float sm = 0.125*0.5*60.0/rep;
  float  n = smoothKaleidoscope(p, sm, rep);
  p.y += TIME*0.125+1.5*g_th;

  g_hf = f;
  vec2 pp = p;

  vec2 vx = g_vx;
  vec2 vy = g_vy;

  vec2 wx = g_wx;
  vec2 wy = g_wy;


  //float aa = mix(0.95, 0.25, tanh_approx(pp.x));
  float aa = 0.5;

  v = vec2(fbm(p + vx, aa), fbm(p + vy, aa))*f;
  w = vec2(fbm(p + 3.0*v + wx, aa), fbm(p + 3.0*v + wy, aa))*f;
  
  return -tanh_approx(fbm(p + 2.25*w, aa)*f);
}

vec3 normal(vec2 p) {
  vec2 v;
  vec2 w;
  vec2 e = vec2(4.0/RESOLUTION.y, 0);
  
  vec3 n;
  n.x = warp(p + e.xy, v, w) - warp(p - e.xy, v, w);
  n.y = 2.0*e.x;
  n.z = warp(p + e.yx, v, w) - warp(p - e.yx, v, w);
  
  return normalize(n);
}

void compute_globals() {

  vec2 vx = vec2(0.0, 0.0);
  vec2 vy = vec2(3.2, 1.3);

  vec2 wx = vec2(1.7, 9.2);
  vec2 wy = vec2(8.3, 2.8);

  vx *= ROT(TTIME/1000.0);
  vy *= ROT(TTIME/900.0);

  wx *= ROT(TTIME/800.0);
  wy *= ROT(TTIME/700.0);
  
  g_vx = vx;
  g_vy = vy;
  
  g_wx = wx;
  g_wy = wy;
}

vec3 weird(vec2 p) {
  const vec3 up  = vec3(0.0, 1.0, 0.0);
  const vec3 lp1 = 1.0*vec3(1.0, 1.25, 1.0);
  const vec3 lp2 = 1.0*vec3(-1.0, 2.5, 1.0);

  vec3 ro = vec3(0.0, 10.0, 0.0);
  vec3 pp = vec3(p.x, 0.0, p.y);

  vec2 v;
  vec2 w;
 
  float h  = warp(p, v, w);
  float hf = g_hf;
  vec3  n  = normal(p);

  vec3 lcol1 = hsv2rgb(vec3(0.7, 0.5, 1.0)); 
  vec3 lcol2 = hsv2rgb(vec3(0.4, 0.5, 1.0));
  vec3 po  = vec3(p.x, 0.0, p.y);
  vec3 rd  = normalize(po - ro);
  
  vec3 ld1 = normalize(lp1 - po);
  vec3 ld2 = normalize(lp2 - po);
 
  float diff1 = max(dot(n, ld1), 0.0);
  float diff2 = max(dot(n, ld2), 0.0);

  vec3  ref   = reflect(rd, n);
  float ref1  = max(dot(ref, ld1), 0.0);
  float ref2  = max(dot(ref, ld2), 0.0);

  const vec3 col1 = vec3(0.1, 0.7, 0.8).xzy;
  const vec3 col2 = vec3(0.7, 0.3, 0.5).zyx;
  
  float a = length(p);
  vec3 col = vec3(0.0);
//  col -= 0.5*hsv2rgb(vec3(fract(0.3*TIME+0.25*a+0.5*v.x), 0.85, abs(tanh_approx(v.y))));
//  col -= 0.5*hsv2rgb(vec3(fract(-0.5*TIME+0.25*a+0.125*w.x), 0.85, abs(tanh_approx(w.y))));
  col += hsv2rgb(vec3(fract(-0.1*TIME+0.125*a+0.5*v.x+0.125*w.x), abs(0.5+tanh_approx(v.y*w.y)), tanh_approx(0.1+abs(v.y-w.y))));
  col -= 0.5*(length(v)*col1 + length(w)*col2*1.0);
  /*
  col += 0.25*diff1;
  col += 0.25*diff2;
  */
  col += 0.5*lcol1*pow(ref1, 20.0);
  col += 0.5*lcol2*pow(ref2, 10.0);
  col *= hf;

  return col;
}

vec4 plane3(vec3 ro, vec3 rd, vec3 pp, vec3 off, float aa, float n) {
  float h = hash(n+1234.4);
  float th = TAU*h;
  g_th = th;
  float s = 1.*mix(0.2, 0.3, h);

  vec3 hn;
  vec2 p = (pp-off*vec3(1.0, 1.0, 0.0)).xy;
  p *= ROT(0.2*mix(-1.0, 1.0, h));
  p /= s;
  float lp = length(p); 
  p -= -iris_center;
  const float lw = 0.005;
  vec4 de = completeEye(p, th)*s;
  float ax = smoothstep(-aa, aa, de.x);
  float ay = smoothstep(-aa, aa, de.y);
  float az = smoothstep(-aa, aa, -de.z);
  float aw = smoothstep(-aa, aa, 0.0125*(de.w+0.025));

  float df = 1.0-tanh_approx(0.5*lp);
  vec3 acol = vec3(df);
  vec3 icol = weird(p);
  vec3 ecol = mix(vec3(0.0), vec3(1.0), ax);
  vec3 bcol = mix(icol, ecol, az*0.5*df);
  vec4 col = vec4(bcol, aw);

  return col;
}

vec4 plane(vec3 ro, vec3 rd, vec3 pp, vec3 off, float aa, float n) {
  return plane3(ro, rd, pp, off, aa, n);
}


vec3 skyColor(vec3 ro, vec3 rd) {
  float ld = max(dot(rd, vec3(0.0, 0.0, 1.0)), 0.0);
  vec3 baseCol = 1.0*vec3(2.0, 1.0, 3.0)*(pow(ld, 100.0));
  return vec3(baseCol);
}

vec3 color(vec3 ww, vec3 uu, vec3 vv, vec3 ro, vec2 p) {
  float lp = length(p);
  vec2 np = p + 1.0/RESOLUTION.xy;
  const float per = 10.0;
  float rdd = (1.0+0.5*lp*tanh_approx(lp+0.9*PCOS(per*p.x)*PCOS(per*p.y)));
  vec3 rd = normalize(p.x*uu + p.y*vv + rdd*ww);
  vec3 nrd = normalize(np.x*uu + np.y*vv + rdd*ww);

  const float planeDist = 1.0-0.0;
  const int furthest = 4;
  const int fadeFrom = max(furthest-3, 0);
  const float fadeDist = planeDist*float(furthest - fadeFrom);
  float nz = floor(ro.z / planeDist);

  vec3 skyCol = skyColor(ro, rd);

  // Steps from nearest to furthest plane and accumulates the color

  vec4 acol = vec4(0.0);
  const float cutOff = 0.95;
  bool cutOut = false;
  
  for (int i = 1; i <= furthest; ++i) {
    float pz = planeDist*nz + planeDist*float(i);

    float pd = (pz - ro.z)/rd.z;

    if (pd > 0.0 && acol.w < cutOff) {
      vec3 pp = ro + rd*pd;
      vec3 npp = ro + nrd*pd;

      float aa = 3.0*length(pp - npp);

      vec3 off = offset(pp.z);

      vec4 pcol = plane(ro, rd, pp, off, aa, nz+float(i));

      float nz = pp.z-ro.z;
      float fadeIn = exp(-2.5*max((nz - planeDist*float(fadeFrom))/fadeDist, 0.0));
      float fadeOut = smoothstep(0.0, planeDist*0.1, nz);
      pcol.xyz = mix(skyCol, pcol.xyz, (fadeIn));
      pcol.w *= fadeOut;

      pcol = clamp(pcol, 0.0, 1.0);

      acol = alphaBlend(pcol, acol);
    } else {
      cutOut = true;
      break;
    }

  }

  vec3 col = alphaBlend(skyCol, acol);
// To debug cutouts due to transparency  
//  col += cutOut ? vec3(1.0, -1.0, 0.0) : vec3(0.0);
  return col;
}

// Classic post processing
vec3 postProcess(vec3 col, vec2 q) {
  col = clamp(col, 0.0, 1.0);
  col = pow(col, 1.0/std_gamma);
  col = col*0.6+0.4*col*col*(3.0-2.0*col);
  col = mix(col, vec3(dot(col, vec3(0.33))), -0.4);
  col *=0.5+0.5*pow(19.0*q.x*q.y*(1.0-q.x)*(1.0-q.y),0.7);
  return col;
}

vec3 effect(vec2 p, vec2 q) {
  compute_globals();
  
  float tm  = TIME*0.5*BPM/60.0;
  vec3 ro   = offset(tm);
  vec3 dro  = doffset(tm);
  vec3 ddro = ddoffset(tm);

  vec3 ww = normalize(dro);
  vec3 uu = normalize(cross(normalize(vec3(0.0,1.0,0.0)+ddro), ww));
  vec3 vv = normalize(cross(ww, uu));

  vec3 col = color(ww, uu, vv, ro, p);
  return col;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 q = fragCoord/RESOLUTION.xy;
  vec2 p = -1. + 2. * q;
  p.x *= RESOLUTION.x/RESOLUTION.y;

  vec3 col = effect(p, q);
  col += smoothstep(3.0, 0.0, TIME);
  col = postProcess(col, q);

  fragColor = vec4(col, 1.0);
}`,
    "Apollian with a Twist": `// License CC0: Apollian with a twist
// Playing around with apollian fractal

#define TIME            iTime
#define RESOLUTION      iResolution
#define PI              3.141592654
#define TAU             (2.0*PI)
#define L2(x)           dot(x, x)
#define ROT(a)          mat2(cos(a), sin(a), -sin(a), cos(a))
#define PSIN(x)         (0.5+0.5*sin(x))

vec3 hsv2rgb(vec3 c) {
  const vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float apollian(vec4 p, float s) {
  float scale = 1.0;

  for(int i=0; i<7; ++i) {
    p        = -1.0 + 2.0*fract(0.5*p+0.5);

    float r2 = dot(p,p);
    
    float k  = s/r2;
    p       *= k;
    scale   *= k;
  }
  
  return abs(p.y)/scale;
}

float weird(vec2 p) {
  float z = 4.0;
  p *= ROT(TIME*0.1);
  float tm = 0.2*TIME;
  float r = 0.5;
  vec4 off = vec4(r*PSIN(tm*sqrt(3.0)), r*PSIN(tm*sqrt(1.5)), r*PSIN(tm*sqrt(2.0)), 0.0);
  vec4 pp = vec4(p.x, p.y, 0.0, 0.0)+off;
  pp.w = 0.125*(1.0-tanh(length(pp.xyz)));
  pp.yz *= ROT(tm);
  pp.xz *= ROT(tm*sqrt(0.5));
  pp /= z;
  float d = apollian(pp, 1.2);
  return d*z;
}

float df(vec2 p) {
  const float zoom = 0.5;
  p /= zoom;
  float d0 = weird(p);
  return d0*zoom;
}

vec3 color(vec2 p) {
  float aa   = 2.0/RESOLUTION.y;
  const float lw = 0.0235;
  const float lh = 1.25;

  const vec3 lp1 = vec3(0.5, lh, 0.5);
  const vec3 lp2 = vec3(-0.5, lh, 0.5);

  float d = df(p);

  float b = -0.125;
  float t = 10.0;

  vec3 ro = vec3(0.0, t, 0.0);
  vec3 pp = vec3(p.x, 0.0, p.y);

  vec3 rd = normalize(pp - ro);

  vec3 ld1 = normalize(lp1 - pp);
  vec3 ld2 = normalize(lp2 - pp);

  float bt = -(t-b)/rd.y;
  
  vec3 bp   = ro + bt*rd;
  vec3 srd1 = normalize(lp1-bp);
  vec3 srd2 = normalize(lp2-bp);
  float bl21= L2(lp1-bp);
  float bl22= L2(lp2-bp);

  float st1= (0.0-b)/srd1.y;
  float st2= (0.0-b)/srd2.y;
  vec3 sp1 = bp + srd1*st1;
  vec3 sp2 = bp + srd2*st1;

  float bd = df(bp.xz);
  float sd1= df(sp1.xz);
  float sd2= df(sp2.xz);

  vec3 col  = vec3(0.0);
  const float ss =15.0;
  
  col       += vec3(1.0, 1.0, 1.0)*(1.0-exp(-ss*(max((sd1+0.0*lw), 0.0))))/bl21;
  col       += vec3(0.5)*(1.0-exp(-ss*(max((sd2+0.0*lw), 0.0))))/bl22;
  float l   = length(p);
  float hue = fract(0.75*l-0.3*TIME)+0.3+0.15;
  float sat = 0.75*tanh(2.0*l);
  vec3 hsv  = vec3(hue, sat, 1.0);
  vec3 bcol = hsv2rgb(hsv);
  col       *= (1.0-tanh(0.75*l))*0.5;
  col       = mix(col, bcol, smoothstep(-aa, aa, -d));  
  col       += 0.5*sqrt(bcol.zxy)*(exp(-(10.0+100.0*tanh(l))*max(d, 0.0)));

  return col;
}

vec3 postProcess(vec3 col, vec2 q)  {
  col=pow(clamp(col,0.0,1.0),vec3(1.0/2.2)); 
  col=col*0.6+0.4*col*col*(3.0-2.0*col);  // contrast
  col=mix(col, vec3(dot(col, vec3(0.33))), -0.4);  // saturation
  col*=0.5+0.5*pow(19.0*q.x*q.y*(1.0-q.x)*(1.0-q.y),0.7);  // vigneting
  return col;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 q = fragCoord/RESOLUTION.xy;
  vec2 p = -1. + 2. * q;
  p.x *= RESOLUTION.x/RESOLUTION.y;
  
  vec3 col = color(p);
  col = postProcess(col, q);
  
  fragColor = vec4(col, 1.0);
}`,
    "Stars and Galaxy": `// License CC0: Stars and galaxy
// Bit of sunday tinkering lead to stars and a galaxy
// Didn't turn out as I envisioned but it turned out to something
// that I liked so sharing it.

// Controls how many layers of stars
#define LAYERS            5.0

#define PI                3.141592654
#define TAU               (2.0*PI)
#define TIME              mod(iTime, 30.0)
#define TTIME             (TAU*TIME)
#define RESOLUTION        iResolution
#define ROT(a)            mat2(cos(a), sin(a), -sin(a), cos(a))

// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
float sRGB(float t) { return mix(1.055*pow(t, 1./2.4) - 0.055, 12.92*t, step(t, 0.0031308)); }
// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
vec3 sRGB(in vec3 c) { return vec3 (sRGB(c.x), sRGB(c.y), sRGB(c.z)); }

// License: Unknown, author: Matt Taylor (https://github.com/64), found: https://64.github.io/tonemapping/
vec3 aces_approx(vec3 v) {
  v = max(v, 0.0);
  v *= 0.6;
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp((v*(a*v+b))/(v*(c*v+d)+e), 0.0, 1.0);
}

// License: Unknown, author: Unknown, found: don't remember
float tanh_approx(float x) {
  //  Found this somewhere on the interwebs
  //  return tanh(x);
  float x2 = x*x;
  return clamp(x*(27.0 + x2)/(27.0+9.0*x2), -1.0, 1.0);
}


// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
const vec4 hsv2rgb_K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www);
  return c.z * mix(hsv2rgb_K.xxx, clamp(p - hsv2rgb_K.xxx, 0.0, 1.0), c.y);
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
vec2 mod2(inout vec2 p, vec2 size) {
  vec2 c = floor((p + size*0.5)/size);
  p = mod(p + size*0.5,size) - size*0.5;
  return c;
}

// License: Unknown, author: Unknown, found: don't remember
vec2 hash2(vec2 p) {
  p = vec2(dot (p, vec2 (127.1, 311.7)), dot (p, vec2 (269.5, 183.3)));
  return fract(sin(p)*43758.5453123);
}

vec2 shash2(vec2 p) {
  return -1.0+2.0*hash2(p);
}

vec3 toSpherical(vec3 p) {
  float r   = length(p);
  float t   = acos(p.z/r);
  float ph  = atan(p.y, p.x);
  return vec3(r, t, ph);
}


// License: CC BY-NC-SA 3.0, author: Stephane Cuillerdier - Aiekick/2015 (twitter:@aiekick), found: https://www.shadertoy.com/view/Mt3GW2
vec3 blackbody(float Temp) {
  vec3 col = vec3(255.);
  col.x = 56100000. * pow(Temp,(-3. / 2.)) + 148.;
  col.y = 100.04 * log(Temp) - 623.6;
  if (Temp > 6500.) col.y = 35200000. * pow(Temp,(-3. / 2.)) + 184.;
  col.z = 194.18 * log(Temp) - 1448.6;
  col = clamp(col, 0., 255.)/255.;
  if (Temp < 1000.) col *= Temp/1000.;
  return col;
}


// License: MIT, author: Inigo Quilez, found: https://www.shadertoy.com/view/XslGRr
float noise(vec2 p) {
  // Found at https://www.shadertoy.com/view/sdlXWX
  // Which then redirected to IQ shader
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f*f*(3.-2.*f);
  
  float n =
         mix( mix( dot(shash2(i + vec2(0.,0.) ), f - vec2(0.,0.)), 
                   dot(shash2(i + vec2(1.,0.) ), f - vec2(1.,0.)), u.x),
              mix( dot(shash2(i + vec2(0.,1.) ), f - vec2(0.,1.)), 
                   dot(shash2(i + vec2(1.,1.) ), f - vec2(1.,1.)), u.x), u.y);

  return 2.0*n;              
}

float fbm(vec2 p, float o, float s, int iters) {
  p *= s;
  p += o;

  const float aa = 0.5;
  const mat2 pp = 2.04*ROT(1.0);

  float h = 0.0;
  float a = 1.0;
  float d = 0.0;
  for (int i = 0; i < iters; ++i) {
    d += a;
    h += a*noise(p);
    p += vec2(10.7, 8.3);
    p *= pp;
    a *= aa;
  }
  h /= d;
  
  return h;
}

float height(vec2 p) {
  float h = fbm(p, 0.0, 5.0, 5);
  h *= 0.3;
  h += 0.0;
  return (h);
}

vec3 stars(vec3 ro, vec3 rd, vec2 sp, float hh) {
  vec3 col = vec3(0.0);
  
  const float m = LAYERS;
  hh = tanh_approx(20.0*hh);

  for (float i = 0.0; i < m; ++i) {
    vec2 pp = sp+0.5*i;
    float s = i/(m-1.0);
    vec2 dim  = vec2(mix(0.05, 0.003, s)*PI);
    vec2 np = mod2(pp, dim);
    vec2 h = hash2(np+127.0+i);
    vec2 o = -1.0+2.0*h;
    float y = sin(sp.x);
    pp += o*dim*0.5;
    pp.y *= y;
    float l = length(pp);
  
    float h1 = fract(h.x*1667.0);
    float h2 = fract(h.x*1887.0);
    float h3 = fract(h.x*2997.0);

    vec3 scol = mix(8.0*h2, 0.25*h2*h2, s)*blackbody(mix(3000.0, 22000.0, h1*h1));

    vec3 ccol = col + exp(-(mix(6000.0, 2000.0, hh)/mix(2.0, 0.25, s))*max(l-0.001, 0.0))*scol;
    col = h3 < y ? ccol : col;
  }
  
  return col;
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/articles/spherefunctions
vec2 raySphere(vec3 ro, vec3 rd, vec4 sph) {
  vec3 oc = ro - sph.xyz;
  float b = dot( oc, rd );
  float c = dot( oc, oc ) - sph.w*sph.w;
  float h = b*b - c;
  if( h<0.0 ) return vec2(-1.0);
  h = sqrt( h );
  return vec2(-b - h, -b + h);
}


vec4 moon(vec3 ro, vec3 rd, vec2 sp, vec3 lp, vec4 md) {
  vec2 mi = raySphere(ro, rd, md);
  
  vec3 p    = ro + mi.x*rd;
  vec3 n    = normalize(p-md.xyz);
  vec3 r    = reflect(rd, n);
  vec3 ld   = normalize(lp - p);
  float fre = dot(n, rd)+1.0;
  fre = pow(fre, 15.0);
  float dif = max(dot(ld, n), 0.0);
  float spe = pow(max(dot(ld, r), 0.0), 8.0);
  float i = 0.5*tanh_approx(20.0*fre*spe+0.05*dif);
  vec3 col = blackbody(1500.0)*i+hsv2rgb(vec3(0.6, mix(0.6, 0.0, i), i));

  float t = tanh_approx(0.25*(mi.y-mi.x));
 
  return vec4(vec3(col), t);
}

vec3 sky(vec3 ro, vec3 rd, vec2 sp, vec3 lp, out float cf) {
  float ld = max(dot(normalize(lp-ro), rd),0.0);
  float y = -0.5+sp.x/PI;
  y = max(abs(y)-0.02, 0.0)+0.1*smoothstep(0.5, PI, abs(sp.y));
  vec3 blue = hsv2rgb(vec3(0.6, 0.75, 0.35*exp(-15.0*y)));
  float ci = pow(ld, 10.0)*2.0*exp(-25.0*y); 
  vec3 yellow = blackbody(1500.0)*ci;
  cf = ci;
  return blue+yellow;
}

vec3 galaxy(vec3 ro, vec3 rd, vec2 sp, out float sf) {
  vec2 gp = sp;
  gp *= ROT(0.67);
  gp += vec2(-1.0, 0.5);
  float h1 = height(2.0*sp);
  float gcc = dot(gp, gp);
  float gcx = exp(-(abs(3.0*(gp.x))));
  float gcy = exp(-abs(10.0*(gp.y)));
  float gh = gcy*gcx;
  float cf = smoothstep(0.05, -0.2, -h1);
  vec3 col = vec3(0.0);
  col += blackbody(mix(300.0, 1500.0, gcx*gcy))*gcy*gcx;
  col += hsv2rgb(vec3(0.6, 0.5, 0.00125/gcc));
  col *= mix(mix(0.15, 1.0, gcy*gcx), 1.0, cf);
  sf = gh*cf;
  return col;
}

vec3 grid(vec3 ro, vec3 rd, vec2 sp) {
  const float m = 1.0;

  const vec2 dim = vec2(1.0/8.0*PI);
  vec2 pp = sp;
  vec2 np = mod2(pp, dim);

  vec3 col = vec3(0.0);

  float y = sin(sp.x);
  float d = min(abs(pp.x), abs(pp.y*y));
  
  float aa = 2.0/RESOLUTION.y;
  
  col += 2.0*vec3(0.5, 0.5, 1.0)*exp(-2000.0*max(d-0.00025, 0.0));
  
  return 0.25*tanh(col);
}

vec3 color(vec3 ro, vec3 rd, vec3 lp, vec4 md) {
  vec2 sp = toSpherical(rd.xzy).yz;

  float sf = 0.0;
  float cf = 0.0;
  vec3 col = vec3(0.0);

  vec4 mcol = moon(ro, rd, sp, lp, md);

  col += stars(ro, rd, sp, sf)*(1.0-tanh_approx(2.0*cf));
  col += galaxy(ro, rd, sp, sf);
  col = mix(col, mcol.xyz, mcol.w);
  col += sky(ro, rd, sp, lp, cf);
  col += grid(ro, rd, sp);

  if (rd.y < 0.0)
  {
    col = vec3(0.0);
  }

  return col;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 q = fragCoord/iResolution.xy;
  vec2 p = -1.0 + 2.0*q;
  p.x *= RESOLUTION.x/RESOLUTION.y;

  vec3 ro = vec3(0.0, 0.0, 0.0);
  vec3 lp = 500.0*vec3(1.0, -0.25, 0.0);
  vec4 md = 50.0*vec4(vec3(1.0, 1., -0.6), 0.5);
  vec3 la = vec3(1.0, 0.5, 0.0);
  vec3 up = vec3(0.0, 1.0, 0.0);
  la.xz *= ROT(TTIME/60.0-PI/2.0);
  
  vec3 ww = normalize(la - ro);
  vec3 uu = normalize(cross(up, ww));
  vec3 vv = normalize(cross(ww,uu));
  vec3 rd = normalize(p.x*uu + p.y*vv + 2.0*ww);
  vec3 col= color(ro, rd, lp, md);
  
  col *= smoothstep(0.0, 4.0, TIME)*smoothstep(30.0, 26.0, TIME);
  col = aces_approx(col);
  col = sRGB(col);

  fragColor = vec4(col,1.0);
}`,
    "Mandala": `#define PI  3.141592654
#define TAU (2.0*PI)

vec3 saturate(vec3 col) {
  return clamp(col, 0.0, 1.0);
}


void rot(inout vec2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  p = vec2(c*p.x + s*p.y, -s*p.x + c*p.y);
}

vec2 mod2(inout vec2 p, vec2 size)  {
  vec2 c = floor((p + size*0.5)/size);
  p = mod(p + size*0.5,size) - size*0.5;
  return c;
}

vec2 modMirror2(inout vec2 p, vec2 size) {
  vec2 halfsize = size*0.5;
  vec2 c = floor((p + halfsize)/size);
  p = mod(p + halfsize, size) - halfsize;
  p *= mod(c,vec2(2.0))*2.0 - vec2(1.0);
  return c;
}


vec2 toSmith(vec2 p)  {
  // z = (p + 1)/(-p + 1)
  // (x,y) = ((1+x)*(1-x)-y*y,2y)/((1-x)*(1-x) + y*y)
  float d = (1.0 - p.x)*(1.0 - p.x) + p.y*p.y;
  float x = (1.0 + p.x)*(1.0 - p.x) - p.y*p.y;
  float y = 2.0*p.y;
  return vec2(x,y)/d;
}

vec2 fromSmith(vec2 p)  {
  // z = (p - 1)/(p + 1)
  // (x,y) = ((x+1)*(x-1)+y*y,2y)/((x+1)*(x+1) + y*y)
  float d = (p.x + 1.0)*(p.x + 1.0) + p.y*p.y;
  float x = (p.x + 1.0)*(p.x - 1.0) + p.y*p.y;
  float y = 2.0*p.y;
  return vec2(x,y)/d;
}

vec2 toRect(vec2 p) {
  return vec2(p.x*cos(p.y), p.x*sin(p.y));
}

vec2 toPolar(vec2 p) {
  return vec2(length(p), atan(p.y, p.x));
}

float box(vec2 p, vec2 b) {
  vec2 d = abs(p)-b;
  return length(max(d,vec2(0))) + min(max(d.x,d.y),0.0);
}

float circle(vec2 p, float r) {
  return length(p) - r;
}

float mandala_soundAmp() {
  float bass = texture(iChannel0, vec2(0.05, 0.0)).x;
  float mid  = texture(iChannel0, vec2(0.25, 0.0)).x;
  float high = texture(iChannel0, vec2(0.70, 0.0)).x;
  float amp = bass * 0.5 + mid * 0.35 + high * 0.15;
  return clamp(pow(amp, 0.05) * 1.35, 0.0, 1.0);
}

float mandala_df(float localTime, vec2 p) {
  vec2 pp = toPolar(p);
  float a = TAU/64.0;
  float np = pp.y/a;
  pp.y = mod(pp.y, a);
  float m2 = mod(np, 2.0);
  if (m2 > 1.0) {
    pp.y = a - pp.y;
  }
  pp.y += localTime/40.0;
  p = toRect(pp);
  p = abs(p);
  p -= vec2(0.5);
  
  float d = 10000.0;
  
  for (int i = 0; i < 4; ++i) {
    mod2(p, vec2(1.0));
    float da = -0.2 * cos(localTime*0.25);
    float sb = box(p, vec2(0.35)) + da ;
    float cb = circle(p + vec2(0.2), 0.25) + da;
    
    float dd = max(sb, -cb);
    d = min(dd, d);
    
    p *= 1.5 + 1.0*(0.5 + 0.5*sin(0.5*localTime));
    rot(p, 1.0);
  }

  
  return d;
}

vec3 mandala_postProcess(float localTime, vec3 col, vec2 uv) 
{
  float r = length(uv);
  float a = atan(uv.y, uv.x);
  col = clamp(col, 0.0, 1.0);   
  col=pow(col,mix(vec3(0.5, 0.75, 1.5), vec3(0.45), r)); 
  col=col*0.6+0.4*col*col*(3.0-2.0*col);  // contrast
  col=mix(col, vec3(dot(col, vec3(0.33))), -0.4);  // satuation
  col*=sqrt(1.0 - sin(-localTime + (50.0 - 25.0*sqrt(r))*r))*(1.0 - sin(0.5*r));
  col = clamp(col, 0.0, 1.0);
  float ff = pow(1.0-0.75*sin(20.0*(0.5*a + r + -0.1*localTime)), 0.75);
  col = pow(col, vec3(ff*0.9, 0.8*ff, 0.7*ff));
  col *= 0.5*sqrt(max(4.0 - r*r, 0.0));
  return clamp(col, 0.0, 1.0);
}

vec2 mandala_distort(float localTime, vec2 uv) {
  float lt = 0.1*localTime;
  vec2 suv = toSmith(uv);
  suv += 1.0*vec2(cos(lt), sin(sqrt(2.0)*lt));
//  suv *= vec2(1.5 + 1.0*sin(sqrt(2.0)*time), 1.5 + 1.0*sin(time));
  uv = fromSmith(suv);
  modMirror2(uv, vec2(2.0+sin(lt)));
  return uv;
}

vec3 mandala_sample(float localTime, vec2 p)
{
  float lt = 0.1*localTime;
  vec2 uv = p;
  uv *=8.0;
  rot(uv, lt);
  //uv *= 0.2 + 1.1 - 1.1*cos(0.1*iTime);

  vec2 nuv = mandala_distort(localTime, uv);
  vec2 nuv2 = mandala_distort(localTime, uv + vec2(0.0001));

  float nl = length(nuv - nuv2);
  float nf = 1.0 - smoothstep(0.0, 0.002, nl);

  uv = nuv;
  
  float d = mandala_df(localTime, uv);

  vec3 col = vec3(0.0);
 
  const float r = 0.065;

  float nd = d / r;
  float md = mod(d, r);
  
  if (abs(md) < 0.025) {
    col = (d > 0.0 ? vec3(0.25, 0.65, 0.25) : vec3(0.65, 0.25, 0.65) )/abs(nd);
  }

  if (abs(d) < 0.0125) {
    col = vec3(1.0);
  }

  col += 1.0 - pow(nf, 5.0);
  
  col = mandala_postProcess(localTime, col, uv);;
  
  col += 1.0 - pow(nf, 1.0);

  return saturate(col);
}

vec3 mandala_main(vec2 p) {

  float localTime = iTime + 30.0;
  vec3 col  = vec3(0.0);
  vec2 unit = 1.0/iResolution.xy;
  const int aa = 2;
  for(int y = 0; y < aa; ++y)
  {
    for(int x = 0; x < aa; ++x)
    {
      col += mandala_sample(localTime, p - 0.5*unit + unit*vec2(x, y));
    }
  }

  col /= float(aa*aa);
  return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
  float time = 0.1*iTime;
  vec2 uv = fragCoord/iResolution.xy - vec2(0.5);
  uv.x *= iResolution.x/iResolution.y;

  vec3 col = mandala_main(uv);
  col *= 0.55 + 0.75 * mandala_soundAmp();
    
  fragColor = vec4(col, 1.0);
    
}`,
    "Kali Sky": `// More Kali-de explorations +1
// orgiginally https://www.shadertoy.com/view/MtlGR2
// License aGPL v3
// 2015, stefan berke 
// credits to eiffie and kali


// http://www.musicdsp.org/showone.php?id=238
float Tanh(in float x) { return clamp(x * ( 27. + x * x ) / ( 27. + 9. * x * x ), -1., 1.); }

// two different traps and colorings
#define mph (.5 + .5 * Tanh(sin(iTime/17.123+1.2)*4.))


vec3 kali_sky(in vec3 pos, in vec3 dir)
{
    float time = iTime;
    
	vec4 col = vec4(0,0,0,1);
	
	float t = 0., pln;
    for (int k=0; k<50; ++k)
	{
		vec4 p = vec4(pos + t * dir, 1.);

		vec3 param = mix(
            vec3(1.2+.4*sin(time/6.13-.4)*min(1.,(time-70.)/10.), .5, 0.09+0.08*sin(time/4.)),
			vec3(.51, .5, 1.+0.5*sin(iTime/40.)), mph);

        // "kali-set" by Kali
		float d = 10.; pln=16.;
        vec3 av = vec3(0.);
		for (int i=0; i<11; ++i)
		{
            p = abs(p) / dot(p.xyz, p.xyz);
            // distance to prism/cylinder
            d = min(d, mix(p.x+p.y+p.z, length(p.xy), mph) / p.w);
            // disc
            if (i == 1)	pln = min(pln, dot(p.xyz, vec3(0,0,1)) / p.w);
			av += p.xyz/(4.+p.w);
            p.xyz -= param - 100.*col.x*mph*(1.-mph);
		}
        // blend the gems a bit 
		d += .03*(1.-mph)*smoothstep(0.1,0., t);
		if (d <= 0.0) break;
        // something like a light trap
		col.w = min(col.w, d);
        
#if 1
        // a few more steps for texture
        for (int i=0; i<5; ++i)
        {
            p = abs(p) / dot(p.xyz, p.xyz);
            av += p.xyz/(4.+p.w);
            p.xyz -= vec3(.83)-0.1*p.xyz;
        }
#endif        
		col.xyz += max(av / 9000., p.xyz / 8200.);
		
		t += min(0.1, mix(d*d, d, mph));
	}
	
	return mix(col.xyz/col.w*(2.1-2.*mph)/(1.+.2*t), 
               mph-0.0003*length(pos)/col.www - (1.-mph*0.4)*vec3(0.6,0.4,0.1)/(1.+pln), 
               mph);
}


vec2 rotate(in vec2 v, float r) { float s = sin(r), c = cos(r);	return vec2(v.x * c - v.y * s, v.x * s + v.y * c); }

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = (fragCoord.xy - iResolution.xy*.5) / iResolution.y * 2.;
    
    vec3 dir = normalize(vec3(uv, (.9+.2*mph) - 0.4*length(uv)));
    
    float t = iTime/2.;
	vec3 pos = vec3((1.-mph*.5)*sin(t/2.), (.3-.2*mph)*cos(t/2.), (.3+2.*mph)*(-1.5+sin(t/4.13)));
    pos.xy /= 1.001 - mph + 0.2 * -pos.z;
    dir.yz = rotate(dir.yz, -1.4+mph+(1.-.6*mph)*(-.5+0.5*sin(t/4.13+2.+1.*sin(t/1.75))));
    dir.xy = rotate(dir.xy, sin(t/2.)+0.2*sin(t+sin(t/3.)));
    
	fragColor = vec4(kali_sky(pos, dir), 1.);
}`,
    "Chameleon": `// Created by sebastien durand - 2015
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
//-----------------------------------------------------

// Change this to improve quality - Rq only applied on edge
#define ANTIALIASING 5

// decomment this to see where antialiasing is applied
//#define SHOW_EDGES

#define RAY_STEP 48
//#define NOISE_SKIN
#define ZERO int(min(0.,iFrame))
#define PI 3.14159279


bool WithChameleon;	    // for optim : true if the ray intersect the bounding sphere of the chameleon

float Anim;				// pos in animation
mat2 Rotanim, Rotanim2, Rot3; // rotation matrix
float ca3, sa3;         // pre calculater sin and cos
float closest;			// min distance to chameleon on the ray (use for glow light) 

// ----------------------------------------------------

float hash( float n ) { return fract(sin(n)*43758.5453123); }


#ifdef NOISE_SKIN
// By Shane -----

// Tri-Planar blending function. Based on an old Nvidia tutorial.
vec3 tex3D( sampler2D tex, in vec3 p, in vec3 n ){
    n = max((abs(n) - 0.2)*7., 0.001); // n = max(abs(n), 0.001), etc.
    n /= (n.x + n.y + n.z );  
    
	return (texture(tex, p.yz)*n.x + texture(tex, p.zx)*n.y + texture(tex, p.xy)*n.z).xyz;
}

vec3 doBumpMap( sampler2D tex, in vec3 p, in vec3 nor, float bumpfactor){
    const float eps = 0.001;
    float ref = (tex3D(tex,  p , nor)).x;                 
    vec3 grad = vec3( (tex3D(tex, vec3(p.x-eps, p.y, p.z), nor).x)-ref,
                      (tex3D(tex, vec3(p.x, p.y-eps, p.z), nor).x)-ref,
                      (tex3D(tex, vec3(p.x, p.y, p.z-eps), nor).x)-ref )/eps;
             
    grad -= nor*dot(nor, grad);          
                      
    return normalize( nor + grad*bumpfactor );
}

#endif


// ----------------------------------------------------

bool intersectSphere(in vec3 ro, in vec3 rd, in vec3 c, in float r) {
    ro -= c;
	float b = dot(rd,ro), d = b*b - dot(ro,ro) + r*r;
	return (d>0. && -sqrt(d)-b > 0.);
}

// ----------------------------------------------------

float udRoundBox( vec3 p, vec3 b, float r ){
  	return length(max(abs(p)-b,0.))-r;
}

// capsule with bump in the middle -> use for arms and legs
vec2 sdCapsule( vec3 p, vec3 a, vec3 b, float r ) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa,ba)/dot(ba,ba), 0., 1. );
    float dd = cos(3.14*h*2.5);  // Little adaptation
    return vec2(length(pa - ba*h) - r*(1.-.1*dd+.4*h), 30.-15.*dd); 
}

vec2 smin(in vec2 a, in vec2 b, in float k ) {
	float h = clamp( .5 + (b.x-a.x)/k, 0., 1. );
	return mix( b, a, h ) - k*h*(1.-h);
}

float smin(in float a, in float b, in float k ) {
	float h = clamp( .5 + (b-a)/k, 0., 1. );
	return mix(b, a, h) - k*h*(1.-h);
}

vec2 min2(in vec2 a, in vec2 b) {
	return a.x<b.x?a:b;
}

// ----------------------------------------------------

vec2 spiralTail(in vec3 p) {
    float a = atan(p.y,p.x)+.2*Anim;
	float r = length(p.xy);
    float lr = log(r);
    float th = 0.475-.25*r; // thickness according to distance
    float d = fract(.5*(a-lr*10.)/PI); //apply rotation and scaling.
	
    d = (0.5-abs(d-0.5))*2.*PI*r/10.;
  	d *= 1.1-1.1*lr;  // space fct of distance
   
    r+=.05*cos(a*60.); // radial bumps
    r+=(.2-.2*(smoothstep(0.,.08, abs(p.z))));

    return vec2(
        max(max(sqrt(d*d+p.z*p.z)-th*r, length(p.xy-vec2(.185,-.14))-1.05), -length(p.xy-vec2(.4,1.5))+.77),
        abs(30.*cos(10.*d)) + abs(20.*cos(a*10.)));
}

vec2 body(in vec3 p) {
    const float scale = 3.1;
    
    p.y=-p.y;
    p.x += 2.;
    p/=scale;
    
    float a = atan(p.y,p.x);
	float r = length(p.xy);
    float d = (.5*a-log(r))/PI; //apply rotation and scaling.
    float th = .4*(1.-smoothstep(.0,1.,abs(a+.35-Anim*.05)));    
 
    d = (1.-2.*abs(d-.5))*r*1.5;
    
   // r +=.005*cos(80.*d); // longitudinal bumps
    r+=.01*cos(a*200.); // radial bumps
    r-=.2*(smoothstep(0.,.1,abs(p.z)));
    
    float dis = sqrt(d*d+p.z*p.z)-th*r;
 	dis *= scale;
    dis = max(dis, length(p.xy-vec2(.86,-.07))-.7);
    return vec2(dis, abs(30.*cos(17.*d)) + abs(20.*cos(a*20.)));
}

vec2 head(in vec3 p) {
 //   p.yz *= Rotanim;  // small rotation of head 
   
   
    p.z = abs(p.z);
    
    p.y += .25+.03*Anim;
    p.x += .03+.03*Anim;
    p.xy *= Rotanim;

    vec3 pa1 = p, ba = vec3(1.,-.2,-.3);
    pa1.z = p.z-.22;
    
    float h = clamp(dot(pa1, ba), 0.0, 1.0 );
    pa1.x -= h;

    // Head
	float dh = length(pa1) - .8*(-.5+1.3*sqrt(abs(cos(1.5701+h*1.5701))))+.08*(1.+h)*smoothstep(0.,.2,abs(p.z));
    dh = max(-p.y-.2, dh); 
    dh += -.04+.04*(smoothstep(0.,.2,abs(p.z)));
    dh = min(dh, max(p.x-1.35,max(p.y+.3, length(p-vec3(1.-.035*Anim,.25,-.1))-.85)));
    dh += .01*cos(40.*h) -.06;
    
    // Eyes
    vec3 eye = vec3(-.2,-.0105,.15);
  	eye.zy *= Rotanim2;
    float de = max(length(p-vec3(.7,.26,.45))-.3, -(length(p-vec3(.7,.26,.45) - eye)-.13*clamp(Anim+.2,.7,1.1)));
    vec2 dee = min2(vec2(de,20.+1000.*abs(dot(p,eye))), vec2(length(p-vec3(.7,.26,.45))-.2, -102.));
  
    return smin(dee, vec2(dh*.8, 40.- abs(20.*cos(h*3.))) ,.06); 
}
    
vec2 support(vec3 p, vec2 c, float th) {
    p-=vec3(-2.5,-.7,0);
    float d1 = length(p-vec3(0,-6.5,0)) - 3.;          
    float d = length(max(abs(p-vec3(0,-2,.75))-vec3(.5,2.5,.1),0.))-.11;     
    p.xy *= Rot3; 
    d = min(d, max(length(max(abs(p)-vec3(4,3,.1),0.))-.1,
                  -length(max(abs(p)-vec3(3.5,2.5,.5),0.))+.1));
    return min2(vec2(d1,-105.),
        min2(vec2(d,-100.), 
                 vec2(length(max(abs(p-vec3(0,0,.2))-vec3(3.4,2.4,.01),0.))-.3, -103.)));
}


//----------------------------------------------------------------------

vec2 map(in vec3 pos) {
    // Ground
    vec2 res1 = vec2( pos.y+4.2, -101.0 );
    // Screen
	res1 = min2(support(pos+vec3(2.5,-0.56,0), vec2(.1,15.), 0.05), res1);
    
    if (WithChameleon) {
        // Tail + Body
        vec2 res = smin(spiralTail(pos.xyz-vec3(0,-.05-.05*Anim,0)), body( pos.xyz-vec3(-.49,1.5,0)),.1 ); 
        // Head
        res = smin(res, head(pos - vec3(-2.8,3.65,0)), .5);
        pos.z = abs(pos.z);
        // legs
        res = min2(res, min2(sdCapsule(pos, vec3(.23,-.1*Anim+1.3,.65), vec3(.75,-.1*Anim+.6,.05),.16),
                             sdCapsule(pos, vec3(.23,-.1*Anim+1.3,.65), vec3(-.35,1.35,.3),.16)));
        res = min2(res, vec2(length(pos-vec3(-.35,1.35,.1))- .33, 30.));   
        // arms 
        res = smin(res, min2(sdCapsule(pos, vec3(-.8+.06*Anim,2.5,.85),vec3(-1.25+.03*Anim,3.,.2), .16),
                             sdCapsule(pos, vec3(-.8+.06*Anim,2.5,.85), vec3(-1.25,2.1,.3),.16)),.15);
        res = min2(res, vec2(length(pos-vec3(-1.55,1.9,.1))- .3, 30.));
        
        if (res.x < closest) {
            closest = abs(res.x);
        }
        return min2(res, res1);
    }
    else {
        return res1;
    }
}


//----------------------------------------------------------------------
#define EDGE_WIDTH 0.15

vec2 castRay(in vec3 ro, in vec3 rd, in float maxd, inout float hmin) {
    closest = 9999.; // reset closest trap
	float precis = .0006, h = EDGE_WIDTH+precis, t = 2., m = -1.;
    hmin = 0.;
    for( int i=ZERO; i<RAY_STEP; i++) {
        if( abs(h)<t*precis || t>maxd ) break;
        t += h;
	    vec2 res = map(ro+rd*t);
        if (h < EDGE_WIDTH && res.x > h + 0.001) {
			hmin = 10.0;
		}
        h = res.x;
	    m = res.y;
    }
    
	//if (hmin != h) hmin = 10.;
    if( t>maxd ) m = -200.0;
    return vec2( t, m );
}

float softshadow( in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k) {
	float res = 1.0;
    float t = mint;
    for( int i=ZERO; i<26; i++ ) {
		if( t>maxt ) break;
        float h = map( ro + rd*t ).x;
        res = min( res, k*h/t );
        t += h;
    }
    return clamp( res, 0., 1.);
}

// normal with kali edge finder
float Edge=0.;
vec3 calcNormal(vec3 p, vec3 rd, float t) { 
    float pitch = .2 * t / iResolution.x; 
	pitch = max( pitch, .015 );

	vec3 e = vec3(0.0,2.*pitch,0.0);
	float d1=map(p-e.yxx).x,d2=map(p+e.yxx).x;
	float d3=map(p-e.xyx).x,d4=map(p+e.xyx).x;
	float d5=map(p-e.xxy).x,d6=map(p+e.xxy).x;
	float d=map(p).x;
    
	Edge=abs(d-0.5*(d2+d1))+abs(d-0.5*(d4+d3))+abs(d-0.5*(d6+d5)); //edge finder
	Edge=min(1.,pow(Edge,.55)*15.);
    
    vec3 grad = vec3(d2-d1,d4-d3,d6-d5);
	return normalize(grad - max(.0,dot (grad,rd ))*rd);
}




float calcAO( in vec3 pos, in vec3 nor) {
	float totao = 0.0;
    float sca = 1.0;
    for( int aoi=ZERO; aoi<5; aoi++ ) {
        float hr = 0.01 + 0.05*float(aoi);
        vec3 aopos =  nor * hr + pos;
        float dd = map( aopos ).x;
        totao += -(dd-hr)*sca;
        sca *= .75;
    }
    return clamp( 1.0 - 4.0*totao, 0.0, 1.0 );
}

vec3 mandelbrot(in vec2 uv, vec3 col) {
    uv.x += 1.5;
    uv.x = -uv.x;

    float a=.05*sqrt(abs(Anim)), ca = cos(a), sa = sin(a);
    mat2 rot = mat2(ca,-sa,sa,ca);
    uv *= rot;
	float kk=0., k = abs(.15+.01*Anim);
    uv *= mix(.02, 2., k);
	uv.x-=(1.-k)*1.8;
    vec2 z = vec2(0);
    vec3 c = vec3(0);
    for(int i=ZERO;i<50;i++) {
        if(length(z) >= 4.0) break;
        z = vec2(z.x*z.x-z.y*z.y, 2.*z.y*z.x) + uv;
        if(length(z) >= 4.0) {
            kk = float(i)*.07;
            break; // does not works on some engines !
        }
    }
    return clamp(mix(vec3(.1,.1,.2), clamp(col*kk*kk,0.,1.), .6+.4*Anim),0.,1.);
}

vec3 screen(in vec2 uv, vec3 scrCol) {
    // tv effect with horizontal lines and color switch
    vec3 oricol = mandelbrot(vec2(uv.x,uv.y), scrCol);
    vec3 col;
	float colorShift = .2*cos(.5*iTime);
    col.r = mandelbrot(vec2(uv.x,uv.y+colorShift), scrCol).x;
    col.g = oricol.y;
    col.b = mandelbrot(vec2(uv.x,uv.y-colorShift), scrCol).z;
    
	uv *= Rot3;	
	col =(.5*scrCol+col)*(.5+.5*cos(iTime*5.))*cos(iTime*10.+40.*uv.y);  
    return col*col;
}

// clean lines on the ground
float isGridLine(vec2 p, vec2 v) {
    vec2 k = smoothstep(.1,.9,abs(mod(p+v*.5, v)-v*.5)/.08);
    return k.x * k.y;
}

vec3 render( in vec3 ro, in vec3 rd, inout float hmin) { 
    // Test bounding sphere (optim)
    WithChameleon = intersectSphere(ro,rd,vec3(-.5,1.65,0),3.15); //2.95);
    
    vec2 res = castRay(ro,rd,60.0, hmin);
    float distCham = abs(closest);
    
#ifdef SHOW_EDGES
     if( res.y>-150.)  {
           vec3 pos = ro + res.x*rd;
     	vec3 nor = calcNormal(pos, rd, res.x);
     }
    return vec3(1);
#else
    
    float t = res.x;
	float m = res.y;
    vec3 cscreen = vec3(sin(.1+1.1*iTime), cos(.1+1.1*iTime),.5);
    cscreen *= cscreen;
 
    vec3 col;
	float dt;
    float glow = 1.-smoothstep(Anim + cos(iTime),.9+1.15,2.2);
    glow *= step(.3, hash(iTime)); //floor(.01+10.5*iTime)));
    
    if( m>-150.)  { 
        vec3 pos = ro + t*rd;
        vec3 nor = calcNormal(pos, rd, t);

        if( m>0. ) { // Chameleon
			col = vec3(.4) + .35*cscreen + .3*sin(1.57*.5*iTime + vec3(.05,.09,.1)*(m-1.) );
#ifdef NOISE_SKIN
            nor = doBumpMap(iChannel0, pos*.5, nor, 0.05);
#endif            
        } else if (m<-104.5) {  // bottom of screen
            col = vec3(.92);
            dt = dot(normalize(pos-vec3(-4,-4,0)), vec3(0,0,-1));
            col += (dt>0.) ? (.75*glow+.3)*dt*cscreen: vec3(0); 
        } else if (m<-102.5) {
           	if (pos.z<0.) { // screen
            	col = screen(pos.xy,cscreen);
                col += 20.*glow*col;
            } else { // back of screen
                col = vec3(.92);
            	distCham *= .25; // Hack for chameleon light on screen
            }
        } else if (m<-101.5) {
            col = .2+3.5*cscreen*glow;
            
        } else if(m<-100.5) {  // Ground
            float f = mod( floor(2.*pos.z) + floor(2.*pos.x), 2.0);
            col = 0.4 + 0.1*f*vec3(1.);
            col = .1+.9*col*isGridLine(pos.xz, vec2(2.));
            dt = dot(normalize(pos-vec3(-4,-4,0)), vec3(0,0,-1));
 			col += (dt>0.) ? (.75*glow+.3)*dt*cscreen: vec3(0);     
    		//col = clamp(col,0.,1.);
        } else {  // Screen
            col = vec3(.92);
            distCham *= .25; // Hack for chameleon light on screen
        }
		
        float ao = calcAO( pos, nor );

		vec3 lig = normalize( vec3(-0.6, 0.7, -0.5) );
		float amb = clamp( 0.5+0.5*nor.y, 0.0, 1.0 );
        float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
        float bac = clamp( dot( nor, normalize(vec3(-lig.x,0.0,-lig.z))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);

		float sh = 1.0;
		if( dif>0.02 ) { 
            WithChameleon = intersectSphere(pos,lig,vec3(-.5,1.65,0),2.95);
            sh = softshadow( pos, lig, 0.02, 13., 8.0 ); 
            dif *= sh; 
        }

		vec3 brdf = vec3(0.0);
		brdf += 1.80*amb*vec3(0.10,0.11,0.13)*ao;
        brdf += 1.80*bac*vec3(0.15,0.15,0.15)*ao;
        brdf += 0.8*dif*vec3(1.00,0.90,0.70)*ao;

		float pp = clamp( dot( reflect(rd,nor), lig ), 0.0, 1.0 );
		float spe = 1.2*sh*pow(pp,16.0);
		float fre = ao*pow( clamp(1.0+dot(nor,rd),0.0,1.0), 2.0 );

		col = col*brdf*(.5+.5*sh) + vec3(.25)*col*spe + 0.2*fre*(0.5+0.5*col);
        
        float rimMatch =  1. - max( 0. , dot( nor , -rd ) );
        col += vec3((.1+cscreen*.1 )* pow( rimMatch, 10.));
	}

	col *= 2.5*exp( -0.01*t*t );
    float BloomFalloff = 15000.; //mix(1000.,5000., Anim);
 	col += .5*glow*cscreen/(1.+distCham*distCham*distCham*BloomFalloff);
    
	return vec3( clamp(col,0.0,1.0) );
#endif    
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    // animation
    float GlobalTime = iTime; // + .1*hash(iTime);
    
    Anim = clamp(5.6*cos(GlobalTime)*cos(4.*GlobalTime),-2.5,1.2);
    ca3 = cos(.275+.006*Anim); sa3 = sin(.275+.006*Anim);   
	Rot3 = mat2(ca3,-sa3,sa3,ca3);
    float a=.1+.05*Anim, ca = cos(a), sa = sin(a);
    Rotanim = mat2(ca,-sa,sa,ca);
    float b = mod(GlobalTime,12.)>10.?cos(8.*GlobalTime):.2*cos(4.*GlobalTime), cb = cos(b), sb = sin(b);
    Rotanim2 = mat2(cb,-sb,sb,cb);
    float time = 17. + /*14.5 +*/ GlobalTime;
    
    // timed camera orbit (replaces mouse)
    float moX = 0.5 + 0.5*sin(iTime * 0.15);
    float moY = 0.5 + 0.5*cos(iTime * 0.12);
    
    // camera	
    float dist = 13.;
    vec3 ro = vec3( -0.5+dist*cos(0.1*time + 6.0*moX), 3.5 + 10.0*moY, 0.5 + dist*sin(0.1*time + 6.0*moX) );
    vec3 ta = vec3( -3.5, .5, 0. );

    // camera tx
    vec3 cw = normalize( ta-ro );
    vec3 cp = vec3( 0.0, 1.0, 0.0 );
    vec3 cu = normalize( cross(cw,cp) );
    vec3 cv = normalize( cross(cu,cw) );

    // render
    vec3 colorSum = vec3(0);
    int nbSample = 0;
    
 #if (ANTIALIASING == 1)	
	int i=0;
#else
	for (int i=ZERO;i<ANTIALIASING;i++) {
#endif
        float randPix = 0.; //hash(iTime); // Use frame rate to improve antialiasing ... not sure of result
		vec2 subPix = .4*vec2(cos(randPix+6.28*float(i)/float(ANTIALIASING)),
                              sin(randPix+6.28*float(i)/float(ANTIALIASING)));
		//vec3 ray = Ray(2.0,fragCoord.xy+subPix);
		vec2 q = (fragCoord.xy+subPix)/iResolution.xy;
		//vec2 q = (fragCoord.xy+.4*vec2(cos(6.28*float(i)/float(ANTIALIASING)),sin(6.28*float(i)/float(ANTIALIASING))))/iResolution.xy;
        vec2 p = -1.0+2.0*q;
        p.x *= iResolution.x/iResolution.y;
        vec3 rd = normalize( p.x*cu + p.y*cv + 2.5*cw );
        
        nbSample++;
        float hmin = 100.;
        colorSum += sqrt(render( ro, rd, hmin));
        
#ifdef SHOW_EDGES
 		colorSum = vec3(1);
        if (Edge>0.3) colorSum = vec3(.6);  
        if (hmin>0.5) colorSum = vec3(0,0,0);   
        break;
#endif
        
#if (ANTIALIASING > 1)
        // optim : use antialiasing only on objects edges //exit if far from objects
        if (Edge<0.3 && hmin<0.5 ) break;
	}
#endif
    
    fragColor = vec4(colorSum/float(nbSample), 1.);
}`,

    "Quadtree Smile": `#define C(u,a,b) cross(vec3(u-a,0), vec3(b-a,0)).z > 0.
#define S(a) smoothstep(2./R.y, -2./R.y, a)

// https://www.shadertoy.com/view/4djSRW
float h11(float p)
{
    p = fract(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float h12(vec2 p)
{
	vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// https://www.shadertoy.com/view/MsXGDj
float catrom(float t) 
{
    float f = floor(t),
          x = t - f;
    float v0 = h11(f), v1 = h11(f+1.), v2 = h11(f+2.), v3 = h11(f+3.);
	float c2 = -.5 * v0	+ 0.5*v2;
	float c3 = v0		- 2.5*v1 + 2.0*v2 - 0.5*v3;
	float c4 = -.5 * v0	+ 1.5*v1 - 1.5*v2 + 0.5*v3;
	return(((c4 * x + c3) * x + c2) * x + v1);
}

// https://iquilezles.org/articles/distfunctions2d/
float sdArc( in vec2 p, in vec2 sc, in float ra, float rb )
{
    // sc is the sin/cos of the arc's aperture
    p.x = abs(p.x);
    return ((sc.y*p.x>sc.x*p.y) ? length(p-sc*ra) : 
                                  abs(length(p)-ra)) - rb;
}

float sdPoly( in vec2[4] v, in vec2 p )
{
    float d = dot(p-v[0],p-v[0]);
    float s = 1.0;
    for( int i=0, j=3; i<4; j=i, i++ )
    {
        vec2 e = v[j] - v[i];
        vec2 w =    p - v[i];
        vec2 b = w - e*clamp( dot(w,e)/dot(e,e), 0.0, 1.0 );
        d = min( d, dot(b,b) );
        bvec3 c = bvec3(p.y>=v[i].y,p.y<v[j].y,e.x*w.y>e.y*w.x);
        if( all(c) || all(not(c)) ) s*=-1.0;  
    }
    return s*sqrt(d);
}

void mainImage( out vec4 o, vec2 u )
{
    vec2 R = iResolution.xy,
         tl = vec2(-1, 1) * R/R.y,
         tr = vec2( 1, 1) * R/R.y,
         bl = vec2(-1,-1) * R/R.y,
         br = vec2( 1,-1) * R/R.y;
         
    vec2 v = u,
         w = u / R;
    u = (u+u-R)/R.y;
   // u /= .98 +  .02 *dot(u,u);
    
    float t = iTime/2.2;// + mod(v.x+v.y,2.)*.002;
    vec4 bg = vec4(.33);
    
    vec2 ID; 
    float area;
    float threshold = mix(.5,.032, smoothstep(2.,5., iTime + .0*mod(v.x+v.y,2.)));
    
    for (int i = 0; i < 14; i++)
    { 
        t += 7.3*h12(ID);
        
        float k = float(i)+1.;
        float K = 1./k;
        
        float to = mix(.01*(10.-k)*cos(t*k + (u.x+u.y)*k/2. + h12(ID)*10.), 3.,
                       smoothstep(5.,2., iTime));
        float mx1 = catrom(t);
        float mx2 = catrom(t+to);
        vec2 x1, x2;
  
        if (i%2 == 0)
        {
            x1 = mix(tl, tr, mx1);
            x2 = mix(bl, br, mx2);
            if (C(u,x1,x2)) tr = x1, br = x2, ID += vec2(K,0);       
            else            tl = x1, bl = x2, ID -= vec2(K,0);
        }
        else 
        {
            x1 = mix(tl, bl, mx1);
            x2 = mix(tr, br, mx2);
            if (C(u,x1,x2)) tl = x1, tr = x2, ID += vec2(0,K);        
            else            bl = x1, br = x2, ID -= vec2(0,K);
        }    

        area = tl.x*bl.y + bl.x*br.y + br.x*tr.y + tr.x*tl.y
             - tl.y*bl.x - bl.y*br.x - br.y*tr.x - tr.y*tl.x;
        
        if (h12(ID) < threshold) break;
    }
        
    float h = h12(ID); 
    
    vec2 c = (tl+tr+bl+br)/4.;
    vec2 uc = u-c;
    vec4 co = vec4(0, floor(t+9.*(uc.x+uc.y))+.24, .64, 0);    
    vec4 col = .8+.4*cos(3.14*(h*1e2 + co));
    o = col; // * vec4(C(u,tl,tr) && C(u,tr,br) && C(u,br,bl) && C(u,bl,tl));

    float l = max(length(tl-br), length(tr-bl));    
    o *= .25 + .75*S( sdPoly(vec2[] (tl, bl, br, tr), u) + .0033 );
    o *= .75 + .25*S( sdPoly(vec2[] (tl, bl, br, tr), u+.07*l)   );
    
    if (h > threshold) o = mix(o, o.rrrr, .9);
    
    float s = h > .5 ? 1. : -1.;
    o *= .97+.03*S(abs(fract(20.*(uc.x+s*uc.y))-.5)-.25);

    o = clamp(o,0.,1.);
    o = mix(bg, o, exp(-.0007/area));
    
    float dm = min(length((bl+br)/2. - (tl+tr)/2.), 
                   length((bl+tl)/2. - (br+tr)/2.));
    float sdm = smoothstep(.35, .47, dm);
    
    vec2 p = vec2(-.09,.02*cos(15.*h12(ID+.31)*t))*.75;
    vec2 q = vec2( .09,.02*sin(15.*h12(ID+.31)*t))*.75;
        
    uc.y -= .02;
    float sp1 = S(length(uc-p)       - .06);
    float sp2 = S(length(uc-p-.0075) - .045);
    float sp3 = S(length(uc-p-.0075) - .0225);
    float sq1 = S(length(uc-q)       - .06);
    float sq2 = S(length(uc-q-.0075) - .045);
    float sq3 = S(length(uc-q-.0075) - .0225);
    
    vec2 r = vec2(0, -.045 + .02*cos(20.*h12(ID+.4)*t));
    vec2 sc = vec2(sin(.8),cos(.8));        
    float sr1 = S(sdArc(-uc-.0075-r, sc, .18, .03));
    float sr2 = S(sdArc(-uc-r, sc, .18, .015));
    
    o = mix(o, vec4(0), sdm*sp1);
    o = mix(o, vec4(1), sdm*sp2);
    o = mix(o, vec4(0), sdm*sp3);
    o = mix(o, vec4(0), sdm*sq1);
    o = mix(o, vec4(1), sdm*sq2);
    o = mix(o, vec4(0), sdm*sq3);      
    o = mix(o, vec4(0), sdm*sr1);
    o = mix(o, vec4(1), sdm*sr2);      
       
    o = mix(bg, o, pow(tanh(64.*w.x*(1.-w.x)*w.y*(1.-w.y)),.23));
}`,
    "Snow Kaleidoscope": `
#define time iTime
#define resolution iResolution.xy

float snow(vec2 uv,float scale)
{

	float _t = time*0.3;
	 uv.x+=_t/scale; 
	uv*=scale;vec2 s=floor(uv),f=fract(uv),p;float k=40.,d;
	p=.5+.35*sin(11.*fract(sin((s+p+scale)*mat2(7,3,6,5))*5.))-f;d=length(p);k=min(d,k);
	k=smoothstep(0.,k,sin(f.x+f.y)*0.003);
    	return k;
}
float makePoint(float x,float y,float fx,float fy,float sx,float sy,float t){
   float xx=x+sin(t*fx)*sx;
   float yy=y+cos(t*fy)*sy;
   return 1.0/sqrt(xx*xx+yy*yy);
}

vec3 palette( float t ) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.263,0.416,0.557);

    return a + b*cos( 6.28318*(c*t+d) );
}
float happy_star(vec2 uv, float anim)
{
    uv = abs(uv);
    vec2 pos = min(uv.xy/uv.yx, anim);
    float p = (2.0 - pos.x - pos.y);
    return (2.0+p*(p*p-1.5)) / (uv.x+uv.y);      
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ){
	vec2 uv=(fragCoord.xy*2.-resolution.xy)/min(resolution.x,resolution.y); 
	float dd = 1.0-length(uv);
vec2 pos = uv;
vec2 uv2 = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
	const float pi = 3.14159;
	const float n = 16.0;
	 
    vec2 uv0 = uv2;
    vec3 finalColor2 = vec3(0.0);
    
    for (float i = 0.0; i < 4.0; i++) {
        uv2 = fract(uv2 * 1.5) - 0.5;

        float d = length(uv) * exp(-length(uv0));

        vec3 col = palette(length(uv0) + i*.4 + iTime*.4);

        d = sin(d*8. + iTime)/8.;
        d = abs(d);

        d = pow(0.01 / d, 1.2);

        finalColor2 += col * d;
    }
	float radius = length(pos) * 2.0 - 0.4;
	float t = atan(pos.y, pos.x);
	
	float color = 0.0;
	float color2 = 0.0;
	for (float i = 9.0; i <= n; i++){
		color += 0.002 / abs(0.2 * sin(
			3. * (t + i/n * time * 2.1)
		    ) - radius
		);
        
        
	}
    for (float i = 9.0; i <= n; i++){
		color2 += 0.005 / abs(0.2 * sin(
			5. * (t + i/n * time * 0.1)
		    ) - radius*0.5
		);
        
        
	}
    
    
	vec2 p=(fragCoord.xy/resolution.x)*2.0-vec2(1.0,resolution.y/resolution.x);
     p=p*2.0;
   
   float x=p.x;
   float y=p.y;

   float a=
       makePoint(x,y,3.3,2.9,0.3,0.3,time);
   a=a+makePoint(x,y,1.9,2.0,0.4,0.4,time);
   a=a+makePoint(x,y,0.8,0.7,0.4,0.5,time);
   a=a+makePoint(x,y,2.3,0.1,0.6,0.3,time);
   a=a+makePoint(x,y,0.8,1.7,0.5,0.4,time);
   a=a+makePoint(x,y,0.3,1.0,0.4,0.4,time);
   a=a+makePoint(x,y,1.4,1.7,0.4,0.5,time);
   a=a+makePoint(x,y,1.3,2.1,0.6,0.3,time);
   a=a+makePoint(x,y,1.8,1.7,0.5,0.4,time);   
   
   float b=
       makePoint(x,y,1.2,1.9,0.3,0.3,time);
   b=b+makePoint(x,y,0.7,2.7,0.4,0.4,time);
   b=b+makePoint(x,y,1.4,0.6,0.4,0.5,time);
   b=b+makePoint(x,y,2.6,0.4,0.6,0.3,time);
   b=b+makePoint(x,y,0.7,1.4,0.5,0.4,time);
   b=b+makePoint(x,y,0.7,1.7,0.4,0.4,time);
   b=b+makePoint(x,y,0.8,0.5,0.4,0.5,time);
   b=b+makePoint(x,y,1.4,0.9,0.6,0.3,time);
   b=b+makePoint(x,y,0.7,1.3,0.5,0.4,time);

   float c=
       makePoint(x,y,3.7,0.3,0.3,0.3,time);
   c=c+makePoint(x,y,1.9,1.3,0.4,0.4,time);
   c=c+makePoint(x,y,0.8,0.9,0.4,0.5,time);
   c=c+makePoint(x,y,1.2,1.7,0.6,0.3,time);
   c=c+makePoint(x,y,0.3,0.6,0.5,0.4,time);
   c=c+makePoint(x,y,0.3,0.3,0.4,0.4,time);
   c=c+makePoint(x,y,1.4,0.8,0.4,0.5,time);
   c=c+makePoint(x,y,0.2,0.6,0.6,0.3,time);
   c=c+makePoint(x,y,1.3,0.5,0.5,0.4,time);
   vec3 d=vec3(a,b,c)/32.0;
	uv.y += sin(uv.x*1.4)*0.2;
	uv.x *= 0.79;
	float c2=snow(uv,30.)*.3;
	c2+=snow(uv,20.)*.5;
	c2+=snow(uv,15.)*.8;
	c2+=snow(uv,10.);
	c2+=snow(uv,8.);
	c2+=snow(uv,6.);
	c2+=snow(uv,5.);
	c2*=0.2/dd;
	vec3 finalColor=(vec3(0.0,0.8,5.9))*c2*color*3000.0+finalColor2;
	fragColor = vec4(finalColor*d*color2,1);
    uv *= 2.0 * ( cos(iTime * 2.0) -2.5); // scale
    float anim = sin(iTime * 12.0) * 0.1 + 1.0;  // anim between 0.9 - 1.1 
    fragColor+= vec4(happy_star(uv, anim) * vec3(0.35,0.2,1.15)*1.75*finalColor2, 1.0);
}`,
    "Fractal Dive": `vec4 light;
float ui;
mat2 m,n,nn;
float map (vec3 p) {
    float d = length(p-light.xyz)-light.w;
    d = min(d,max(10.-p.z,0.));
    float t = 2.5;
    for (int i = 0; i < 13; i++) {
        t = t*0.66;
        p.xy = m*p.xy;
        p.yz = n*p.yz;
        p.zx = nn*p.zx;
        p.xz = abs(p.xz) - t;

    }
    d = min(d,length(p)-1.4*t);

    return d;
}
vec3 norm (vec3 p) {
    vec2 e = vec2 (.001,0.);
    return normalize(vec3(
        map(p+e.xyy) - map(p-e.xyy),
        map(p+e.yxy) - map(p-e.yxy),
        map(p+e.yyx) - map(p-e.yyx)
    ));
}
vec3 dive (vec3 p, vec3 d) {
    for (int i = 0; i < 20; i++) {
        p += d*map(p);
    }
    return p;
}
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 v = fragCoord/iResolution.xy*2.-1.;
	v.x *= iResolution.x/iResolution.y;
    ui = 100.*iTime;
    float y = -0.001*ui;
    m = mat2(sin(y),cos(y),-cos(y),sin(y));
    y = 0.0035*ui;
    n = mat2(sin(y),cos(y),-cos(y),sin(y));
    y = 0.0023*ui;
    nn = mat2(sin(y),cos(y),-cos(y),sin(y));
    vec3 r = vec3(0,0,-15.+2.*sin(0.01*ui));
    light = vec4(10.*sin(0.01*ui),2,-23,1);
    vec3 d = normalize(vec3(v,5.));
    vec3 p = dive(r,d);
    d = normalize(light.xyz-p);
    vec3 no = norm(p);
    vec3 col = vec3(.7,.8,.9);
    vec3 bounce = dive(p+0.01*d,d);
    col = mix(col,vec3(0),dot(no, normalize(light.xyz-p)));
    if (length(bounce-light.xyz) > light.w+0.1) col *= 0.2;

    fragColor = vec4(col,1.0);
}`,
    "Oil Nebula": `#define iterations 13
#define formuparam 0.53

#define volsteps 20
#define stepsize 0.1

#define zoom   0.800
#define tile   0.850
#define speed  0.010

#define brightness 0.0015
#define darkmatter 0.300
#define distfading 0.730
#define saturation 0.850

#define time iTime

#define PI 3.141592
#define TWOPI 6.283184

#define R2D 180.0/PI*
#define D2R PI/180.0*

mat2 rotMat(in float r){float c = cos(r);float s = sin(r);return mat2(c,-s,s,c);}

//fract -> -0.5 -> ABS  : coordinate absolute Looping
float abs1d(in float x){return abs(fract(x)-0.5);}
vec2 abs2d(in vec2 v){return abs(fract(v)-0.5);}
float cos1d(float p){ return cos(p*TWOPI)*0.25+0.25;}
float sin1d(float p){ return sin(p*TWOPI)*0.25+0.25;}

#define OC 15.0
vec3 Oilnoise(in vec2 pos, in vec3 RGB)
{
    vec2 q = vec2(0.0);
    float result = 0.0;
   
    float s = 2.2;
    float gain = 0.44;
    vec2 aPos = abs2d(pos)*0.5;//add pos

    for(float i = 0.0; i < OC; i++)
    {
        pos *= rotMat(D2R 30.);
        float time = (sin(iTime)*0.5+0.5)*0.2+iTime*0.8;
        q =  pos * s + time;
        q =  pos * s + aPos + time;
        q = vec2(cos(q));

        result += sin1d(dot(q, vec2(0.3))) * gain;

        s *= 1.07;
        aPos += cos(smoothstep(0.0,0.15,q));
        aPos*= rotMat(D2R 5.0);
        aPos*= 1.232;
    }
   
    result = pow(result,4.504);
    return clamp( RGB / abs1d(dot(q, vec2(-0.240,0.000)))*.5 / result, vec3(0.0), vec3(1.0));
}


float easeFade(float x)
{
    return 1.-(2.*x-1.)*(2.*x-1.)*(2.*x-1.)*(2.*x-1.);
}
float holeFade(float t, float life, float lo)//lifeOffset
{
    return easeFade(mod(t-lo,life)/life);
}
vec2 getPos(float t, float life, float offset, float lo)
{
    return vec2(cos(offset+floor((t-lo)/life)*life)*iResolution.x/2.,
    sin(2.*offset+floor((t-lo)/life)*life)*iResolution.y/2.);

}

void mainVR( out vec4 fragColor, in vec2 fragCoord, in vec3 ro, in vec3 rd )
{
//get coords and direction
vec3 dir=rd;
vec3 from=ro;

//volumetric rendering
float s=0.1,fade=1.;
vec3 v=vec3(0.);
for (int r=0; r<volsteps; r++) {
vec3 p=from+s*dir*.5;
p = abs(vec3(tile)-mod(p,vec3(tile*2.))); // tiling fold
float pa,a=pa=0.;
for (int i=0; i<iterations; i++) {
p=abs(p)/dot(p,p)-formuparam;
            p.xy*=mat2(cos(iTime*0.01),sin(iTime*0.01),-sin(iTime*0.01),cos(iTime*0.01) );// the magic formula
a+=abs(length(p)-pa); // absolute sum of average change
pa=length(p);
}
float dm=max(0.,darkmatter-a*a*.001); //dark matter
a*=a*a; // add contrast
if (r>6) fade*=1.3-dm; // dark matter, don't render near
//v+=vec3(dm,dm*.5,0.);
v+=fade;
v+=vec3(s,s*s,s*s*s*s)*a*brightness*fade; // coloring based on distance
fade*=distfading; // distance fading
s+=stepsize;
}
v=mix(vec3(length(v)),v,saturation); //color adjust
fragColor = vec4(v*.01,1.);
}
float happy_star(vec2 uv, float anim)
{
    uv = abs(uv);
    vec2 pos = min(uv.xy/uv.yx, anim);
    float p = (2.0 - pos.x - pos.y);
    return (2.0+p*(p*p-1.5)) / (uv.x+uv.y);      
}
#define Q(p) p *= 2.*r(round(atan(p.x, p.y) * 4.) / 4.)
#define r(a) mat2(cos(a + asin(vec4(0,1,-1,0))))
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
vec4 o =fragColor;
    vec2 u =fragCoord;
vec2 uv=fragCoord.xy/iResolution.xy-.5;
    vec2 cPos = -1.0 + 2.0 * fragCoord.xy / iResolution.xy;
   
    // distance of current pixel from center
float cLength = length(cPos);


     vec2 st = (fragCoord/iResolution.xy);
            st.x = ((st.x - 0.5) *(iResolution.x / iResolution.y)) + 0.5;
    float stMask = step(0.0, st.x * (1.0-st.x));


    //st-=.5; //st move centor. Oil noise sampling base to 0.0 coordinate
    st*=3.;
   
    vec3 rgb = vec3(0.30, .8, 1.200);
   
   
    //berelium, 2024-06-07 - anti-aliasing
    float AA = 1.0;
    vec2 pix = 1.0 / iResolution.xy;
    vec2 aaST = vec2(0.0);
    vec3 col = vec3(0.0);
    for(float i = 0.0; i < AA; i++)
    {
        for(float j = 0.0; j < AA; j++)
        {
            aaST = st + pix * vec2( (i+0.5)/AA, (j+0.5)/AA );
            col += Oilnoise(aaST, rgb);
        }
   
    }
   
    col /= AA * AA;
uv.y*=iResolution.y/iResolution.x;
    vec2 v = iResolution.xy,
         w,
         k = u = .2*(u+u-v)/v.y;    
         
    o = vec4(1,2,3,0);
     
    for (float a = .5, t = iTime*0.21, i;
         ++i < 19.;
         o += (1.+ cos(vec4(0,1,3,0)+t))
           / length((1.+i*dot(v,v)) * sin(w*3.-9.*u.yx+t))
         )  
        v = cos(++t - 7.*u*pow(a += .03, i)) - 5.*u,        
        u *= mat2(cos(i+t*.02 - vec4(0,11,33,0))),
        u += .005 * tanh(40.*dot(u,u)*cos(1e2*u.yx+t))
           + .2 * a * u
           + .003 * cos(t+4.*exp(-.01*dot(o,o))),      
        w = u / (1. -2.*dot(u,u));
             
    o = pow(o = 1.-sqrt(exp(-o*o*o/2e2)), .3*o/o)
      - dot(k-=u,k) / 250.;
vec3 dir=vec3(uv*zoom,1.);

vec2  resolution = iResolution.xy;

    // Initialize color and texture accumulators
    vec4 color = vec4(1.0, 2.0, 3.0, 0.0);
    vec4 baseColor = color;
   
    // Initialize time and amplitude variables

    float amplitude = 0.5;
  vec2 coord = fragCoord * 2. - iResolution.xy;
    // Normalized pixel coordinates (from 0 to 1)
 
   
   
    float holeSize = iResolution.y/10.;
    float holeLife = 2.;
   
   
    vec3 final;
    for (int i = 0; i<45; i++) {
        vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(float(i),2.*float(i)+4.,4.*float(i)+16.));

        float s = holeSize;
        float lifeOffset = float(i)/2.;

        vec2 pos = getPos(iTime, holeLife, float(i)*4.5,lifeOffset);

        float d = distance(coord,pos)/s;
        d = 1./d-.1;
       
        final += mix(vec3(0),col, d)*holeFade(iTime,holeLife,lifeOffset);
    }
  vec2 pos = 0.5 - uv;
    // Adjust y by aspect for uniform transforms
    pos.y /= iResolution.x/iResolution.y;
   
    //**********         Glow        **********
   
    // Equation 1/x gives a hyperbola which is a nice shape to use for drawing glow as
    // it is intense near 0 followed by a rapid fall off and an eventual slow fade
    float dist = 1.0/length(pos);
   
    //**********        Radius       **********
   
    // Dampen the glow to control the radius
    dist *= 0.1;
   
    //**********       Intensity     **********
   
    // Raising the result to a power allows us to change the glow fade behaviour
    // See https://www.desmos.com/calculator/eecd6kmwy9 for an illustration
    // (Move the slider of m to see different fade rates)
    dist = pow(dist, 0.8);
   
    // Knowing the distance from a fragment to the source of the glow, the above can be
    // written compactly as:
    // float getGlow(float dist, float radius, float intensity){
    // return pow(radius/dist, intensity);
// }
    // The returned value can then be multiplied with a colour to get the final result
       
    // Add colour
    vec3 col2 = dist * vec3(1.0, 2.5, 1.25);

    // Tonemapping. See comment by P_Malin
    col2 = 1.0 - exp( -col );
     
    vec2 uv2 = tanh(uv);
     uv2*=10.;
 
    // Final color adjustment for visual output
   
vec3 from=vec3(1.,.5,0.5);
    Q(from.xy);
from.xy+= (cPos/max(cLength,0.0001))*cos(cLength*8.0-iTime*2.0) * 0.03;
     
mainVR(fragColor, fragCoord, from, dir);
   
  fragColor*=vec4(final*vec3(0.4,1.,1.)+o.xyz,1.);
       fragColor+=vec4(col2,1.);
}`,

    "Fireworks (KuKo 227)": `/*
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
        
    ▓              🌟  KuKo Day 227  🌟  
    
    ▓ forked from @Froxel 
    ▓ https://www.shadertoy.com/view/wdlGW4
    
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
*/

// A simple fireworks demo. This was written a few years back, so I do not
// recall some of the finer implementation details... But the basic idea was
// to reduce the number of particles that needs to be tested by dividing each
// firework (big particle) into angles/sectors for the explosion (smaller
// particles). Using a very simple accumulation buffer style of temporal
// accumulation (without history clamping) also gives the nice light trails :)
//
// Author: Fredrik Nysjö (2018?)

#define NUM_SUBPARTICLES 32.
#define BLAST_RADIUS 0.63
#define PARTICLE_RADIUS 0.005
#define SUBPARTICLE_RADIUS 0.003
#define USE_TEMPORAL_ACCUMULATION 1
#define USE_JITTER 1
#define boxstep(a, b, x) clamp(((x)-(a))/((b)-(a)),0.0,1.0)

uniform float iWriteFeedback;

float fwHash11(float p)
{
    p = fract(p * 0.1031);
    p *= p + 33.33;
    return fract(p * p);
}

float fwHash21(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec2 fwHash22(vec2 p)
{
    return vec2(fwHash21(p), fwHash21(p + 17.0));
}

vec3 fireworkHue(float hue)
{
    return 0.55 + 0.45 * cos(6.28318 * (hue + vec3(0.0, 0.33, 0.67)));
}

vec3 laplacian(vec2 uv, vec2 px, sampler2D ch)
{
    vec3 c = texture(ch, uv).rgb;
    vec3 r = texture(ch, uv + vec2( px.x, 0.0)).rgb;
    vec3 l = texture(ch, uv + vec2(-px.x, 0.0)).rgb;
    vec3 u = texture(ch, uv + vec2(0.0,  px.y)).rgb;
    vec3 d = texture(ch, uv + vec2(0.0, -px.y)).rgb;

    return r + l + u + d - 4.0*c;
}

vec3 sky(vec2 uv)
{
    return mix(vec3(0.0, 0.002, 0.004), vec3(0.0), smoothstep(0.0, 1.0, uv.y));
}

float subparticle(vec2 xy, vec2 uv, float t, float r, int i)
{
    vec2 p = (xy - uv) / r;
    float s  = NUM_SUBPARTICLES / 360.0;
    float a  = round((degrees(atan(p.y, p.x) + 3.141592)) * s) / s;
    vec2 del = vec2(cos(radians(a)), sin(radians(a)));
    float delay = fwHash11(a / 24.0 + float(i) * 1.618) * 0.5 + 0.5;
    float sub_r = SUBPARTICLE_RADIUS * (1.0 + t);
    float fade  = (length((xy + del * (r * 0.9) * delay) - uv) < sub_r) ? 4.0 : 0.0;
    return fade;
}

vec3 particle(vec2 uv, float aspect, vec2 jitter, int i)
{
    vec2 seed = vec2(float(i & 63), float(i >> 6)) + 0.1;
    vec2 posData = fwHash22(seed);
    vec2 velData = fwHash22(seed + 3.7);
    vec2 xy = posData * vec2(aspect, 0.5) + vec2(0.0, 0.5);
    vec2 del = velData * vec2(0.4, 0.05) + vec2(-0.2, 0.1);
    vec3 color = fireworkHue(fract(float(i) * 0.127 + posData.x * 0.37 + 0.11));
    color *= color;
    color *= bool(USE_TEMPORAL_ACCUMULATION) ? 2.0 : 1.0;

    float t = mod((iFrame / 60.0) * del.y, xy.y) / xy.y;
    float r = boxstep(0.3, 1.0, t) * BLAST_RADIUS + PARTICLE_RADIUS;
    float fade = boxstep(1.0, 0.2, t);

    xy.x += t * del.x;
    xy.y = 0.8 * pow(t * xy.y, 1.0 / 3.0);
    xy += bool(USE_JITTER) ? jitter : vec2(10.0);

    fade = (r > PARTICLE_RADIUS) ? subparticle(xy, uv, t, r, i) * fade : 2.0;
    return (length(xy - uv) < r) ? fade * color : vec3(0.0);
}

void mainImage(out vec4 O, in vec2 I)
{
    vec2 uv  = I / iResolution.xy;
    float as = iResolution.x / iResolution.y;
    vec2 px  = 1.0 / iResolution.xy;

    vec3 rnd = vec3(fwHash22(mod(I, 64.0)), fwHash11(dot(I, vec2(12.9898, 78.233))));
    vec2 jitter = (fract(rnd.rg + 1.61803 * mod(floor(iFrame), 64.0)) - 0.5) * 0.0025;

    vec3 colSky = sky(uv);
    colSky += (rnd - 0.5) / 255.0;

    vec3 fw = vec3(0.0);
    for(int i = 0; i < 32 + int(min(0., iFrame)); i++)
    {
        fw += particle(vec2(uv.x * as, uv.y), as, jitter, i);
    }

    vec3 prev = texture(iChannel1, uv).rgb;
    vec3 lap = laplacian(uv, px, iChannel1);

    float D      = 0.45;
    float decay  = 1.0;
    float inject = 2.5;

    vec3 smoke = prev + D * lap - decay * prev + inject * fw;
    smoke = clamp(smoke, 0.0, 9.0);

    vec3 col = colSky + fw + smoke * 0.49;

    O = iWriteFeedback > 0.5 ? vec4(smoke, 1.0) : vec4(col, 1.0);
}`,

    "Sunset Train": `float hash1(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 x){
    vec2 f = fract(x);
    vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    
    vec2 p = floor(x);
    float a = hash1(p + vec2(0.0, 0.0));
    float b = hash1(p + vec2(1.0, 0.0));
    float c = hash1(p + vec2(0.0, 1.0));
    float d = hash1(p + vec2(1.0, 1.0));
    
    return a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y;
}

float fbm(vec2 x, int detail){
    float a = 0.0;
    float b = 1.0;
    float t = 0.0;
    for(int i = 0; i < detail; i++){
        float n = noise(x);
        a += b*n;
        t += b;
        b *= 0.7;
        x *= 2.0; 
    
    }
    return a/t;
}

float fbm2(vec2 x, int detail){
    float a = 0.0;
    float b = 1.0;
    float t = 0.0;
    for(int i = 0; i < detail; i++){
        float n = noise(x);
        a += b*n;
        t += b;
        b *= 0.9;
        x *= 2.0; 
    
    }
    return a/t;
}

float box(vec2 uv, float x1, float x2, float y1, float y2){
    return (uv.x > x1 && uv.x < x2 && uv.y > y1 && uv.y < y2)?1.0:0.0;
} 

#define dot2(v) dot(v, v)
#define layer(dh, v)  if (uv.y < h + midlevel - (dh) ) return vec4(v, 1.);

vec4 foreground(vec2 uv, float t){
    float midlevel;
    float h;
    float disp;
    float dist;
    vec2 uv2;
    
    uv.y -= 0.2;
    // clouds foreground //////////////////////////////////////////////////////////////
    
    // c14
    midlevel = -0.1;
    disp = 1.7;
    dist = 1.0;
    uv2 = uv + vec2(t/dist + 40.0, 0.0);
    h = (fbm(uv2, 8) - 0.5)*disp;
    layer(0.12, vec3(0.43, 0.32, 0.31));
    layer(0.08, vec3(0.55, 0.42, 0.41));
    layer(0.04, vec3(0.66, 0.42, 0.40));
    layer(0., vec3(0.77, 0.48, 0.46));
    
    // c13
    
    midlevel = 0.05;
    disp = 1.7;
    dist = 2.0;
    uv2 = uv + vec2(t/dist + 38.0, 0.0);
    h = (fbm(uv2, 8) - 0.5)*disp;
    layer(0.1, vec3(0.95, 0.66, 0.48));
    layer(0.04, vec3(0.98, 0.76, 0.64));
    layer(0., vec3(0.95, 0.80, 0.77));
    
    return vec4(0.95, 0.80, 0.77, 0.);
}

vec4 background(vec2 uv, float t){
    float midlevel;
    float h;
    float disp;
    float dist;
    vec2 uv2;
    
    // clouds ///////////////////////////////////////////////////////
    
    // c12
    midlevel = 0.3;
    disp = 0.9;
    dist = 10.0;
    uv2 = uv + vec2(t/dist + 32.5, 0.0);
    h = (fbm(uv2, 8) - 0.5)*disp;
    layer(0.14, vec3(0.48, 0.19, 0.20));
    layer(0.1, vec3(0.68, 0.28, 0.19));
    layer(0.07, vec3(0.88, 0.38, 0.24));
    layer(0., vec3(0.95, 0.45, 0.30));
    
    // c11
    midlevel = 0.35;
    disp = 1.0;
    dist = 15.0;
    uv2 = uv + vec2(t/dist + 30.0, 0.0);
    h = (fbm(uv2, 8) - 0.5)*disp;
    layer(0.04, vec3(0.98, 0.76, 0.64));
    layer(0., vec3(0.95, 0.80, 0.77));
    
    // c10
    midlevel = 0.35;
    disp = 3.5;
    dist = 20.0;
    uv2 = uv + vec2(t/dist + 27.5, 0.0);
    h = (fbm(uv2, 8) - 0.5)*disp;
    layer(0.12, vec3(0.43, 0.32, 0.31));
    layer(0.08, vec3(0.55, 0.42, 0.41));
    layer(0.04, vec3(0.66, 0.42, 0.40));
    layer(0., vec3(0.77, 0.48, 0.46));
    
    // c9
    midlevel = 0.45;
    disp = 2.0;
    dist = 25.0;
    uv2 = uv + vec2(t/dist + 23.0, 0.0);
    h = (fbm(uv2, 8) - 0.5)*disp;
    layer(0.04, vec3(0.98, 0.57, 0.36));
    layer(0., vec3(1.0, 0.62, 0.44));
    
    // c8
    midlevel = 0.5;
    disp = 2.3;
    dist = 30.0;
    uv2 = uv + vec2(t/dist + 20.5, 0.0);
    h = (fbm(uv2, 8) - 0.5)*disp;
    layer(0.12, vec3(0.41, 0.27, 0.27));
    layer(0.08, vec3(0.53, 0.35, 0.32));
    layer(0.04, vec3(0.80, 0.24, 0.17));
    layer(0., vec3(0.99, 0.29, 0.20));
    
    // c7
    midlevel = 0.5;
    disp = 2.5;
    dist = 35.0;
    uv2 = uv + vec2(t/dist + 18.0, 0.0);
    h = (fbm(uv2, 8) - 0.5)*disp;
    layer(0.1, vec3(0.88, 0.38, 0.24));
    layer(0.05, vec3(0.98, 0.42, 0.28));
    layer(0., vec3(1.0, 0.48, 0.35));
    
    // c6
    midlevel = 0.6;
    disp = 2.0;
    dist = 40.0;
    uv2 = uv + vec2(t/dist + 18.0, 0.0);
    h = (fbm(uv2, 8) - 0.5)*disp;
    layer(0.1, vec3(0.95, 0.66, 0.48));
    layer(0., vec3(1.0, 0.76, 0.60));
    
    // c5
    midlevel = 0.75;
    disp = 3.5;
    dist = 45.0;
    uv2 = uv + vec2(t/dist + 15.5, 0.0);
    h = (fbm(uv2, 8) - 0.5)*disp;
    layer(0.2, vec3(1.0, 0.55, 0.33));
    layer(0.15, vec3(0.98, 0.50, 0.24));
    layer(0.1, vec3(0.90, 0.55, 0.40));
    layer(0., vec3(1.0, 0.62, 0.44));
    
    // c4
    midlevel = 0.7;
    disp = 2.7;
    dist = 50.0;
    uv2 = uv + vec2(t/dist + 12.0, 0.0);
    h = (fbm(uv2, 8) - 0.5)*disp;
    layer(0.04, vec3(0.73, 0.36, 0.30));
    layer(0., vec3(0.80, 0.40, 0.34));
    
    // c3
    midlevel = 0.8;
    disp = 2.7;
    dist = 60.0;
    uv2 = uv + vec2(t/dist + 9.5, 0.0);
    h = (fbm(uv2, 8) - 0.5)*disp;
    layer(0.1, vec3(0.93, 0.58, 0.35));
    layer(0., vec3(1.0, 0.76, 0.60));
    
    // c2
    midlevel = 0.9;
    disp = 3.0;
    dist = 70.0;
    uv2 = uv + vec2(t/dist + 7.0, 0.0);
    h = (fbm(uv2, 8) - 0.5)*disp;
    layer(0.1, vec3(0.56, 0.25, 0.22));
    layer(0.05, vec3(0.60, 0.30, 0.27));
    layer(0., vec3(0.74, 0.35, 0.30));
    
    // c1
    midlevel = 1.0;
    disp = 5.0;
    dist = 100.0;
    uv2 = uv + vec2(t/dist + 3.5, 0.0);
    h = (fbm(uv2, 8) - 0.5)*disp;
    layer(0.1, vec3(0.92, 0.85, 0.82));
    layer(0., vec3(1.0, 0.94, 0.91));
    
    return vec4(0.58, 0.7, 1.0, 1.);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.y;
    float t = iTime*4.0;
    vec4 bg = background(uv, t);
    
    vec4 fg = vec4(0.);
    int n = 5;
    if (uv.y < 0.5)
    for (int i = 0; i < n; i++){
        fg += foreground(uv, t+4.*float(i)/float(n)/60.) / (float(n));
    }
    
    vec3 col = bg.rgb;
    // train /////////////////////////////////////////////////////////////////////
    float k;
    float midlevel;
    float h;
    float disp;
    float dist;
    vec2 uv2;
    uv.y -= 0.2;
    // choo choo
    k = 1.0;
    uv2 = fract(uv*9.0);
    float wagon = 1.0;
    wagon *= 1.0 - step(0.45, uv.x);
    wagon *= 1.0 - step(0.115, uv.y);
    wagon *= step(0.103, uv.y);
    wagon *= step(0.05, 1.0 - abs(uv2.x*2.0 - 1.0));
    
    float join = 1.0; 
    join *= 1.0 - step(0.45, uv.x);
    join *= 1.0 - step(0.11, uv.y);
    join *= step(0.107, uv.y);
    
    
    float roof = 1.0;
    roof *= 1.0 - step(0.45, uv.x);
    roof *= 1.0 - step(0.117, uv.y);
    roof *= step(0.11, uv.y);
    roof *= step(0.15, 1.0 - abs(uv2.x*2.0 - 1.0));
    
    float loco = box(uv, 0.45, 0.5, 0.103, 0.112);
    float chem1 = box(uv, 0.49, 0.495, 0.103, 0.12);
    float chem2 = box(uv, 0.488, 0.496, 0.12, 0.123);
    float locoRoof = box(uv, 0.443, 0.47, 0.11, 0.117);
    
    float wheel = 1.0 - step(0.00004, dot2(uv - vec2(0.457, 0.106)));
    wheel += 1.0 - step(0.00002, dot2(uv - vec2(0.487, 0.105)));
    wheel += 1.0 - step(0.00002, dot2(uv - vec2(0.497, 0.105)));
    
    if (uv.x < 0.45 && uv.y > 0.025 && uv.y < 0.2){
        wheel += 1.0 - step(0.002, dot2(uv2 - vec2(0.2, 0.95)));
        wheel += 1.0 - step(0.002, dot2(uv2 - vec2(0.8, 0.95)));
    }
    col = mix(col, vec3(0.18, 0.12, 0.15), join);
    col =  mix(col, vec3(0.48, 0.19, 0.20), wagon);
    col = mix(col, vec3(0.18, 0.12, 0.15), roof);
    
    col = mix(col, vec3(0.38, 0.19, 0.20), loco);
    col = mix(col, vec3(0.38, 0.19, 0.20), chem1);
    col = mix(col, vec3(0.18, 0.12, 0.15), locoRoof);
    col = mix(col, vec3(0.18, 0.12, 0.15), chem2 + wheel);
    // loco smoke //////
    
    dist = 5.0;
    uv2 = uv + vec2(t/dist + 3.5, 0.0);
    uv2.x -= t/dist*0.2;
    h = fbm2(uv2, 8) - 0.55;
    
    if(uv.x < 0.49){
        float x = -uv.x + 0.49;
        float smokeWobble = 0.5 + 0.5*sin(iTime*0.3);
        float y = abs(uv.y + h*0.4 - smokeWobble * 0.35 * sqrt(x) - 0.12) - 0.8*x*exp(-x*10.0);
        if(y < 0.0) col = vec3(1.0, 0.94, 0.91);
        if(y < - 0.02) col = vec3(0.92 - tan(sin(iTime*0.7)), 0.85-sin(iTime*0.5), 0.82 - cos(iTime*0.6));
    }
    
    //bridge ///////
    dist = 5.0;
    uv2 = uv + vec2(t/dist + 32.5, 0.0);
    uv2.x = fract(uv2.x*3.0);
    k = 1.0;
    k *= smoothstep(0.001, 0.003, abs(uv2.y - pow(uv2.x - 0.5, 2.0)*0.15 - 0.12));
    k *= min(step(0.05, 1.0 - abs(uv2.x*2.0 - 1.0))
         +   step(0.17, uv2.y), 1.0);
    k *= min(smoothstep(0.02, 0.05, 1.0 - abs(uv2.x*2.0 - 1.0))
         +   step(0.177, uv2.y), 1.0);
         
    k *= min(step(0.1, uv2.y)
           + smoothstep(-0.09, -0.085, -uv2.y - 0.001/(1.0 - abs(uv2.x*2.0 - 1.0))), 1.0);
           
    k *= min(smoothstep(0.05, 0.2, 1.0 - abs(fract(uv2.x*16.0)*2.0 - 1.0))
         +   step(0.12, uv2.y - pow(uv2.x - 0.5, 2.0)*0.15)
         +   step(-0.1, -uv2.y), 1.0);
    col = mix(vec3(0.29, 0.09, 0.08)*smoothstep(-0.08, 0.08, uv.y), col, k);
    
    
    
    col = mix(col, fg.rgb, fg.a);

    // Output to screen
    uv = fragCoord/iResolution.xy;
    col = mix(col, texture(iChannel1, uv).rgb, 0.3);
    fragColor = vec4(col,1.0);
}`,

    "Ocean Waves": `// Math
const float PI = 3.1415926535897932384626433832795;
const float PI_2 = 1.57079632679489661923;
const float PI_4 = 0.785398163397448309616;

mat2 rot(float phi)
{
    float c = cos(phi);
    float s = sin(phi);

    return mat2(
        vec2(c, -s),
        vec2(s, c));
}

// SDF
float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return length(max(d,0.0));
}

float sdTriPrism( vec3 p, vec2 h )
{
  vec3 q = abs(p);
  return max(q.z-h.y,max(q.x*0.866025+p.y*0.5,-p.y)-h.x*0.5);
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}

float sdCappedCylinder( vec3 p, float h, float r )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdEllipsoid( vec3 p, vec3 r )
{
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
}


float sdSphere( vec3 p, float s )
{
  return length(p)-s;
}

// SDF Operators
float opSmoothSubtraction( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h); }

float opSmoothUnion( float d1, float d2, float k ) {
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h); }

float opSmoothIntersection( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) + k*h*(1.0-h); }

// Noise
vec2 random(vec2 st)
{
    st = vec2( dot(st,vec2(127.1,311.7)),
              dot(st,vec2(269.5,183.3)) );
    return -1.0 + 2.0*fract(sin(st)*43758.5453123);
}

float noiseo(vec2 st)
{
    vec2 f = fract(st);
    vec2 i = floor(st);
    
    vec2 u = f * f * f * (f * (f * 6. - 15.) + 10.);
    
    float r = mix( mix( dot( random(i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ),
                     dot( random(i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( random(i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ),
                     dot( random(i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
    return r * .5 + .5;
}

float fbm(vec2 st)
{
    float value = 0.;
    float amplitude = .5;
    float frequency = 0.;
    
    for (int i = 0; i < 8; i++)
    {
        value += amplitude * noiseo(st);
        st *= 2.;
        amplitude *= .5;
    }
    
    return value;
}

// Noise and FBM (as seen on iq tutorials)
//==========================================================================================
// hashes
//==========================================================================================

float hash1( vec2 p )
{
    p  = 50.0*fract( p*0.3183099 );
    return fract( p.x*p.y*(p.x+p.y) );
}

float hash1( float n )
{
    return fract( n*17.0*fract( n*0.3183099 ) );
}

vec2 hash2( float n ) { return fract(sin(vec2(n,n+1.0))*vec2(43758.5453123,22578.1459123)); }


vec2 hash2( vec2 p ) 
{
    const vec2 k = vec2( 0.3183099, 0.3678794 );
    p = p*k + k.yx;
    return fract( 16.0 * k*fract( p.x*p.y*(p.x+p.y)) );
}

float hash12(vec2 p)
{
	vec3 p3  = fract(vec3(p.xyx) * 443.8975);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

//==========================================================================================
// noises
//==========================================================================================

// value noise, and its analytical derivatives
vec4 noised( in vec3 x )
{
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);

    float n = p.x + 317.0*p.y + 157.0*p.z;
    
    float a = hash1(n+0.0);
    float b = hash1(n+1.0);
    float c = hash1(n+317.0);
    float d = hash1(n+318.0);
    float e = hash1(n+157.0);
	float f = hash1(n+158.0);
    float g = hash1(n+474.0);
    float h = hash1(n+475.0);

    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    return vec4( -1.0+2.0*(k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z), 
                      2.0* du * vec3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
                                      k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
                                      k3 + k6*u.x + k5*u.y + k7*u.x*u.y ) );
}

float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    
    float n = p.x + 317.0*p.y + 157.0*p.z;
    
    float a = hash1(n+0.0);
    float b = hash1(n+1.0);
    float c = hash1(n+317.0);
    float d = hash1(n+318.0);
    float e = hash1(n+157.0);
	float f = hash1(n+158.0);
    float g = hash1(n+474.0);
    float h = hash1(n+475.0);

    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    return -1.0+2.0*(k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z);
}

vec3 noised( in vec2 x )
{
    vec2 p = floor(x);
    vec2 w = fract(x);
    
    vec2 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec2 du = 30.0*w*w*(w*(w-2.0)+1.0);
    
    float a = hash1(p+vec2(0,0));
    float b = hash1(p+vec2(1,0));
    float c = hash1(p+vec2(0,1));
    float d = hash1(p+vec2(1,1));

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k4 = a - b - c + d;

    return vec3( -1.0+2.0*(k0 + k1*u.x + k2*u.y + k4*u.x*u.y), 
                      2.0* du * vec2( k1 + k4*u.y,
                                      k2 + k4*u.x ) );
}

float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 w = fract(x);
    vec2 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    
#if 0
    p *= 0.3183099;
    float kx0 = 50.0*fract( p.x );
    float kx1 = 50.0*fract( p.x+0.3183099 );
    float ky0 = 50.0*fract( p.y );
    float ky1 = 50.0*fract( p.y+0.3183099 );

    float a = fract( kx0*ky0*(kx0+ky0) );
    float b = fract( kx1*ky0*(kx1+ky0) );
    float c = fract( kx0*ky1*(kx0+ky1) );
    float d = fract( kx1*ky1*(kx1+ky1) );
#else
    float a = hash1(p+vec2(0,0));
    float b = hash1(p+vec2(1,0));
    float c = hash1(p+vec2(0,1));
    float d = hash1(p+vec2(1,1));
#endif
    
    return -1.0+2.0*( a + (b-a)*u.x + (c-a)*u.y + (a - b - c + d)*u.x*u.y );
}

//==========================================================================================
// fbm constructions
//==========================================================================================

const mat3 m3  = mat3( 0.00,  0.80,  0.60,
                      -0.80,  0.36, -0.48,
                      -0.60, -0.48,  0.64 );
const mat3 m3i = mat3( 0.00, -0.80, -0.60,
                       0.80,  0.36, -0.48,
                       0.60, -0.48,  0.64 );
const mat2 m2 = mat2(  0.80,  0.60,
                      -0.60,  0.80 );
const mat2 m2i = mat2( 0.80, -0.60,
                       0.60,  0.80 );

//------------------------------------------------------------------------------------------

float fbm_2( in vec3 x )
{
    float f = 2.0;
    float s = 0.5;
    float a = 0.0;
    float b = 0.5;
    for( int i=0; i<2; i++ )
    {
        float n = noise(x);
        a += b*n;
        b *= s;
        x = f*m3*x;
    }
	return a;
}

float fbm_4( in vec3 x )
{
    float f = 2.0;
    float s = 0.5;
    float a = 0.0;
    float b = 0.5;
    for( int i=0; i<4; i++ )
    {
        float n = noise(x);
        a += b*n;
        b *= s;
        x = f*m3*x;
    }
	return a;
}

#define DRAG_MULT 0.048

vec2 wavedx(vec2 position, vec2 direction, float speed, float frequency, float timeshift) {
    float x = dot(direction, position) * frequency + timeshift * speed;
    float wave = exp(sin(x) - 1.0);
    float dx = wave * cos(x);
    return vec2(wave, -dx);
}

float getwaves(vec2 position, int iterations, float time){
	float iter = 0.0;
    float phase = 6.0;
    float speed = 2.0;
    float weight = 1.0;
    float w = 0.0;
    float ws = 0.0;
    for(int i=0;i<iterations;i++){
        vec2 p = vec2(sin(iter), cos(iter));
        vec2 res = wavedx(position, p, speed, phase, time);
        position += normalize(p) * res.y * weight * DRAG_MULT;
        w += res.x * weight;
        iter += 12.0;
        ws += weight;
        weight = mix(weight, 0.0, 0.2);
        phase *= 1.18;
        speed *= 1.07;
    }
    return w / ws;
}



#define AA 1

#define ZERO int(min(iFrame, 0.))
#define MAX_STEPS			200
#define MAX_DIST			25.0
#define SURFACE_DIST		0.001

#define Time iTime
#define clamp01(x) max(min(x, 1.0), 0.0)

vec3 ro;

float getwaves(vec2 position, int iterations, float time);
float fbm_2( in vec3 x );

vec2 map(vec3 p, bool complete)
{
    vec2 v = vec2(MAX_DIST, 0.0);
    
    // water
    float final = getwaves(p.xz * .35, 20, iTime * .5) * (getwaves(p.xz * .15 + vec2(2.2, 2.2), 3, iTime * .5) * 1.5 + .4) * 1.05;
    float f = dot(p, vec3(0.0, 1.0, 0.0)) - final;
    v = vec2(f, 1.0);
    
    return v;
}

vec3 calcNormal(vec3 p)
{
    // inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
    vec3 n = vec3(0.0);
    for( int i=ZERO; i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(p+0.0001*e, true).x;
    }
    return normalize(n);
}

vec2 rayMarch(vec3 ro, vec3 rd)
{
    float t = 0.0;
    vec3 p;
    vec2 obj;
    for (int i = 0; i < MAX_STEPS; i++)
    {
        p = ro + t * rd;
       	
        obj = map(p, true);
        
        if (abs(obj.x) < SURFACE_DIST || abs(t) > MAX_DIST) break;
        
        t += obj.x;
    }
    
    obj.x = t;
    return obj;
}

// Lighting
float ambientOcclusion(vec3 p, vec3 n)
{
	float stepSize = 0.0026;
	float t = stepSize;
	float oc = 0.0;
	for(int i = 0; i < 10; ++i)
	{
		vec2 obj = map(p + n * t, true);
		oc += t - obj.x;
		t += pow(float(i), 1.85) * stepSize;
	}

	return 1.0 - clamp(oc * 0.5, 0.0, 1.0);
}

float getVisibility(vec3 p0, vec3 p1, float k)
{
	vec3 rd = normalize(p1 - p0);
	float t = 10.0 * SURFACE_DIST;
	float maxt = length(p1 - p0);
	float f = 1.0;
	while(t < maxt || t < MAX_DIST)
	{
		vec2 o = map(p0 + rd * t, false);

		if(o.x < SURFACE_DIST)
			return 0.0;

		f = min(f, k * o.x / t);

		t += o.x;
	}

	return f;
}

#define iterations 17
#define formuparam 0.53

#define volsteps 20
#define stepsize 0.1

#define zoom   0.800
#define tile   0.850
#define speed  0.010 

#define brightness 0.0015
#define darkmatter 0.300
#define distfading 0.730
#define saturation 0.850

vec3 stars(vec2 uv)
{
	vec3 dir=vec3(uv*zoom,1.);
	float time=iTime*speed+.25;

	// timed rotation (replaces mouse)
	float a1 = .5 + sin(iTime * 0.2) * 1.0;
	float a2 = .8 + cos(iTime * 0.17) * 1.0;
	mat2 rot1=mat2(cos(a1),sin(a1),-sin(a1),cos(a1));
	mat2 rot2=mat2(cos(a2),sin(a2),-sin(a2),cos(a2));
	dir.xz*=rot1;
	dir.xy*=rot2;
	vec3 from=vec3(1.,.5,0.5);
	from+=vec3(time*2.,time,-2.);
	from.xz*=rot1;
	from.xy*=rot2;
	
	//volumetric rendering
	float s=0.1,fade=1.;
	vec3 v=vec3(0.);
	for (int r=0; r<volsteps; r++) {
		vec3 p=from+s*dir*.5;
		p = abs(vec3(tile)-mod(p,vec3(tile*2.))); // tiling fold
		float pa,a=pa=0.;
		for (int i=0; i<iterations; i++) { 
			p=abs(p)/dot(p,p)-formuparam; // the magic formula
			a+=abs(length(p)-pa); // absolute sum of average change
			pa=length(p);
		}
		float dm=max(0.,darkmatter-a*a*.001); //dark matter
		a*=a*a; // add contrast
		if (r>6) fade*=1.-dm; // dark matter, don't render near
		//v+=vec3(dm,dm*.5,0.);
		v+=fade;
		v+=vec3(s,s*s,s*s*s*s)*a*brightness*fade; // coloring based on distance
		fade*=distfading; // distance fading
		s+=stepsize;
	}
	v=mix(vec3(length(v)),v,saturation); //color adjust
  return v;
}

// Renderer
vec4 render(vec2 obj, vec3 p, vec3 rd, vec2 uv)
{
    vec3 col;
    
    vec3 normal = calcNormal(p);
    
    const vec3 background_color = vec3(0.0, 0.01, 0.02);
    vec3 background = background_color;
    
    vec2 pos = uv - vec2(0.0, 0.2) - vec2(0.0, 0.2) * sin(iTime * 0.5) * 0.1;
    background += pow(clamp01(1.0 - length(pos * 1.5)), 1.9) * background * 20.0;
    background += pow(clamp01(1.0 - length(pos * 6.5)), 3.9) * background * 90.0;
    
    // float n = fbm_2(vec3(pos * 52.0 + iTime * 0.5, 1.0)) * 1.8;
    // n = smoothstep(0.72, 0.78, n) * 8.5;

    vec3 stars = stars(uv);    
    background += 0.001*stars;
    float c = 1.0;
    
    if (obj.x >= MAX_DIST)
    {
        col = background;
    }
    else
    {
        vec3 albedo = vec3(0.0, 0.0, 0.0);
        
        float a = pow(1.0 - clamp(dot(-rd, normal), 0.0, 1.0), 2.6);
        float m = pow(length(ro - p) * 0.2, 1.4) * 0.8;

        c = pow(clamp01(1.0 - length((uv - vec2(0.0, -0.4)) * .4)), 5.0) * 3.0;

        float diff_mask = a * m * c;
        float ambient_mask = a * m + .06;
        albedo = vec3(0.0, 0.044, 0.09) * 10.0;
        float spec_power = 80.0;
        float spec_mask = 6.7 * m;
        
        // Moon Light
        #if 1
        {
            const vec3 light_pos = vec3(-0.0, 40.0, 100.4);
            const vec3 light_col = vec3(0.2, 0.2, 0.2);
			vec3 refd = reflect(rd, normal);
            vec3 light_dir = normalize(light_pos - p);
            
            float diffuse = dot(light_dir, normal);
            float visibility = getVisibility(p, light_pos, 10.0);
        	float spec = pow(max(0.0, dot(refd, light_dir)), spec_power);

            col += diff_mask * diffuse * albedo * visibility * light_col * 1.86;
            col += spec * (light_col * albedo) * spec_mask * visibility * c;
        }
        #endif
        
        // Fill Light
        #if 1
        {
            const vec3 light_pos = vec3(0.0, 100.0, 0.0);
            const vec3 light_col = vec3(0.0, 0.4, 0.2);
			vec3 refd = reflect(rd, normal);
            vec3 light_dir = normalize(light_pos - p);
            
            float diffuse = dot(light_dir, normal);
            float visibility = getVisibility(p, light_pos, 10.0);
        	float spec = pow(max(0.0, dot(refd, light_dir)), spec_power);

            col += diff_mask * diffuse * albedo * visibility * light_col * .1;
            col += spec * (light_col * albedo) * spec_mask * visibility * .03;
        }
        #endif
        
        
        // Ambient light
        #if 1
        col += albedo * 0.2 * ambient_mask;
        #endif
    }
    
    return vec4(col, obj.x);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
  
    float v = 1.7 + sin(iTime * 0.5) * 0.5;
    const vec3 ta = vec3(0.0, 0.0, 20.0);
    vec3 ro = vec3(
        0.0,
        v,
        0.0
    );

    vec4 tot = vec4(0.0);
#if AA>1
    for(int m=ZERO; m<AA; m++)
    for(int n=ZERO; n<AA; n++)
    {
        vec2 o = vec2(float(m), float(n)) / float(AA) - 0.5;
        vec2 uv = (2.0 * (fragCoord + o) - iResolution.xy) / iResolution.y;
#else    
    	vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
#endif       
        // Ray direction
        vec3 ww = normalize(ta - ro);
        vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
        vec3 vv = normalize(cross(uu, ww));
        
        vec3 rd = normalize(uv.x * uu + uv.y * vv + 2.3 * ww);
        
        // render	
        vec2 obj = rayMarch(ro, rd);
        vec3 p = ro + obj.x * rd;
    
   		vec4 col = render(obj, p, rd, uv);
        
        tot += col;
#if AA>1
    }
    tot /= float(AA*AA);
#endif

    fragColor = vec4(tot)*2.1;
}`,

    "Pianoscope Text": `// Source edited by David Hoskins - 2013.

// I took and completed this http://glsl.heroku.com/e#9743.20 - just for fun! 8|
// Locations in 3x7 font grid, inspired by http://www.claudiocc.com/the-1k-notebook-part-i/
// Had to edit it to remove some duplicate lines.
// ABC  a:GIOMJL b:AMOIG c:IGMO d:COMGI e:OMGILJ f:CBN g:OMGIUS h:AMGIO i:EEHN j:GHTS k:AMIKO l:BN m:MGHNHIO n:MGIO
// DEF  o:GIOMG p:SGIOM q:UIGMO r:MGI s:IGJLOM t:BNO u:GMOI v:GJNLI w:GMNHNOI x:GOKMI y:GMOIUS z:GIMO
// GHI
// JKL 
// MNO
// PQR
// STU

vec2 coord;

#define font_size 20. 
#define font_spacing .045
vec2 caret_origin = vec2(1.0, .62);
vec2 caret;

#define STROKEWIDTH 0.05
#define PI 3.14159265359

#define A_ vec2(0.,0.)
#define B_ vec2(1.,0.)
#define C_ vec2(2.,0.)

//#define D_ vec2(0.,1.)
#define E_ vec2(1.,1.)
//#define F_ vec2(2.,1.)

#define G_ vec2(0.,2.)
#define H_ vec2(1.,2.)
#define I_ vec2(2.,2.)

#define J_ vec2(0.,3.)
#define K_ vec2(1.,3.)
#define L_ vec2(2.,3.)

#define M_ vec2(0.,4.)
#define N_ vec2(1.,4.)
#define O_ vec2(2.,4.)

//#define P_ vec2(0.,5.)
//#define Q_ vec2(1.,5.)
//#define R_ vec2(1.,5.)

#define S_ vec2(0.,6.)
#define T_ vec2(1.,6.)
#define U_ vec2(2.0,6.)

#define A(p) t(G_,I_,p) + t(I_,O_,p) + t(O_,M_, p) + t(M_,J_,p) + t(J_,L_,p);caret.x += 1.0;
#define B(p) t(A_,M_,p) + t(M_,O_,p) + t(O_,I_, p) + t(I_,G_,p);caret.x += 1.0;
#define C(p) t(I_,G_,p) + t(G_,M_,p) + t(M_,O_,p);caret.x += 1.0;
#define D(p) t(C_,O_,p) + t(O_,M_,p) + t(M_,G_,p) + t(G_,I_,p);caret.x += 1.0;
#define E(p) t(O_,M_,p) + t(M_,G_,p) + t(G_,I_,p) + t(I_,L_,p) + t(L_,J_,p);caret.x += 1.0;
#define F(p) t(C_,B_,p) + t(B_,N_,p) + t(G_,I_,p);caret.x += 1.0;
#define G(p) t(O_,M_,p) + t(M_,G_,p) + t(G_,I_,p) + t(I_,U_,p) + t(U_,S_,p);caret.x += 1.0;
#define H(p) t(A_,M_,p) + t(G_,I_,p) + t(I_,O_,p);caret.x += 1.0;
#define I(p) t(E_,E_,p) + t(H_,N_,p);caret.x += 1.0;
#define J(p) t(E_,E_,p) + t(H_,T_,p) + t(T_,S_,p);caret.x += 1.0;
#define K(p) t(A_,M_,p) + t(M_,I_,p) + t(K_,O_,p);caret.x += 1.0;
#define L(p) t(B_,N_,p);caret.x += 1.0;
#define M(p) t(M_,G_,p) + t(G_,I_,p) + t(H_,N_,p) + t(I_,O_,p);caret.x += 1.0;
#define N(p) t(M_,G_,p) + t(G_,I_,p) + t(I_,O_,p);caret.x += 1.0;
#define O(p) t(G_,I_,p) + t(I_,O_,p) + t(O_,M_, p) + t(M_,G_,p);caret.x += 1.0;
#define P(p) t(S_,G_,p) + t(G_,I_,p) + t(I_,O_,p) + t(O_,M_, p);caret.x += 1.0;
#define Q(p) t(U_,I_,p) + t(I_,G_,p) + t(G_,M_,p) + t(M_,O_, p);caret.x += 1.0;
#define R(p) t(M_,G_,p) + t(G_,I_,p);caret.x += 1.0;
#define S(p) t(I_,G_,p) + t(G_,J_,p) + t(J_,L_,p) + t(L_,O_,p) + t(O_,M_,p);caret.x += 1.0;
#define T(p) t(B_,N_,p) + t(N_,O_,p) + t(G_,I_,p);caret.x += 1.0;
#define U(p) t(G_,M_,p) + t(M_,O_,p) + t(O_,I_,p);caret.x += 1.0;
#define V(p) t(G_,J_,p) + t(J_,N_,p) + t(N_,L_,p) + t(L_,I_,p);caret.x += 1.0;
#define W(p) t(G_,M_,p) + t(M_,O_,p) + t(N_,H_,p) + t(O_,I_,p);caret.x += 1.0;
#define X(p) t(G_,O_,p) + t(I_,M_,p);caret.x += 1.0;
#define Y(p) t(G_,M_,p) + t(M_,O_,p) + t(I_,U_,p) + t(U_,S_,p);caret.x += 1.0;
#define Z(p) t(G_,I_,p) + t(I_,M_,p) + t(M_,O_,p);caret.x += 1.0;
#define STOP(p) t(N_,N_,p);caret.x += 1.0;


//-----------------------------------------------------------------------------------
float minimum_distance(vec2 v, vec2 w, vec2 p)
{	// Return minimum distance between line segment vw and point p
  	float l2 = (v.x - w.x)*(v.x - w.x) + (v.y - w.y)*(v.y - w.y); //length_squared(v, w);  // i.e. |w-v|^2 -  avoid a sqrt
  	if (l2 == 0.0) {
		return distance(p, v);   // v == w case
	}
	
	// Consider the line extending the segment, parameterized as v + t (w - v).
  	// We find projection of point p onto the line.  It falls where t = [(p-v) . (w-v)] / |w-v|^2
  	float t = dot(p - v, w - v) / l2;
  	if(t < 0.0) {
		// Beyond the 'v' end of the segment
		return distance(p, v);
	} else if (t > 1.0) {
		return distance(p, w);  // Beyond the 'w' end of the segment
	}
  	vec2 projection = v + t * (w - v);  // Projection falls on the segment
	return distance(p, projection);
}

//-----------------------------------------------------------------------------------
float textColor(vec2 from, vec2 to, vec2 p)
{
	p *= font_size;
	float inkNess = 0., nearLine, corner;
	nearLine = minimum_distance(from,to,p); // basic distance from segment, thanks http://glsl.heroku.com/e#6140.0
	inkNess += smoothstep(0., 1., 1.- 14.*(nearLine - STROKEWIDTH)); // ugly still
	inkNess += smoothstep(0., 2.5, 1.- (nearLine  + 5. * STROKEWIDTH)); // glow
	return inkNess;
}

//-----------------------------------------------------------------------------------
vec2 grid(vec2 letterspace) 
{
	return ( vec2( (letterspace.x / 2.) * .65 , 1.0-((letterspace.y / 2.) * .95) ));
}

//-----------------------------------------------------------------------------------
float count = 0.0;
float gtime;
float t(vec2 from, vec2 to, vec2 p) 
{
	count++;
	if (count > gtime*20.0) return 0.0;
	return textColor(grid(from), grid(to), p);
}

//-----------------------------------------------------------------------------------
vec2 r()
{
	vec2 pos = coord.xy/iResolution.xy;
	pos.y -= caret.y;
	pos.x -= font_spacing*caret.x;
	return pos;
}

//-----------------------------------------------------------------------------------
void _()
{
	caret.x += 1.5;
}

//-----------------------------------------------------------------------------------
void newline()
{
	caret.x = caret_origin.x;
	caret.y -= .18;
}

//-----------------------------------------------------------------------------------
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float time = mod(iTime, 11.0);
    gtime = time;

	float d = 0.;
	vec3 col = vec3(0.1, .07+0.07*(.5+sin(fragCoord.y*3.14159*1.1+time*2.0)) + sin(fragCoord.y*.01+time+2.5)*0.05, 0.1);
    
    coord = fragCoord;
	
	caret = caret_origin;

	// pianoscope (centered)
	caret.x += 5.0;
	d += P(r()); d += I(r()); d += A(r()); d += N(r()); d += O(r());
	d += S(r()); d += C(r()); d += O(r()); d += P(r()); d += E(r());
	newline();
    // the listening floor
	d += T(r()); d += H(r()); d += E(r()); _();
	d += L(r()); d += I(r()); d += S(r()); d += T(r()); d += E(r());
	d += N(r()); d += I(r()); d += N(r()); d += G(r()); _();
	d += F(r()); d += L(r()); d += O(r()); d += O(r()); d += R(r());
	d = clamp(d* (.75+sin(fragCoord.x*PI*.5-time*4.3)*.5), 0.0, 1.0);
      
    col += vec3(d*.5, d, d*.85);
	vec2 xy = fragCoord.xy / iResolution.xy;
	col *= vec3(.4, .4, .3) + 0.5*pow(100.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), .4 );	
    fragColor = vec4( col, 1.0 );
}`,

    "Neon Pianoscope Text": `// pianoscope logo — heart-style glow + horizontal neon gradient
// https://www.shadertoy.com/view/7l3GDS

#define PS_FONT 20.0
#define PS_SPACING 0.045
#define PS_Y 0.52
#define PS_LETTERS 10.0
#define PS_CARET0 6.0
#define PS_RADIUS (0.00075 * PS_FONT)
#define PS_CORE_A (0.00030 * PS_FONT)
#define PS_CORE_B (0.00010 * PS_FONT)
#define PS_BLOOM 2.4
#define PS_POWER 1.3

vec2 coord;
vec2 caret;
float textDist;

#define A_ vec2(0.0, 0.0)
#define B_ vec2(1.0, 0.0)
#define C_ vec2(2.0, 0.0)
#define E_ vec2(1.0, 1.0)
#define G_ vec2(0.0, 2.0)
#define H_ vec2(1.0, 2.0)
#define I_ vec2(2.0, 2.0)
#define J_ vec2(0.0, 3.0)
#define K_ vec2(1.0, 3.0)
#define L_ vec2(2.0, 3.0)
#define M_ vec2(0.0, 4.0)
#define N_ vec2(1.0, 4.0)
#define O_ vec2(2.0, 4.0)
#define S_ vec2(0.0, 6.0)
#define T_ vec2(1.0, 6.0)
#define U_ vec2(2.0, 6.0)

float getGlow(float dist, float radius, float intensity) {
    return pow(radius / max(dist, 1e-4), intensity);
}

float minimum_distance(vec2 v, vec2 w, vec2 p) {
    float l2 = dot(v - w, v - w);
    if (l2 == 0.0) return distance(p, v);
    float t = clamp(dot(p - v, w - v) / l2, 0.0, 1.0);
    return distance(p, v + t * (w - v));
}

vec2 grid(vec2 letterspace) {
    return vec2((letterspace.x / 2.0) * 0.65, 1.0 - ((letterspace.y / 2.0) * 0.95));
}

void addSeg(vec2 from, vec2 to, vec2 p) {
    vec2 pp = p * PS_FONT;
    textDist = min(textDist, minimum_distance(grid(from), grid(to), pp));
}

#define SEG(f, t, p) addSeg(f, t, p)
#define A(p) SEG(G_, I_, p); SEG(I_, O_, p); SEG(O_, M_, p); SEG(M_, J_, p); SEG(J_, L_, p); caret.x += 1.0;
#define B(p) SEG(A_, M_, p); SEG(M_, O_, p); SEG(O_, I_, p); SEG(I_, G_, p); caret.x += 1.0;
#define C(p) SEG(I_, G_, p); SEG(G_, M_, p); SEG(M_, O_, p); caret.x += 1.0;
#define D(p) SEG(C_, O_, p); SEG(O_, M_, p); SEG(M_, G_, p); SEG(G_, I_, p); caret.x += 1.0;
#define E(p) SEG(O_, M_, p); SEG(M_, G_, p); SEG(G_, I_, p); SEG(I_, L_, p); SEG(L_, J_, p); caret.x += 1.0;
#define F(p) SEG(C_, B_, p); SEG(B_, N_, p); SEG(G_, I_, p); caret.x += 1.0;
#define G(p) SEG(O_, M_, p); SEG(M_, G_, p); SEG(G_, I_, p); SEG(I_, U_, p); SEG(U_, S_, p); caret.x += 1.0;
#define H(p) SEG(A_, M_, p); SEG(G_, I_, p); SEG(I_, O_, p); caret.x += 1.0;
#define I(p) SEG(E_, E_, p); SEG(H_, N_, p); caret.x += 1.0;
#define J(p) SEG(E_, E_, p); SEG(H_, T_, p); SEG(T_, S_, p); caret.x += 1.0;
#define K(p) SEG(A_, M_, p); SEG(M_, I_, p); SEG(K_, O_, p); caret.x += 1.0;
#define L(p) SEG(B_, N_, p); caret.x += 1.0;
#define M(p) SEG(M_, G_, p); SEG(G_, I_, p); SEG(H_, N_, p); SEG(I_, O_, p); caret.x += 1.0;
#define N(p) SEG(M_, G_, p); SEG(G_, I_, p); SEG(I_, O_, p); caret.x += 1.0;
#define O(p) SEG(G_, I_, p); SEG(I_, O_, p); SEG(O_, M_, p); SEG(M_, G_, p); caret.x += 1.0;
#define P(p) SEG(S_, G_, p); SEG(G_, I_, p); SEG(I_, O_, p); SEG(O_, M_, p); caret.x += 1.0;
#define Q(p) SEG(U_, I_, p); SEG(I_, G_, p); SEG(G_, M_, p); SEG(M_, O_, p); caret.x += 1.0;
#define R(p) SEG(M_, G_, p); SEG(G_, I_, p); caret.x += 1.0;
#define S(p) SEG(I_, G_, p); SEG(G_, J_, p); SEG(J_, L_, p); SEG(L_, O_, p); SEG(O_, M_, p); caret.x += 1.0;
#define T(p) SEG(B_, N_, p); SEG(N_, O_, p); SEG(G_, I_, p); caret.x += 1.0;
#define U(p) SEG(G_, M_, p); SEG(M_, O_, p); SEG(O_, I_, p); caret.x += 1.0;
#define V(p) SEG(G_, J_, p); SEG(J_, N_, p); SEG(N_, L_, p); SEG(L_, I_, p); caret.x += 1.0;
#define W(p) SEG(G_, M_, p); SEG(M_, O_, p); SEG(N_, H_, p); SEG(O_, I_, p); caret.x += 1.0;
#define X(p) SEG(G_, O_, p); SEG(I_, M_, p); caret.x += 1.0;
#define Y(p) SEG(G_, M_, p); SEG(M_, O_, p); SEG(I_, U_, p); SEG(U_, S_, p); caret.x += 1.0;
#define Z(p) SEG(G_, I_, p); SEG(I_, M_, p); SEG(M_, O_, p); caret.x += 1.0;

float psAspect() { return iResolution.x / iResolution.y; }

float psMidCaret() { return PS_CARET0 + (PS_LETTERS - 1.0) * 0.5; }

vec2 textPos() {
    vec2 uv = coord.xy / iResolution.xy;
    float aspect = psAspect();
    vec2 p;
    p.x = (uv.x - 0.5) * aspect + (psMidCaret() - caret.x) * PS_SPACING * aspect;
    p.y = uv.y - PS_Y;
    return p;
}

float wordT(vec2 uv) {
    float halfW = (PS_LETTERS - 1.0) * PS_SPACING * 0.5;
    return smoothstep(0.5 - halfW, 0.5 + halfW, uv.x);
}

vec3 gradTint(float t) {
    vec3 pink = vec3(1.0, 0.12, 0.48);
    vec3 cyan = vec3(0.18, 0.62, 1.0);
    vec3 purp = vec3(0.72, 0.22, 0.95);
    vec3 c = mix(pink, cyan, smoothstep(0.0, 0.48, t));
    return mix(c, purp, smoothstep(0.52, 1.0, t));
}

vec3 neonLit(float dist, vec3 tint) {
    vec3 col = vec3(0.0);
    col += getGlow(dist, PS_RADIUS * PS_BLOOM, PS_POWER * 0.9) * tint * 0.22;
    col += getGlow(dist, PS_RADIUS, PS_POWER) * tint * 0.65;
    col += vec3(1.0) * smoothstep(PS_CORE_A, PS_CORE_B, dist) * 8.0;
    return col;
}

float drawLogo() {
    caret = PS_CARET0;
    textDist = 1e4;
    P(textPos()); I(textPos()); A(textPos()); N(textPos()); O(textPos());
    S(textPos()); C(textPos()); O(textPos()); P(textPos()); E(textPos());
    return textDist;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    coord = fragCoord;
    vec2 uv = fragCoord.xy / iResolution.xy;
    float dist = drawLogo();
    vec3 col = neonLit(dist, gradTint(wordT(uv)));

    col = 1.0 - exp(-col);
    col = pow(col, vec3(0.4545));

    fragColor = vec4(col, 1.0);
}`,
    "Wave Zoom": `const int numWaves = 8;
const float numFreqs = 10.0;
const float meanFreq = 4.0;
const float stdDev = 1.5;
const float period = .32 * 4.;
const float ln2 = log(2.0);
const float mean = meanFreq * .69314718;

float wavething(int n, float x){
    float l = ln2 * float(n) + log(x);
    l -= mean;
    return exp(-l * l / stdDev) / 2.0;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord)
{
    vec2 textureUV = fragCoord / iResolution.xy;
    float pi = 4.0 * atan(1.0);
	float pi2 = 2.5 * pi;
    
    // oscillate numStripes based on time
    float max = 1.;
    float min = .23;
    float oscilationRange = (max - min)/2.0;
    float oscilationOffset = oscilationRange + min;
    float numStripes = oscilationOffset + sin(iTime / period) * oscilationRange;
    
    // oscillate pi values for additional movement
    max = 4.25 * atan(1.0);
    min = 3.0 * atan(1.0);
    oscilationRange = (max - min)/2.0;
    oscilationOffset = oscilationRange + min;
    pi = oscilationOffset + sin(iTime / period * 0.1) * oscilationRange;
    pi2 = 2.5 * pi;
    
    
    vec2 xy = pi2 * numStripes
        * (2.0 * fragCoord.xy / iResolution.y - vec2(iResolution.x / iResolution.y, 1.0));
    
    // oscillate position offset based on time
    vec2 xyCalc = pi2 * numStripes * (2.0 * vec2(0.0) / iResolution.y - vec2(iResolution.x / iResolution.y, 1.0));
    float xCalc = xyCalc.x;
    float yCalc = xyCalc.y;
    
    max = xCalc * -0.025;
    min = xCalc * 0.025;
    oscilationRange = (max - min)/2.0;
    oscilationOffset = oscilationRange + min;
    float xVal = oscilationOffset + sin(iTime / 1.) * oscilationRange;
    
    max = yCalc * -0.025;
    min = yCalc * 0.025;
    oscilationRange = (max - min)/2.0;
    oscilationOffset = oscilationRange + min;
    float yVal = oscilationOffset + sin(iTime / .5) * oscilationRange;
    
    xy.x -= xVal;
    xy.y -= yVal;
    
    // pattern calculations
    float scale = exp2(-fract(iTime / period));
    float sum1 = 0.0;
    for(int n = 0; n < int(numFreqs); n++)
    {
        sum1 += wavething(n, scale);
    }
    float sum2 = 0.0;
    for(int n = 0; n < numWaves; n++)
    {
        float theta = pi * float(n) / float(numWaves);
        vec2 waveVec = vec2(cos(theta), sin(theta));
        float phase = dot(xy, waveVec);
        for(int k = 0; k < int(numFreqs); k++){
            sum2 += cos(phase * scale * exp2(float(k))) * wavething(k, scale);
        }
    }
    
    // overal rgb tint
    float rTint = 0.00;
    float gTint = 0.75;
    float bTint = 0.75;
    fragColor = vec4(rTint, gTint, bTint, 1.0);
    
    // main zoom pattern rgb
    float rVal = sum2 / sum1 * sin(iTime / period * .5);
    float gVal = sum2 / sum1 * cos(iTime / period * .5);
    float bVal = sum2 / sum1 * cos(iTime / period * .5);
	fragColor += vec4(rVal, gVal, bVal, 1.0);
    
    // the intensity of the rgb vals (just used to tone stuff down)
    float rIntensity = 0.3;
    float gIntensity = 0.3;
    float bIntensity = 0.4;
    fragColor = vec4(fragColor.r * rIntensity, fragColor.g * gIntensity, fragColor.b * bIntensity, 1.0);
    
    // apply texture to specific colors
    float rTexAmt = 0.25;
    float gTexAmt = 0.0;
    float bTexAmt = 0.0;
    vec3 col = texture(iChannel0, textureUV).rgb;
    col.r *= fragColor.r * rTexAmt;
    col.g *= fragColor.g * gTexAmt;
    col.b *= fragColor.b * bTexAmt;
    fragColor += vec4(col, 1.0);
    
    // apply vignette
    float vignette = distance(textureUV, vec2(0.5));
    vignette = mix(1.5, -0.5, vignette);
    fragColor *= vignette;
}`,
    "Mandel's Infinite 4D Circus": `// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ⚛️ Mandel's Infinite 4D Circus (Spherical/Equirectangular projection)
// License CC0-1.0
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Raymarched 4D mandlebulbs! Mostly a success :)
//
// A bit out of my depth here, but it works. I'm no stranger to 4D space
// though still this is probably overkill for my first foray into raymarching.
// 
//
// EDIT: June 29, 2024
// Finally figured out how to ID the bulbs for psuedo-unique 
// colors. My repeat function was hiding the secret info vis-via dropped quotient.
// Also found a way to combine normal + distance lighting that I was happy with, 
// added some rotation to the camera, generally re-factored a few things, and made 
// the mandlebulbs snakes via vec3 raymarchRepeat=vec3(3.,1.,3).
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


bool flatmode=false; // < Removes equirectangular projection
bool trippy4Dnormals=false; // < Shows you the 4D normals
vec3 raymarchRepeat=vec3(3.,3.,3.); // The reapeat-frequency of bulbs in XYZ used for ray-marching 
//                      ^= (3.,1.,3) for mandlebulbsnakes

// Raymarch Settings:
const int MAX_STEPS = 100;
const float MAX_DIST = 100.0;
const float SURF_DIST = 0.01; //< Smaller for higher quality

// Mandel SDF Settings:
const int ITERATIONS = 10;
const float BAILOUT = 2.0;
const float POWER = 8.0;//< Play with this one

// Tasty SDF function to calculate the distance to a 4D Mandelbulb. 
float sdMandelbulb4D(vec4 p) {


    vec4 z = p;
    float dr = 1.0;
    float r = 0.0;

    for (int i = 0; i < ITERATIONS; i++) {
        r = length(z);
        if (r > BAILOUT) break;

        // Convert to polar coordinates
        float theta = acos(z.w / r);
        float phi = atan(z.z, z.x);
        float psi = atan(z.y, length(z.xz));

        // Scale and rotate the point
        dr = pow(r, POWER - 1.0) * POWER * dr + 1.0;
        float zr = pow(r, POWER);
        theta *= POWER;
        phi *= POWER;
        psi *= POWER;

        // Convert back to Cartesian coordinates
        z = zr * vec4(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta) * cos(psi), cos(theta) * sin(psi));
        z += p;
    }
    return 0.5 * log(r) * r / dr;
}

// Function to repeat space in 4D
vec4 repeat(vec4 p, vec4 c, out vec4 quotient) {
    quotient = floor(p/c);
    return mod(p,c) - 0.5 * c;
}

// Function to compute the normal using the gradient of the SDF
vec4 getNormal(vec4 p) {
    vec4 eps = vec4(0.001, 0.0, 0.0, 0.0);
    float d = sdMandelbulb4D(p);
    vec4 normal;
    normal.x = sdMandelbulb4D(p + eps.xwww) - d;
    normal.y = sdMandelbulb4D(p + eps.wxww) - d;
    normal.z = sdMandelbulb4D(p + eps.wwxw) - d;
    normal.w = sdMandelbulb4D(p + eps.wwwx) - d;
    return normalize(normal);
}

// Does raymarching stuff w/mandelBulb4D SDF.
// \`out float marches\` => used in our final color-mix.
// \`out vec4 p\`        => used for normal calculations
// \`out vec4 id\`       => usef for psuedo-unique bulb-color.
float raymarch(vec4 ro, vec4 rd, out vec4 p, out float marches, out vec4 id) {
    float t = 0.0;

    marches=0.;
    vec4 qTotal=vec4(0.);
    for (int i = 0; i < MAX_STEPS; i++) {
        vec4 pos4D = ro + t * rd;
        vec4 quotient;
        vec4 repeatPos4D = repeat(pos4D, vec4(raymarchRepeat, 2.),quotient); // Repeat space in 4D
        marches+=1.;
        float d = sdMandelbulb4D(repeatPos4D);
        if (d < SURF_DIST) {
            p = repeatPos4D; // p for normals
            id = quotient; // id for psuedo-unique color
            return t;// return distance from bulb
        }
        t += d;
        if (t > MAX_DIST) {
            break;
        }
    }
    return MAX_DIST;
}

// This function let's us jump from 2D-UV to spherical 3D-XYZ position
// The jist is that XY of UV can represent 2-Sphere angles to get a point on the sphere.
// The 2-Sphere point than gives you an XYZ normalized [-1,1].
vec3 uv3D(vec2 uv) {
    float theta = uv.x * 2.0 * 3.14159265359; // Longitude
    float phi = uv.y * 3.14159265359; // Latitude
    float x = sin(phi) * cos(theta);
    float y = sin(phi) * sin(theta);
    float z = cos(phi);
    // { Dev Note }
    // If you're porting this shader to a material, I strongly recommend you skip this function 
    // and just use the XYZ of your \`varying vNormal\` in place of the result you would get here.
    // Should be suitable for all spheres and most round geometries
    return vec3(x, y, z);
}
vec3 randomColor(float seed) {
    float r = fract(sin(seed * 69.42069 + 70.333) * 43758.5453);
    float g = mod(fract(sin((seed + 1.0) * 39.3467 + 57.583) * 43758.5453)+iTime,1.);
    float b = fract(sin((seed + 2.0) * 73.1562 + 25.345) * 43758.5453);
    return vec3(r, g, b);
}
vec3 rotateX(vec3 v, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    mat3 rotationMatrix = mat3(
        1.0, 0.0, 0.0,
        0.0, c,   -s,
        0.0, s,   c
    );
    return rotationMatrix * v;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{

    // Default color is black
    vec3 color = vec3(0.0);

    // Handle UVs for spherical 3D projection
    vec2 uv = fragCoord/iResolution.xy; 
    vec3 pos = uv3D(uv);
    if(flatmode){
        vec2 uv2D = (fragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;
        pos = vec3(uv2D,-1.);
    }
    pos=rotateX(pos,iTime/10.);

    
    // Ray Origin 
    // (XYZ)=> Your position in 3D space;
    // (W)=>The 3D representation of 4D Mandel Bulb
    // Ray Direction is based on UV/3D-Cartesian spherical XYZ
    vec4 ro = vec4(cos(iTime)*1.5-1.5,iTime*2. , sin(iTime)*1.5-1.5, 1.+sin(iTime)/2.); 
    vec4 rd = normalize(vec4(pos,0.)); // Ray direction with zero w component

    // Raymarch to find the distance to the bulb
    vec4 p, id;
    float marches;
    float dist = raymarch(ro, rd, p, marches, id);

    
    // If we hit a MandelBulb do lighting and color calculations.
    if (dist < 100.0) {
    
        vec4 normal = getNormal(p);
        vec3 lightDir = normalize((p - rd).xyz);
        float diffuse = max(dot(normalize(normal.xyz), lightDir), .8);// Dropping W & Z instead using XYY as it gave a more interesting effect IMO.
        
        // mix distance + 
        color = mix(1.-vec3(marches/100.),randomColor(length(floor(id.xzy))),.5);//mix(1.-vec3(marches/100.),(randomColor(length(ceil(p.xyz)))),.33);
        // 4D normal color is very trippy
        // You can rotate the normals by doing variations of XYZ/W, XWZ/Y, YZW/X, etc....
        if(trippy4Dnormals){
            color=normal.xyz/normal.w;
        }
        
        //Scrapped diffuse lighting
        color = color/(diffuse);
    }

    fragColor = vec4(color, 1.0);
}`,
    "Droste Zoom": `// Created by Danil (2024+) https://github.com/danilw
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// self https://www.shadertoy.com/view/MfdyW2

#define STTA_PERIOD 10.0
#define STTA2_PERIOD 16.0
#define STTA2_ACTIVE 8.0

// this is just
// mix of my https://www.shadertoy.com/view/NslGRN
// with
// copy of FabriceNeyret2
// https://www.shadertoy.com/view/4c3cD2

// that is fork of other shaders


// Fork of "Droste Zoom" by roywig. 
// https://shadertoy.com/view/Xs3SWj  
// https://www.shadertoy.com/view/Ml33R7
// cf article http://www.josleys.com/article_show.php?id=82

bool stta = false;
bool stta2 = false;
void mainImage_texture(out vec4 fragColor, in vec2 fragCoord);
void mainImage( out vec4 O, vec2 z )
{
    vec2  R = iResolution.xy;
          z = ( z+z - R ) / R.y; 
    float s = 2.3,   // log(10)
          a = .366,   // s/2PI
          t = iTime;
    
    z = mat2(1,-a,a,1) * vec2( log(length(z)), atan(z.y,z.x) ); // Conformal spiral mapping
    z = exp( mod(z.x-t, s) ) * cos(z.y +vec2(0,11)) ;           // Droste transform
    
    stta = mod(iTime, STTA_PERIOD) < STTA_PERIOD * 0.5;
    stta2 = mod(iTime, STTA2_PERIOD) < STTA2_ACTIVE;
    stta = stta || stta2;
    mainImage_texture(O,(((.5+.5*z*.1)-0.5)*R.y/R.xy+0.5)*R.xy);
    
    // bad fix to dfd border - black line
    if(stta2)O*=vec4((1.-smoothstep(0.,0.5,length(z)-9.5))*smoothstep(0.,0.051,length(z)-1.));
    // bad fix - run texture func twice with px shift - giga overhead in this single shader case
    // correct fix - is move mainImage_texture to BufA(as 2d texture) and do multiple texture reading-filtering here   
}









// everything below is 
// https://www.shadertoy.com/view/NslGRN

// MODIFIED

// this have most impact on image 

// color cut smoothstep (-0.5,0.5) a<b
#define SMA vec2(-0.1,0.042+0.4*float(stta))

// static SHAPE form, default 0.5
#define STATIC_SHAPE 0.0815

// speed of ROTATION
#define ROTATION_SPEED 1.8999

//#define USE_COLOR
const vec3 color_blue=vec3(0.5,0.65,0.8);
const vec3 color_red=vec3(0.99,0.2,0.1);


#define tshift 15.

// static CAMERA position, 0.49 on top, 0.001 horizontal
#define CAMERA_POS (0.49999-0.499*float(stta2))


//-----------------------------------------



// Created by Danil (2021+) https://cohost.org/arugl
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// self https://www.shadertoy.com/view/NslGRN


// --defines for "DESKTOP WALLPAPERS" that use this shader--
// comment or uncomment every define to make it work (add or remove "//" before #define)


// this shadertoy use ALPHA, NO_ALPHA set alpha to 1, BG_ALPHA set background as alpha
// iChannel0 used as background if alpha ignored by wallpaper-app
#define NO_ALPHA
//#define BG_ALPHA
//#define SHADOW_ALPHA
#define ONLY_BOX


// save PERFORMANCE by disabling shadow
#define NO_SHADOW


// static SCALE far/close to camera, 2.0 is default, exampe 0.5 or 10.0
#define CAMERA_FAR 5.1


// ANIMATION shape change
//#define ANIM_SHAPE


// ANIMATION color change
#define ANIM_COLOR


// custom COLOR, and change those const values



// use 4xAA for cube only (set 2-4-etc level of AA)
//#define AA_CUBE 4

// use 4xAA for everything - derivative filtering will not be used, look fcos2
// this is very slow - DO NOT USE
//#define AA_ALL 4



// --shader code--

// Layers sorted and support transparency and self-intersection-transparency
// Antialiasing is only dFd. (with some dFd fixes around edges)

// using iq's intersectors: https://iquilezles.org/articles/intersectors
// using https://www.shadertoy.com/view/ltKBzG
// using https://www.shadertoy.com/view/tsVXzh
// using https://www.shadertoy.com/view/WlffDn
// using https://www.shadertoy.com/view/WslGz4



// reflect back side
//#define backside_refl

// Camera with mouse
//#define MOUSE_control

// min(iFrame,0) does not speedup compilation in ANGLE
#define ANGLE_loops 0


// this shader discover Nvidia bug with arrays https://www.shadertoy.com/view/NslGR4
// use DEBUG with BUG, BUG trigger that bug and one layer will be white on Nvidia in OpenGL
//#define DEBUG
//#define BUG

#define FDIST 3.1
#define PI 3.1415926
#define GROUNDSPACING 0.5
#define GROUNDGRID 0.05
#define BOXDIMS vec3(0.75, 0.75, 1.25)

#define IOR 1.33

mat3 rotx(float a){float s = sin(a);float c = cos(a);return mat3(vec3(1.0, 0.0, 0.0), vec3(0.0, c, s), vec3(0.0, -s, c));  }
mat3 roty(float a){float s = sin(a);float c = cos(a);return mat3(vec3(c, 0.0, s), vec3(0.0, 1.0, 0.0), vec3(-s, 0.0, c));}
mat3 rotz(float a){float s = sin(a);float c = cos(a);return mat3(vec3(c, s, 0.0), vec3(-s, c, 0.0), vec3(0.0, 0.0, 1.0 ));}

vec3 fcos1(vec3 x) {
    vec3 w = fwidth(x);
    //if((length(w)==0.))return vec3(0.); // dFd fix2
    //w*=0.; //test
    float lw=length(w);
    if((lw==0.)||isnan(lw)||isinf(lw)){vec3 tc=vec3(0.); for(int i=0;i<8;i++)tc+=cos(x+x*float(i-4)*(0.01*400./iResolution.y));return tc/8.;}
    
    return cos(x) * smoothstep(3.14 * 2.0, 0.0, w);
}

vec3 fcos2( vec3 x){return cos(x);}
vec3 fcos( vec3 x){
#ifdef AA_ALL 
    return fcos2(x);
#else
    return fcos1(x);
#endif
}

vec3 getColor(vec3 p)
{

    // dFd fix, dFd broken on borders, but it fix only top level dFd, self intersection has border
    //if (length(p) > 0.99)return vec3(0.);
    p = abs(p);

    p *= 01.25;
    p = 0.5 * p / dot(p, p);
#ifdef ANIM_COLOR
    p+=-0.42072*(iTime+tshift);
#endif

    float t = (0.13) * length(p);
    vec3 col = vec3(0.3, 0.4, 0.5);
    col += 0.12 * fcos(6.28318 * t * 1.0 + vec3(0.0, 0.8, 1.1));
    col += 0.11 * fcos(6.28318 * t * 3.1 + vec3(0.3, 0.4, 0.1));
    col += 0.10 * fcos(6.28318 * t * 5.1 + vec3(0.1, 0.7, 1.1));
    col += 0.10 * fcos(6.28318 * t * 17.1 + vec3(0.2, 0.6, 0.7));
    col += 0.10 * fcos(6.28318 * t * 31.1 + vec3(0.1, 0.6, 0.7));
    col += 0.10 * fcos(6.28318 * t * 65.1 + vec3(0.0, 0.5, 0.8));
    col += 0.10 * fcos(6.28318 * t * 115.1 + vec3(0.1, 0.4, 0.7));
    col += 0.10 * fcos(6.28318 * t * 265.1 + vec3(1.1, 1.4, 2.7));
    col = clamp(col, 0., 1.);
    

    return col;
    
}

void calcColor(vec3 ro, vec3 rd, vec3 nor, float d, float len, int idx, bool si, float td, out vec4 colx,
               out vec4 colsi)
{

    vec3 pos = (ro + rd * d);
#ifdef DEBUG
    float a = 1. - smoothstep(len - 0.15, len + 0.00001, length(pos));
    if (idx == 0)colx = vec4(1., 0., 0., a);
    if (idx == 1)colx = vec4(0., 1., 0., a);
    if (idx == 2)colx = vec4(0., 0., 1., a);
    if (si)
    {
        pos = (ro + rd * td);
        float ta = 1. - smoothstep(len - 0.15, len + 0.00001, length(pos));
        if (idx == 0)colsi = vec4(1., 0., 0., ta);
        if (idx == 1)colsi = vec4(0., 1., 0., ta);
        if (idx == 2)colsi = vec4(0., 0., 1., ta);
    }
#else
    float a = 1. - smoothstep(len - 0.15*0.5, len + 0.00001, length(pos));
a*=(1.-smoothstep(SMA.x,SMA.y,-pos.z));
    //a=1.;
    vec3 col = getColor(pos);
    colx = vec4(col, a);
    if (si)
    {
        pos = (ro + rd * td);
        float ta = 1. - smoothstep(len - 0.15*0.5, len + 0.00001, length(pos));
        //ta=1.;
        col = getColor(pos);
        colsi = vec4(col, ta);
    }
#endif
}

// xSI is self intersect data, fade to fix dFd on edges
bool iBilinearPatch(in vec3 ro, in vec3 rd, in vec4 ps, in vec4 ph, in float sz, out float t, out vec3 norm,
                    out bool si, out float tsi, out vec3 normsi, out float fade, out float fadesi)
{
    vec3 va = vec3(0.0, 0.0, ph.x + ph.w - ph.y - ph.z);
    vec3 vb = vec3(0.0, ps.w - ps.y, ph.z - ph.x);
    vec3 vc = vec3(ps.z - ps.x, 0.0, ph.y - ph.x);
    vec3 vd = vec3(ps.xy, ph.x);
    t = -1.;
    tsi = -1.;
    si = false;
    fade = 1.;
    fadesi = 1.;
    norm=vec3(0.,1.,0.);normsi=vec3(0.,1.,0.);

    float tmp = 1.0 / (vb.y * vc.x);
    float a = 0.0;
    float b = 0.0;
    float c = 0.0;
    float d = va.z * tmp;
    float e = 0.0;
    float f = 0.0;
    float g = (vc.z * vb.y - vd.y * va.z) * tmp;
    float h = (vb.z * vc.x - va.z * vd.x) * tmp;
    float i = -1.0;
    float j = (vd.x * vd.y * va.z + vd.z * vb.y * vc.x) * tmp - (vd.y * vb.z * vc.x + vd.x * vc.z * vb.y) * tmp;

    float p = dot(vec3(a, b, c), rd.xzy * rd.xzy) + dot(vec3(d, e, f), rd.xzy * rd.zyx);
    float q = dot(vec3(2.0, 2.0, 2.0) * ro.xzy * rd.xyz, vec3(a, b, c)) + dot(ro.xzz * rd.zxy, vec3(d, d, e)) +
              dot(ro.yyx * rd.zxy, vec3(e, f, f)) + dot(vec3(g, h, i), rd.xzy);
    float r =
        dot(vec3(a, b, c), ro.xzy * ro.xzy) + dot(vec3(d, e, f), ro.xzy * ro.zyx) + dot(vec3(g, h, i), ro.xzy) + j;

    if (abs(p) < 0.000001)
    {
        float tt = -r / q;
        if (tt <= 0.)
            return false;
        t = tt;
        // normal

        vec3 pos = ro + t * rd;
        if(length(pos)>sz)return false;
        vec3 grad =
            vec3(2.0) * pos.xzy * vec3(a, b, c) + pos.zxz * vec3(d, d, e) + pos.yyx * vec3(f, e, f) + vec3(g, h, i);
        norm = -normalize(grad);
        return true;
    }
    else
    {
        float sq = q * q - 4.0 * p * r;
        if (sq < 0.0)
        {
            return false;
        }
        else
        {
            float s = sqrt(sq);
            float t0 = (-q + s) / (2.0 * p);
            float t1 = (-q - s) / (2.0 * p);
            float tt1 = min(t0 < 0.0 ? t1 : t0, t1 < 0.0 ? t0 : t1);
            float tt2 = max(t0 > 0.0 ? t1 : t0, t1 > 0.0 ? t0 : t1);
            float tt0 = tt1;
            if (tt0 <= 0.)
                return false;
            vec3 pos = ro + tt0 * rd;
            // black border on end of circle and self intersection with alpha come because dFd
            // uncomment this to see or rename fcos2 to fcos
            //sz+=0.3; 
            bool ru = step(sz, length(pos)) > 0.5;
            if (ru)
            {
                tt0 = tt2;
                pos = ro + tt0 * rd;
            }
            if (tt0 <= 0.)
                return false;
            bool ru2 = step(sz, length(pos)) > 0.5;
            if (ru2)
                return false;

            // self intersect
            if ((tt2 > 0.) && ((!ru)) && !(step(sz, length(ro + tt2 * rd)) > 0.5))
            {
                si = true;
                fadesi=s;
                tsi = tt2;
                vec3 tpos = ro + tsi * rd;
                // normal
                vec3 tgrad = vec3(2.0) * tpos.xzy * vec3(a, b, c) + tpos.zxz * vec3(d, d, e) +
                             tpos.yyx * vec3(f, e, f) + vec3(g, h, i);
                normsi = -normalize(tgrad);
            }
            
            fade=s;
            t = tt0;
            // normal
            vec3 grad =
                vec3(2.0) * pos.xzy * vec3(a, b, c) + pos.zxz * vec3(d, d, e) + pos.yyx * vec3(f, e, f) + vec3(g, h, i);
            norm = -normalize(grad);

            return true;
        }
    }
}

float dot2( in vec3 v ) { return dot(v,v); }

float segShadow( in vec3 ro, in vec3 rd, in vec3 pa, float sh )
{
    float dm = dot(rd.yz,rd.yz);
    float k1 = (ro.x-pa.x)*dm;
    float k2 = (ro.x+pa.x)*dm;
    vec2  k5 = (ro.yz+pa.yz)*dm;
    float k3 = dot(ro.yz+pa.yz,rd.yz);
    vec2  k4 = (pa.yz+pa.yz)*rd.yz;
    vec2  k6 = (pa.yz+pa.yz)*dm;
    
    for( int i=0; i<4 + ANGLE_loops; i++ )
    {
        vec2  s = vec2(i&1,i>>1);
        float t = dot(s,k4) - k3;
        
        if( t>0.0 )
        sh = min(sh,dot2(vec3(clamp(-rd.x*t,k1,k2),k5-k6*s)+rd*t)/(t*t));
    }
    return sh;
}

float boxSoftShadow( in vec3 ro, in vec3 rd, in vec3 rad, in float sk ) 
{
    rd += 0.0001 * (1.0 - abs(sign(rd)));
	vec3 rdd = rd;
	vec3 roo = ro;

    vec3 m = 1.0/rdd;
    vec3 n = m*roo;
    vec3 k = abs(m)*rad;
	
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

    float tN = max( max( t1.x, t1.y ), t1.z );
	float tF = min( min( t2.x, t2.y ), t2.z );
	
    if( tN<tF && tF>0.0) return 0.0;
    
    float sh = 1.0;
    sh = segShadow( roo.xyz, rdd.xyz, rad.xyz, sh );
    sh = segShadow( roo.yzx, rdd.yzx, rad.yzx, sh );
    sh = segShadow( roo.zxy, rdd.zxy, rad.zxy, sh );
    sh = clamp(sk*sqrt(sh),0.0,1.0);
    return sh*sh*(3.0-2.0*sh);
}

float box(in vec3 ro, in vec3 rd, in vec3 r, out vec3 nn, bool entering)
{
    rd += 0.0001 * (1.0 - abs(sign(rd)));
    vec3 dr = 1.0 / rd;
    vec3 n = ro * dr;
    vec3 k = r * abs(dr);

    vec3 pin = -k - n;
    vec3 pout = k - n;
    float tin = max(pin.x, max(pin.y, pin.z));
    float tout = min(pout.x, min(pout.y, pout.z));
    if (tin > tout)
        return -1.;
    if (entering)
    {
        nn = -sign(rd) * step(pin.zxy, pin.xyz) * step(pin.yzx, pin.xyz);
    }
    else
    {
        nn = sign(rd) * step(pout.xyz, pout.zxy) * step(pout.xyz, pout.yzx);
    }
    return entering ? tin : tout;
}

vec3 bgcol(in vec3 rd)
{
    return mix(vec3(0.01), vec3(0.336, 0.458, .668), 1. - pow(abs(rd.z+0.25), 1.3));
}

vec3 background(in vec3 ro, in vec3 rd , vec3 l_dir, out float alpha)
{
#ifdef ONLY_BOX
alpha=0.;
return vec3(0.01);
#endif
    float t = (-BOXDIMS.z - ro.z) / rd.z;
    alpha=0.;
    vec3 bgc = bgcol(rd);
    if (t < 0.)
        return bgc;
    vec2 uv = ro.xy + t * rd.xy;
#ifdef NO_SHADOW
    float shad=1.;
#else
    float shad = boxSoftShadow((ro + t * rd), normalize(l_dir+vec3(0.,0.,1.))*rotz(PI*0.65) , BOXDIMS, 1.5);
#endif
    float aofac = smoothstep(-0.95, .75, length(abs(uv) - min(abs(uv), vec2(0.45))));
    aofac = min(aofac,smoothstep(-0.65, 1., shad));
    float lght=max(dot(normalize(ro + t * rd+vec3(0.,-0.,-5.)), normalize(l_dir-vec3(0.,0.,1.))*rotz(PI*0.65)), 0.0);
    vec3 col = mix(vec3(0.4), vec3(.71,.772,0.895), lght*lght* aofac+ 0.05) * aofac;
    alpha=1.-smoothstep(7.,10.,length(uv));
#ifdef SHADOW_ALPHA
    //alpha=clamp(alpha*max(lght*lght*0.95,(1.-aofac)*1.25),0.,1.);
    alpha=clamp(alpha*(1.-aofac)*1.25,0.,1.);
#endif
    return mix(col*length(col)*0.8,bgc,smoothstep(7.,10.,length(uv)));
}

#define swap(a,b) tv=a;a=b;b=tv

vec4 insides(vec3 ro, vec3 rd, vec3 nor_c, vec3 l_dir, out float tout)
{
    tout = -1.;
    vec3 trd=rd;

    vec3 col = vec3(0.);

    float pi = 3.1415926;

    if (abs(nor_c.x) > 0.5)
    {
        rd = rd.xzy * nor_c.x;
        ro = ro.xzy * nor_c.x;
    }
    else if (abs(nor_c.z) > 0.5)
    {
        l_dir *= roty(pi);
        rd = rd.yxz * nor_c.z;
        ro = ro.yxz * nor_c.z;
    }
    else if (abs(nor_c.y) > 0.5)
    {
        l_dir *= rotz(-pi * 0.5);
        rd = rd * nor_c.y;
        ro = ro * nor_c.y;
    }

#ifdef ANIM_SHAPE
    float curvature = (0.001+1.5-1.5*smoothstep(0.,8.5,mod((iTime+tshift)*0.44,20.))*(1.-smoothstep(10.,18.5,mod((iTime+tshift)*0.44,20.))));
    // curvature(to not const above) make compilation on Angle 15+ sec
#else
#ifdef STATIC_SHAPE
    const float curvature = STATIC_SHAPE;
#else
    const float curvature = .5;
#endif
#endif
    float bil_size = 1.;
    vec4 ps = vec4(-bil_size, -bil_size, bil_size, bil_size) * curvature;
    vec4 ph = vec4(-bil_size, bil_size, bil_size, -bil_size) * curvature;
    
    vec4 [3]colx=vec4[3](vec4(0.),vec4(0.),vec4(0.));
    vec3 [3]dx=vec3[3](vec3(-1.),vec3(-1.),vec3(-1.));
    vec4 [3]colxsi=vec4[3](vec4(0.),vec4(0.),vec4(0.));
    int [3]order=int[3](0,1,2);

    for (int i = 0; i < 3 + ANGLE_loops; i++)
    {
        if (abs(nor_c.x) > 0.5)
        {
            ro *= rotz(-pi * (1. / float(3)));
            rd *= rotz(-pi * (1. / float(3)));
        }
        else if (abs(nor_c.z) > 0.5)
        {
            ro *= rotz(pi * (1. / float(3)));
            rd *= rotz(pi * (1. / float(3)));
        }
        else if (abs(nor_c.y) > 0.5)
        {
            ro *= rotx(pi * (1. / float(3)));
            rd *= rotx(pi * (1. / float(3)));
        }
        vec3 normnew;
        float tnew;
        bool si;
        float tsi;
        vec3 normsi;
        float fade;
        float fadesi;

        if (iBilinearPatch(ro, rd, ps, ph, bil_size, tnew, normnew, si, tsi, normsi, fade, fadesi))
        {
            if (tnew > 0.)
            {
                vec4 tcol, tcolsi;
                calcColor(ro, rd, normnew, tnew, bil_size, i, si, tsi, tcol, tcolsi);
                if (tcol.a > 0.0)
                {
                    {
                        vec3 tvalx = vec3(tnew, float(si), tsi);
                        dx[i]=tvalx;
                    }
#ifdef DEBUG
                    colx[i]=tcol;
                    if (si)colxsi[i]=tcolsi;
#else

                    float dif = clamp(dot(normnew, l_dir), 0.0, 1.0);
                    float amb = clamp(0.5 + 0.5 * dot(normnew, l_dir), 0.0, 1.0);

                    {
#ifdef USE_COLOR
                        vec3 shad = 0.57 * color_blue * amb + 1.5*color_blue.bgr * dif;
                        const vec3 tcr = color_red;
#else
                        vec3 shad = vec3(0.32, 0.43, 0.54) * amb + vec3(1.0, 0.9, 0.7) * dif;
                        const vec3 tcr = vec3(1.,0.21,0.11);
#endif
                        float ta = clamp(length(tcol.rgb),0.,1.);
                        tcol=clamp(tcol*tcol*2.,0.,1.);
                        vec4 tvalx =
                            vec4((tcol.rgb*shad*1.4 + 3.*(tcr*tcol.rgb)*clamp(1.-(amb+dif),0.,1.)), min(tcol.a,ta));
                        tvalx.rgb=clamp(2.*tvalx.rgb*tvalx.rgb,0.,1.);
                        tvalx*=(min(fade*5.,1.));
                        colx[i]=tvalx;
                    }
                    if (si)
                    {
                        dif = clamp(dot(normsi, l_dir), 0.0, 1.0);
                        amb = clamp(0.5 + 0.5 * dot(normsi, l_dir), 0.0, 1.0);
                        {
#ifdef USE_COLOR
                            vec3 shad = 0.57 * color_blue * amb + 1.5*color_blue.bgr * dif;
                            const vec3 tcr = color_red;
#else
                            vec3 shad = vec3(0.32, 0.43, 0.54) * amb + vec3(1.0, 0.9, 0.7) * dif;
                            const vec3 tcr = vec3(1.,0.21,0.11);
#endif
                            float ta = clamp(length(tcolsi.rgb),0.,1.);
                            tcolsi=clamp(tcolsi*tcolsi*2.,0.,1.);
                            vec4 tvalx =
                                vec4(tcolsi.rgb * shad + 3.*(tcr*tcolsi.rgb)*clamp(1.-(amb+dif),0.,1.), min(tcolsi.a,ta));
                            tvalx.rgb=clamp(2.*tvalx.rgb*tvalx.rgb,0.,1.);
                            tvalx.rgb*=(min(fadesi*5.,1.));
                            colxsi[i]=tvalx;
                        }
                    }
#endif
                }
            }
        }
    }
    // transparency logic and layers sorting 
    float a = 1.;
    if (dx[0].x < dx[1].x){{vec3 swap(dx[0], dx[1]);}{int swap(order[0], order[1]);}}
    if (dx[1].x < dx[2].x){{vec3 swap(dx[1], dx[2]);}{int swap(order[1], order[2]);}}
    if (dx[0].x < dx[1].x){{vec3 swap(dx[0], dx[1]);}{int swap(order[0], order[1]);}}

    tout = max(max(dx[0].x, dx[1].x), dx[2].x);

    if (dx[0].y < 0.5)
    {
        a=colx[order[0]].a;
    }

#if !(defined(DEBUG)&&defined(BUG))
    
    // self intersection
    bool [3] rul= bool[3](
        ((dx[0].y > 0.5) && (dx[1].x <= 0.)),
        ((dx[1].y > 0.5) && (dx[0].x > dx[1].z)),
        ((dx[2].y > 0.5) && (dx[1].x > dx[2].z))
    );
    for(int k=0;k<3;k++){
        if(rul[k]){
            vec4 tcolxsi = vec4(0.);
            tcolxsi=colxsi[order[k]];
            vec4 tcolx = vec4(0.);
            tcolx=colx[order[k]];

            vec4 tvalx = mix(tcolxsi, tcolx, tcolx.a);
            colx[order[k]]=tvalx;

            vec4 tvalx2 = mix(vec4(0.), tvalx, max(tcolx.a, tcolxsi.a));
            colx[order[k]]=tvalx2;
        }
    }

#endif

    float a1 = (dx[1].y < 0.5) ? colx[order[1]].a : ((dx[1].z > dx[0].x) ? colx[order[1]].a : 1.);
    float a2 = (dx[2].y < 0.5) ? colx[order[2]].a : ((dx[2].z > dx[1].x) ? colx[order[2]].a : 1.);
    col = mix(mix(colx[order[0]].rgb, colx[order[1]].rgb, a1), colx[order[2]].rgb, a2);
    a = max(max(a, a1), a2);
    return vec4(col, a);
}

void mainImage_texture(out vec4 fragColor, in vec2 fragCoord)
{
    float osc = 0.5;
    vec3 l_dir = normalize(vec3(0., 1., 0.));
    l_dir *= rotz(0.5);
    float mouseY = 1.0 * 0.5 * PI;
#ifdef MOUSE_control
    mouseY = (1.0 - 1.15 * iMouse.y / iResolution.y) * 0.5 * PI;
    if(iMouse.y < 1.)
#endif
#ifdef CAMERA_POS
    mouseY = PI*CAMERA_POS;
#else
    mouseY = PI*0.49 - smoothstep(0.,8.5,mod((iTime+tshift)*0.33,25.))*(1.-smoothstep(14.,24.0,mod((iTime+tshift)*0.33,25.))) * 0.55 * PI;
#endif
#ifdef ROTATION_SPEED
    float mouseX = -2.*PI-0.25*(iTime*ROTATION_SPEED+tshift);
#else
    float mouseX = -2.*PI-0.25*(iTime+tshift);
#endif
#ifdef MOUSE_control
    mouseX+=-(iMouse.x / iResolution.x) * 2. * PI;
#endif
    
#ifdef CAMERA_FAR
    vec3 eye = (2. + CAMERA_FAR) * vec3(cos(mouseX) * cos(mouseY), sin(mouseX) * cos(mouseY), sin(mouseY));
#else
    vec3 eye = 4. * vec3(cos(mouseX) * cos(mouseY), sin(mouseX) * cos(mouseY), sin(mouseY));
#endif
    vec3 w = normalize(-eye);
    vec3 up = vec3(0., 0., 1.);
    vec3 u = normalize(cross(w, up));
    vec3 v = cross(u, w);

    vec4 tot=vec4(0.);
#if defined(AA_CUBE)||defined(AA_ALL)
#ifdef AA_CUBE
    const int AA = AA_CUBE;
#else
    const int AA = AA_ALL;
#endif
    vec3 incol_once=vec3(0.);
    bool in_once=false;
    vec4 incolbg_once=vec4(0.);
    bool bg_in_once=false;
    vec4 outcolbg_once=vec4(0.);
    bool bg_out_once=false;
    for( int mx=0; mx<AA; mx++ )
    for( int nx=0; nx<AA; nx++ )
    {
    vec2 o = vec2(mod(float(mx+AA/2),float(AA)),mod(float(nx+AA/2),float(AA))) / float(AA) - 0.5;
    vec2 uv = (fragCoord + o - 0.5 * iResolution.xy) / iResolution.x;
#else
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.x;
#endif
    vec3 rd = normalize(w * FDIST + uv.x * u + uv.y * v);

    vec3 ni;
    float t = box(eye, rd, BOXDIMS, ni, true);
    vec3 ro = eye + t * rd;
    vec2 coords = ro.xy * ni.z/BOXDIMS.xy + ro.yz * ni.x/BOXDIMS.yz + ro.zx * ni.y/BOXDIMS.zx;
    float fadeborders = (1.-smoothstep(0.915,1.05,abs(coords.x)))*(1.-smoothstep(0.915,1.05,abs(coords.y)));

    if (t > 0.)
    {
        float ang = -iTime * 0.33;
        vec3 col = vec3(0.);
#ifdef AA_CUBE
        if(in_once)col=incol_once;
        else{
        in_once=true;
#endif
        float R0 = (IOR - 1.) / (IOR + 1.);
        R0 *= R0;

        vec2 theta = vec2(0.);
        vec3 n = vec3(cos(theta.x) * sin(theta.y), sin(theta.x) * sin(theta.y), cos(theta.y));

        vec3 nr = n.zxy * ni.x + n.yzx * ni.y + n.xyz * ni.z;
        vec3 rdr = reflect(rd, nr);
        float talpha;
        vec3 reflcol = background(ro, rdr, l_dir,talpha);

        vec3 rd2 = refract(rd, nr, 1. / IOR);

        float accum = 1.;
        vec3 no2 = ni;
        vec3 ro_refr = ro;

        vec4 [2] colo = vec4[2](vec4(0.),vec4(0.));

        for (int j = 0; j < 2 + ANGLE_loops; j++)
        {
            float tb;
            vec2 coords2 = ro_refr.xy * no2.z + ro_refr.yz * no2.x + ro_refr.zx * no2.y;
            vec3 eye2 = vec3(coords2, -1.);
            vec3 rd2trans = rd2.yzx * no2.x + rd2.zxy * no2.y + rd2.xyz * no2.z;

            rd2trans.z = -rd2trans.z;
            vec4 internalcol = insides(eye2, rd2trans, no2, l_dir, tb);
            if (tb > 0.)
            {
                internalcol.rgb *= accum;
                colo[j]=internalcol;
            }
        }
        float fresnel = R0 + (1. - R0) * pow(1. - dot(-rd, nr), 5.);
        col = mix(mix(colo[1].rgb * colo[1].a, colo[0].rgb, colo[0].a)*fadeborders, reflcol, pow(fresnel, 1.5));
        col=clamp(col,0.,1.);
#ifdef AA_CUBE
        }
        incol_once=col;
        if(!bg_in_once){
        bg_in_once=true;
        float alpha;
        incolbg_once = vec4(background(eye, rd, l_dir, alpha), 0.15);
#if defined(BG_ALPHA)||defined(ONLY_BOX)||defined(SHADOW_ALPHA)
        incolbg_once.w = alpha;
#endif
        }
#endif
        
        float cineshader_alpha = 0.;
        cineshader_alpha = clamp(0.15*dot(eye,ro),0.,1.);
        vec4 tcolx = vec4(col, cineshader_alpha);
#if defined(BG_ALPHA)||defined(ONLY_BOX)||defined(SHADOW_ALPHA)
        tcolx.w = 1.;
#endif
        tot += tcolx;
    }
    else
    {
        vec4 tcolx = vec4(0.);
#ifdef AA_CUBE
        if(!bg_out_once){
        bg_out_once=true;
#endif
        float alpha;
        tcolx = vec4(background(eye, rd, l_dir, alpha), 0.15);
#if defined(BG_ALPHA)||defined(ONLY_BOX)||defined(SHADOW_ALPHA)
        tcolx.w = alpha;
#endif
#ifdef AA_CUBE
        outcolbg_once=tcolx;
        }else tcolx=max(outcolbg_once,incolbg_once);
#endif
        tot += tcolx;
    }
#if defined(AA_CUBE)||defined(AA_ALL)
    }
    tot /= float(AA*AA);
#endif
    fragColor = tot;
#ifdef NO_ALPHA
    fragColor.w = 1.;
#endif
    fragColor.rgb=clamp(fragColor.rgb,0.,1.);
#if defined(BG_ALPHA)||defined(ONLY_BOX)||defined(SHADOW_ALPHA)
    fragColor.rgb=fragColor.rgb*fragColor.w+texture(iChannel0, fragCoord/iResolution.xy).rgb*(1.-fragColor.w);
#endif
    //fragColor=vec4(fragColor.w);
}`,

    "The Rabbit Hole": `// Author: Rigel rui@gil.com
// licence: https://creativecommons.org/licenses/by/4.0/
// link: https://www.shadertoy.com/view/Md3yRf

#define HOLE_PERIOD 14.0
#define HOLE_ACTIVE 7.0

/*
This was inpired by Escher painting "Print Gallery" and this lecture
https://youtu.be/clQA6WhwCeA?t=7m50s

I wanted to do something with the Escher/Droste effect, and I discovered 
this blog post http://roy.red/droste-.html#droste 
by user Roy Wiggins https://www.shadertoy.com/user/roywig

And his other post about KIFS (Kaleidoscopic Iterated Function Systems)
http://roy.red/folding-the-koch-snowflake-.html#folding-the-koch-snowflake

An this sended me along a rabbit hole of folding space, and constructing 
KIFS with escher like spiral zooms :) 

There are plenty of Escher/Droste effect on shadertoy, but this one by reinder
is like total magic. https://www.shadertoy.com/view/Mdf3zM
*/


// utility functions
// conversion from cartesian to polar
vec2 toPolar(vec2 uv) { return vec2(length(uv),atan(uv.y,uv.x)); }
// conversion from polar to cartesian
vec2 toCarte(vec2 z) { return z.x*vec2(cos(z.y),sin(z.y)); }
// complex division in polar form z = vec2(radius,angle)
vec2 zdiv(vec2 z1, vec2 z2) { return vec2(z1.x/z2.x,z1.y-z2.y); }
// complex log in polar form z = vec2(radius,angle)
vec2 zlog(vec2 z) { return toPolar(vec2(log(z.x),z.y)); }
// complex exp in polar form z = vec2(radius,angle)
vec2 zexp(vec2 z) { z = toCarte(z); return vec2(exp(z.x),z.y); }
// smoothstep antialias with fwidth
float ssaa(float v) { return smoothstep(-1.,1.,v/fwidth(v)); }
// stroke an sdf 'd', with a width 'w', and a fill 'f' 
float stroke(float d, float w, bool f) {  return abs(ssaa(abs(d)-w*.5) - float(f)); }
// fills an sdf 'd', and a fill 'f'. false for the fill means inverse 
float fill(float d, bool f) { return abs(ssaa(d) - float(f)); }
// a signed distance function for a rectangle 's' is size
float sdfRect(vec2 uv, vec2 s) { vec2 auv = abs(uv); return max(auv.x-s.x,auv.y-s.y); }
// a signed distance function for a circle, 'r' is radius
float sdfCircle(vec2 uv, float r) { return length(uv)-r; }
// a signed distance function for a hexagon
float sdfHex(vec2 uv) { vec2 auv = abs(uv); return max(auv.x * .866 + auv.y * .5, auv.y)-.5; }
// a signed distance function for a equilateral triangle
float sdfTri(vec2 uv) { return max(abs(uv.x) * .866 + uv.y * .5, -uv.y)-.577; }
// a 'fold' is a kind of generic abs(). 
// it reflects half of the plane in the other half
// the variable 'a' represents the angle of an axis going through the origin
// so in normalized coordinates uv [-1,1] 
// fold(uv,radians(0.)) == abs(uv.y) and fold(uv,radians(90.)) == abs(uv.x) 
vec2 fold(vec2 uv, float a) { a -= 1.57; vec2 axis = vec2(cos(a),sin(a)); return uv-(2.*min(dot(uv,axis),.0)*axis); }
// 2d rotation matrix
mat2 uvRotate(float a) { return mat2(cos(a),sin(a),-sin(a),cos(a)); }


// this functions 'folds' space with the symmetries of the Koch Snowflake
// https://en.wikipedia.org/wiki/Koch_snowflake
// it returns a coordinate system uv, where you can draw whatever you like
// 'n' is the number of iterations
vec2 uvKochSnowflake(vec2 uv, int n) {
    uv = fold(vec2(-abs(uv.x),uv.y),radians(150.))-vec2(.0,.44);
    for (int i=0; i<n; i++) 
        uv = fold(vec2(abs(uv.x),uv.y)*3.-vec2(.75,.0),radians(60.))-vec2(.75,.0);
    return uv;
}

// this functions 'folds' space with the symmetries of the Sierpinski Carpet
// https://en.wikipedia.org/wiki/Sierpinski_carpet
// it's like the 2d equivalent of the menger sponge
// it returns a coordinate system uv, where you can draw whatever you like
// 'n' is the number of iterations
vec2 uvSierpinskiCarpet(vec2 uv, int n) {
    for (int i=0; i<n; i++) {
        uv = fold(abs(uv*3.),radians(45.))-vec2(2.0,1.0);
        uv = vec2(uv.x,abs(uv.y)-1.);
    }
    return uv;    
}

// the scene
vec3 TheRabbitHole(vec2 uv) {

    // a flag for the scene
    float sc = 1.;
	// save current uv for the rabbit
    vec2 uvr = uv;

    // timed Escher/Droste transform (active for HOLE_ACTIVE seconds each HOLE_PERIOD)
    if (mod(iTime, HOLE_PERIOD) < HOLE_ACTIVE) {
        float scale = log(4.);
        float angle = atan(scale/6.283);
        
        // this line is an infinite zoom
        uv /= exp(mod(iTime*.8,6.283/angle));
        
        // this line is the Escher Deformation with a vec2(scale,rotation)
        uv = toCarte(zexp(zdiv(zlog(toPolar(uv)),vec2(cos(angle),angle))));
        
        // this line is the Droste Effect for the size of the frame
        uv /= exp(scale*floor(log(sdfRect(uv*vec2(.8,.66),vec2(0.)))/scale));
        sc = -1.;
    }

    // the frame
    float frame = min(
        stroke(sdfRect(uv,vec2(1.5,1.75)),.5,true),
        // drawing a simple rectangle in the sierpinsi carpet coordinate system
        fill(sdfRect(uvSierpinskiCarpet(mod((uv-vec2(.25,.0))*6.,3.)-1.5,2),vec2(1.)),false));

    // the canvas behind the rabbit
    float canvas = fill(sdfRect(uv,vec2(1.4,1.6)),true)*(1.-sdfRect(uv,vec2(.4,.6)));
    
    // uv for the rabbit
    uvr = sc == 1. ? uvr*.2+vec2(0.,.15) : uvr*.5*uvRotate(iTime*4.);
    
    // uv for the rabbit ears
    vec2 uvears = vec2(abs(uvr.x),uvr.y)*uvRotate(radians(-20.)); 
    float ears = stroke(sdfCircle(vec2(-abs(uvears.x),uvears.y)-vec2(.16,.3),.2),.04,true); 
    
    // uv for the rabbit eyes
    vec2 uveyes = vec2(abs(uvr.x),uvr.y)*uvRotate(radians(-40.)); 
    float eyes = fill(sdfCircle(vec2(-abs(uveyes.x),uveyes.y)-vec2(.05,.1),.07),false); 
    
    // nose ant teeth
    float nose = fill(sdfCircle(vec2(abs(uvr.x),uvr.y)-vec2(.008,.0),.02),false);
    float teeth = fill(sdfRect(vec2(abs(uvr.x),uvr.y)-vec2(.007,-.045),vec2(.005,.015)),false);

    // the face is just a bunch of circles
    float face = max(max(
        fill(sdfCircle(uvr-vec2(.0,.0),.07),true),
        fill(sdfCircle(vec2(abs(uvr.x),uvr.y)-vec2(.078,.05),.07),true)),
        fill(sdfCircle(uvr-vec2(.0,.1),.12),true));
    
    // compose the rabbit
    float rabbit = min(min(min(eyes,nose),teeth),max(ears,face));

    // a coodinate system uv for the Koch Snowflake KIFS
    vec2 uvka = uvKochSnowflake(vec2(abs(uv.x),uv.y)*.7-vec2(2.3,.0),2);
    vec2 uvkb = vec2(uvka.x,mod(uvka.y+iTime,.8)-.4);
    // drawing a pattern with this uv
    float kifs = max(max(max(min(
        fill(sdfCircle(uvkb,.4),false),
        fill(sdfRect(uvka-vec2(.0,-1.5),vec2(.6,6.)),true)),
        stroke(sdfRect(uvka,vec2(1.,.2)),.3,true)),
        fill(sdfHex(uvka-vec2(cos(iTime),sin(iTime)*2.)),true)), 
        fill(sdfRect(uvkb,vec2(.2)),true));

    // the small clock on the left
    vec2 uvc = (uv+vec2(3.3,.0))*1.2;
    vec2 uvch = sc==1. ? uvc : uvc*uvRotate(radians(iTime*60.));
    float chronos = min(min(
        fill(sdfHex(uvc),true),
        stroke(sdfCircle(uvc,.4),.1,false)+
        stroke(mod(atan(uvc.y,uvc.x)+radians(15.),radians(30.))-radians(15.),.15,false)),
        fill(sdfRect(uvch-vec2(.0,.15),vec2(.03,.15)),false));
    
    // the small card figure on the right    
    vec2 uvh = (uv-vec2(3.3,.0))*1.2;
    float card = max(max(stroke(sdfHex(uvh),.1,true),
    fill(sdfCircle(vec2(uvh.x,-sc*uvh.y-sqrt(abs(uvh.x)*.25)+.15)*1.2,.3),true)),
    fill(sdfTri(uvh*7.+vec2(.0,2.2)),true)*sc);

	// background
    vec3 c = vec3(.9)* (sc == 1. ? 1. : 1.2-length(uv)*.15);
    // mixing all components together
    c = mix(c,vec3(.1),canvas);
    c = mix(c,vec3(.6)-sc*vec3(.3),kifs);
    c = mix(c,vec3(.3)-sc*vec3(.3,.0,.0),chronos);
    c = mix(c,vec3(.3)-sc*vec3(.3,.0,.0),card);
    c = mix(c,vec3(.2),frame);
    c = mix(c,vec3(sc),rabbit);
    return c;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
	vec2 uv = ( fragCoord.xy - iResolution.xy * .5) / iResolution.y;

	fragColor = vec4( TheRabbitHole(uv*6.), 1.0 );
}`,
    "Altair": `// mi-ku/Altair

#define STEPS 10
#define SPHERE_R 3.0
#define LINES 12.0

//#define TORUS_VER
//#define BOX_VER

// iq's polynomial smin
float smin( float a, float b, float k )
{
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float sphere( vec3 p, float r )
{
#ifdef TORUS_VER
	vec2 q = vec2(length(p.xy)-r*1.0,p.z);
	return length(q)-r * 0.25;	
#elif defined( BOX_VER )
	return length(max(abs(p)-vec3(r),0.0))-r*0.4;
#else
	return length( p ) - r;
#endif
}

vec3 colorizeDF( vec3 p, vec3 n )
{
	float pattern = max( 0.0, abs( pow( sin( ( p.y + n.y ) * LINES + p.x * .0 ), 1.0 ) ) - 0.85 ) * 4.0;
	vec3 c = vec3( pattern );
	return c;
}

float rm( vec3 p, vec4 s1, vec4 s2, vec4 s3, vec4 s4 )
{
	return smin( sphere( p + s1.xyz, s1.w ), 
			 smin( sphere( p + s2.xyz, s2.w ), 
			   smin( sphere( p + s4.xyz, s4.w ), sphere( p + s3.xyz, s3.w )
				  , 0.5 ), 0.5 ), 0.5 );
}

vec3 colorize( vec2 uv )
{
	float yOff = sin( iTime ) * 2.0;
	vec3 ro = vec3( 0.0, yOff, -10.0 );
	vec3 rd = vec3( uv, 1.0 );
	rd.y -= yOff * 0.1;
	rd = normalize( rd );
	
	vec3 p = ro;
	
	float time = iTime;
	
	float o1 = 3.0, o2 = 2.0, o3 = 3.25, o4 = 2.2;
	vec3  a1 = vec3(  .1,  .2 , 2.1   );
	vec3  a2 = vec3(  .5,  .4,  2.5   );
	vec3  a3 = vec3(  .9,  .6,  2.9   );
	vec3  a4 = vec3( 1.3,  .8,  2.2   );
	vec3  i1 = vec3(  .13, .43,  .87  );
	vec3  i2 = vec3(  .93, .23,  .57  );
	vec3  i3 = vec3(  .33, .13,  .37  );
	vec3  i4 = vec3(  .13, .73,  .127 );
	
	a1 = i1 + time * i2.z;
	a2 = i2 + time * i1.y;
	a3 = i3 + time * i1.z;
	a4 = i4 + time * i2.x;
	
	vec4 s1 = vec4( vec3( sin( a1.x ), cos( a1.y ), sin( a1.z ) * 0.45 ) * o1, SPHERE_R * 0.6 );
	vec4 s2 = vec4( vec3( sin( a2.x ), cos( a2.y ), sin( a2.z ) * 0.45 ) * o2, SPHERE_R * 0.7 );
	vec4 s3 = vec4( vec3( sin( a3.x ), cos( a3.y ), sin( a3.z ) * 0.45 ) * o3, SPHERE_R * 0.8 );
	vec4 s4 = vec4( vec3( sin( a4.x ), cos( a4.y ), sin( a4.z ) * 0.45 ) * o4, SPHERE_R * 0.9 );
	
	for( int i = 0; i < STEPS; i++ )
	{
		p += rd * rm( p, s1, s2, s3, s4 ) * 0.9;
	}
	
	const float nm = 1.0;
	const vec3 dx = vec3( 1.0, 0.0, 0.0 ) * nm;
	const vec3 dy = vec3( 0.0, 1.0, 0.0 ) * nm;
	const vec3 dz = vec3( 0.0, 0.0, 1.0 ) * nm;
	vec3 n = vec3( rm( p + dx, s1, s2, s3, s4 ) - rm( p - dx, s1, s2, s3, s4 ), 
				   rm( p + dy, s1, s2, s3, s4 ) - rm( p - dy, s1, s2, s3, s4 ), 
				   rm( p + dz, s1, s2, s3, s4 ) - rm( p - dz, s1, s2, s3, s4 ) );
	n = normalize( n );
	
	vec3 spC = colorizeDF( p, n );
	
	vec3 l = vec3( 1.0, 1.0, -1.0 );
	l = normalize( l );
	float t = length( p - ro );
	
	return vec3( max( 0.0, min( 1.0, length( p - ro ) * rm( p + 0.1 * l, s1, s2, s3, s4 ) ) ) ) + spC;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
	
	vec2 uv2 = uv - 0.5;
	uv2.x *= 1.78;
	uv2 *= 1.33;
	
	vec3 color = colorize( uv2 );
	fragColor = vec4(color,1.0);
}`,
    "Trippy white fractals": `
#define PI 3.14159
const float scale = 512.;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (fragCoord.yx - .5*iResolution.yx) / iResolution.x;
    if (abs(uv.x) > .5 || abs(uv.y) > .5) return;

    float time = 2.*PI*(iTime/2.5);
    float total = 0.;
    for (int i = 0; i < 7; i++)
    {
        float angle = PI*float(i)/7.;
        float len = dot(uv, vec2(cos(angle), sin(angle)));
        total += (cos(2.*scale*len + time)+1.)/2.;
    }

    // Average 3 low-freq bins to stabilize the reading
    float bass = (texture(iChannel0, vec2(0.01, 0.25)).x
                + texture(iChannel0, vec2(0.03, 0.25)).x
                + texture(iChannel0, vec2(0.05, 0.25)).x) / 3.0;

    float brightness = 1. - abs(2.*fract(.5*total) - 1.);

    // Hue drifts slowly, bass peaks push it
    float hue = fract(iTime * 0.04 + bass * 0.25);
    // Saturation stays low at rest, spikes on bass hit
    float sat = 0.1 + 0.7 * pow(bass, 2.0);

    fragColor = vec4(hsv2rgb(vec3(hue, sat, brightness)), 1.0);
}
`,
    "FTL Needles": `/*
    "Fragments" by @XorDev
    
    https://x.com/XorDev/status/1963618494861258842
*/
void mainImage( out vec4 O, vec2 I)
{
    //Iterator, raymarch depth, turbulence frequency and time
    float i,z,f,t=iTime;
    //Raymarch sample point
    vec3 p;
    //Clear fragColor and raymarch 30 steps
    for(O*=i;i++<3e1;
        //Distance field to cylinder + gyroid
        z+=f=.003+abs(length(p.xy)-5.+dot(cos(p),sin(p).yzx))/8.,
        //Color in waves and attenuate light
        O+=(1.+sin(i*.3+z+t+vec4(6,1,2,0)))/f)
        //Compute sample point and iterate through turbulence waves
        //https://mini.gmshaders.com/p/turbulence
        for(p=z*normalize(vec3(I+I,0)-iResolution.xyy),p.z-=t,f=1.;f++<6.;
            //Blocky, stretched waves
            p+=sin(round(p.yxz*6.)/3.*f)/f);
    //Tanh tonemapping
    //https://mini.gmshaders.com/p/func-tanh
    O=tanh(O/1e3);
}
    
//http://shadertoy.com/view/3cScWy
`,
    "Colorful Nebula": `// just cruising through some colorful noise 😎
#define P(z) vec3(cos(vec2(.15,.2)*(z))*5.,z)  
#define R(a) mat2(cos(a+vec4(0,33,11,0)))
#define hue(v) (.6 + .6 * cos(6.3*(v) + vec4(0,23,21,0)))

// Simple hash function for randomness
float hash(float n) { return fract(sin(n) * 43758.5453123); }

void mainImage(out vec4 o, vec2 u) {
    float i,d,s,n,t=iTime*2.;
    
    // Sample low frequencies for beat detection
    float bass = texture(iChannel0, vec2(0.05, 0.0)).x;
    float kick = texture(iChannel0, vec2(0.02, 0.0)).x;
    
    // Smoother beat detection with gentler thresholds
    float beatThreshold = 0.25;
    float beatStrength = smoothstep(beatThreshold, beatThreshold + 0.3, bass);
    float kickStrength = smoothstep(beatThreshold, beatThreshold + 0.4, kick);
    
    // Smooth the beat strength over time to reduce abrupt changes
    beatStrength = mix(beatStrength, smoothstep(0.2, 0.8, bass), 0.3);
    kickStrength = mix(kickStrength, smoothstep(0.2, 0.8, kick), 0.4);
    
    // Beat-synced random values with smoother interpolation
    float beatTime = floor(t * 1.5); // Slower beat quantization
    float rand1 = hash(beatTime + 1.0) * 2.0 - 1.0;
    float rand2 = hash(beatTime + 2.0) * 2.0 - 1.0;
    float rand3 = hash(beatTime + 3.0) * 2.0 - 1.0;
    
    // Smooth the random values
    float smoothBeat = fract(t * 1.5);
    smoothBeat = smoothstep(0.0, 1.0, smoothBeat);
    
    vec3  q = iResolution,
          p = P(t),
          Z = normalize(P(t+1.) - p),
          X = normalize(vec3(Z.z,0,-Z.x)),
          D = vec3(R(tanh(2.*sin(p.z*.1))) * (u-q.xy/2.)/q.y, 1) 
              * mat3(-X, cross(X, Z), Z);
    
    for(o*=i; i++<1e2;) {
        p += D * s;
        s = tanh(length((p-P(p.z)).xy));
        for(n = .3; n < 16.;
            s -= abs(dot(sin(.05*t+p*n), p-p+.2)) / n,
            n += n);
        d += s = .001 + .8 * abs(s);
        
        // Much more subtle hue shifting
        float hueShift = rand1 * beatStrength * 0.15; // Reduced from 0.5
        vec4 baseColor = hue(.3*p.z + 5.0 + hueShift) / s;
        
        // Gentler beat-reactive brightness (reduced intensity)
        baseColor *= (1.0 + kickStrength * 0.3); // Reduced from 0.8
        
        // Very subtle color pops
        vec3 colorPop = vec3(rand2, rand3, rand1) * beatStrength * 0.08; // Reduced from 0.2
        baseColor.rgb += colorPop * smoothBeat; // Smooth transition
        
        o += baseColor;
    }
    o = tanh(o / 2e4);
    
    // Much gentler final beat pop
    o.rgb += vec3(kickStrength * 0.03) * vec3(rand1, rand2, rand3); // Reduced from 0.1
}


// https://www.shadertoy.com/view/WfjcD3`,
    "Sunyy Ray": `vec2 rotate(vec2 xy, float a) {
  return vec2(xy.x*cos(a)-xy.y*sin(a), xy.x*sin(a)+xy.y*cos(a));
}

float checkerBit(float val, vec2 p, vec2 period, vec2 shiftBias)
{
    float shift = mod(abs(p.x), period.x) * shiftBias.x + (shiftBias.y - mod(abs(p.y), period.y));
    return mod(floor(val / exp2(shift)), 2.0);
}

void mainImage(out vec4 rgba, in vec2 xy)
{
    xy = (xy - iResolution.xy * 0.5);

    float x = abs(xy.x);
    float y = xy.y;
    float z = x - y;
    float w = x + y;
    float frame = floor(iFrame);

    // sky
    float skyVal1 = -frame * 16.0 + x * x + y * y;
    float skyVal2 = -frame * 16.0 + z * z + w * w;
    rgba = abs(vec4(1.0,
        1.0 - checkerBit(skyVal1, vec2(x, y), vec2(4.0, 4.0), vec2(4.0, 4.0)) - 0.1,
        1.0 - checkerBit(skyVal2, vec2(w, z), vec2(4.0, 4.0), vec2(4.0, 4.0)) - 0.1,
        1.0));

    // gull
    xy -= iResolution.xy * 0.25;
    xy.y *= 1.2;
    xy *= 1.5;
    xy = rotate(xy, cos(iTime * 2.0) / 6.0)
       / clamp(abs(cos(iTime / 4.0)) + 0.6, 0.7, 1.5)
       + iResolution.xy * 0.25
       - vec2(-18.0 + cos(iTime) * 45.0, 24.0 + sin(iTime * 45.0));

    vec2 gullCenter = iResolution.xy * 0.25;
    if ((length(xy - gullCenter) < 36.0 || length(xy - gullCenter - vec2(64.0, 0.0)) < 36.0)
     && (length(xy - gullCenter + vec2(9.0, 18.0)) > 40.0
      && length(xy - gullCenter - vec2(73.0, -18.0)) > 40.0))
    {
        rgba += vec4(1.0);
        if (xy.y - iResolution.y * 0.25 < sin(x + iTime * 45.0)) rgba *= vec4(0.25);
        if (mod(y, 2.0) < 1.0) rgba *= vec4(0.75);
        if (mod(x, 2.0) < 1.0) rgba *= vec4(0.75);
    }

    // sea
    if (y + sqrt(abs(cos(x / 16.0 + iTime * 3.0))) * 8.0 < 1.0)
    {
        float seaVal = frame * 16.0 + x * x + y * y - z * z - w * w;
        float seaBit = checkerBit(seaVal, vec2(x, y), vec2(3.0, 2.0), vec2(2.0, 6.0));
        rgba = vec4(0.0, seaBit, seaBit, 1.0);
    }
}`,
    "Synthic": `#define time iTime

float box(vec3 p, vec3 s)
{
	vec3 q = fract(p)*2.0 -1.0;
    return length(max(abs(q)-s,0.0));
}


float trace (vec3 o,vec3 r)
{
    float t = 0.0;
    for(int i=0;i<50;i++)
    {
        vec3 p = o+r*t;
        float d0 = box(p-vec3(0,0,0),vec3(0.25,1,0.5));
        t+=d0*0.5;
    }
    return t;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{

    vec2 uv = vec2(fragCoord.x/iResolution.x,fragCoord.y/iResolution.y);
    uv-= 1.0;
    uv/= vec2(iResolution.y/iResolution.x,1.0);

    vec3 r = normalize(vec3(uv,0.1));
    float tt = time*0.25;
    r.yz *= sin((r.yz*100.0+tt));
    vec3 o = vec3(0,0,tt);
    
    float t = trace(o,r);

    float fog = 0.5/(1.0+t*t*0.1); 
    fragColor = vec4(vec3(fog+vec3(1,0.3,0.0)),1);
}`,

    "Fractal Sounders": `#define pi 3.14159265359

#define iTime tan(iTime*.1)+iTime*.1

float bassBoostLow = 0.0;
float bassBoostHigh = 0.0;
float time = 0.0;

vec3 hsv(in float h, in float s, in float v)
{
	return mix(vec3(1.0), clamp((abs(fract(h + vec3(3, 2, 1) / 3.0) * 6.0 - 3.0) - 1.0), 0.0 , 1.0), s) * v;
}

vec3 formula(in vec2 p, in vec2 c)
{
	const float n = 2.0;
	const int iters = 5;

	//float time = iTime*0.1;
	vec3 col = vec3(0);
	float t = 1.0;
	float dpp = dot(p, p);
	float lp = sqrt(dpp);
	float r = smoothstep(0.0, 0.2, lp);
	
	for (int i = 0; i < iters; i++) {
		// The transformation
        //p+=vec2(sin(c.x+p.x)*.01,
        //        cos(c.y+p.y)*.01);
        float to = bassBoostHigh;
        float index = mod(float(i)*1234.1234, 2.0);
        
        
        if(index < .1)
        {
        	p = p*mat2(cos(cos(time+to)+time+to), -sin(cos(time+to)+time+to),
                   sin(cos(time+to)+time+to), cos(cos(time+to)+time+to));
			p = abs(mod(p*(1.0) + c, n) - (n)/2.0);
        }
        else if(index < 1.1)
			p = abs(mod(p*(1.0) + c, n) - (n)/2.0);//mod(p/dpp + c, n) - n/2.0;
        else if(index < 2.1)
			p = p+to;
		
		dpp = dot(p, p);
        p /= dpp;
		lp = pow(dpp, 1.5);
        
        
        //if(int(14.0*sin(iTime))+iters < i) break;

		//Shade the lines of symmetry black
#if 0
		// Get constant width lines with fwidth()
		float nd = fwidth(dpp);
		float md = fwidth(lp);
		t *= smoothstep(0.0, 0.5, abs((n/2.0-p.x)/nd*n))
		   * smoothstep(0.0, 0.5, abs((n/2.0-p.y)/nd*n))
		   * smoothstep(0.0, 0.5, abs(p.x/md))
		   * smoothstep(0.0, 0.5, abs(p.y/md));
#else
		// Variable width lines
		t *= smoothstep(0.0, 0.01, abs(n/2.0-p.x)*lp)
		   * smoothstep(0.0, 0.01, abs(n/2.0-p.y)*lp)
		   * smoothstep(0.0, 0.01, abs(p.x)*2.0) 
		   * smoothstep(0.0, 0.01, abs(p.y)*2.0);
#endif

		// Fade out the high density areas, they just look like noise
		r *= smoothstep(0.0, 0.2, lp);
		
		// Add to colour using hsv
		col += lp+bassBoostHigh;
		
	}
	
    col = vec3(sin(col.x+time*.125),
               sin(col.y+time*.125+4.0*pi/3.0),
               sin(col.z+time*.125+2.0*pi/3.0))*.5+.5;
    
	return col*t;
}

float lowAverage()
{
    const int iters = 32;
    float sum = 0.0;
    
    float last = length(texture(iChannel0, vec2(0.0)));
    float next;
    for(int i = 1; i < iters/2; i++)
    {
        next = length(texture(iChannel0, vec2(float(i)/float(iters), 0.0)));
        sum += last;//pow(abs(last-next), 1.0);
        last = next;
    }
    return sum/float(iters)*2.0;
}

float highAverage()
{
    const int iters = 32;
    float sum = 0.0;
    
    float last = length(texture(iChannel0, vec2(0.0)));
    float next;
    for(int i = 17; i < iters; i++)
    {
        next = length(texture(iChannel0, vec2(float(i)/float(iters), 0.0)));
        sum += last;//pow(abs(last-next), 1.0);
        last = next;
    }
    return sum/float(iters)*2.0;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
	vec2 p = -1.0 + 2.0 * fragCoord.xy / iResolution.xy;
    
    bassBoostLow += lowAverage()*1.0;
    bassBoostHigh += highAverage()*1.0;
    time = iTime+bassBoostLow*8.0*pi;
    
    p += .125;
    
    p += .5*vec2(cos(time), sin(time));
    
	p.x *= iResolution.x / iResolution.y;
	p *= 1.5+1.125*sin(time*.25);
    
	const vec2 e = vec2(0.06545465634, -0.05346356485);
	vec2 c = time*e;
	//c = 8.0*iMouse.xy/iResolution.xy;
	float d = 1.0;
	vec3 col = vec3(0.0);
	const float blursamples = 4.0;
	float sbs = sqrt(blursamples);
	float mbluramount = 1.0/iResolution.x/length(e)/blursamples*2.0;
	float aabluramount = 1.0/iResolution.x/sbs*4.0;
	for (float b = 0.0; b < blursamples; b++) {
		col += formula(
			p + vec2(mod(b, sbs)*aabluramount, b/sbs*aabluramount), 
			c + e*mbluramount*b);
	}
	col /= blursamples;
	fragColor = vec4(col, 1.0);
}`,
    "Intergalactic offices": `// Raymarching sketch inspired by the work of Marc-Antoine Mathieu
// Leon 2017-11-21
// using code from IQ, Mercury, LJ, Duke, Koltes
// Enhanced with subtle bass-responsive camera bob

// tweak it
#define donut 30.
#define cell 4.
#define height 2.
#define thin .04
#define radius 15.
#define speed 1.

#define STEPS 100.
#define VOLUME 0.001
#define PI 3.14159
#define TAU (2.*PI)
#define time iTime

// raymarching toolbox
float rng (vec2 seed) { return fract(sin(dot(seed*.1684,vec2(54.649,321.547)))*450315.); }
mat2 rot (float a) { float c=cos(a),s=sin(a); return mat2(c,-s,s,c); }
float sdSphere (vec3 p, float r) { return length(p)-r; }
float sdCylinder (vec2 p, float r) { return length(p)-r; }
float sdDisk (vec3 p, vec3 s) { return max(max(length(p.xz)-s.x, s.y), abs(p.y)-s.z); }
float sdIso(vec3 p, float r) { return max(0.,dot(p,normalize(sign(p))))-r; }
float sdBox( vec3 p, vec3 b ) { vec3 d = abs(p) - b; return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0)); }
float sdTorus( vec3 p, vec2 t ) { vec2 q = vec2(length(p.xz)-t.x,p.y); return length(q)-t.y; }
float amod (inout vec2 p, float count) { float an = TAU/count; float a = atan(p.y,p.x)+an/2.; float c = floor(a/an); c = mix(c,abs(c),step(count*.5,abs(c))); a = mod(a,an)-an/2.; p.xy = vec2(cos(a),sin(a))*length(p); return c; }
float amodIndex (vec2 p, float count) { float an = TAU/count; float a = atan(p.y,p.x)+an/2.; float c = floor(a/an); c = mix(c,abs(c),step(count*.5,abs(c))); return c; }
float repeat (float v, float c) { return mod(v,c)-c/2.; }
vec2 repeat (vec2 v, vec2 c) { return mod(v,c)-c/2.; }
vec3 repeat (vec3 v, float c) { return mod(v,c)-c/2.; }
float smoo (float a, float b, float r) { return clamp(.5+.5*(b-a)/r, 0., 1.); }
float smin (float a, float b, float r) { float h = smoo(a,b,r); return mix(b,a,h)-r*h*(1.-h); }
float smax (float a, float b, float r) { float h = smoo(a,b,r); return mix(a,b,h)+r*h*(1.-h); }
vec2 displaceLoop (vec2 p, float r) { return vec2(length(p.xy)-r, atan(p.y,p.x)); }
float map (vec3);
float getShadow (vec3 pos, vec3 at, float k) {
    vec3 dir = normalize(at - pos);
    float maxt = length(at - pos);
    float f = 01.;
    float t = VOLUME*50.;
    for (float i = 0.; i <= 1.; i += 1./15.) {
        float dist = map(pos + dir * t);
        if (dist < VOLUME) return 0.;
        f = min(f, k * dist / t);
        t += dist;
        if (t >= maxt) break;
    }
    return f;
}
vec3 getNormal (vec3 p) { vec2 e = vec2(.01,0); return normalize(vec3(map(p+e.xyy)-map(p-e.xyy),map(p+e.yxy)-map(p-e.yxy),map(p+e.yyx)-map(p-e.yyx))); }

void camera (inout vec3 p) {
    p.xz *= rot(PI/8.);
    p.yz *= rot(PI/6.);
}

float windowCross (vec3 pos, vec4 size, float salt) {
    vec3 p = pos;
    float sx = size.x * (.6+salt*.4);
    float sy = size.y * (.3+salt*.7);
    vec2 sxy = vec2(sx,sy);
    p.xy = repeat(p.xy+sxy/2., sxy);
    float scene = sdBox(p, size.zyw*2.);
    scene = min(scene, sdBox(p, size.xzw*2.));
    scene = max(scene, sdBox(pos, size.xyw));
    return scene;
}

float window (vec3 pos, vec2 dimension, float salt) {
    float thinn = .008;
    float depth = .04;
    float depthCadre = .06;
    float padding = .08;
    float scene = windowCross(pos, vec4(dimension,thinn,depth), salt);
    float cadre = sdBox(pos, vec3(dimension, depthCadre));
    cadre = max(cadre, -sdBox(pos, vec3(dimension - padding, depthCadre*2.)));
    scene = min(scene, cadre);
    return scene;
}

float boxes (vec3 pos, float salt) {
    vec3 p = pos;
    float ry = cell * .43*(.3+salt);
    float rz = cell * .2*(.5+salt);
    float salty = rng(vec2(floor(pos.y/ry), floor(pos.z/rz)));
    pos.y = repeat(pos.y, ry);
    pos.z = repeat(pos.z, rz);
    float scene = sdBox(pos, vec3(.1+.8*salt+salty,.1+.2*salt,.1+.2*salty));
    scene = max(scene, sdBox(p, vec3(cell*.2)));
    return scene;
}

float map (vec3 pos) {
    vec3 camOffset = vec3(-4,0,0.);

    float scene = 1000.;
    vec3 p = pos + camOffset;
    float segments = PI*radius;
    float indexX, indexY, salt;
    vec2 seed;

    // donut distortion
    vec3 pDonut = p;
    pDonut.x += donut;
    pDonut.y += radius;
    pDonut.xz = displaceLoop(pDonut.xz, donut);
    pDonut.z *= donut;
    pDonut.xzy = pDonut.xyz;
    pDonut.xz *= rot(time*.05*speed);

    // ground
    p = pDonut;
    scene = min(scene, sdCylinder(p.xz, radius-height));

    // walls
    p = pDonut;
    float py = p.y + time * speed;
    indexY = floor(py / (cell+thin));
    p.y = repeat(py, cell+thin);
    scene = min(scene, max(abs(p.y)-thin, sdCylinder(p.xz, radius)));
    amod(p.xz, segments);
    p.x -= radius;
    scene = min(scene, max(abs(p.z)-thin, p.x));

    // horizontal windot
    p = pDonut;
    p.xz *= rot(PI/segments);
    py = p.y + time * speed;
    indexY = floor(py / (cell+thin));
    p.y = repeat(py, cell+thin);
    indexX = amodIndex(p.xz, segments);
    amod(p.xz, segments);
    seed = vec2(indexX, indexY);
    salt = rng(seed);
    p.x -= radius;
    vec2 dimension = vec2(.75,.5);
    p.x +=  dimension.x * 1.5;
    scene = max(scene, -sdBox(p, vec3(dimension.x, .1, dimension.y)));
    scene = min(scene, window(p.xzy, dimension, salt));

    // vertical window
    p = pDonut;
    py = p.y + cell/2. + time * speed;
    indexY = floor(py / (cell+thin));
    p.y = repeat(py, cell+thin);
    indexX = amodIndex(p.xz, segments);
    amod(p.xz, segments);
    seed = vec2(indexX, indexY);
    salt = rng(seed);
    p.x -= radius;
    dimension.y = 1.5;
    p.x +=  dimension.x * 1.25;
    scene = max(scene, -sdBox(p, vec3(dimension, .1)));
    scene = min(scene, window(p, dimension, salt));

    // elements
    p = pDonut;
    p.xz *= rot(PI/segments);
    py = p.y + cell/2. + time * speed;
    indexY = floor(py / (cell+thin));
    p.y = repeat(py, cell+thin);
    indexX = amodIndex(p.xz, segments);
    amod(p.xz, segments);
    seed = vec2(indexX, indexY);
    salt = rng(seed);
    p.x -= radius - height;
    scene = min(scene, boxes(p, salt));

    return scene;
}

void mainImage( out vec4 color, in vec2 coord ) {
    vec2 uv = (coord.xy-.5*iResolution.xy)/iResolution.y;
    
    // Sample bass frequencies for camera bob
    float bass = texture2D(iChannel0, vec2(0.05, 0.0)).x;
    
    // Create subtle vertical camera bob with bass
    float bassBob = bass * 0.8; // Subtle multiplier for rave music
    
    vec3 eye = vec3(0, bassBob, -20);
    vec3 ray = normalize(vec3(uv, 1.3));
    camera(eye);
    camera(ray);
    float dither = rng(uv+fract(time));
    vec3 pos = eye;
    float shade = 0.;
    for (float i = 0.; i <= 1.; i += 1./STEPS) {
        float dist = map(pos);
        if (dist < VOLUME) {
            shade = 1.-i;
            break;
        }
        dist *= .5 + .1 * dither;
        pos += ray * dist;
    }
    vec3 light = vec3(40.,100.,-10.);
    float shadow = getShadow(pos, light, 4.);
    color = vec4(1);
    color *= shade;
    color *= shadow;
    color = smoothstep(.0, .5, color);
    color.rgb = sqrt(color.rgb);
}`,
    "Pianoscope": `// CC0: Sunday morning random results with subtle audio reactivity
//  Tinkering around on sunday morning

#define TIME        iTime
#define RESOLUTION  iResolution
#define PI          3.141592654
#define TAU         (2.0*PI)
#define ROT(a)      mat2(cos(a), sin(a), -sin(a), cos(a))

const int max_iter = 5;

// Audio reactivity helper functions
float getBass() {
    // Sample bass frequencies (0.05-0.1 range in frequency domain)
    float bass = 0.0;
    for (int i = 0; i < 5; i++) {
        bass += texture(iChannel0, vec2(0.01 + 0.02*float(i), 0.0)).x;
    }
    bass /= 30.0;
    
    // Dampen to avoid extreme reactions
    return 0.4 + 0.3 * smoothstep(0.0, 0.7, bass);
}

float getMids() {
    // Sample mid frequencies
    float mids = texture(iChannel0, vec2(0.3, 0.0)).x;
    
    // Provide subtle response
    return 0.8 + 0.2 * smoothstep(0.0, 0.7, mids);
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
vec2 mod2(inout vec2 p, vec2 size) {
  vec2 c = floor((p + size*0.5)/size);
  p = mod(p + size*0.5,size) - size*0.5;
  return c;
}

// License: Unknown, author: Unknown, found: don't remember
float hash(vec2 co) {
  return fract(sin(dot(co.xy ,vec2(12.9898,58.233))) * 13758.5453);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/articles/distfunctions2d
float box(vec2 p, vec2 b) {
  vec2 d = abs(p)-b;
  return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

// License: MIT, author: Inigo Quilez, found: https://www.iquilezles.org/www/articles/smin/smin.htm
float pmin(float a, float b, float k) {
  float h = clamp(0.5+0.5*(b-a)/k, 0.0, 1.0);
  return mix(b, a, h) - k*h*(1.0-h);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float pabs(float a, float k) {
  return -pmin(a, -a, k);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec2 toPolar(vec2 p) {
  return vec2(length(p), atan(p.y, p.x));
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec2 toRect(vec2 p) {
  return vec2(p.x*cos(p.y), p.x*sin(p.y));
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
float modMirror1(inout float p, float size) {
  float halfsize = size*0.5;
  float c = floor((p + halfsize)/size);
  p = mod(p + halfsize,size) - halfsize;
  p *= mod(c, 2.0)*2.0 - 1.0;
  return c;
}

float smoothKaleidoscope(inout vec2 p, float sm, float rep) {
  vec2 hp = p;

  vec2 hpp = toPolar(hp);
  float rn = modMirror1(hpp.y, TAU/rep);

  float sa = PI/rep - pabs(PI/rep - abs(hpp.y), sm);
  hpp.y = sign(hpp.y)*(sa);

  hp = toRect(hpp);

  p = hp;

  return rn;
}

float shape(vec2 p) {
  // Get audio bass for subtle movement modification
  float audioMod = getBass();
  
  // Slightly adjust amplitude based on bass
  const float amp = 10.0;
  
  // Adjust motion speed very subtly with audio
  p += amp*sin(vec2(1.0, sqrt(0.5))*0.026*TIME*TAU/amp * audioMod);
  
  vec2 cp = p;
  vec2 np = round(p);
  cp -= np;

  float h0 = hash(np+123.4); 
  if (h0 > 0.5) {
    cp = vec2(-cp.y, cp.x);
  }

  vec2 cp0 = cp;
  cp0 -= -0.5;
  float d0 = (length(cp0)-0.5);
  vec2 cp1 = cp;
  cp1 -= 0.5;
  float d1 = (length(cp1)-0.5);
  
  float d = d0;
  d = min(d, d1);
  
  // Subtly adjust shape thickness with audio
  d = abs(d)-0.125 * mix(0.95, 1.05, getMids());
  
  return d;
}

vec2 df(vec2 p, out int ii, out bool inside) {
  float sz = 0.9;
  float ds = shape(p);
  vec2 pp = p;

  float r = 0.0;

  ii = max_iter;
  for (int i=0; i<max_iter; ++i) {
    pp = p;
    vec2 nn = mod2(pp, vec2(sz));
  
    vec2 cp = nn*sz;
    float d = shape(cp);
    
    r = sz*0.5; 

    if (abs(d) > 0.5*sz*sqrt(2.0)) {
      ii = i;
      inside = d < 0.0;
      break;
    }

    sz /= 3.0;
  }
  
  float aa = 0.25*sz;

  float d0 = box(pp, vec2(r-aa))-aa; 
  float d1 = length(pp);
  return vec2(d0, d1);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
  vec2 q = fragCoord/RESOLUTION.xy;
  vec2 p = -1. + 2. * q;
  vec2 pp = p;
  p.x *= RESOLUTION.x/RESOLUTION.y;
  float aa = 4.0/RESOLUTION.y;
  vec2 op = p;
  
  // Get audio influences
  float bassValue = getBass();
  float midsValue = getMids();
  
  // Subtly adjust rotation speed with bass
  mat2 rot = ROT(0.0125*TIME * mix(0.9, 1.1, bassValue)); 
  p *= rot;
  
  // Very slight adjustment to kaleidoscope with mid frequencies
  float kaleidoRepetitions = 34.0 * mix(0.98, 1.02, midsValue);
  smoothKaleidoscope(p, 0.025, kaleidoRepetitions);
  
  p *= ROT(0.25*length(op));
  p *= transpose(rot);

  int i;
  bool inside;
  vec2 d2 = df(p, i, inside);
  float ii = float(i)/float(max_iter);
  vec3 col = vec3(0.0);
  
  // Apply subtle color variation with audio
  vec3 rgb = 0.5*(1.0+cos(0.5*TIME-0.5*PI*length(p) + 
                 vec3(0.0, 1.0, 2.0) + PI*ii + 
                 (inside ? (2.0*(dot(p,pp)+1.0)) : 0.0)));
                 
  // Glow reacts to bass but is kept subtle
  rgb += 0.0025/max(d2.y, 0.005) * mix(0.9, 1.1, bassValue);
  
  col = mix(col, rgb, smoothstep(0.0, -aa, d2.x));
  col -= vec3(0.25)*(length(op)+0.0);
  col *= smoothstep(1.5, 0.5, length(pp));
  
  // Very subtle overall color boost with audio
  col *= 1.0 + 0.05 * bassValue;
  
  col = sqrt(col);

  fragColor = vec4(col, 1.0);
}`,
"Multiversal Web": `#define PI 3.141592654
mat2 rot(float x)
{
    return mat2(cos(x), sin(x), -sin(x), cos(x));
}
vec2 foldRotate(in vec2 p, in float s) {
    float a = PI / s - atan(p.x, p.y);
    float n = PI * 2. / s;
    a = floor(a / n) * n;
    p *= rot(a);
    return p;
}
float sdRect( vec2 p, vec2 b )
{
  vec2 d = abs(p) - b;
  return min(max(d.x, d.y),0.0) + length(max(d,0.0));
}
// TheGrid by dila
// https://www.shadertoy.com/view/llcXWr
float tex(vec2 p, float z)
{
    p = foldRotate(p, 8.0);
    vec2 q = (fract(p / 10.0) - 0.5) * 10.0;
    for (int i = 0; i < 3; ++i) {
        for(int j = 0; j < 2; j++) {
        	q = abs(q) - .25;
        	q *= rot(PI * .25);
        }
        q = abs(q) - vec2(1.0, 1.5);
        q *= rot(PI * .25 * z);
		q = foldRotate(q, 3.0);  
    }
	float d = sdRect(q, vec2(1., 1.));
    float f = 1.0 / (1.0 + abs(d));
    return smoothstep(.9, 1., f);
}
// The Drive Home by BigWings
// https://www.shadertoy.com/view/MdfBRX
float Bokeh(vec2 p, vec2 sp, float size, float mi, float blur)
{
    float d = length(p - sp);
    float c = smoothstep(size, size*(1.-blur), d);
    c *= mix(mi, 1., smoothstep(size*.8, size, d));
    return c;
}
vec2 hash( vec2 p ){
	p = vec2( dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3)));
	return fract(sin(p)*43758.5453) * 2.0 - 1.0;
}
float dirt(vec2 uv, float n)
{
    vec2 p = fract(uv * n);
    vec2 st = (floor(uv * n) + 0.5) / n;
    vec2 rnd = hash(st);
    return Bokeh(p, vec2(0.5, 0.5) + vec2(0.2) * rnd, 0.05, abs(rnd.y * 0.4) + 0.3, 0.25 + rnd.x * rnd.y * 0.2);
}
float sm(float start, float end, float t, float smo)
{
    return smoothstep(start, start + smo, t) - smoothstep(end - smo, end, t);
}
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
    uv = uv * 2.0 - 1.0;
    uv.x *= iResolution.x / iResolution.y;
    uv *= 2.0;
    
    vec3 col = vec3(0.0);
    #define N 6
    #define NN float(N)
    #define INTERVAL 3.0
    #define INTENSITY vec3((NN * INTERVAL - t) / (NN * INTERVAL))
    
    float time = iTime;
    for(int i = 0; i < N; i++) {
        float t;
        float ii = float(N - i);
        t = ii * INTERVAL - mod(time - INTERVAL * 0.75, INTERVAL);
        col = mix(col, INTENSITY, dirt(mod(uv * max(0.0, t) * 0.1 + vec2(.2, -.2) * time, 1.2), 3.5));
        
        t = ii * INTERVAL - mod(time + INTERVAL * 0.5, INTERVAL);
        col = mix(col, INTENSITY * vec3(0.7, 0.8, 1.0) * 1.3,tex(uv * max(0.0, t), 4.45));
        
        t = ii * INTERVAL - mod(time - INTERVAL * 0.25, INTERVAL);
        col = mix(col, INTENSITY * vec3(1.), dirt(mod(uv * max(0.0, t) * 0.1 + vec2(-.2, -.2) *  time, 1.2), 3.5));
        
        t = ii * INTERVAL - mod(time, INTERVAL);
    	float r = length(uv * 2.0 * max(0.0, t));
    	float rr = sm(-24.0, -0.0, (r - mod(time * 30.0, 90.0)), 10.0);
        col = mix(col, mix(INTENSITY * vec3(1.), INTENSITY * vec3(0.7, 0.5, 1.0) * 3.0, rr),tex(uv * 2.0 * max(0.0, t), 0.27 + (2.0 * rr)));
    }
	fragColor = vec4(col, 1.0);
}

`,
"Paper Gear": `// CC0: Truchet + Kaleidoscope FTW
//  Bit of experimenting with kaleidoscopes and truchet turned out nice
//  Quite similar to an earlier shader I did but I utilized a different truchet pattern this time
#define PI              3.141592654
#define TAU             (2.0*PI)
#define RESOLUTION      iResolution
#define TIME            iTime
#define ROT(a)          mat2(cos(a), sin(a), -sin(a), cos(a))
#define PCOS(x)         (0.5+0.5*cos(x))

// License: Unknown, author: Unknown, found: don't remember
vec4 alphaBlend(vec4 back, vec4 front) {
  float w = front.w + back.w*(1.0-front.w);
  vec3 xyz = (front.xyz*front.w + back.xyz*back.w*(1.0-front.w))/w;
  return w > 0.0 ? vec4(xyz, w) : vec4(0.0);
}

// License: Unknown, author: Unknown, found: don't remember
vec3 alphaBlend(vec3 back, vec4 front) {
  return mix(back, front.xyz, front.w);
}

// License: Unknown, author: Unknown, found: don't remember
float hash(float co) {
  return fract(sin(co*12.9898) * 13758.5453);
}

// License: Unknown, author: Unknown, found: don't remember
float hash(vec2 p) {
  float a = dot(p, vec2 (127.1, 311.7));
  return fract(sin (a)*43758.5453123);
}

// License: Unknown, author: Unknown, found: don't remember
float tanh_approx(float x) {
  //  Found this somewhere on the interwebs
  //  return tanh(x);
  float x2 = x*x;
  return clamp(x*(27.0 + x2)/(27.0+9.0*x2), -1.0, 1.0);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/articles/smin
float pmin(float a, float b, float k) {
  float h = clamp(0.5+0.5*(b-a)/k, 0.0, 1.0);
  return mix(b, a, h) - k*h*(1.0-h);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/index.htm
vec3 postProcess(vec3 col, vec2 q) {
  col = clamp(col, 0.0, 1.0);
  col = pow(col, vec3(1.0/2.2));
  col = col*0.6+0.4*col*col*(3.0-2.0*col);
  col = mix(col, vec3(dot(col, vec3(0.33))), -0.4);
  col *=0.5+0.5*pow(19.0*q.x*q.y*(1.0-q.x)*(1.0-q.y),0.7);
  return col;
}

float pmax(float a, float b, float k) {
  return -pmin(-a, -b, k);
}

float pabs(float a, float k) {
  return pmax(a, -a, k);
}

vec2 toPolar(vec2 p) {
  return vec2(length(p), atan(p.y, p.x));
}

vec2 toRect(vec2 p) {
  return vec2(p.x*cos(p.y), p.x*sin(p.y));
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
float modMirror1(inout float p, float size) {
  float halfsize = size*0.5;
  float c = floor((p + halfsize)/size);
  p = mod(p + halfsize,size) - halfsize;
  p *= mod(c, 2.0)*2.0 - 1.0;
  return c;
}

float smoothKaleidoscope(inout vec2 p, float sm, float rep) {
  vec2 hp = p;

  vec2 hpp = toPolar(hp);
  float rn = modMirror1(hpp.y, TAU/rep);

  float sa = PI/rep - pabs(PI/rep - abs(hpp.y), sm);
  hpp.y = sign(hpp.y)*(sa);

  hp = toRect(hpp);

  p = hp;

  return rn;
}

// The path function
vec3 offset(float z) {
  float a = z+texture(iChannel0, vec2(0.1, 0.0)).x;;
  vec2 p = -0.075*(vec2(cos(a), sin(a*sqrt(2.0))) + vec2(cos(a*sqrt(0.75)), sin(a*sqrt(0.5))));
  return vec3(p, z);
}

// The derivate of the path function
//  Used to generate where we are looking
vec3 doffset(float z) {
  float eps = 0.1;
  return 0.5*(offset(z + eps) - offset(z - eps))/eps;
}

// The second de=rivate of the path function
//  Used to generate tilt
vec3 ddoffset(float z) {
  float eps = 0.1;
  return 0.125*(doffset(z + eps) - doffset(z - eps))/eps;
}

vec2 cell_df(float r, vec2 np, vec2 mp, vec2 off) {
  const vec2 n0 = normalize(vec2(1.0, 1.0));
  const vec2 n1 = normalize(vec2(1.0, -1.0));

  np += off;
  mp -= off;
  
  float hh = hash(np);
  float h0 = hh;

  vec2  p0 = mp;  
  p0 = abs(p0);
  p0 -= 0.5;
  float d0 = length(p0);
  float d1 = abs(d0-r); 

  float dot0 = dot(n0, mp);
  float dot1 = dot(n1, mp);

  float d2 = abs(dot0);
  float t2 = dot1;
  d2 = abs(t2) > sqrt(0.5) ? d0 : d2;

  float d3 = abs(dot1);
  float t3 = dot0;
  d3 = abs(t3) > sqrt(0.5) ? d0 : d3;


  float d = d0;
  d = min(d, d1);
  if (h0 > .85)
  {
    d = min(d, d2);
    d = min(d, d3);
  }
  else if(h0 > 0.5)
  {
    d = min(d, d2);
  }
  else if(h0 > 0.15)
  {
    d = min(d, d3);
  }
  
  return vec2(d, d0-r);
}

vec2 truchet_df(float r, vec2 p) {
  vec2 np = floor(p+0.5);
  vec2 mp = fract(p+0.5) - 0.5;
  return cell_df(r, np, mp, vec2(0.0));
}

vec4 plane(vec3 ro, vec3 rd, vec3 pp, vec3 off, float aa, float n) {
  float h_ = hash(n);
  float h0 = fract(1777.0*h_);
  float h1 = fract(2087.0*h_);
  float h2 = fract(2687.0*h_);
  float h3 = fract(3167.0*h_);
  float h4 = fract(3499.0*h_);

  float l = length(pp - ro);

  vec3 hn;
  vec2 p = (pp-off*vec3(1.0, 1.0, 0.0)).xy;
  p *= ROT(0.5*(h4 - 0.5)*TIME);
  float rep = 2.0*round(mix(5.0, 30.0, h2));
  float sm = 0.05*20.0/rep;
  float sn = smoothKaleidoscope(p, sm, rep);
  p *= ROT(TAU*h0+0.025*TIME);
  float z = mix(0.2, 0.4, h3);
  p /= z;
  p+=0.5+floor(h1*1000.0);
  float tl = tanh_approx(0.33*l);
  float r = mix(0.30, 0.45, PCOS(0.1*n));
  vec2 d2 = truchet_df(r, p);
  d2 *= z;
  float d = d2.x;
  float lw =0.025*z; 
  d -= lw;
  
  vec3 col = mix(vec3(1.0), vec3(0.0), smoothstep(aa, -aa, d));
  col = mix(col, vec3(0.0), smoothstep(mix(1.0, -0.5, tl), 1.0, sin(PI*100.0*d)));
//  float t0 = smoothstep(aa, -aa, -d2.y-lw);
  col = mix(col, vec3(0.0), step(d2.y, 0.0));
  //float t = smoothstep(3.0*lw, 0.0, -d2.y);
//  float t = smoothstep(aa, -aa, -d2.y-lw);
  float t = smoothstep(aa, -aa, -d2.y-3.0*lw)*mix(0.5, 1.0, smoothstep(aa, -aa, -d2.y-lw));
  return vec4(col, t);
}

vec3 skyColor(vec3 ro, vec3 rd) {
  float d = pow(max(dot(rd, vec3(0.0, 0.0, 1.0)), 0.0), 20.0);
  return vec3(d);
}

vec3 color(vec3 ww, vec3 uu, vec3 vv, vec3 ro, vec2 p) {
  float lp = length(p);
  vec2 np = p + 1.0/RESOLUTION.xy;
  float rdd = (2.0+1.0*tanh_approx(lp));
//  float rdd = 2.0;
  vec3 rd = normalize(p.x*uu + p.y*vv + rdd*ww);
  vec3 nrd = normalize(np.x*uu + np.y*vv + rdd*ww);

  const float planeDist = 1.0-0.25;
  const int furthest = 6;
  const int fadeFrom = max(furthest-5, 0);

  const float fadeDist = planeDist*float(furthest - fadeFrom);
  float nz = floor(ro.z / planeDist);

  vec3 skyCol = skyColor(ro, rd);


  vec4 acol = vec4(0.0);
  const float cutOff = 0.95;
  bool cutOut = false;

  // Steps from nearest to furthest plane and accumulates the color 
  for (int i = 1; i <= furthest; ++i) {
    float pz = planeDist*nz + planeDist*float(i);

    float pd = (pz - ro.z)/rd.z;

    if (pd > 0.0 && acol.w < cutOff) {
      vec3 pp = ro + rd*pd;
      vec3 npp = ro + nrd*pd;

      float aa = 3.0*length(pp - npp);

      vec3 off = offset(pp.z);

      vec4 pcol = plane(ro, rd, pp, off, aa, nz+float(i));

      float nz = pp.z-ro.z;
      float fadeIn = smoothstep(planeDist*float(furthest), planeDist*float(fadeFrom), nz);
      float fadeOut = smoothstep(0.0, planeDist*0.1, nz);
      pcol.xyz = mix(skyCol, pcol.xyz, fadeIn);
      pcol.w *= fadeOut;
      pcol = clamp(pcol, 0.0, 1.0);

      acol = alphaBlend(pcol, acol);
    } else {
      cutOut = true;
      break;
    }

  }

  vec3 col = alphaBlend(skyCol, acol);
// To debug cutouts due to transparency  
//  col += cutOut ? vec3(1.0, -1.0, 0.0) : vec3(0.0);
  return col;
}

vec3 effect(vec2 p, vec2 q) {
  float tm  = TIME*0.25;
  vec3 ro   = offset(tm);
  vec3 dro  = doffset(tm);
  vec3 ddro = ddoffset(tm);

  vec3 ww = normalize(dro);
  vec3 uu = normalize(cross(normalize(vec3(0.0,1.0,0.0)+ddro), ww));
  vec3 vv = normalize(cross(ww, uu));

  vec3 col = color(ww, uu, vv, ro, p);
  
  return col;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 q = fragCoord/RESOLUTION.xy;
  vec2 p = -1. + 2. * q;
  p.x *= RESOLUTION.x/RESOLUTION.y;
  
  vec3 col = effect(p, q);
  col *= smoothstep(0.2, 4.0, TIME);
  col = postProcess(col, q);
 
  fragColor = vec4(col, 1.0);
}

`,
"Paper Shapes": `// Define variables with proper initialization
vec2 softPolyOffset = vec2(0.0, 0.0);
int softPolyIndex = 0;

float poly(vec2 p, int n, float r)
{
    // Use the actual n value instead of hardcoded 8
    float d = 0.0;
    float a_step = 3.14159265359 * 2.0 / float(n);
    
    for(int i = 0; i < 12; ++i) // Increase max iteration to handle large n values
    {
        if(i >= n) break; // Break if we've done enough iterations for this polygon
        float a = float(i) * a_step;
        float b = max(0.0, dot(p, vec2(cos(a), sin(a))) - r);
        d += b * b;
    }
    return sqrt(d);
}

float heart(vec2 p, float s)
{
    float d = max(dot(p, vec2(-1., -1.2) * 8.), dot(p, vec2(1., -1.2) * 8.));
    float u = abs(p.x) + 1.7;
    float v = max(0.0, p.y + 0.9);
    return length(vec2(d, length(vec2(u, v)))) - 1.8 - s;
}

float softPoly(vec2 p, int n, float r, float s)
{
    if(softPolyIndex == 12)
    {
        float d = heart(p, r - s);
        return clamp(smoothstep(0.0, s * 2.0, d), 0.0, 1.0);
    }
    
    p = abs(p);
    if(p.x > p.y)
        p = p.yx;
    
    float aa = 3.14159265359 / float(n); // Correct angle calculation
    mat2 rotMat = mat2(cos(aa), sin(aa), -sin(aa), cos(aa));
    p *= rotMat;
    p -= softPolyOffset;
    
    float d = poly(p, n, r - s);
    return clamp(smoothstep(0.0, s * 2.0, d), 0.0, 1.0);
}

// Audio reactivity helper
float getAudioReactivity() {
    // Sample audio data from low-mid range (adjust position as needed)
    float bass = texture(iChannel0, vec2(0.1, 0.0)).x;
    float mids = texture(iChannel0, vec2(0.3, 0.0)).x;
    
    // Smooth and scale the reactivity
    return 0.8 * (bass * 0.035 + mids * 0.015);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec2 t = (uv - vec2(0.5)) * 1.5;
    t.x *= iResolution.x / iResolution.y;
    
    // Add subtle audio reactivity to the background
    float reactivity = getAudioReactivity();
    vec3 col = vec3(0.2, 0.15, 0.1) * 
        (cos((t.y * 100.0 + sin(t.x + t.y * 5.0) * 10.0 * 
             cos(t.x * 3.0) * sin(t.y * 20.0)) * 2.0) * 0.1 + 0.9);
    
    // Make the background slightly audio reactive
    col *= 0.8 + 0.2 * reactivity;
    
    float depth = 0.0;
    float shad0 = 1.0, shad1 = 1.0;

    // Adjust animation speed with audio reactivity
    float timeScale = 0.3 * (0.8 + 0.2 * reactivity);
    
    for(int i = 0; i < 20; ++i)
    {
        softPolyIndex = i;
        
        // Make the polygon movement subtly audio reactive
        softPolyOffset = vec2(
            cos(float(i) + iTime * timeScale),
            sin(float(i) * 2.0 + iTime * timeScale * 0.5)
        ) * 0.4 * (0.9 + 0.1 * reactivity);
        
        vec2 p = t.xy;
        vec2 p2 = p;
        
        int n = 3 + int(mod(float(i), 7.0));
        // Slightly vary size with audio
        float r = 0.2 * (0.95 + 0.05 * reactivity);
        
        float a = 1.0 - softPoly(p2, n, r, 0.003);
        float as0 = softPoly(p2 + 2.0 * vec2(0.002, 0.005) * (1.0 + float(i) - depth), 
                            n, r, 0.01 + 0.003 * (1.0 + float(i) - depth));
        float as1 = softPoly(p2, n, r, 0.01 + 0.01 * (1.0 + float(i) - depth));
        
        shad0 *= as0;
        shad1 *= as1;
        shad0 = mix(shad0, 1.0, a);
        shad1 = mix(shad1, 1.0, a);
        
        // Make the heart color pulse with the audio
        vec3 c = (i == 12) ? 
            vec3(1.0, 0.3 + 0.2 * reactivity, 0.6 - 0.1 * reactivity) : 
            vec3(1.0);
            
        col = mix(col, c, a);
        depth = mix(depth, float(i + 1), a);
    }

    // Add more brightness to prevent everything from being black
    col = (0.6 * 0.5 * col * mix(0.2, 1.0, shad0) * vec3(1.0, 1.0, 0.6) + 
           0.5 * vec3(0.8, 0.8, 1.0) * col * mix(0.2, 1.0, shad1));

    col += pow(1.0 - smoothstep(0.0, 3.0, -t.y + 1.0), 4.0) * vec3(1.0, 1.0, 0.6) * 0.3;
    
    // Add a little extra brightness based on audio
    col += vec3(0.05 * reactivity);

    // Apply gamma correction
    fragColor.rgb = sqrt(max(col, vec3(0.001))); // Prevent negative values
    fragColor.a = 1.0;
}`,
"Muah Sound": `float getAudioReactivity() {
    // Sample multiple frequency ranges
    float bass = texture(iChannel0, vec2(0.05, 0.0)).x;
    float lowMids = texture(iChannel0, vec2(0.15, 0.0)).x;
    float mids = texture(iChannel0, vec2(0.3, 0.0)).x;
    
    // Create a smooth, musical response
    float beatPulse = pow(bass * 0.7 + lowMids * 0.3, 1.5);
    float smoothReact = mix(mids, beatPulse, 0.5);
    
    return smoothstep(0.0, 1.0, smoothReact);
}

#define MARCHLIMIT 70

vec3 camPos = vec3(0.0, 0.0, -1.0);
vec3 ld = vec3(0.0, 0.0, 1.0);
vec3 up = vec3(0.0, 1.0, 0.0);
vec3 right = vec3(1.0, 0.0, 0.0);
vec3 lightpos = vec3(1.5, 1.5, 1.5);

// Smooth HSV to RGB conversion 
vec3 hsv2rgb_smooth( in vec3 c )
{
    vec3 rgb = clamp( abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0 );
	rgb = rgb*rgb*(3.0-2.0*rgb); // cubic smoothing	
	return c.z * mix( vec3(1.0), rgb, c.y);
}

vec4 range(vec3 p)
{
    // Get audio reactivity
    float audioReact = getAudioReactivity();
    
    // Sphere with Radius
    vec3 spherepos = vec3(0.0, 0.0, 0.0);
    float radius = log(sin(iTime*0.1)*0.05+1.0)+0.1;
    // Subtle audio influence on radius
    radius += audioReact * 0.015;
	
    float anim = smoothstep(0., .1, cos(iTime*0.4)+1.0);
    float anim2 = smoothstep(0., .1, -cos(iTime*0.4)+1.0);
    
    // Audio-reactive wave amplitude
    float audioWave = 1.0 + audioReact * 0.2;
    float xampl = sin(iTime*1.3)*0.4*anim * audioWave;
    float yampl = (sin(iTime*1.3)*0.4-(anim2*0.3)) * audioWave;
    
    p.x += cos((max(-2.0+p.z-camPos.z,0.)))*xampl-xampl;
    p.y += sin((max(-2.0+p.z-camPos.z,0.)))*yampl;
    
    p = mod(p + vec3(0.5,0.5,0.5), vec3(1.0,1.0,1.0)) - vec3(0.5,0.5,0.5);
    spherepos = mod(spherepos + vec3(0.5,0.5,0.5), vec3(1.0,1.0,1.0)) - vec3(0.5,0.5,0.5);
    
    vec3 diff = p - spherepos;
    vec3 normal = normalize(diff);
    
    return vec4(normal, length(diff)-radius);
}

vec3 lerp(vec3 a, vec3 b, float p)
{
    p = clamp(p,0.,1.);
 	return a*(1.0-p)+b*p;   
}

vec4 march(vec3 cam, vec3 n)
{
    float len = 1.0;
    vec4 ret;
    
    for(int i = 0; i < MARCHLIMIT; i++)
    {
        ret = range(camPos + len*n)*0.5;
		len += ret.w;
    }
    
	return vec4(ret.xyz, len);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float audioReact = getAudioReactivity();
    
    // Audio-reactive color angle offset
    float colorangle = audioReact * 30.0; // Shift hue by up to 30 degrees with beat
    
	vec2 uv = (fragCoord.xy*2.0) / iResolution.xy - vec2(1, 1);
    uv.x *= iResolution.x / iResolution.y;
    
    // Base rotation with subtle audio influence
    float rotangle = iTime*0.08 + audioReact * 0.02;
    vec2 newuv;
    newuv.x = uv.x*cos(rotangle)-uv.y*sin(rotangle);
    newuv.y = uv.x*sin(rotangle)+uv.y*cos(rotangle);
    uv = newuv;
    
    camPos = vec3(0.5, 0.5, iTime*1.0);
    float zoom = 0.6;
    vec3 n = normalize(vec3(sin(uv.x*3.1415*zoom),sin(uv.y*3.1415*zoom) ,ld.z*cos(uv.x*3.1415*zoom)*cos(uv.y*3.1415*zoom)));
    vec4 rangeret = march(camPos, n);
    float d = log(rangeret.w / 1.0 + 1.0);
    vec3 normal = rangeret.xyz;
    
    vec3 p = camPos + n*d;
    float angle = acos(dot(normal, n)/length(normal)*length(n));
    
    // Original color calculation with audio enhancement
    vec3 color1 = vec3(
        d*0.1 + (colorangle + iTime)*0.01 + atan(uv.y/uv.x)*3.1415,
        2.0 + audioReact * 0.3, // Saturation boost with audio
        max(1.0 - log(d), 0.0) * (1.0 + audioReact * 0.2) // Brightness pulse
    );
    
    vec3 color2 = vec3(
        d*0.1 + ((colorangle + iTime)+120.0)*0.01,
        2.0 + audioReact * 0.3, // Saturation boost with audio
        max(1.0 - log(d), 0.0) * (1.0 + audioReact * 0.2) // Brightness pulse
    );
    
    // Original lerp based on angle, with audio influence on the mix
    float mixFactor = cos(angle/10.0 + audioReact * 0.5);
    
	fragColor = vec4(hsv2rgb_smooth(lerp(color1, color2, mixFactor)), 1.0);
}`,
"Firestorm": `/// 
/// This post cloned from below post.
/// I also reduce code and add a sound.
/// I respect the post.
///
/// [[ Fire Storm Cube ]]
/// 
/// https://www.shadertoy.com/view/ldyyWm
///

#define R iResolution

float burn;

mat2 rot(float a)
{
    float s = sin(a);
    float c = cos(a);
    
    return mat2(s, c, -c, s);
}

float map(vec3 p)
{
    float i = texture(iChannel0, vec2(0.2, 0.5)).x;
    
    float d1 = length(p) - 1. * i;
    
    //mat2 r = rot(-iTime / 3.0 + length(p));
    mat2 r = rot(iTime * 2.0 + length(p));
    p.xy *= r;
    p.zy *= r;
    
    p = abs(p);// - iTime;
    p = abs(p - round(p)) *  2.5 * i;
    
    //r = rot(iTime);
    //p.xy *= r;
    //p.xz *= r;
    
    float l1 = length(p.xy);
    float l2 = length(p.yz);
    float l3 = length(p.xz);
    
    float g = 0.01;
    float d2 = min(min(l1, l2), l3) + g;
    
    burn = pow(d2 - d1, 2.0);
    
    return min(d1, d2);
}

void mainImage( out vec4 O, in vec2 U )
{
    vec2 uv = (2.0 * U - R.xy) / R.y;
    vec3 ro = normalize(vec3(uv, 1.5));
    
    vec3 ta = vec3(0, 0, -2);
    
    float t = 0.;
    for  (int i = 0; i < 30; i++)
    {
        t += map(ta + ro * t) * 0.5;
    }

    O = vec4(1.0 - burn, 0, 0, 1.0);
}`,
"Needs Work - Big Bang": `void mainImage(out vec4 o, vec2 F) {
    vec2 R = iResolution.xy; 
    o-=o;
    
    // Audio reactivity with dampening
    float bass = texture(iChannel0, vec2(0.05, 0.0)).x;  
    float mids = texture(iChannel0, vec2(0.3, 0.0)).x;   
    float high = texture(iChannel0, vec2(0.7, 0.0)).x;   
    
    // Apply smoothing to audio values (very important for stability)
    bass = mix(bass, 0.5, 0.8);  // Heavily dampen towards neutral value
    mids = mix(mids, 0.5, 0.7);  // Less sensitive to rapid changes
    high = mix(high, 0.5, 0.6);  // Slightly more responsive for highs
    
    // Reduce overall movement speed by 2x
    float baseSpeed = 0.05;  // Was effectively 0.1 (t * 0.1)
    
    for(float d, t = iTime * baseSpeed, i = 0.; i > -1.; i -= .06) {
        // Greatly reduced audio impact on speed
        float speedMod = 1.0 + bass * 0.1;  // Was 0.5, now only 10% variation
        d = fract(i - 3.*t * speedMod);
        
        // Minimal scaling adjustment from mids
        float scale = 28.0 + mids * 2.0;  // Was 10.0, reduced to 2.0
        vec4 c = vec4((F - R *.5) / R.y * d, i, 0) * scale;
        
        for (int j=0; j++ < 27;) {
            // Keep some parameters static, reduce audio influence on others
            float param1 = 7.0 - 0.2 * sin(t);  // Removed bass influence
            float param2 = 6.3 + high * 0.5;    // Was 1.5, reduced to 0.5
            float param3 = 0.7;                 // Removed mids influence completely
            float param4 = 1.0 - cos(t/0.8) + high * 0.2;  // Was 0.5, reduced to 0.2
            
            c.xzyw = abs(c / dot(c,c) - vec4(param1, param2, param3, param4)/7.0);
        }
        
        // Much gentler color intensity modulation
        float intensity = 1.0 + bass * 0.5 + mids * 0.3;  // Was 2.0 & 1.0
        o -= c * c.yzww * d--*d / vec4(3, 5, 1, 1) * intensity;
    }
    
    // Retain subtle color variation
    o.r += high * 0.1;  // Was 0.2
    o.g += mids * 0.05; // Was 0.1
    o.b += bass * 0.15; // Was 0.3
}`,
"Black Hole": `
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
"Procedural Circuitry": `// This content is under the MIT License.

#define time iTime*.01
#define width .005
float zoom = .12;

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
	
	fragColor = mix(freqs[3]-.03, 1., v) * vec4(1.5*freqs[2] * t * t* t , 1.2*freqs[1] * t * t, freqs[3]*t, 1.0)+c2+starcolor;
}`,
"Temporal Fractal Mic'd": `// CC0: Appolloian with a twist II - Subtle Audio Reactive Version
// Original shader with very gentle audio reactivity - line segments stabilized

#define RESOLUTION  iResolution
#define TIME        iTime
#define MAX_MARCHES 320
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
  float bassBoost = 1.12 + getBassIntensity() * 0.0005;
  float scale = 1.0;
  
  for(int i=0; i < 5; ++i) {
    // Remove mid-frequency distortion only
    p = -1.0 + 2.0*fract(0.5*p + 0.5);
    
    float r2 = dot(p,p);
    float k = (s * bassBoost)/r2;
    p *= k;
    scale *= k;
  }
  
  vec3 ap = abs(p/scale);  
  float d = length(ap.xy);
  
  // Check if this is a line segment (z-axis)
  bool isLine = ap.z < d;
  float dz = ap.z;
  d = min(d, dz);
  
  float hh = 0.0;
  if (d == dz){
    // Very subtle high frequency influence
    hh += 0.5 + getHighIntensity() * 0.02;
  }
  
  // Use h to pass information about whether this is a line
  h = isLine ? 1.0 : hh;
  return d;
}

float df(vec2 p, out float h) {
  // Subtle overall intensity modulation
  float intensityMod = getOverallIntensity() * 0.03;
  const float fz = 1.0 - 0.0;
  float z = 1.75 * fz * (1.0 + intensityMod);
  
  p /= z;
  
  // Very subtle mid-frequency modulation
  float midMod = getMidIntensity() * 0.01;
  
  vec3 p3 = vec3(p, 0.1 + midMod);
  p3.xz *= g_rot0;
  p3.yz *= g_rot1;
  
  float isLine;
  float d = apolloian(p3, 1.0/fz, isLine);
  h = isLine;
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
  
  float isLine;
  float d = df(p, isLine);
  float ss = shadow(p, lightDir, 0.005, lightLen);
  
  // Subtle color modulation with audio, but different for line segments
  float bassHueShift = getBassIntensity() * 0.03;
  float highSatMod = getHighIntensity() * 0.05;
  
  vec3 bcol;
  if (isLine > 0.5) {
    // Special color handling for line segments - less audio reactivity
    bcol = hsv2rgb(vec3(
        fract(0.5 - 0.1 * length(p) + 0.25 * TIME), 
        0.7, 
        1.0
    ));
  } else {
    // Normal fractal parts - full audio reactivity
    bcol = hsv2rgb(vec3(
        fract(isLine - 0.2 * length(p) + 0.75 * TIME + bassHueShift), 
        0.666 + highSatMod, 
        1.0
    ));
  }
  
  vec3 col = vec3(0.0);
  
  // Subtle lighting intensity modulation
  float lightIntensity = 0.5 * (1.0 + getOverallIntensity() * 0.1);
  col += mix(0., 1.0, diff) * lightIntensity * mix(0.1, 1.0, ss)/(lightLen3 * lightLen3);
  
  // Subtle glow modulation with bass
  float glowIntensity = 300.0 + getBassIntensity() * 20.0;
  col += exp(-glowIntensity * abs(d)) * sqrt(bcol);
  
  // Subtle bloom  smodulation with high frequencies
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
"Wavey McStrings":`#define S smoothstep

vec4 Line(vec2 uv, float speed, float height, vec3 col) {
    // Get audio data - simple single sample per line
    float audio = texture(iChannel0, vec2(height * 0.05, 0.0)).x;
    
    // Adjust wave amplitude with audio
    float amp = 0.2 * (1.0 + audio);
    
    // Create the wave shape
    uv.y += S(1., 0.05, abs(uv.x)) * sin(iTime * 0.5 * speed + uv.x * height) * amp;
    
    // Create the line with original thickness
    return vec4(S(.006 * S(.002, .9, abs(uv.x)), 0., abs(uv.y) - .004) * col, 1.0) 
           * S(1., .3, abs(uv.x));
}

void mainImage(out vec4 O, in vec2 I) {
    vec2 uv = (I - .5 * iResolution.xy) / iResolution.y;
    
    // Overall audio intensity for global effect
    float audioSum = texture(iChannel0, vec2(0.1, 0.0)).x;
    
    O = vec4(0.);
    for (float i = 0.; i <= 5.; i += 1.) {
        float t = i / 5.;
        
        // Simple audio multiplier
        float audioMult = 0.50 + audioSum * 3.0;
        
        // Create each line - keeping close to the original
        O += Line(
            uv, 
            1. + t,                     // Original speed
            4. + t,                     // Original height 
            vec3(sin(audioSum) + t * .7, .2 + t * .4, 0.3) * audioMult  // Color enhanced by audio
        );
    }
}`,

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
    float T = iTime*0.55;
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
}

`,


"Trippy Cheese Rail":`// "Fractal Cartoon" with procedural cat replacement
// Modified from original "DE edge detection" by Kali

// There are no lights and no AO, only color by normals and dark edges.

//#define SHOWONLYEDGES
#define NYAN 
#define WAVES
#define BORDER

#define RAY_STEPS 150

#define BRIGHTNESS 1.2
#define GAMMA 1.4
#define SATURATION .65

#define detail .001
#define t iTime*.5

const vec3 origin=vec3(-1.,.7,0.);
float det=0.0;

// 2D rotation function
mat2 rot(float a) {
    return mat2(cos(a),sin(a),-sin(a),cos(a));    
}

// "Amazing Surface" fractal
vec4 formula(vec4 p) {
    p.xz = abs(p.xz+1.)-abs(p.xz-1.)-p.xz;
    p.y-=.25;
    p.xy*=rot(radians(35.));
    p=p*2./clamp(dot(p.xyz,p.xyz),.2,1.);
    return p;
}

// Distance function
float de(vec3 pos) {
#ifdef WAVES
    pos.y+=sin(pos.z-t*6.)*.15; //waves!
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

// Camera path
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

// Modified rainbow trail function - no need for external texture
vec4 rainbow(vec2 p)
{
    float q = max(p.x,-0.1);
    float s = sin(p.x*7.0+t*70.0)*0.08;
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
// Improved procedural cat function with better proportions
vec4 proceduralCat(vec2 p)
{
    // Calculate aspect ratio to make cat proportional to screen resolution
    float aspectRatio = iResolution.x/iResolution.y;
    float scale = min(1.0, aspectRatio); // Adjust scale based on aspect ratio
    
    // Simple cat shape made of circles and rectangles - use less scaling
    vec2 uv = p*vec2(0.9*scale, 0.9); // Better proportions
    
    // Cat animation - reduced for subtle effect
    float wiggle = sin(iTime*2.0)*0.05;
    
    // Cat head - better positioning
    float headSize = 0.2;
    float headOffset = 0.0; // Center the head
    
    // Cat body (head)
    float catBody = length(uv - vec2(headOffset, 0.0)) - headSize;
    
    // Cat ears - wider positioning and better size
    float earSize = 0.1;
    float earSpacing = 0.15;
    float catEar1 = length(uv - vec2(headOffset-earSpacing, 0.2 + wiggle)) - earSize;
    float catEar2 = length(uv - vec2(headOffset+earSpacing, 0.2 - wiggle)) - earSize;
    
    // Cat face features - better proportions
    float eyeSize = 0.04;
    float eyeSpacing = 0.1;
    float catEye1 = length(uv - vec2(headOffset-eyeSpacing, 0.05)) - eyeSize;
    float catEye2 = length(uv - vec2(headOffset+eyeSpacing, 0.05)) - eyeSize;
    float catNose = length(uv - vec2(headOffset, -0.05)) - eyeSize*0.8;
    
    // Whiskers - simple lines
    float whiskerThickness = 0.01;
    float leftWhisker = smoothstep(whiskerThickness, 0.0, abs(uv.y+0.05) - 
                       whiskerThickness * abs((uv.x+0.15)/0.2));
    float rightWhisker = smoothstep(whiskerThickness, 0.0, abs(uv.y+0.05) - 
                        whiskerThickness * abs((uv.x-0.15)/0.2));
    
    // Combine all elements
    float catShape = min(catBody, min(catEar1, catEar2));
    
    // Add the face features
    vec4 color = vec4(0.0);
    
    // Main cat color (pink)
    if (catShape < 0.0) {
        color = vec4(0.9, 0.5, 0.6, 0.9);
    }
    
    // Eyes (black)
    if (catEye1 < 0.0 || catEye2 < 0.0) {
        color = vec4(0.1, 0.1, 0.1, 1.0);
    }
    
    // Nose (red)
    if (catNose < 0.0) {
        color = vec4(0.9, 0.2, 0.2, 1.0);
    }
    
    // Whiskers (white)
    if (leftWhisker > 0.5 || rightWhisker > 0.5) {
        color = vec4(1.0, 1.0, 1.0, 0.8);
    }
    
    // Position constraint
    float edgeConstraint = 0.4;
    if (length(uv) > edgeConstraint) color.a = 0.0;
    
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
    float sunsize=9.-max(0.,texture(iChannel0,vec2(.6,.2)).x)*5.; // responsive sun size
    float an=atan(dir.x,dir.y)+iTime*1.5; // angle for drawing and rotating sun
    float s=pow(clamp(1.0-length(dir.xy)*sunsize-abs(.2-mod(an,.4)),0.,1.),.1); // sun
    float sb=pow(clamp(1.0-length(dir.xy)*(sunsize-.2)-abs(.2-mod(an,.4)),0.,1.),.1); // sun border
    float sg=pow(clamp(1.0-length(dir.xy)*(sunsize-3.5)-.5*abs(.2-mod(an,.4)),0.,1.),3.); // sun rays
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
    dir.yx*=rot(dir.x);
    vec2 ncatpos=(dir.xy+vec2(-3.+mod(-t,6.),-.27));
    vec4 ncat=proceduralCat(ncatpos*5.); // Use our procedural cat instead
    vec4 rain=rainbow(ncatpos*10.+vec2(.8,.5));
    if (totdist>8.) col=mix(col,max(vec3(.2),rain.xyz),rain.a*.9);
    if (totdist>8.) col=mix(col,max(vec3(.2),ncat.xyz),ncat.a*.9);
#endif
#endif
    return col;
}

// get camera position
vec3 move(inout vec3 dir) {
    vec3 go=path(t);
    vec3 adv=path(t+.7);
    float hd=de(adv);
    vec3 advec=normalize(adv-go);
    float an=adv.x-go.x; an*=min(1.,abs(adv.z-go.z))*sign(adv.z-go.z)*.7;
    dir.xy*=mat2(cos(an),sin(an),-sin(an),cos(an));
    an=advec.y*1.7;
    dir.yz*=mat2(cos(an),sin(an),-sin(an),cos(an));
    an=atan(advec.x,advec.z);
    dir.xz*=mat2(cos(an),sin(an),-sin(an),cos(an));
    return go;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
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
"Trippy Cheese Rail 2": `// "Fractal Land & Clouds" - by Lynxx

// based on "Fractal Cartoon" - former "DE edge detection" by Kali https://www.shadertoy.com/view/XsBXWt
// and "2D Clouds" by drift: https://www.shadertoy.com/view/4tdSWr
// There are no lights and no AO, only color by normals and dark edges.
// update: Nyan Cat cameo, thanks to code from mu6k: https://www.shadertoy.com/view/4dXGWH

//#define SHOWONLYEDGES
#define NYAN 
#define WAVES
#define BORDER

#define RAY_STEPS 150

#define BRIGHTNESS 1.2
#define GAMMA 1.4
#define SATURATION .65

#define detail .001
#define t iTime*.5

const float cloudscale = 1.1;
const float speed = 0.04;
const float clouddark = 0.4;
const float cloudlight = 0.2;
const float cloudcover = 0.1;
const float cloudalpha = 5.0;
const float skytint = 0.4;
const vec3 skycolour1 = vec3(0.2, 0.4, 0.6);
const vec3 skycolour2 = vec3(0.4, 0.6, 0.9);
const mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );

vec2 hash( vec2 p ) {
	p = vec2(dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise( in vec2 p ) {
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;
	vec2 i = floor(p + (p.x+p.y)*K1);	
    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0); //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
    vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;
    vec3 h = max(0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
	vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot(n, vec3(70.0));	
}

float fbm(vec2 n) {
	float total = 0.0, amplitude = 0.1;
	for (int i = 0; i < 7; i++) {
		total += noise(n) * amplitude;
		n = m * n;
		amplitude *= 0.4;
	}
	return total;
}

const vec3 origin=vec3(-1.,.7,0.);
float det=0.0;

// 2D rotation function
mat2 rot(float a) {
	return mat2(cos(a),sin(a),-sin(a),cos(a));	
}

// "Amazing Surface" fractal
vec4 formula(vec4 p) {
    p.xz = abs(p.xz+1.)-abs(p.xz-1.)-p.xz;
    p.y-=.25;
    p.xy*=rot(radians(35.));
    p=p*2.0/clamp(dot(p.xyz,p.xyz),.2,1.);
	return p;
}

// Distance function
float de(vec3 pos) {
#ifdef WAVES
	pos.y+=cos(pos.z-t*6.)*.25; //waves!
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

// Camera path
vec3 path(float ti) {
	ti*=1.5;
	vec3  p=vec3(sin(ti),(1.-sin(ti*2.))*.5,-ti*5.)*.5;
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
	return normalize(vec3(d5-d6,d1-d2,d3-d4));
}

// Used Nyan Cat code by mu6k, with some mods
vec4 rainbow(vec2 p) {
	float q = max(p.x,-0.1);
	float s = sin(p.x*7.0+t*70.0)*0.08;
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

vec4 nyan(vec2 p) {
	vec2 uv = p*vec2(0.4,1.0);
	float ns=3.0;
	float nt = iTime*ns; nt-=mod(nt,240.0/256.0/6.0); nt = mod(nt,240.0/256.0);
	float ny = mod(iTime*ns,1.0); ny-=mod(ny,0.75); ny*=-0.05;
	vec4 color = texture(iChannel1,vec2(uv.x/3.0+210.0/256.0-nt+0.05,.5-uv.y-ny));
	if (uv.x<-0.3) color.a = 0.0;
	if (uv.x>0.2) color.a=0.0;
	return color;
}

vec3 clouds(in vec3 from, in vec3 dir) {
    vec2 p = (from*dir).xy;
    p.y+=(dir.y*3.);
    p.x/=from.x;
    
	vec2 uv = p*vec2(iResolution.x/iResolution.y,1.0);    
    float time = iTime * speed;
    float q = fbm(uv * cloudscale * 0.5);
    
    //ridged noise shape
	float r = 0.0;
	uv *= cloudscale;
    uv -= q - time;
    float weight = 0.8;
    for (int i=0; i<8; i++){
		r += abs(weight*noise( uv ));
        uv = m*uv + time;
		weight *= 0.7;
    }
    
    //noise shape
	float f = 0.0;
    uv = p*vec2(iResolution.x/iResolution.y,1.0);
	uv *= cloudscale;
    uv -= q - time;
    weight = 0.7;
    for (int i=0; i<8; i++){
		f += weight*noise( uv );
        uv = m*uv + time;
		weight *= 0.6;
    }
    
    f *= r + f;
    
    //noise colour
    float c = 0.0;
    time = iTime * speed * 2.0;
    uv = p*vec2(iResolution.x/iResolution.y,1.0);
	uv *= cloudscale*2.0;
    uv -= q - time;
    weight = 0.4;
    for (int i=0; i<7; i++){
		c += weight*noise( uv );
        uv = m*uv + time;
		weight *= 0.6;
    }
    
    //noise ridge colour
    float c1 = 0.0;
    time = iTime * speed * 3.0;
    uv = p*vec2(iResolution.x/iResolution.y,1.0);
	uv *= cloudscale*3.0;
    uv -= q - time;
    weight = 0.4;
    for (int i=0; i<7; i++){
		c1 += abs(weight*noise( uv ));
        uv = m*uv + time;
		weight *= 0.6;
    }
	
    c += c1;
    
    vec3 skycolour = mix(skycolour2, skycolour1, p.y);
    vec3 cloudcolour = vec3(1.1, 1.1, 0.9) * clamp((clouddark + cloudlight*c), 0.0, 1.0);
    f = cloudcover + cloudalpha*f*r;
	return mix(skycolour, clamp(skytint * skycolour + cloudcolour, 0.0, 1.0), clamp(f + c, 0.0, 1.0));;
}

// Raymarching and 2D graphics
vec3 raymarch(in vec3 from, in vec3 dir) {
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
        else
            break;
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
	float an=atan(dir.x,dir.y)+iTime*1.1; // angle for drawing and rotating sun
	float s=pow(clamp(0.9-length(dir.xy)*sunsize-abs(.2-mod(an,.4)),0.,1.),.1); // sun
	float sb=pow(clamp(1.-length(dir.xy)*(sunsize-.2)-abs(.2-mod(an,.4)),0.,1.),.1); // sun border
	float sg=pow(clamp(1.2-length(dir.xy)*(sunsize-4.5)-.5*abs(.2-mod(an,.4)),0.,1.),3.); // sun rays
	
	// set up background with clouds and sun
    vec3 backg = clouds(from,dir);
		 backg=max(vec3(.84,.9,.1)*s, backg);
		 backg=max(backg,sg*vec3(.84,.9,.1));
	
	col=mix(vec3(1.,.9,.3),col,exp(-.004*totdist*totdist));// distant fading to sun color
	if (totdist>25.) col=backg; // hit background
	col=pow(col,vec3(GAMMA))*BRIGHTNESS;
	col=mix(vec3(length(col)),col,SATURATION);
#ifdef SHOWONLYEDGES
	col=1.-vec3(length(col));
#else
	col*=vec3(1.,.9,.85);
#ifdef NYAN
    if (iTime<180.) {
        dir.yx*=rot(dir.x);
        dir.yx/=(1.+(iTime/180.));
        vec2 ncatpos=(dir.xy+vec2(-3.+mod(-t,6.),-.27));
        vec4 ncat=nyan(ncatpos*5.);
        vec4 rain=rainbow(ncatpos*10.+vec2(.8,.5));
        if (totdist>8.) col=mix(col,max(vec3(.2),rain.xyz),rain.a*.9);
        if (totdist>8.) col=mix(col,max(vec3(.2),ncat.xyz),ncat.a*.9);
    }
#endif
#endif
	return col;
}

// get camera position
vec3 move(inout vec3 dir) {
	vec3 go=path(t);
	vec3 adv=path(t+.7);
	float hd=de(adv);
	vec3 advec=normalize(adv-go);
	float an=adv.x-go.x; an*=min(1.,abs(adv.z-go.z))*sign(adv.z-go.z)*.7;
	dir.xy*=mat2(cos(an),sin(an),-sin(an),cos(an));
    an=advec.y*1.7;
	dir.yz*=mat2(cos(an),sin(an),-sin(an),cos(an));
	an=atan(advec.x,advec.z);
	dir.xz*=mat2(cos(an),sin(an),-sin(an),cos(an));
	return go;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
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
	color=mix(vec3(0.),color,pow(max(0.,.95-length(oriuv*oriuv*oriuv*vec2(.95,.95))),.4));
	#endif
	fragColor = vec4(color,1.);
}`,
"Jellyfish Migration": `// Luminescence by Martijn Steinrucken aka BigWings - 2017
// Email:countfrolic@gmail.com Twitter:@The_ArtOfCode
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

// My entry for the monthly challenge (May 2017) on r/proceduralgeneration 
// Use the mouse to look around. Uncomment the SINGLE define to see one specimen by itself.
// Code is a bit of a mess, too lazy to clean up. Hope you like it!

// Music by Klaus Lunde
// https://soundcloud.com/klauslunde/zebra-tribute

// YouTube: The Art of Code -> https://www.youtube.com/channel/UCcAlTqd9zID6aNX3TzwxJXg
// Twitter: @The_ArtOfCode

#define INVERTMOUSE -1.

#define MAX_STEPS 100.
#define VOLUME_STEPS 8.
//#define SINGLE
#define MIN_DISTANCE 0.1
#define MAX_DISTANCE 100.
#define HIT_DISTANCE .01

#define S(x,y,z) smoothstep(x,y,z)
#define B(x,y,z,w) S(x-z, x+z, w)*S(y+z, y-z, w)
#define sat(x) clamp(x,0.,1.)
#define SIN(x) sin(x)*.5+.5

const vec3 lf=vec3(1., 0., 0.);
const vec3 up=vec3(0., 1., 0.);
const vec3 fw=vec3(0., 0., 1.);

const float halfpi = 1.570796326794896619;
const float pi = 3.141592653589793238;
const float twopi = 6.283185307179586;


vec3 accentColor1 = vec3(1., .1, .5);
vec3 secondColor1 = vec3(.1, .5, 1.);

vec3 accentColor2 = vec3(1., .5, .1);
vec3 secondColor2 = vec3(.1, .5, .6);

vec3 bg;	 	// global background color
vec3 accent;	// color of the phosphorecence

float N1( float x ) { return fract(sin(x)*5346.1764); }
float N2(float x, float y) { return N1(x + y*23414.324); }

float N3(vec3 p) {
    p  = fract( p*0.3183099+.1 );
	p *= 17.0;
    return fract( p.x*p.y*p.z*(p.x+p.y+p.z) );
}

struct ray {
    vec3 o;
    vec3 d;
};

struct camera {
    vec3 p;			// the position of the camera
    vec3 forward;	// the camera forward vector
    vec3 left;		// the camera left vector
    vec3 up;		// the camera up vector
	
    vec3 center;	// the center of the screen, in world coords
    vec3 i;			// where the current ray intersects the screen, in world coords
    ray ray;		// the current ray: from cam pos, through current uv projected on screen
    vec3 lookAt;	// the lookat point
    float zoom;		// the zoom factor
};

struct de {
    // data type used to pass the various bits of information used to shade a de object
	float d;	// final distance to field
    float m; 	// material
    vec3 uv;
    float pump;
    
    vec3 id;
    vec3 pos;		// the world-space coordinate of the fragment
};
    
struct rc {
    // data type used to handle a repeated coordinate
	vec3 id;	// holds the floor'ed coordinate of each cell. Used to identify the cell.
    vec3 h;		// half of the size of the cell
    vec3 p;		// the repeated coordinate
    //vec3 c;		// the center of the cell, world coordinates
};
    
rc Repeat(vec3 pos, vec3 size) {
	rc o;
    o.h = size*.5;					
    o.id = floor(pos/size);			// used to give a unique id to each cell
    o.p = mod(pos, size)-o.h;
    //o.c = o.id*size+o.h;
    
    return o;
}
    
camera cam;


void CameraSetup(vec2 uv, vec3 position, vec3 lookAt, float zoom) {
	
    cam.p = position;
    cam.lookAt = lookAt;
    cam.forward = normalize(cam.lookAt-cam.p);
    cam.left = cross(up, cam.forward);
    cam.up = cross(cam.forward, cam.left);
    cam.zoom = zoom;
    
    cam.center = cam.p+cam.forward*cam.zoom;
    cam.i = cam.center+cam.left*uv.x+cam.up*uv.y;
    
    cam.ray.o = cam.p;						// ray origin = camera position
    cam.ray.d = normalize(cam.i-cam.p);	// ray direction is the vector from the cam pos through the point on the imaginary screen
}


// ============== Functions I borrowed ;)

//  3 out, 1 in... DAVE HOSKINS
vec3 N31(float p) {
   vec3 p3 = fract(vec3(p) * vec3(.1031,.11369,.13787));
   p3 += dot(p3, p3.yzx + 19.19);
   return fract(vec3((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y, (p3.y+p3.z)*p3.x));
}

// DE functions from IQ
float smin( float a, float b, float k )
{
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float smax( float a, float b, float k )
{
	float h = clamp( 0.5 + 0.5*(b-a)/k, 0.0, 1.0 );
	return mix( a, b, h ) + k*h*(1.0-h);
}

float sdSphere( vec3 p, vec3 pos, float s ) { return (length(p-pos)-s); }

// From http://mercury.sexy/hg_sdf
vec2 pModPolar(inout vec2 p, float repetitions, float fix) {
	float angle = twopi/repetitions;
	float a = atan(p.y, p.x) + angle/2.;
	float r = length(p);
	float c = floor(a/angle);
	a = mod(a,angle) - (angle/2.)*fix;
	p = vec2(cos(a), sin(a))*r;

	return p;
}
    
// -------------------------


float Dist( vec2 P,  vec2 P0, vec2 P1 ) {
    //2d point-line distance
    
	vec2 v = P1 - P0;
    vec2 w = P - P0;

    float c1 = dot(w, v);
    float c2 = dot(v, v);
    
    if (c1 <= 0. )  // before P0
    	return length(P-P0);
    
    float b = c1 / c2;
    vec2 Pb = P0 + b*v;
    return length(P-Pb);
}

vec3 ClosestPoint(vec3 ro, vec3 rd, vec3 p) {
    // returns the closest point on ray r to point p
    return ro + max(0., dot(p-ro, rd))*rd;
}

vec2 RayRayTs(vec3 ro1, vec3 rd1, vec3 ro2, vec3 rd2) {
	// returns the two t's for the closest point between two rays
    // ro+rd*t1 = ro2+rd2*t2
    
    vec3 dO = ro2-ro1;
    vec3 cD = cross(rd1, rd2);
    float v = dot(cD, cD);
    
    float t1 = dot(cross(dO, rd2), cD)/v;
    float t2 = dot(cross(dO, rd1), cD)/v;
    return vec2(t1, t2);
}

float DistRaySegment(vec3 ro, vec3 rd, vec3 p1, vec3 p2) {
	// returns the distance from ray r to line segment p1-p2
    vec3 rd2 = p2-p1;
    vec2 t = RayRayTs(ro, rd, p1, rd2);
    
    t.x = max(t.x, 0.);
    t.y = clamp(t.y, 0., length(rd2));
                
    vec3 rp = ro+rd*t.x;
    vec3 sp = p1+rd2*t.y;
    
    return length(rp-sp);
}

vec2 sph(vec3 ro, vec3 rd, vec3 pos, float radius) {
	// does a ray sphere intersection
    // returns a vec2 with distance to both intersections
    // if both a and b are MAX_DISTANCE then there is no intersection
    
    vec3 oc = pos - ro;
    float l = dot(rd, oc);
    float det = l*l - dot(oc, oc) + radius*radius;
    if (det < 0.0) return vec2(MAX_DISTANCE);
    
    float d = sqrt(det);
    float a = l - d;
    float b = l + d;
    
    return vec2(a, b);
}


vec3 background(vec3 r) {
	
    float x = atan(r.x, r.z);		// from -pi to pi	
	float y = pi*0.5-acos(r.y);  		// from -1/2pi to 1/2pi		
    
    vec3 col = bg*(1.+y);
    
	float t = iTime;				// add god rays
    
    float a = sin(r.x);
    
    float beam = sat(sin(10.*x+a*y*5.+t));
    beam *= sat(sin(7.*x+a*y*3.5-t));
    
    float beam2 = sat(sin(42.*x+a*y*21.-t));
    beam2 *= sat(sin(34.*x+a*y*17.+t));
    
    beam += beam2;
    col *= 1.+beam*.05;

    return col;
}




float remap(float a, float b, float c, float d, float t) {
	return ((t-a)/(b-a))*(d-c)+c;
}



de map( vec3 p, vec3 id ) {

    float t = iTime*2.;
    
    float N = N3(id);
    
    de o;
    o.m = 0.;
    
    float x = (p.y+N*twopi)*1.+t;
    float r = 1.;
    
    float pump = cos(x+cos(x))+sin(2.*x)*.2+sin(4.*x)*.02;
    
    x = t + N*twopi;
    p.y -= (cos(x+cos(x))+sin(2.*x)*.2)*.6;
    p.xz *= 1. + pump*.2;
    
    float d1 = sdSphere(p, vec3(0., 0., 0.), r);
    float d2 = sdSphere(p, vec3(0., -.5, 0.), r);
    
    o.d = smax(d1, -d2, .1);
    o.m = 1.;
    
    if(p.y<.5) {
        float sway = sin(t+p.y+N*twopi)*S(.5, -3., p.y)*N*.3;
        p.x += sway*N;	// add some sway to the tentacles
        p.z += sway*(1.-N);
        
        vec3 mp = p;
    	mp.xz = pModPolar(mp.xz, 6., 0.);
        
        float d3 = length(mp.xz-vec2(.2, .1))-remap(.5, -3.5, .1, .01, mp.y);
    	if(d3<o.d) o.m=2.;
        d3 += (sin(mp.y*10.)+sin(mp.y*23.))*.03;
        
        float d32 = length(mp.xz-vec2(.2, .1))-remap(.5, -3.5, .1, .04, mp.y)*.5;
        d3 = min(d3, d32);
        o.d = smin(o.d, d3, .5);
        
        if( p.y<.2) {
             vec3 op = p;
    		op.xz = pModPolar(op.xz, 13., 1.);
            
        	float d4 = length(op.xz-vec2(.85, .0))-remap(.5, -3., .04, .0, op.y);
    		if(d4<o.d) o.m=3.;
            o.d = smin(o.d, d4, .15);
        }
    }    
    o.pump = pump;
    o.uv = p;
    
    o.d *= .8;
    return o;
}

vec3 calcNormal( de o ) {
	vec3 eps = vec3( 0.01, 0.0, 0.0 );
	vec3 nor = vec3(
	    map(o.pos+eps.xyy, o.id).d - map(o.pos-eps.xyy, o.id).d,
	    map(o.pos+eps.yxy, o.id).d - map(o.pos-eps.yxy, o.id).d,
	    map(o.pos+eps.yyx, o.id).d - map(o.pos-eps.yyx, o.id).d );
	return normalize(nor);
}

de CastRay(ray r) {
    float d = 0.;
    float dS = MAX_DISTANCE;
    
    vec3 pos = vec3(0., 0., 0.);
    vec3 n = vec3(0.);
    de o, s;
    
    float dC = MAX_DISTANCE;
    vec3 p;
    rc q;
    float t = iTime;
    vec3 grid = vec3(6., 30., 6.);
        
    for(float i=0.; i<MAX_STEPS; i++) {
        p = r.o + r.d*d;
        
        #ifdef SINGLE
        s = map(p, vec3(0.));
        #else
        p.y -= t;  // make the move up
        p.x += t;  // make cam fly forward
            
        q = Repeat(p, grid);
    	
        vec3 rC = ((2.*step(0., r.d)-1.)*q.h-q.p)/r.d;	// ray to cell boundary
        dC = min(min(rC.x, rC.y), rC.z)+.01;		// distance to cell just past boundary
        
        float N = N3(q.id);
        q.p += (N31(N)-.5)*grid*vec3(.5, .7, .5);
        
		if(Dist(q.p.xz, r.d.xz, vec2(0.))<1.1)
        //if(DistRaySegment(q.p, r.d, vec3(0., -6., 0.), vec3(0., -3.3, 0)) <1.1) 
        	s = map(q.p, q.id);
        else
            s.d = dC;
        
        
        #endif
           
        if(s.d<HIT_DISTANCE || d>MAX_DISTANCE) break;
        d+=min(s.d, dC);	// move to distance to next cell or surface, whichever is closest
    }
    
    if(s.d<HIT_DISTANCE) {
        o.m = s.m;
        o.d = d;
        o.id = q.id;
        o.uv = s.uv;
        o.pump = s.pump;
        
        #ifdef SINGLE
        o.pos = p;
        #else
        o.pos = q.p;
        #endif
    }
    
    return o;
}

float VolTex(vec3 uv, vec3 p, float scale, float pump) {
    // uv = the surface pos
    // p = the volume shell pos
    
	p.y *= scale;
    
    float s2 = 5.*p.x/twopi;
    float id = floor(s2);
    s2 = fract(s2);
    vec2 ep = vec2(s2-.5, p.y-.6);
    float ed = length(ep);
    float e = B(.35, .45, .05, ed);
    
   	float s = SIN(s2*twopi*15. );
	s = s*s; s = s*s;
    s *= S(1.4, -.3, uv.y-cos(s2*twopi)*.2+.3)*S(-.6, -.3, uv.y);
    
    float t = iTime*5.;
    float mask = SIN(p.x*twopi*2. + t);
    s *= mask*mask*2.;
    
    return s+e*pump*2.;
}

vec4 JellyTex(vec3 p) { 
    vec3 s = vec3(atan(p.x, p.z), length(p.xz), p.y);
    
    float b = .75+sin(s.x*6.)*.25;
    b = mix(1., b, s.y*s.y);
    
    p.x += sin(s.z*10.)*.1;
    float b2 = cos(s.x*26.) - s.z-.7;
   
    b2 = S(.1, .6, b2);
    return vec4(b+b2);
}

vec3 render( vec2 uv, ray camRay, float depth ) {
    // outputs a color
    
    bg = background(cam.ray.d);
    
    vec3 col = bg;
    de o = CastRay(camRay);
    
    float t = iTime;
    vec3 L = up;
    

    if(o.m>0.) {
        vec3 n = calcNormal(o);
        float lambert = sat(dot(n, L));
        vec3 R = reflect(camRay.d, n);
        float fresnel = sat(1.+dot(camRay.d, n));
        float trans = (1.-fresnel)*.5;
        vec3 ref = background(R);
        float fade = 0.;
        
        if(o.m==1.) {	// hood color
            float density = 0.;
            for(float i=0.; i<VOLUME_STEPS; i++) {
                float sd = sph(o.uv, camRay.d, vec3(0.), .8+i*.015).x;
                if(sd!=MAX_DISTANCE) {
                    vec2 intersect = o.uv.xz+camRay.d.xz*sd;

                    vec3 uv = vec3(atan(intersect.x, intersect.y), length(intersect.xy), o.uv.z);
                    density += VolTex(o.uv, uv, 1.4+i*.03, o.pump);
                }
            }
            vec4 volTex = vec4(accent, density/VOLUME_STEPS); 
            
            
            vec3 dif = JellyTex(o.uv).rgb;
            dif *= max(.2, lambert);

            col = mix(col, volTex.rgb, volTex.a);
            col = mix(col, vec3(dif), .25);

            col += fresnel*ref*sat(dot(up, n));

            //fade
            fade = max(fade, S(.0, 1., fresnel));
        } else if(o.m==2.) {						// inside tentacles
            vec3 dif = accent;
    		col = mix(bg, dif, fresnel);
            
            col *= mix(.6, 1., S(0., -1.5, o.uv.y));
            
            float prop = o.pump+.25;
            prop *= prop*prop;
            col += pow(1.-fresnel, 20.)*dif*prop;
            
            
            fade = fresnel;
        } else if(o.m==3.) {						// outside tentacles
        	vec3 dif = accent;
            float d = S(100., 13., o.d);
    		col = mix(bg, dif, pow(1.-fresnel, 5.)*d);
        }
        
        fade = max(fade, S(0., 100., o.d));
        col = mix(col, bg, fade);
        
        if(o.m==4.)
            col = vec3(1., 0., 0.);
    } 
     else
        col = bg;
    
    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	float t = iTime*.04;
    
    vec2 uv = (fragCoord.xy / iResolution.xy);
    uv -= .5;
    uv.y *= iResolution.y/iResolution.x; 
    
    vec2 m = iMouse.xy/iResolution.xy;
    
    if(m.x<0.05 || m.x>.95) {				// move cam automatically when mouse is not used
    	m = vec2(t*.5, SIN(t*pi)*.75+.75);
    }
	
    accent = mix(accentColor1, accentColor2, SIN(t*15.456));
    bg = mix(secondColor1, secondColor2, SIN(t*7.345231));
    
    float turn = (.1-m.x)*twopi;
    float s = sin(turn);
    float c = cos(turn);
    mat3 rotX = mat3(c,  0., s, 0., 1., 0., s,  0., -c);
    
    #ifdef SINGLE
    float camDist = -10.;
    #else
    float camDist = -.1;
    #endif
    
    vec3 lookAt = vec3(0., -1., 0.);
    
    vec3 camPos = vec3(0., INVERTMOUSE*camDist*cos((m.y)*pi), camDist)*rotX;
   	
    CameraSetup(uv, camPos+lookAt, lookAt, 1.);
    
    vec3 col = render(uv, cam.ray, 0.);
    
    col = pow(col, vec3(mix(1.5, 2.6, SIN(t+pi))));		// post-processing
    float d = 1.-dot(uv, uv);		// vignette
    col *= (d*d*d)+.1;
    
    fragColor = vec4(col, 1.);
}`,
"3D Play": `/*
    Abstract Corridor with Procedural Textures - Collapsing Tunnel with Rocks
    -----------------------------------------------------------------------

    Modified version with original wall textures and falling rocks on strong beats.
*/

#define PI 3.1415926535898
#define FH 1.0 // Floor height.

// Grey scale.
float getGrey(vec3 p){ return p.x*0.299 + p.y*0.587 + p.z*0.114; }

// Non-standard vec3-to-vec3 hash function.
vec3 hash33(vec3 p){
    float n = sin(dot(p, vec3(7, 157, 113)));
    return fract(vec3(2097152, 262144, 32768)*n);
}

// Hash function for procedural textures
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// 2x2 matrix rotation.
mat2 rot2(float a){
    float c = cos(a); float s = sin(a);
    return mat2(c, s, -s, c);
}

// Value noise for procedural textures
float valueNoise(in vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    // Smoothstep interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);

    // Four corners
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal Brownian Motion for richer textures
float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    // 5 octaves of noise
    for (int i = 0; i < 5; i++) {
        value += amplitude * valueNoise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

// 3D noise for volume texturing
float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);

    f = f * f * (3.0 - 2.0 * f);

    vec2 uv = (i.xy + vec2(37.0, 17.0) * i.z) + f.xy;
    vec2 rg = mix(
        vec2(hash21(uv), hash21(uv + vec2(37.0, 17.0))),
        vec2(hash21(uv + vec2(1.0, 0.0)), hash21(uv + vec2(1.0, 0.0) + vec2(37.0, 17.0))),
        f.x);

    return mix(rg.x, rg.y, f.z);
}

// Wall texture (original)
vec3 wallTexture(vec2 p) {
    float scale = 4.0;
    p *= scale;
    float n1 = fbm(p);
    float n2 = fbm(p * 2.0 + vec2(8.76, 3.42));
    float cracks = smoothstep(0.3, 0.4, fbm(p * 2.5));
    vec3 baseColor = mix(vec3(0.3, 0.3, 0.4), vec3(0.2, 0.1, 0.3), n1);
    vec3 accentColor = mix(vec3(0.5, 0.4, 0.6), vec3(0.6, 0.5, 0.7), n2);
    vec3 col = baseColor * (0.8 + 0.2 * n2);
    col = mix(col, accentColor, cracks * 0.7);
    col *= 0.8 + 0.4 * n1;
    return col;
}

// Floor texture (original)
vec3 floorTexture(vec2 p) {
    float scale = 8.0;
    p *= scale;
    vec2 grid = abs(fract(p) - 0.5);
    float line = smoothstep(0.05, 0.0, min(grid.x, grid.y));
    float n = fbm(p * 0.5) * 0.5 + 0.5;
    vec3 col = mix(vec3(0.2), vec3(0.15), n);
    col = mix(col, vec3(0.1), line * 0.8);
    col *= 0.7 + 0.3 * n;
    return col;
}

// Tri-Planar procedural texturing
vec3 proceduralTex3D(in vec3 p, in vec3 n, bool isFloor) {
    n = max((abs(n) - 0.2) * 7.0, 0.001);
    n /= (n.x + n.y + n.z);

    vec3 texX, texY, texZ;

    if (isFloor) {
        texX = floorTexture(p.yz);
        texY = floorTexture(p.zx);
        texZ = floorTexture(p.xy);
    } else {
        texX = wallTexture(p.yz);
        texY = wallTexture(p.zx);
        texZ = wallTexture(p.xy);
    }

    return texX * n.x + texY * n.y + texZ * n.z;
}

// The triangle function.
vec3 tri(in vec3 x){return abs(x-floor(x)-.5);}

// The function used to perturb the walls of the cavern (slight increase)
float surfFunc(in vec3 p){
    return dot(tri(p*0.5 + tri(p*0.25).yzx), vec3(0.666)) * 1.1;
}

// The path is a 2D sinusoid that varies over time
vec2 path(in float z){ float s = sin(z/24.)*cos(z/12.); return vec2(s*12., 0.); }

// Standard tunnel distance function (slight increase in base perturbation)
float map(vec3 p){
    float sf = surfFunc(p - vec3(0, cos(p.z/3.)*.15, 0));
    vec2 tun = abs(p.xy - path(p.z))*vec2(0.5, 0.7071);
    float n = 1. - max(tun.x, tun.y) + (0.55 - sf);
    return min(n, p.y + FH);
}

// Bump mapping
vec3 doBumpMap(in vec3 p, in vec3 nor, float bumpfactor, bool isFloor){
    const float eps = 0.001;
    float ref = getGrey(proceduralTex3D(p, nor, isFloor));
    vec3 grad = vec3(
        getGrey(proceduralTex3D(vec3(p.x - eps, p.y, p.z), nor, isFloor)) - ref,
        getGrey(proceduralTex3D(vec3(p.x, p.y - eps, p.z), nor, isFloor)) - ref,
        getGrey(proceduralTex3D(vec3(p.x, p.y, p.z - eps), nor, isFloor)) - ref
    ) / eps;

    grad -= nor * dot(nor, grad);
    return normalize(nor + grad * bumpfactor);
}

// Surface normal.
vec3 getNormal(in vec3 p) {
    const float eps = 0.001;
    return normalize(vec3(
        map(vec3(p.x + eps, p.y, p.z)) - map(vec3(p.x - eps, p.y, p.z)),
        map(vec3(p.x, p.y + eps, p.z)) - map(vec3(p.x, p.y - eps, p.z)),
        map(vec3(p.x, p.y, p.z + eps)) - map(vec3(p.x, p.y, p.z - eps))
    ));
}

// Based on original by IQ.
float calculateAO(vec3 p, vec3 n){
    const float AO_SAMPLES = 5.0;
    float r = 0.0, w = 1.0, d;

    for (float i = 1.0; i<AO_SAMPLES + 1.1; i++){
        d = i/AO_SAMPLES;
        r += w*(d - map(p + n*d));
        w *= 0.5;
    }

    return 1.0 - clamp(r, 0.0, 1.0);
}

// Cool curve function, by Shadertoy user, Nimitz.
float curve(in vec3 p, in float w){
    vec2 e = vec2(-1., 1.)*w;

    float t1 = map(p + e.yxx), t2 = map(p + e.xxy);
    float t3 = map(p + e.xyx), t4 = map(p + e.yyy);

    return 0.125/(w*w) *(t1 + t2 + t3 + t4 - 4.*map(p));
}

// Audio reaction functions
float getAudioLevel(float freq) {
    return texture(iChannel0, vec2(freq, 0.0)).x;
}

// Function to generate a simple rock shape
float rockShape(vec3 p) {
    float s = 1.0;
    s *= 1.0 - smoothstep(0.4, 0.6, length(p + vec3(0.1, 0.1, 0.1) * sin(iTime * 5.0)));
    s *= 1.0 - smoothstep(0.4, 0.6, length(p + vec3(-0.1, -0.2, 0.0) * cos(iTime * 7.0)));
    s *= 1.0 - 0.8 * noise3D(p * 3.0);
    return s;
}

// Distance function for a falling rock
float fallingRock(vec3 p, float beat, float fallTime) {
    if (beat > 0.6) { // Only show rocks on strong beats
        vec3 rockPos = vec3(sin(fallTime * 2.0) * 2.0, 2.0 - fallTime * 3.0, iTime * 5.0 + cos(fallTime * 1.5) * 2.0);
        return rockShape(p - rockPos) * 0.3; // Scale down the rock
    }
    return -100.0; // Return a large negative value to ensure it's not the closest surface
}

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    // Screen coordinates.
    vec2 uv = (fragCoord - iResolution.xy*0.5)/iResolution.y;

    // Get audio levels
    float bassLevel = getAudioLevel(0.05);

    // Stronger beat detection
      float beatThreshold = 0.6;
    float beat = smoothstep(beatThreshold, beatThreshold + 0.1, bassLevel);


    // Camera setup with more pronounced shaking on the beat
    vec3 camPos = vec3(0.0, 0.0, iTime * 5.0);
    float shakeAmplitude = 0.08;
    camPos.x += shakeAmplitude * beat * sin(iTime * 30.0);
    camPos.y += shakeAmplitude * beat * cos(iTime * 25.0);
    vec3 lookAt = camPos + vec3(0.0, 0.1, 0.5);

    // Light positioning
    vec3 light_pos = camPos + vec3(0.0, 0.125, -0.125);
    vec3 light_pos2 = camPos + vec3(0.0, 0.0, 6.0);

    lookAt.xy += path(lookAt.z);
    camPos.xy += path(camPos.z);
    light_pos.xy += path(light_pos.z);
    light_pos2.xy += path(light_pos2.z);

    // Ray direction.
    float FOV = PI/3.;
    vec3 forward = normalize(lookAt-camPos);
    vec3 right = normalize(vec3(forward.z, 0., -forward.x));
    vec3 up = cross(forward, right);
    vec3 rd = normalize(forward + FOV*uv.x*right + FOV*uv.y*up);

    // Camera rotation with slight beat-induced wobble
    float wobbleIntensity = 0.008;
    rd.xy = rot2(path(lookAt.z).x/32. + beat * wobbleIntensity * sin(iTime * 40.0)) * rd.xy;

    // Ray marching
    float t = 0.0, dt;
    float rockFallTime = 0.0;
    bool hitRock = false;
    for(int i=0; i<128; i++){
        vec3 currentPos = camPos + rd * t;
        float rockDist = fallingRock(currentPos, beat, rockFallTime);
        float tunnelDist = map(currentPos);

        if (rockDist > 0.0) {
            dt = min(tunnelDist, rockDist);
            if(dt == rockDist){
                hitRock = true;
            }
        } else {
            dt = tunnelDist;
        }

        if(dt<0.005 || t>150.){ break; }
        t += dt*0.75;
        if(hitRock){
           rockFallTime += 0.1;
        }
    }

    vec3 sceneCol = vec3(0.);

    if(dt<0.005){
        vec3 sp = t * rd+camPos;
        vec3 sn = getNormal(sp);

        const float tSize0 = 1.0;
        const float tSize1 = 1.0 / 4.0;

        bool isFloor = (sp.y < -(FH-0.005));

         if (isFloor) {
            sn = doBumpMap(sp * tSize1, sn, 0.025, true);
        } else {
            sn = doBumpMap(sp * tSize0, sn, 0.025, false);
        }

        float ao = calculateAO(sp, sn);

        vec3 ld = light_pos-sp;
        vec3 ld2 = light_pos2-sp;
        float distlpsp = max(length(ld), 0.001);
        float distlpsp2 = max(length(ld2), 0.001);
        ld /= distlpsp;
        ld2 /= distlpsp2;

        float atten = min(1./(distlpsp) + 1./(distlpsp2), 1.0) * (1.0 + bassLevel * 0.5);
        float ambience = 0.25;
        float diff = max(dot(sn, ld), 0.0);
        float diff2 = max(dot(sn, ld2), 0.0);
        float spec = pow(max(dot(reflect(-ld, sn), -rd), 0.0), 8.0);
        float spec2 = pow(max(dot(reflect(-ld2, sn), -rd), 0.0), 8.0);
        float crv = clamp(curve(sp, 0.125)*0.5 + 0.5, .0, 1.);
        float fre = pow(clamp(dot(sn, rd) + 1., .0, 1.), 1.0);
        vec3 texCol;
        if(hitRock){
            texCol = vec3(0.6,0.2,0.1);
        }
        else{
            texCol = proceduralTex3D(sp * (isFloor ? tSize1 : tSize0), sn, isFloor);
        }
        float shading = crv * 0.5 + 0.5;

        // Slight brightness boost on beat
        float brightnessBoost = 1.0 + beat * 0.2;

        sceneCol = texCol * ((diff + diff2) * 0.75 + ambience * 0.25) * brightnessBoost +
                  (spec + spec2) * texCol * 2.0 +
                  fre * crv * texCol.zyx * 2.0;

        sceneCol *= atten * shading * ao;

        float lineIntensity = clamp(1.0 - abs(curve(sp, 0.0125)), 0., 1.);
        sceneCol *= lineIntensity;
    }

    fragColor = vec4(clamp(sceneCol, 0., 1.), 1.0);
}
`,
"Hexed": `#define r iResolution.xy
#define PI 3.14159265358979323
const float sqrt3 = sqrt(3.);

float hstp(float x) {
    return max(min(x/2.-round(x/6.)*3.,1.),-1.)+round(x/6.)*2.;
}

vec2 hextosqr(vec2 p) {
    return (vec2(p.x,0) + vec2(-1,2) * vec2(p.y) / sqrt3 + vec2(-1,1) * hstp(p.x - sqrt3 * p.y)) / 2.;
}

vec2 sqrtohex(vec2 p) {
    return vec2(1,sqrt3) * (vec2(3,2) * p - vec2(0,p.x));
}

void mainImage(out vec4 c,in vec2 o) {
    for(int x = 0; x++ < 4;)
    for(int y = 0; y++ < 4;) {
        float w = 8.;
        vec2 s = o + vec2(x,y) / 4.;
        vec2 p = (s / r * 2. - 1.) * sqrt(r / r.yx);
        float m = iTime * 0.25;
        float g = m * -0.5 * PI * 3.;
        p *= mat2(cos(g),sin(g),-sin(g),cos(g)) / pow(w * w,fract(m) + 2.);
        vec2 l = round(hextosqr(p));
        vec2 h = p - sqrtohex(l);
        int a;
        for(int i = 0; i++ < 32; a++) {
            vec2 l = round(hextosqr(p));
            vec2 h = p - sqrtohex(l);
            c += clamp(vec4(length(h) * cos(mod(atan(h.y,h.x),PI / 3.) - PI / 6.) / 2.) * 25. - 21.,0.,1.) * 0.1;
            if(l != vec2(0)) {
                c += vec4(1. - sqrt(1. - length(h) / 2.)) * 0.03;
                break;
            }
            float t = m * (float(i % 2) * 2. - 1.) * PI * 3.;
            p *= mat2(cos(t),sin(t),-sin(t),cos(t)) * w;
        }
    }
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
"Sunset on river -  Raph":`// Sunset over the ocean.
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
}`,

"DULL AMAP":`// Amapiano Frequency Vortex
// A psychedelic visualization optimized for house and amapiano music

precision highp float;

// Standard Shadertoy uniforms (compatible with your converter)
uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D iChannel0; // Audio input texture

// Audio analysis zones
#define BASS_FREQ 0.05      // Bass frequencies for beat and log drum detection
#define LOW_MID_FREQ 0.15   // Low-mid for amapiano piano elements
#define HIGH_MID_FREQ 0.3   // High-mid for melody and leads
#define HIGH_FREQ 0.7       // High for hi-hats and percussives

// Configurable parameters - these could be exposed to UI sliders
#define PSYCHEDELIC_INTENSITY 0.8  // Overall intensity 0.0-1.0
#define COLOR_SHIFT_SPEED 0.4     // Color cycling speed
#define FRACTAL_DETAIL 6          // Iteration depth for fractal elements
#define BEAT_SENSITIVITY 1.2      // How reactive the visuals are to beats
#define MOTION_SMOOTH 0.8         // Smoothness of motion (higher = smoother)

// Pre-computed constants
const float PI = 3.14159265;
const float TWO_PI = 6.28318530;

// Various angle rotations for transformations
mat2 rotate2D(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, -s, s, c);
}

// Smooth min function for organic blending
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Get audio level from a specific frequency range
float getAudioLevel(float freq) {
    return texture2D(iChannel0, vec2(freq, 0.0)).x;
}

// Get beat intensity with temporal smoothing for amapiano log drum emphasis
float getBeatIntensity() {
    // Bass frequencies for main beats with emphasis on the characteristic 
    // log drum patterns of amapiano music
    float bassLevel = getAudioLevel(BASS_FREQ) * 1.8;
    float lowMidLevel = getAudioLevel(LOW_MID_FREQ) * 0.7;
    
    // Combine for total beat intensity with temporal smoothing
    float beatIntensity = bassLevel + lowMidLevel * 0.5;
    
    // Create a pulsing effect that's more pronounced on beats but still maintains motion
    return beatIntensity * BEAT_SENSITIVITY * (0.6 + 0.4 * sin(iTime * 2.0));
}

// Create a psychedelic color palette that shifts over time
vec3 psychedelicColor(float t) {
    // Base colors inspired by vibrant club lighting
    vec3 color1 = vec3(0.5, 0.0, 0.8);  // Purple
    vec3 color2 = vec3(0.0, 0.8, 0.8);  // Cyan
    vec3 color3 = vec3(0.9, 0.4, 0.0);  // Orange
    vec3 color4 = vec3(1.0, 0.2, 0.5);  // Pink
    
    // Get audio levels for different frequency ranges
    float highLevel = getAudioLevel(HIGH_FREQ);
    float midLevel = getAudioLevel(HIGH_MID_FREQ);
    
    // Shift colors based on audio
    float adjustedTime = iTime * COLOR_SHIFT_SPEED + midLevel * 2.0;
    
    // Create smooth transitions between colors
    float t1 = fract(t * 0.5 + adjustedTime * 0.2);
    float t2 = fract(t * 0.5 + adjustedTime * 0.2 + 0.33);
    float t3 = fract(t * 0.5 + adjustedTime * 0.2 + 0.67);
    
    // Mix colors based on audio-modified time
    vec3 color = color1 * (0.5 + 0.5 * sin(TWO_PI * t1)) +
                 color2 * (0.5 + 0.5 * sin(TWO_PI * t2)) +
                 color3 * (0.5 + 0.5 * sin(TWO_PI * t3)) +
                 color4 * highLevel;
    
    // Normalize and boost saturation for psychedelic effect
    return normalize(color) * (0.5 + length(color) * 0.5);
}

// Generate vortex-like fractal pattern
float vortexFractal(vec2 uv, float beatIntensity) {
    // Center coordinates
    vec2 p = uv;
    
    // Apply audio-reactive rotation and scaling
    p *= 1.0 + beatIntensity * 0.2;
    p *= rotate2D(iTime * 0.1 + beatIntensity * 0.5);
    
    float intensity = PSYCHEDELIC_INTENSITY;
    
    // Fractal parameters that respond to music
    float midLevel = getAudioLevel(HIGH_MID_FREQ);
    float shape = 0.0;
    
    // Fractal iteration
    for (int i = 0; i < FRACTAL_DETAIL; i++) {
        // Modify position with each iteration
        p = abs(p) / dot(p, p) - intensity * (0.5 + beatIntensity * 0.2);
        
        // Rotate based on beat
        p *= rotate2D(TWO_PI * (0.1 + 0.05 * midLevel) + beatIntensity * 0.1);
        
        // Accumulate shape value
        shape += length(p) * (0.1 + midLevel * 0.1);
    }
    
    return shape;
}

// Amapiano-specific log drum visual effect
float logDrumEffect(vec2 uv, float beatIntensity) {
    // Focus on bass frequencies where log drums are prominent in amapiano
    float logDrum = getAudioLevel(BASS_FREQ * 1.2) * 
                    getAudioLevel(BASS_FREQ * 0.8);
    
    // Create pulsating rings that respond to log drum patterns
    float dist = length(uv);
    float rings = 0.0;
    
    // Multiple rings with different frequencies
    for (int i = 2; i < 7; i++) {
        float size = float(i) * 0.1 + logDrum * 0.2;
        float width = 0.02 + logDrum * 0.01;
        rings += smoothstep(width, 0.0, abs(dist - size));
    }
    
    // Modulate with time and beat
    rings *= 0.5 + 0.5 * sin(dist * 10.0 - iTime * 2.0);
    
    return rings * logDrum * 3.0;
}

// Main shader function
void main() {
    // Normalize coordinates for aspect ratio
    vec2 uv = (2.0 * gl_FragCoord.xy - iResolution.xy) / min(iResolution.x, iResolution.y);
    
    // Get beat intensity from audio analysis
    float beatIntensity = getBeatIntensity();
    
    // Generate vortex fractal pattern
    float pattern = vortexFractal(uv, beatIntensity);
    
    // Add log drum visual effect specific to amapiano
    float logDrumVisual = logDrumEffect(uv, beatIntensity);
    
    // Combine effects
    float finalPattern = pattern + logDrumVisual;
    
    // Get psychedelic color based on pattern
    vec3 color = psychedelicColor(finalPattern);
    
    // Add highlight effect on beats
    color += vec3(1.0, 0.8, 0.4) * beatIntensity * 0.3;
    
    // Add subtle pulsing vignette effect
    float vignette = 1.0 - length(uv) * (0.5 + beatIntensity * 0.1);
    color *= vignette;
    
    // Final color with gamma correction for vibrant display
    gl_FragColor = vec4(pow(color, vec3(0.8)), 1.0);
}`,
"SHINY GLOPPP!": `

// REMEMBER TAN() ON COLOR FOR LIGHTING
#define PI 3.14159265359

// Comment/uncomment this to disable/enable anti-aliasing.
// #define AA

// The scene renderer averages pow(AA_SAMPLES, 2) random ray-marched samples for anti-aliasing.
#define AA_SAMPLES 4

// Enable or disable the rotating of the camera
#define ROTATE_CAMERA 1

// Material properties. Play around with these to change the way how the spheres are shaded.
const vec3 LIGHT_INTENSITY = vec3 (6.0);
const float INDIRECT_INTENSITY = 0.55;
const vec3 INDIRECT_SPECULAR_OFFSET = vec3(0.45, 0.65, 0.85);

const vec3 SPHERE0_RGB = vec3(0.4, 0.5, 0.0);
const vec3 SPHERE1_RGB = vec3(0.5, 0.0, 0.4);
const vec3 SPHERE2_RGB = vec3(0.0, 0.4, 0.5);

const float SPHERE_METALLIC = 0.99;
const float SPHERE_ROUGHNESS = 0.11;

// If you decrease metalness factor above, make sure to also change this to something
// non-metallic. Sample values can be found at: https://learnopengl.com/PBR/Theory
const vec3 SILVER_F0 = vec3(0.988, 0.98, 0.96);


// Modified version of Íñigo Quílez's integer hash3 function (https://www.shadertoy.com/view/llGSzw).
vec2 Hash2(uint n) 
{
	n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    uvec2 k = n * uvec2(n,n*16807U);
    return vec2( k & uvec2(0x7fffffffU))/float(0x7fffffff);
}

float UniformHash(vec2 xy)
{
	return fract(sin(dot(xy.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float SdPlane(vec3 p)
{
    return p.y;
}

float SdSphere(vec3 p, float s)
{
	return length(p) - s;  
}

vec2 SdUnion(vec2 d1, vec2 d2)
{
	return d1.x < d2.x ? d1 : d2;   
}

float SdSmoothMin(float a, float b)
{
    float k = 0.12;
    float h = clamp(0.5 + 0.5 * (b-a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float SdSmoothMin(float a, float b, vec3 mtl1, vec3 mtl0, inout vec3 mtl)
{
    float k = 0.12;
    float h = clamp(0.5 + 0.5 * (b-a) / k, 0.0, 1.0);
    float s = mix(b, a, h) - k * h * (1.0 - h);
    float sA = s - a;
    float sB = s - b;
    float r = sA / (sA + sB);
    mtl = mix(mtl1, mtl0, r);
    return s;
}

const float g_sRepeat = 800.0;

vec2 PosIndex(vec3 pos)
{
	vec2 posIndex;
	posIndex.x = floor((pos.x + 0.5 * g_sRepeat) / g_sRepeat);
	posIndex.y = floor((pos.y + 0.5 * g_sRepeat) / g_sRepeat);
	return posIndex;
}

float UniformHashFromPos(vec3 pos)
{
	pos.xy = PosIndex(pos);
	return UniformHash(pos.xy);
}

vec3 VecOsc(vec3 vecFreq, vec3 vecAmp, float dT)
{
	return vecAmp * sin(vec3((iTime + dT) * 2.0 * PI) * vecFreq);
}

vec2 SdScene(vec3 p)
{
    float uniformRandom = UniformHashFromPos(p);
    vec3 uSphereOsc1 = VecOsc(vec3(1.02389382 / 2.0, 1.0320809 / 3.0, 1.07381 / 4.0),
							vec3(0.25, 0.25, 0.1), uniformRandom);
    vec3 uSphereOsc2 = VecOsc(vec3(1.032038 / 4.0, 1.13328 / 2.0, 1.09183 / 3.0),
							vec3(0.25, 0.25, 0.1), uniformRandom);
    vec3 uSphereOsc3 = VecOsc(vec3(1.123283 / 3.0, 1.13323 / 4.0, 1.2238 / 2.0),
							vec3(0.25, 0.25, 0.1), uniformRandom);
    return SdUnion(vec2(SdPlane(p), 1.0), 
                   vec2(
                       SdSmoothMin(
                           SdSmoothMin(
                               SdSphere(p - vec3(0.0, 0.5, 0.0) + uSphereOsc1, 0.18),
                               SdSphere(p - vec3(0.0, 0.5, 0.0) + uSphereOsc2, 0.2)
		     	    	   ),
                       	   SdSphere(p - vec3(0.0, 0.5, 0.0) + uSphereOsc3, 0.19)
                       ),
                       2.0
                   )
           );
}

vec2 SdScene(in vec3 p, inout vec3 mtl)
{
    float uniformRandom = UniformHashFromPos(p);
    vec3 uSphereOsc1 = VecOsc(vec3(1.02389382 / 2.0, 1.0320809 / 3.0, 1.07381 / 4.0),
							vec3(0.25, 0.25, 0.1), uniformRandom);
    vec3 uSphereOsc2 = VecOsc(vec3(1.032038 / 4.0, 1.13328 / 2.0, 1.09183 / 3.0),
							vec3(0.25, 0.25, 0.1), uniformRandom);
    vec3 uSphereOsc3 = VecOsc(vec3(1.123283 / 3.0, 1.13323 / 4.0, 1.2238 / 2.0),
							vec3(0.25, 0.25, 0.1), uniformRandom);
    
    float smin1 = SdSmoothMin(
                             SdSphere(p - vec3(0.0, 0.5, 0.0) + uSphereOsc1, 0.18),
                             SdSphere(p - vec3(0.0, 0.5, 0.0) + uSphereOsc2, 0.2),
        					 SPHERE0_RGB, SPHERE1_RGB, mtl
				  );
    float smin2 = SdSmoothMin(smin1, SdSphere(p - vec3(0.0, 0.5, 0.0) + uSphereOsc3, 0.19),
                              mtl, SPHERE2_RGB, mtl);
    return SdUnion(vec2(SdPlane(p), 1.0), vec2(smin2, 2.0));
}

vec3 CalcNormal(vec3 p)
{
	vec2 e = vec2(1.0,-1.0) * 0.5773 * 0.0005;
    return normalize( e.xyy * SdScene( p + e.xyy ).x + 
					  e.yyx * SdScene( p + e.yyx ).x + 
					  e.yxy * SdScene( p + e.yxy ).x + 
					  e.xxx * SdScene( p + e.xxx ).x );
}

float ShadowMarch(in vec3 origin, in vec3 rayDirection)
{
	float result = 1.0;
    float t = 0.01;
    for (int i = 0; i < 64; ++i)
    {
        float hit = SdScene(origin + rayDirection * t).x;
        if (hit < 0.001)
            return 0.0;
        result = min(result, 5.0 * hit / t);
        t += hit;
        if (t >= 1.5)
            break;
    }
    
    return clamp(result, 0.0, 1.0);
}
    
vec2 RayMarch(in vec3 origin, in vec3 rayDirection, inout vec3 mtl)
{
    float material = -1.0;
    float t = 0.01;
	for(int i = 0; i < 64; ++i)
    {
        vec3 p = origin + rayDirection * t;
        vec2 hit = SdScene(p, mtl);
        if (hit.x < 0.001 * t || t > 50.0)
			break;
        t += hit.x;
        material = hit.y;
    }
    
    if (t > 50.0)
    {
     	material = -1.0;   
    }
    return vec2(t, material);
}

//-----------------------------------------PBR Functions-----------------------------------------------//

// Trowbridge-Reitz GGX based Normal distribution function
float NormalDistributionGGX(float NdotH, float roughness)
{    
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    
    float numerator = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom *= denom * PI;
    
    return numerator / denom;
}

// Schlick-Beckmann GGX approximation used for smith's method
float GeometrySchlickGGX(float NdotX, float k)
{
    float numerator   = NdotX;
    float denom = NdotX * (1.0 - k) + k;

    return numerator / denom;
}

// Smith's method for calculating geometry shadowing
float GeometrySmith(float NdotV, float NdotL, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;
    //float k = (r*r) * sqrt(2.0 / PI);
    
    float ggx2 = GeometrySchlickGGX(NdotV, k);
    float ggx1 = GeometrySchlickGGX(NdotL, k);

    return ggx1 * ggx2;
}

// Schlick's approximation for Fresnel equation
vec3 FresnelSchlick(float dotProd, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - dotProd, 5.0);
}

float FresnelSchlick(float dotProd, float F0, float F90)
{
    return F0 + (F90 - F0) * pow(1.0 - dotProd, 5.0);
}

// Burley 2012, "Physically-Based Shading at Disney"
float DiffuseBurley(float linearRoughness, float NdotV, float NdotL, float LdotH)
{
    float f90 = 0.5 + 2.0 * linearRoughness * LdotH * LdotH;
    float lightScatter = FresnelSchlick(NdotL, 1.0, f90);
    float viewScatter  = FresnelSchlick(NdotV, 1.0, f90);
    return lightScatter * viewScatter * (1.0 / PI);
}

// Irradiance from "Ditch River" IBL (http://www.hdrlabs.com/sibl/archive.html)
vec3 DiffuseIrradiance(const vec3 n) 
{
    return max(
          vec3( 0.754554516862612,  0.748542953903366,  0.790921515418539)
        + vec3(-0.083856548007422,  0.092533500963210,  0.322764661032516) * (n.y)
        + vec3( 0.308152705331738,  0.366796330467391,  0.466698181299906) * (n.z)
        + vec3(-0.188884931542396, -0.277402551592231, -0.377844212327557) * (n.x)
        , 0.0);
}

// Karis 2014, "Physically Based Material on Mobile"
vec2 PrefilteredEnvApprox(float roughness, float NoV) 
{
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572,  0.022);
    const vec4 c1 = vec4( 1.0,  0.0425,  1.040, -0.040);

    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;

    return vec2(-1.04, 1.04) * a004 + r.zw;
}

//-----------------------------------------------------------------------------------------------------//

// Handle the shading of the scene.
vec3 RenderScene(in vec3 origin, in vec3 rayDirection, out float hitDistance)
{
    vec3 finalColor = vec3(0.0);
    vec3 albedo = vec3(0.0);
    vec3 sphereMaterial = vec3(0.0);

    // returns distance and material
    vec2 hit = RayMarch(origin, rayDirection, sphereMaterial);
    hitDistance = hit.x;
    float material = hit.y;
    
    vec3 sphereColor = vec3(0.13, 0.0, 0.5);
    
    if (material > 0.0)
    {
        // Essential vectors and scalars for lighting calculation
        vec3 position = origin + hitDistance * rayDirection;
        vec3 normal	= CalcNormal(position);
        vec3 viewDir = normalize(-rayDirection);
        vec3 lightDir = normalize(vec3(20.6, 20.7, -20.7) - position);
        vec3 halfVec = normalize(viewDir + lightDir);
        vec3 reflectDir = normalize(reflect(rayDirection, normal));
        
        float NdotL = max(dot(normal, lightDir), 0.0);
        float NdotV = max(dot(normal, viewDir), 0.0);
        float NdotH = max(dot(normal, halfVec), 0.0);
        float LdotH = max(dot(lightDir, halfVec), 0.0);
        float VdotH = max(dot(halfVec, viewDir), 0.0);
        
        float roughness = 0.0, metallic = 0.0;
        vec3 F0 = vec3(0.0);
        
        if (material < 2.0)
        {
            // Checkerboard floor
        	float f = mod(floor(7.5 * position.z) + floor(7.5 * position.x), 2.0);
			albedo = 0.4 + f * vec3(0.6);
            roughness = (f > 0.5 ? 1.0 : 0.18);
            metallic = 0.4;
            // Plastic/Glass F0 value
            F0 = mix(vec3(0.04), albedo, metallic);
        } 
        else if (material < 3.0)
        {
            // Spheres
         	albedo =  sphereMaterial;
            roughness = clamp(SPHERE_ROUGHNESS, 0.0, 1.0);
            metallic = clamp(SPHERE_METALLIC, 0.0, 1.0);
            // Silver F0 value
            F0 = mix(SILVER_F0, albedo, metallic);
        }
         
        
        // Calculate radiance
        //float lightDistance = length(lightDir);
        //float attenuation = 1.0 / (lightDistance * lightDistance);
        float attenuation = ShadowMarch(position, lightDir);
        vec3 radiance = LIGHT_INTENSITY * attenuation;
        
        // Cook-Torrence specular BRDF
        float ndf = NormalDistributionGGX(NdotH, roughness);
        float geometry = GeometrySmith(NdotV, NdotL, roughness);
        vec3 fresnel = FresnelSchlick(VdotH, F0);
        
        vec3 numerator = ndf * geometry * fresnel;
        float denominator = 4.0 * NdotV * NdotL;
        
        vec3 specular = numerator / max(denominator, 0.0001);
        
        // Burley Diffuse BRDF
        float diffuse = DiffuseBurley(roughness * roughness, NdotV, NdotL, LdotH);
        
        // Energy conservation
        vec3 kS = fresnel;
        vec3 kD = vec3(1.0) - kS;
        // Diffuse light decreases as "metal-ness" increases (and vice versa).
        kD *= 1.0 - metallic;
        
        vec3 ambient = 0.05 * albedo;
            
        // Note to self: Hmm, not sure whether to divide diffuse by PI or not. Some implementations
        // do while others don't seem to.
        // Also, note to self: We don't multiply by kS here because it's already done in the calculation
        // of the numerator part of the specular component.
        finalColor += (kD * albedo * diffuse / PI + specular + ambient) * radiance * NdotL;
        
        
        // Indirect Lighting
        sphereMaterial = vec3(0.0);
        vec2 indirectHit = RayMarch(position, reflectDir, sphereMaterial);
        vec3 indirectDiffuse = DiffuseIrradiance(normal) / PI;
        vec3 indirectSpecular = INDIRECT_SPECULAR_OFFSET + reflectDir.y * 0.72;
        
        if (indirectHit.y > 0.0)
        {
            if (indirectHit.y < 2.0)
            {
                vec3 indirectPosition = position + indirectHit.x * reflectDir;
                // Checkerboard floor
                float f = mod(floor(7.5 * indirectPosition.z) + floor(7.5 * indirectPosition.x), 2.0);
				indirectSpecular = 0.4 + f * vec3(0.6);
            }
            else if (indirectHit.y < 3.0)
            {
                // Spheres
                indirectSpecular = sphereMaterial;
            }
        }
        
        vec2 prefilteredSpecularBRDF = PrefilteredEnvApprox(roughness, NdotV);
        vec3 indirectSpecularColor = F0 * prefilteredSpecularBRDF.x + prefilteredSpecularBRDF.y;
        vec3 ibl = (1. - metallic) * albedo * indirectDiffuse + indirectSpecular * indirectSpecularColor;
        
        finalColor += ibl * INDIRECT_INTENSITY;
    }
    
    return finalColor;
}

// Gamma correction
vec3 LinearTosRGB(const vec3 linear)
{
    return pow(linear, vec3(1.0 / 2.2));
}

// Tone mapping
vec3 AcesFilmicToneMap(const vec3 x)
{
    // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

// Generate the camera view matrix
mat3 SetCamera(in vec3 origin, in vec3 target, float rotation)
{
    vec3 forward = normalize(target - origin);
    vec3 orientation = vec3(sin(rotation), cos(rotation), 0.0);
    vec3 left = normalize(cross(forward, orientation));
    vec3 up = normalize(cross(left, forward));
    return mat3(left, up, forward);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Set up the camera view matrix
    vec3 lookAt = vec3(0.0, 0.25, 0.0);
    float phi = iTime * 0.25 + PI; // horizontal plane angle
    float theta = 0.3; // left-right (around y-axis) angle
    
    // === AUDIO-REACTIVE CAMERA BOBBING ===
    // Sample bass frequencies for beat detection with smoothing
    float bassFreq = texture(iChannel0, vec2(0.07, 0.0)).x; // Bass frequency
    
    // Create a proper bounce effect that goes up and down
    // Store the "impact" of the bass hit in a variable that decays over time
    float decay = 2.0; // Higher value = faster decay
    float bassImpact = sin(iTime * decay) * 0.5 + 0.5; // Baseline oscillation
    
    // Add audio impact that amplifies the bounce
    float smoothBass = bassFreq * 0.6; // Reduce intensity
    
    // Create a proper bouncing motion that goes up and returns
    float bounce = smoothBass * (0.25 - 0.15 * bassImpact);
    
    // Clamp to prevent extreme movements
    bounce = clamp(bounce, 0.0, 0.25);
    
#if ROTATE_CAMERA
    // Original orbit motion + vertical audio-reactive bounce
    vec3 eyePosition = vec3(
        cos(theta) * cos(phi), 
        sin(theta) + bounce, // Smooth, limited audio-reactive bounce
        cos(theta) * sin(phi)
    ) * 4.0;
#else
    vec3 eyePosition = vec3(0.0, 1.0 + bounce, -4.0);
#endif
    
    // Calculate the camera matrix.
    mat3 cameraToWorld = SetCamera(eyePosition, lookAt, 0.0);
    
    vec3 color = vec3(0.0);
    float t;
    
#ifdef AA
    
    int totalAASamples = AA_SAMPLES*AA_SAMPLES;
    for (int i = 0; i < totalAASamples; ++i)
    {
        
        // Shamelessly plugged and modified from demofox's reflection/refraction shader.
        // Calculates stratified subpixel jitter for anti-aliasing.
        float x = mod(float(i), float(AA_SAMPLES));
        float y = mod(float(i / AA_SAMPLES), float(AA_SAMPLES));
        
        vec2 jitter = (Hash2(uint(i)) + vec2(x, y)) / float(AA_SAMPLES);

        vec2 uv = 2.0 * (fragCoord.xy + jitter) / iResolution.xy - 1.0;
        uv.x *= iResolution.x / iResolution.y;

#else
        vec2 uv = 2.0 * fragCoord.xy / iResolution.xy - 1.0;
        uv.x *= iResolution.x / iResolution.y;

#endif    
        // Ray direction starts from camera's current position
        vec3 rayDirection = cameraToWorld * normalize(vec3(uv, 5.0));

        // Ray march the scene using sdfs
        color += RenderScene(eyePosition, rayDirection, t);    
#ifdef AA
    }
    
    color /= float(totalAASamples);
#endif
    
    // Add a simple distance fog to the scene
    float fog = 1.0 - exp2(-0.012 * t * t);
    color = mix(color, 0.8 * vec3(0.6, 0.8, 1.0), fog);
    
    // Tone mapping
    color = AcesFilmicToneMap(color);

    // Gamma correction
    color = LinearTosRGB(color);
    
    fragColor = vec4(tan(color), 1.0);
}`,
"SHINY GLOPPP! 2": `vec3 palette( float t)
{
    vec3 a = vec3(0.848, 0.500, 0.588);
    vec3 b = vec3(0.718, 0.500, 0.500);
    vec3 c = vec3(0.750, 1.000, 0.667);
    vec3 d = vec3(-0.082, -0.042, 0.408);
    
    return a + b*cos( 6.28318*(c*t+d) );
}

float distance_from_sphere(in vec3 p, in vec3 c, float r)
{
    return length(p - c) - r;
}

float map_shape(in vec3 p)
{
    // Sample audio data from different frequency bands
    float bassLevel = texture(iChannel0, vec2(0.05, 0.0)).x; // Low frequencies
    float midLevel = texture(iChannel0, vec2(0.2, 0.0)).x;   // Mid frequencies
    float highLevel = texture(iChannel0, vec2(0.5, 0.0)).x;  // High frequencies
    
    // Use bass for the main displacement amplitude
    float audioAmplitude = bassLevel * 2.6 + midLevel * 0.3 + highLevel * 0.1;
    
    // Add a baseline motion so it's always animated even with no audio
    float baselineMotion = sin(iTime * 2.0 + cos(iTime * 12.0));
    
    // Combine audio amplitude with baseline motion (weighted so audio is prominent)
    float combinedAmplitude = mix(baselineMotion * 0.5, audioAmplitude * 0.9, 0.7);
    
    // Calculate displacement using the combined amplitude
    float displacement = sin(3.0 * p.x) * sin(3.0 * p.y) * sin(3.0 * p.z) * 0.25 * combinedAmplitude;
    
    // Make sphere surface slightly ripple based on higher frequencies
    float detailRipple = sin(8.0 * p.x + iTime) * sin(8.0 * p.y + iTime) * sin(8.0 * p.z) * highLevel * 0.15;
    
    float sphere_0 = distance_from_sphere(p, vec3(0.0), 1.8);
    
    // Add both types of displacement to the sphere
    return sphere_0 + displacement + detailRipple;
}

vec3 calculate_normal(in vec3 p)
{
    const vec3 small_step = vec3(0.001, 0.0, 0.0);
    float gradient_x = map_shape(p + small_step.xyy) - map_shape(p - small_step.xyy);
    float gradient_y = map_shape(p + small_step.yxy) - map_shape(p - small_step.yxy);
    float gradient_z = map_shape(p + small_step.yyx) - map_shape(p - small_step.yyx);
    vec3 normal = vec3(gradient_x, gradient_y, gradient_z);
    return normalize(normal);
}

vec3 ray_march(in vec3 ro, in vec3 rd)
{
    float distance_traveled = 0.0;
    const int max_steps = 32;
    const float min_hit_dist = 0.001;
    const float max_trace_dist = 1000.0;
    vec3 col = palette(length(ro) + (iTime * 0.2));
    for (int i = 0; i < max_steps; i++)
    {
        vec3 current_position = ro + distance_traveled * rd;
        float distance_to_closest = map_shape(current_position);
        
        if (distance_to_closest < min_hit_dist) 
        {
           vec3 normal = calculate_normal(current_position);
           vec3 light_position = vec3(2.0, -5.0, 3.0);
           vec3 direction_to_light = normalize(current_position - light_position);
           float diffuse_intensity = max(0.0, dot(normal, direction_to_light));
           return vec3(1.0, 0.0, 0.0) * diffuse_intensity;
        }
        if (distance_traveled > max_trace_dist)
        {
            break;
        }
        
        distance_traveled += distance_to_closest;
    }
    
    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy * 2.0 - 1.0; 
    uv.x *= iResolution.x / iResolution.y;
    
    vec3 camera_position = vec3(0.0, 0.0, -5.0);
    vec3 ro = camera_position;
    vec3 rd = vec3(uv, 1.0);
    vec3 shaded_color = ray_march(ro, rd);
    fragColor = vec4(shaded_color, 1.0);
}`,
"Unseen Vow": `/*
	Perspex Web Lattice
	-------------------
	
	I felt that Shadertoy didn't have enough Voronoi examples, so I made another one. :) I'm
	not exactly sure what it's supposed to be... My best guess is that an Alien race with no 
	common sense designed a monitor system with physics defying materials. :)

	Technically speaking, there's not much to it. It's just some raymarched 2nd order Voronoi.
	The dark perspex-looking web lattice is created by manipulating the Voronoi value slightly 
	and giving the effected region an ID value so as to color it differently, but that's about
	it. The details are contained in the "heightMap" function.

	There's also some subtle edge detection in order to give the example a slight comic look. 
	3D geometric edge detection doesn't really differ a great deal in concept from 2D pixel 
	edge detection, but it obviously involves more processing power. However, it's possible to 
	combine the edge detection with the normal calculation and virtually get it for free. Kali 
	uses it to great effect in his "Fractal Land" example. It's also possible to do a
	tetrahedral version... I think Nimitz and some others may have done it already. Anyway, 
	you can see how it's done in the "nr" (normal) function.

	Geometric edge related examples:

	Fractal Land - Kali
	https://www.shadertoy.com/view/XsBXWt

	Rotating Cubes - Shau
	https://www.shadertoy.com/view/4sGSRc

	Voronoi mesh related:

    // I haven't really looked into this, but it's interesting.
	Weaved Voronoi - FabriceNeyret2 
    https://www.shadertoy.com/view/ltsXRM

*/

#define FAR 2.

int id = 0; // Object ID - Red perspex: 0; Black lattice: 1.


// Tri-Planar blending function. Based on an old Nvidia writeup:
// GPU Gems 3 - Ryan Geiss: https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch01.html
vec3 tex3D( sampler2D tex, in vec3 p, in vec3 n ){
   
    n = max((abs(n) - .2), .001);
    n /= (n.x + n.y + n.z ); // Roughly normalized.
    
	p = (texture(tex, p.yz)*n.x + texture(tex, p.zx)*n.y + texture(tex, p.xy)*n.z).xyz;
    
    // Loose sRGB to RGB conversion to counter final value gamma correction...
    // in case you're wondering.
    return p*p;
}


// Compact, self-contained version of IQ's 3D value noise function. I have a transparent noise
// example that explains it, if you require it.
float n3D(vec3 p){
    
	const vec3 s = vec3(7, 157, 113);
	vec3 ip = floor(p); p -= ip; 
    vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
    p = p*p*(3. - 2.*p); //p *= p*p*(p*(p * 6. - 15.) + 10.);
    h = mix(fract(sin(h)*43758.5453), fract(sin(h + s.x)*43758.5453), p.x);
    h.xy = mix(h.xz, h.yw, p.y);
    return mix(h.x, h.y, p.z); // Range: [0, 1].
}

// vec2 to vec2 hash.
vec2 hash22(vec2 p) { 

    // Faster, but doesn't disperse things quite as nicely. However, when framerate
    // is an issue, and it often is, this is a good one to use. Basically, it's a tweaked 
    // amalgamation I put together, based on a couple of other random algorithms I've 
    // seen around... so use it with caution, because I make a tonne of mistakes. :)
    float n = sin(dot(p, vec2(41, 289)));
    //return fract(vec2(262144, 32768)*n); 
    
    // Animated.
    p = fract(vec2(262144, 32768)*n); 
    // Note the ".45," insted of ".5" that you'd expect to see. When edging, it can open 
    // up the cells ever so slightly for a more even spread. In fact, lower numbers work 
    // even better, but then the random movement would become too restricted. Zero would 
    // give you square cells.
    return sin( p*6.2831853 + iTime )*.45 + .5; 
    
}

// 2D 2nd-order Voronoi: Obviously, this is just a rehash of IQ's original. I've tidied
// up those if-statements. Since there's less writing, it should go faster. That's how 
// it works, right? :)
//
float Voronoi(in vec2 p){
    
	vec2 g = floor(p), o; p -= g;
	
	vec3 d = vec3(1); // 1.4, etc. "d.z" holds the distance comparison value.
    
	for(int y = -1; y <= 1; y++){
		for(int x = -1; x <= 1; x++){
            
			o = vec2(x, y);
            o += hash22(g + o) - p;
            
			d.z = dot(o, o); 
            // More distance metrics.
            //o = abs(o);
            //d.z = max(o.x*.8666 + o.y*.5, o.y);// 
            //d.z = max(o.x, o.y);
            //d.z = (o.x*.7 + o.y*.7);
            
            d.y = max(d.x, min(d.y, d.z));
            d.x = min(d.x, d.z); 
                       
		}
	}
	
    return max(d.y/1.2 - d.x*1., 0.)/1.2;
    //return d.y - d.x; // return 1.-d.x; // etc.
    
}

// The height map values. In this case, it's just a Voronoi variation. By the way, I could
// optimize this a lot further, but it's not a particularly taxing distance function, so
// I've left it in a more readable state.
float heightMap(vec3 p){
    
    id =0;
    float c = Voronoi(p.xy*4.); // The fiery bit.
    
    // For lower values, reverse the surface direction, smooth, then
    // give it an ID value of one. Ie: this is the black web-like
    // portion of the surface.
    if (c<.07) {c = smoothstep(0.7, 1., 1.-c)*.2; id = 1; }

    return c;
}

// Standard back plane height map. Put the plane at vec3(0, 0, 1), then add some height values.
// Obviously, you don't want the values to be too large. The one's here account for about 10%
// of the distance between the plane and the camera.
float m(vec3 p){
   
    float h = heightMap(p); // texture(iChannel0, p.xy/2.).x; // Texture work too.
    
    return 1. - p.z - h*.1;
    
}

/*
// Tetrahedral normal, to save a couple of "map" calls. Courtesy of IQ.
vec3 nr(in vec3 p){

    // Note the slightly increased sampling distance, to alleviate artifacts due to hit point inaccuracies.
    vec2 e = vec2(0.005, -0.005); 
    return normalize(e.xyy * m(p + e.xyy) + e.yyx * m(p + e.yyx) + e.yxy * m(p + e.yxy) + e.xxx * m(p + e.xxx));
}
*/

/*
// Standard normal function - for comparison with the one below.
vec3 nr(in vec3 p) {
	const vec2 e = vec2(0.005, 0);
	return normalize(vec3(m(p + e.xyy) - m(p - e.xyy), m(p + e.yxy) - m(p - e.yxy),	m(p + e.yyx) - m(p - e.yyx)));
}
*/

// The normal function with some edge detection rolled into it.
vec3 nr(vec3 p, inout float edge) { 
	
    vec2 e = vec2(.005, 0);

    // Take some distance function measurements from either side of the hit point on all three axes.
	float d1 = m(p + e.xyy), d2 = m(p - e.xyy);
	float d3 = m(p + e.yxy), d4 = m(p - e.yxy);
	float d5 = m(p + e.yyx), d6 = m(p - e.yyx);
	float d = m(p)*2.;	// The hit point itself - Doubled to cut down on calculations. See below.
     
    // Edges - Take a geometry measurement from either side of the hit point. Average them, then see how
    // much the value differs from the hit point itself. Do this for X, Y and Z directions. Here, the sum
    // is used for the overall difference, but there are other ways. Note that it's mainly sharp surface 
    // curves that register a discernible difference.
    edge = abs(d1 + d2 - d) + abs(d3 + d4 - d) + abs(d5 + d6 - d);
    //edge = max(max(abs(d1 + d2 - d), abs(d3 + d4 - d)), abs(d5 + d6 - d)); // Etc.
    
    // Once you have an edge value, it needs to normalized, and smoothed if possible. How you 
    // do that is up to you. This is what I came up with for now, but I might tweak it later.
    edge = smoothstep(0., 1., sqrt(edge/e.x*2.));
	
    // Return the normal.
    // Standard, normalized gradient mearsurement.
    return normalize(vec3(d1 - d2, d3 - d4, d5 - d6));
}

/*
// I keep a collection of occlusion routines... OK, that sounded really nerdy. :)
// Anyway, I like this one. I'm assuming it's based on IQ's original.
float cAO(in vec3 p, in vec3 n)
{
	float sca = 3., occ = 0.;
    for(float i=0.; i<5.; i++){
    
        float hr = .01 + i*.5/4.;        
        float dd = m(n * hr + p);
        occ += (hr - dd)*sca;
        sca *= 0.7;
    }
    return clamp(1.0 - occ, 0., 1.);    
}
*/

/*
// Standard hue rotation formula... compacted down a bit.
vec3 rotHue(vec3 p, float a){

    vec2 cs = sin(vec2(1.570796, 0) + a);

    mat3 hr = mat3(0.299,  0.587,  0.114,  0.299,  0.587,  0.114,  0.299,  0.587,  0.114) +
        	  mat3(0.701, -0.587, -0.114, -0.299,  0.413, -0.114, -0.300, -0.588,  0.886) * cs.x +
        	  mat3(0.168,  0.330, -0.497, -0.328,  0.035,  0.292,  1.250, -1.050, -0.203) * cs.y;
							 
    return clamp(p*hr, 0., 1.);
}
*/

// Simple environment mapping. Pass the reflected vector in and create some
// colored noise with it. The normal is redundant here, but it can be used
// to pass into a 3D texture mapping function to produce some interesting
// environmental reflections.
//
// More sophisticated environment mapping:
// UI easy to integrate - XT95    
// https://www.shadertoy.com/view/ldKSDm
vec3 eMap(vec3 rd, vec3 sn){
    
    vec3 sRd = rd; // Save rd, just for some mixing at the end.
    
    // Add a time component, scale, then pass into the noise function.
    rd.xy -= iTime*.25;
    rd *= 3.;
    
    //vec3 tx = tex3D(iChannel0, rd/3., sn);
    //float c = dot(tx*tx, vec3(.299, .587, .114));
    
    float c = n3D(rd)*.57 + n3D(rd*2.)*.28 + n3D(rd*4.)*.15; // Noise value.
    c = smoothstep(0.5, 1., c); // Darken and add contast for more of a spotlight look.
    
    //vec3 col = vec3(c, c*c, c*c*c*c).zyx; // Simple, warm coloring.
    vec3 col = vec3(min(c*1.5, 1.), pow(c, 2.5), pow(c, 12.)).zyx; // More color.
    
    // Mix in some more red to tone it down and return.
    return mix(col, col.yzx, sRd*.25+.25); 
    
}

void mainImage(out vec4 c, vec2 u){

    // Unit direction ray, camera origin and light position.
    vec3 r = normalize(vec3(u - iResolution.xy*.5, iResolution.y)), 
         o = vec3(0), l = o + vec3(0, 0, -1);
   
    // Rotate the canvas. Note that sine and cosine are kind of rolled into one.
    vec2 a = sin(vec2(1.570796, 0) + iTime/8.); // Fabrice's observation.
    r.xy = mat2(a, -a.y, a.x) * r.xy;

    
    // Standard raymarching routine. Raymarching a slightly perturbed back plane front-on
    // doesn't usually require many iterations. Unless you rely on your GPU for warmth,
    // this is a good thing. :)
    float d, t = 0.;
    
    for(int i=0; i<32;i++){
        
        d = m(o + r*t);
        // There isn't really a far plane to go beyond, but it's there anyway.
        if(abs(d)<0.001 || t>FAR) break;
        t += d*.7;

    }
    
    t = min(t, FAR);
    
    // Set the initial scene color to black.
    c = vec4(0);
    
    float edge = 0.; // Edge value - to be passed into the normal.
    
    if(t<FAR){
    
        vec3 p = o + r*t, n = nr(p, edge);

        l -= p; // Light to surface vector. Ie: Light direction vector.
        d = max(length(l), 0.001); // Light to surface distance.
        l /= d; // Normalizing the light direction vector.

        
 
        // Obtain the height map (destorted Voronoi) value, and use it to slightly
        // shade the surface. Gives a more shadowy appearance.
        float hm = heightMap(p);
        
        // Texture value at the surface. Use the heighmap value above to distort the
        // texture a bit.
        vec3 tx = tex3D(iChannel0, (p*2. + hm*.2), n);
        //tx = floor(tx*15.999)/15.; // Quantized cartoony colors, if you get bored enough.

        c.xyz = vec3(1.)*(hm*.8 + .2); // Applying the shading to the final color.
        
        c.xyz *= vec3(1.5)*tx; // Multiplying by the texture value and lightening.
        
        
        // Color the cell part with a fiery (I incorrectly spell it firey all the time) 
        // palette and the latticey web thing a very dark color.
        //
        c.x = dot(c.xyz, vec3(.299, .587, .114)); // Grayscale.
        if (id==0) c.xyz *= vec3(min(c.x*1.5, 1.), pow(c.x, 5.), pow(c.x, 24.))*2.;
        else c.xyz *= .1;
        
        // Hue rotation, for anyone who's interested.
        //c.xyz = rotHue(c.xyz, mod(iTime/16., 6.283));
       
        
        float df = max(dot(l, n), 0.); // Diffuse.
        float sp = pow(max(dot(reflect(-l, n), -r), 0.), 32.); // Specular.
        
        if(id == 1) sp *= sp; // Increase specularity on the dark lattice.
        
		// Applying some diffuse and specular lighting to the surface.
        c.xyz = c.xyz*(df + .75) + vec3(1, .97, .92)*sp + vec3(.5, .7, 1)*pow(sp, 32.);
        
        // Add the fake environmapping. Give the dark surface less reflectivity.
        vec3 em = eMap(reflect(r, n), n); // Fake environment mapping.
        if(id == 1) em *= .5;
        c.xyz += em;
        
        // Edges.
        //if(id == 0)c.xyz += edge*.1; // Lighter edges.
        c.xyz *= 1. - edge*.8; // Darker edges.
        
        // Attenuation, based on light to surface distance.    
        c.xyz *= 1./(1. + d*d*.125);
        
        // AO - The effect is probably too subtle, in this case, so we may as well
        // save some cycles.
        //c.xyz *= cAO(p, n);
        
    }
    
    
    // Vignette.
    //vec2 uv = u/iResolution.xy;
    //c.xyz = mix(c.xyz, vec3(0, 0, .5), .1 -pow(16.*uv.x*uv.y*(1.-uv.x)*(1.-uv.y), 0.25)*.1);
    
    // Apply some statistically unlikely (but close enough) 2.0 gamma correction. :)
    c = vec4(sqrt(clamp(c.xyz, 0., 1.)), 1.);
    
    
}`,
"Squeeze me": `// Modified from Peleg Gefen's original shader
// Adapted to work without image textures and to use microphone input

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

// Toggle variable width (on by default)
#define Shrink

#define ZOOM 20.0
#define PI 3.141592654
#define TAU (2.0*PI)

#define MROT(a) mat2(cos(a), sin(a), -sin(a), cos(a))
const mat2 rot120 = MROT(TAU/3.0);

// Procedural noise and patterns to replace texture channels
float hash(in vec2 co) {
  return fract(sin(dot(co.xy, vec2(12.9898, 58.233))) * 13758.5453);
}

// Additional noise functions to replace textures
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f*f*(3.0-2.0*f);
  return mix(
    mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
    mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x),
    f.y
  );
}

// FBM (Fractal Brownian Motion) for more organic textures
float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  vec2 shift = vec2(100.0);
  // Rotate to reduce axial bias
  mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
  for (int i = 0; i < 5; ++i) {
    v += a * noise(p);
    p = rot * p * 2.0 + shift;
    a *= 0.5;
  }
  return v;
}

// Pattern generators to replace texture channels
vec4 patternA(vec2 uv) {
  // Colorful swirly pattern
  uv *= 3.0;
  float n = fbm(uv * 2.0 + iTime * 0.2);
  float n2 = fbm(uv * 1.5 - iTime * 0.1 + vec2(n, 0.0));
  
  vec3 col = vec3(0.5) + 0.5 * cos(n * 8.0 + vec3(0.0, 1.0, 2.0) + iTime);
  col = mix(col, vec3(0.7, 0.3, 0.2), n2 * 0.6);
  
  return vec4(col, 1.0);
}

vec4 patternB(vec2 uv) {
  // More geometric pattern
  uv *= 5.0;
  float n = sin(uv.x * 3.0 + iTime) * sin(uv.y * 3.0 + iTime * 0.7) * 0.5 + 0.5;
  float n2 = fbm(uv * 3.0 + iTime * 0.4);
  
  vec3 col = vec3(0.2, 0.5, 0.8) + 0.3 * cos(n * 6.0 + vec3(0.0, 2.0, 4.0));
  col = mix(col, vec3(0.9, 0.6, 0.1), n2 * 0.5);
  
  return vec4(col, 1.0);
}

vec2 rot(vec2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return p * mat2(c, s, -s, c);
}

float hexDist(vec2 p) {
  p = abs(p);
  // Distance to the diagonal line
  float c = dot(p, normalize(vec2(1.0, 1.73)));
  // Distance to the vertical line
  c = max(c, p.x);
  return c;
}

vec4 hexCoords(vec2 uv) {
  vec2 r = vec2(1.0, 1.73);
  vec2 h = r * 0.5;
  vec2 a = mod(uv, r) - h;
  vec2 b = mod(uv - h, r) - h;

  vec2 gv;
  if (length(a) < length(b))
    gv = a;
  else
    gv = b;

  float y = 0.5 - hexDist(gv);
  float x = atan(gv.x, gv.y);
  vec2 id = uv - gv;
  return vec4(x, y, id.x, id.y);
}

vec4 hexCoordsOffs(vec2 uv) {
  vec2 r = vec2(1.0, 1.73);
  vec2 h = r * 0.5;
  vec2 a = mod(uv, r) - h;
  vec2 b = mod(uv - h, r) - h;

  vec2 gv;
  if (length(a) < length(b))
    gv = a;
  else
    gv = b;

  float y = 0.5 - hexDist((gv - vec2(0.0, 0.5)));
  y = abs(y + 0.25);
  float x = atan(gv.x, gv.y);
  
  vec2 id = uv - gv;
  return vec4(gv, id.x, id.y);
}

vec3 Truchet(vec2 uv, vec2 id, float width, vec2 seed, out float rotations) {
  // Random Rotation
  float h = hash(id + seed);
  h *= 3.0;
  h = floor(h);
  uv = rot(uv, (h * (TAU / 3.0)));
  
  vec2 offs = vec2(0.400, 0.7);
  float a = length(uv + offs);
  float b = length(uv - offs);
  vec2 cUv = uv + offs;
  float aa = atan(cUv.x, cUv.y);
  cUv = uv - offs;
  float bb = atan(cUv.x, cUv.y);
  
  float c = smoothstep(0.70001 + width, 0.7 + width, a); 
  c -= smoothstep(0.70001 - width, 0.7 - width, a); 
  
  float d = smoothstep(0.70001 + width, 0.7 + width, b); 
  d -= smoothstep(0.70001 - width, 0.7 - width, b); 
  
  float l1 = length(uv.x - uv.y * 0.585); // Line gradient
  float w = width * 1.25;
  float l = smoothstep(w, w - 0.01, l1); // Line mask

  float mask = (c + d + l);
  
  float s = length((uv.x + (width * 0.585)) - (uv.y + (width * 0.585)) * 0.585);
  
  float subMask = clamp(l - c - d, 0.0, 1.0);
  float x = (c + d + subMask) * length(uv);

  float y = (
    (((1.0 - abs((a - (0.705 - (w/2.0))) / w - 0.5))) * c) + // Bottom 
    (((1.0 - abs((b - (0.705 - (w/2.0))) / w - 0.5))) * d) + // Top
    clamp(min(l, subMask + w) * (1.0 - (s / w)), 0.0, 1.0)   // Straight line
  );
  
  float m = min(mask, (subMask + c + d));

  if (mod(id.x, 2.0) == 0.0) x = 0.5 - x / m;
  
  rotations = h;
  vec3 tUv = vec3(x, y, m);
  tUv = clamp(tUv, 0.0, 1.0);
  return tUv;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  // Get audio data from microphone (iChannel0)
  float freqs[4];
  freqs[0] = texture(iChannel0, vec2(0.01, 0.25)).x;
  freqs[1] = texture(iChannel0, vec2(0.07, 0.25)).x;
  freqs[2] = texture(iChannel0, vec2(0.15, 0.25)).x;
  freqs[3] = texture(iChannel0, vec2(0.30, 0.25)).x;
  float avgFreq = (freqs[0] + freqs[1] + freqs[2] + freqs[3]) / 4.0;
  
  float time = iTime * 0.25;
  time += 800.0;
  vec2 uv = (fragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;
  uv = rot(uv, sin((time + (1.0 - freqs[0]) * 0.1) * TAU) * 0.5 + 0.5);
  
  vec4 col = vec4(0.0);
  vec2 uv1 = uv;
  uv *= max(ZOOM, 1.1) + sin(time) * 5.0;
  uv += vec2(time * 7.0 + (freqs[1]));
  
  uv -= (iMouse.xy / iResolution.xy) * 15.0;
  
  vec4 uvid = hexCoords(uv);
  vec4 uvidOf = hexCoordsOffs(uv);
  vec4 uvidOf1 = hexCoordsOffs(uv - 0.25);

  vec2 id = uvidOf.zw;
  vec2 id1 = uvidOf1.zw;

  vec2 huv = uvidOf.xy;
  vec2 huv1 = uvidOf1.xy;

  #ifdef Shrink
    float f = (avgFreq - freqs[0]) * freqs[1];
    float l = length((((fragCoord.xy / iResolution.y) - vec2(0.9, 0.5)) / freqs[1]));
    vec2 res = ((fragCoord.xy / iResolution.y) - vec2(0.5, 0.5)) * f;
    res = rot(res, (f + time) * PI);
    float atanUv = atan(res.x + l, res.y + l);
    
    float width = 1.0 - l / atanUv;
    width = clamp(
      fract(width + sin((mix(
        (uv1.x / freqs[1]) + (uv1.y * freqs[2]), 
        uv1.y * uv1.x, 
        cos(time * 0.1) * 1.5 + 1.5
      ))) * 1.5 * freqs[1]) * length((1.0 - uv1) * 0.1),
      0.3 - freqs[1], // mix
      0.1
    );
  #else
    float width = 0.07;
  #endif
  
  float r1, r2, r3;
  vec3 tUv = Truchet(huv, id, width, vec2(0.1, 0.7), r1);
  
  vec4 htUv = hexCoordsOffs(fract(tUv.xy * 0.1));
  
  vec3 tUv1 = Truchet(huv1, id1, width * 0.75, vec2(0.4, 0.2), r2);
  
  float mask = tUv.z;
  
  vec3 hUv = Truchet(htUv.xy, htUv.zw, width, vec2(1.1, 4.2), r3);
  
  // Replace texture lookups with our procedural patterns
  vec4 hTruchet = patternA(hUv.xy * 2.0 + vec2(sin(time), cos(time * 0.7)));
  
  vec4 truchet1 = patternA(tUv.xy * 1.5 + vec2(-time, 0.0)) * mask * tUv.y;
  
  vec4 truchet2 = patternB(tUv1.xy * 1.2 + vec2(-time, 0.0)) * tUv1.z * tUv1.y;
  
  // Audio reactive color modifications
  col += mix(truchet1, truchet2, (tUv.x + (tUv1.y - (tUv.x))) * 0.5);
  col = mix(col * hTruchet, col, sin(freqs[1]) * 0.5 + 0.5);
  col /= vec4(hUv * freqs[0], 1.0);
  
  // Time-based color cycling
  vec4 randCol = vec4(
    sin(time * 0.25) * 0.5 + 0.5,
    sin(time * 0.5) * 0.5 + 0.5,
    sin(time) * 0.5 + 0.5,
    1.0
  );
  
  // Make colors more reactive to audio
  randCol *= mix(1.0, avgFreq * 3.0, 0.3);
  
  col /= randCol;
  col -= randCol * (1.0 - avgFreq);
  
  // Audio reactive glow
  float glow = smoothstep(0.1, 0.7, freqs[1] * 2.0);
  col += vec4(glow * vec3(1.0, 0.7, 0.3) * mask, 0.0);
  
  // Audio reactive pulsing
  float pulse = sin(time * 4.0) * freqs[0] * 2.0;
  col *= 1.0 + pulse * 0.2;
  
  col = clamp(col, 0.0, 1.0);
  fragColor = vec4(col);
}`,
"Squeeze": `/*

@lsdlive
CC-BY-NC-SA

This is my part for "Mist" by Ohno, a 4 kilobytes demo released at the Cookie 2018.

pouet: http://www.pouet.net/prod.php?which=79350
youtube: https://www.youtube.com/watch?v=UUtU3WVB144

Code/Graphics: Flopine
Code/Graphics: Lsdlive
Music: Triace from Desire

Part1 from Flopine here: https://www.shadertoy.com/view/tdBGWD

Information about my process for making this demo here:
https://twitter.com/lsdlive/status/1090627411379716096

*/

float time = 0.;

float random(vec2 uv) {
	return fract(sin(dot(uv, vec2(12.2544, 35.1571))) * 5418.548416);
}

mat2 r2d(float a) {
	float c = cos(a), s = sin(a);
	// Explained here why you still get an anti-clockwise rotation with this matrix:
	// https://www.shadertoy.com/view/wdB3DW
	return mat2(c, s, -s, c);
}

vec3 re(vec3 p, float d) {
	return mod(p - d * .5, d) - d * .5;
}

void amod2(inout vec2 p, float d) {
	// should be atan(p.y, p.x) but I had this function for a while
	// and putting parameters like this add a PI/6 rotation.
	float a = re(vec3(atan(p.x, p.y)), d).x; 
	p = vec2(cos(a), sin(a)) * length(p);
}

void mo(inout vec2 p, vec2 d) {
	p = abs(p) - d;
	if (p.y > p.x)p = p.yx;
}

vec3 get_cam(vec3 ro, vec3 ta, vec2 uv) {
	vec3 fwd = normalize(ta - ro);
	vec3 right = normalize(cross(fwd, vec3(0, 1, 0)));

	//vec3 right = normalize(vec3(-fwd.z, 0, fwd.x));
	return normalize(fwd + right * uv.x + cross(right, fwd) * uv.y);
}

// signed cube
// https://iquilezles.org/articles/distfunctions
float cube(vec3 p, vec3 b) {
	b = abs(p) - b;
	return min(max(b.x, max(b.y, b.z)), 0.) + length(max(b, 0.));
}

// iq's signed cross sc() - https://iquilezles.org/articles/menger
float sc(vec3 p, float d) {
	p = abs(p);
	p = max(p, p.yzx);
	return min(p.x, min(p.y, p.z)) - d;
}


////////////////////////// SHADER LSDLIVE //////////////////////////

float prim(vec3 p) {

	p.xy *= r2d(3.14 * .5 + p.z * .1); // .1

	amod2(p.xy, 6.28 / 3.); // 3.
	p.x = abs(p.x) - 9.; // 9.

	p.xy *= r2d(p.z * .2); // .2

	amod2(p.xy, 6.28 /
		mix(
			mix(10., 5., smoothstep(59.5, 61.5, time)), // T4
			3.,
			smoothstep(77.5, 77.75, time)) // T8
	); // 3.
	mo(p.xy, vec2(2.)); // 2.

	p.x = abs(p.x) - .6; // .6
	return length(p.xy) - .2;//- smoothstep(80., 87., time)*(.5+.5*sin(time)); // .2
}

float g = 0.; // glow
float de(vec3 p) {

	if (time > 109.2) {
		mo(p.xy, vec2(.2));
		p.x -= 10.;
	}

	if (time > 101.4) {
		p.xy *= r2d(time*.2);
	}

	if (time > 106.5) {
		mo(p.xy, vec2(5. + sin(time)*3.*cos(time*.5), 0.));
	}

	if (time > 104.) {
		amod2(p.xy, 6.28 / 3.);
		p.x += 5.;
	}

	if (time > 101.4) {
		mo(p.xy, vec2(2. + sin(time)*3.*cos(time*.5), 0.));
	}

	p.xy *= r2d(time * .05); // .05

	p.xy *= r2d(p.z *
		mix(.05, .002, step(89.5, time)) // P2 - T11
	); // .05 & .002

	p.x += sin(time) * smoothstep(77., 82., time);

	amod2(p.xy, 6.28 /
		mix(
			mix(1., 2., smoothstep(63.5, 68.5, time)), // T6
			5.,
			smoothstep(72., 73.5, time)) // T7
	); // 5.
	p.x -= 21.; // 21.

	vec3 q = p;

	p.xy *= r2d(p.z * .1); // .1

	amod2(p.xy, 6.28 / 3.); // 3.
	p.x = abs(p.x) -
		mix(20., 5., smoothstep(49.5, 55., time)) // T2
		; // 5.

	p.xy *= r2d(p.z *
		mix(1., .2, smoothstep(77.5, 77.75, time)) // T8b
	); // .2

	p.z = re(p.zzz, 3.).x; // 3.

	p.x = abs(p.x);
	amod2(p.xy, 6.28 /
		mix(6., 3., smoothstep(77.75, 78.5, time)) // T10
	); // 3.
	float sc1 = sc(p,
		mix(8., 1., smoothstep(45.5, 51., time)) // T1
	); // 1.

	amod2(p.xz, 6.28 /
		mix(3., 8., smoothstep(61.5, 65.5, time)) // T5
	); // 8.
	mo(p.xz, vec2(.1)); // .1

	p.x = abs(p.x) - 1.;// 1.

	float d = cube(p, vec3(.2, 10, 1)); // fractal primitive: cube substracted by a signed cross
	d = max(d, -sc1) -
		mix(.01, 2., smoothstep(56., 58.5, time)) // T3
		; // 2.


	g += .006 / (.01 + d * d); // first layer of glow

	d = min(d, prim(q)); // add twisted cylinders

	g += .004 / (.013 + d * d); // second layer of glow (after the union of two geometries)

	return d;
}


////////////////////////// RAYMARCHING FUNCTIONS //////////////////////////


vec3 raymarch_lsdlive(vec3 ro, vec3 rd, vec2 uv) {
	vec3 p;
	float t = 0., ri;

	float dither = random(uv);

	for (float i = 0.; i < 1.; i += .02) {// 50 iterations to keep it "fast"
		ri = i;
		p = ro + rd * t;
		float d = de(p);
		d *= 1. + dither * .05; // avoid banding & add a nice "artistic" little noise to the rendering (leon gave us this trick)
		d = max(abs(d), .002); // phantom mode trick from aiekick https://www.shadertoy.com/view/MtScWW
		t += d * .5;
	}

	// Shading: uv, iteration & glow:
	vec3 c = mix(vec3(.9, .8, .6), vec3(.1, .1, .2), length(uv) + ri);
	c.r += sin(p.z * .1) * .2;
	c += g * .035; // glow trick from balkhan https://www.shadertoy.com/view/4t2yW1

	return c;
}

// borrowed from (mmerchante) : https://www.shadertoy.com/view/MltcWs
void glitch(inout vec2 uv, float start_time_stamp, float end_time_stamp)
{
	int offset = int(floor(time)*2.) + int((uv.x + uv.y) * 8.0);
	float res = mix(10., 100.0, random(vec2(offset)));

	// glitch pixellate
	if (time > start_time_stamp && time <= end_time_stamp) uv = floor(uv * res) / res;

	int seedX = int(gl_FragCoord.x + time) / 32;
	int seedY = int(gl_FragCoord.y + time) / 32;
	int seed = mod(time, 2.) > 1. ? seedX : seedY;


	// glitch splitter
	uv.x += (random(vec2(seed)) * 2.0 - 1.0)
		* step(random(vec2(seed)), pow(sin(time * 4.), 7.0))
		* random(vec2(seed))
		* step(start_time_stamp, time)
		* (1. - step(end_time_stamp, time));
}

////////////////////////// MAIN FUNCTION //////////////////////////

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
	vec2 q = fragCoord.xy / iResolution.xy;
    vec2 uv = (q - .5) * iResolution.xx / iResolution.yx;

	/* just code for the shadertoy port */
	time = mod(iTime, 43. + 10.4);
	time = time + 45.;
	if (time > 88. && time <= 98.6) // 98.
		time += 10.6;


	// added glitch
	glitch(uv, 0., 2.);

	glitch(uv, 98., 99.);
	// lsdlive 2nd part
	glitch(uv, 100.5, 101.5);
	glitch(uv, 103., 104.);
	glitch(uv, 105.5, 106.5);

	vec3 lsd_ro = vec3(0, 0, -4. + time * 8.);
	vec3 lsd_target = vec3(0., 0., time * 8.);
	vec3 lsd_cam = get_cam(lsd_ro, lsd_target, uv);

	vec3 col = vec3(0.);

	if (time > 45. && time <= 88.) // 43 seconds
		col = raymarch_lsdlive(lsd_ro, lsd_cam, uv);

	if (time > 98.6 && time <= 109.) // 10.4 seconds
		col = raymarch_lsdlive(lsd_ro, lsd_cam, uv);


	// vignetting (iq)
	col *= 0.5 + 0.5*pow(16.0*q.x*q.y*(1.0 - q.x)*(1.0 - q.y), 0.25);

	// fading out - end of the demo
	//col *= 1. - smoothstep(120., 125., time);

	fragColor = vec4(col, 1.);
}`,
"Choc Candy": `#define PI 3.14159265

float orenNayarDiffuse(
  vec3 lightDirection,
  vec3 viewDirection,
  vec3 surfaceNormal,
  float roughness,
  float albedo) {
  
  float LdotV = dot(lightDirection, viewDirection);
  float NdotL = dot(lightDirection, surfaceNormal);
  float NdotV = dot(surfaceNormal, viewDirection);

  float s = LdotV - NdotL * NdotV;
  float t = mix(1.0, max(NdotL, NdotV), step(0.0, s));

  float sigma2 = roughness * roughness;
  float A = 1.0 + sigma2 * (albedo / (sigma2 + 0.13) + 0.5 / (sigma2 + 0.33));
  float B = 0.45 * sigma2 / (sigma2 + 0.09);

  return albedo * max(0.0, NdotL) * (A + B * s / t) / PI;
}

float gaussianSpecular(
  vec3 lightDirection,
  vec3 viewDirection,
  vec3 surfaceNormal,
  float shininess) {
  vec3 H = normalize(lightDirection + viewDirection);
  float theta = acos(dot(H, surfaceNormal));
  float w = theta / shininess;
  return exp(-w*w);
}

float fogFactorExp2(
  const float dist,
  const float density
) {
  const float LOG2 = -1.442695;
  float d = density * dist;
  return 1.0 - clamp(exp2(d * d * LOG2), 0.0, 1.0);
}

//
// Description : Array and textureless GLSL 2D/3D/4D simplex
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//

vec4 mod289(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0; }

float mod289(float x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0; }

vec4 permute(vec4 x) {
     return mod289(((x*34.0)+1.0)*x);
}

float permute(float x) {
     return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

float taylorInvSqrt(float r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec4 grad4(float j, vec4 ip)
  {
  const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
  vec4 p,s;

  p.xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
  p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
  s = vec4(lessThan(p, vec4(0.0)));
  p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www;

  return p;
  }

// (sqrt(5) - 1)/4 = F4, used once below
#define F4 0.309016994374947451

float snoise(vec4 v)
  {
  const vec4  C = vec4( 0.138196601125011,  // (5 - sqrt(5))/20  G4
                        0.276393202250021,  // 2 * G4
                        0.414589803375032,  // 3 * G4
                       -0.447213595499958); // -1 + 4 * G4

// First corner
  vec4 i  = floor(v + dot(v, vec4(F4)) );
  vec4 x0 = v -   i + dot(i, C.xxxx);

// Other corners

// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
  vec4 i0;
  vec3 isX = step( x0.yzw, x0.xxx );
  vec3 isYZ = step( x0.zww, x0.yyz );
//  i0.x = dot( isX, vec3( 1.0 ) );
  i0.x = isX.x + isX.y + isX.z;
  i0.yzw = 1.0 - isX;
//  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
  i0.y += isYZ.x + isYZ.y;
  i0.zw += 1.0 - isYZ.xy;
  i0.z += isYZ.z;
  i0.w += 1.0 - isYZ.z;

  // i0 now contains the unique values 0,1,2,3 in each channel
  vec4 i3 = clamp( i0, 0.0, 1.0 );
  vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );
  vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );

  //  x0 = x0 - 0.0 + 0.0 * C.xxxx
  //  x1 = x0 - i1  + 1.0 * C.xxxx
  //  x2 = x0 - i2  + 2.0 * C.xxxx
  //  x3 = x0 - i3  + 3.0 * C.xxxx
  //  x4 = x0 - 1.0 + 4.0 * C.xxxx
  vec4 x1 = x0 - i1 + C.xxxx;
  vec4 x2 = x0 - i2 + C.yyyy;
  vec4 x3 = x0 - i3 + C.zzzz;
  vec4 x4 = x0 + C.wwww;

// Permutations
  i = mod289(i);
  float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);
  vec4 j1 = permute( permute( permute( permute (
             i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
           + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
           + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
           + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));

// Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
// 7*7*6 = 294, which is close to the ring size 17*17 = 289.
  vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;

  vec4 p0 = grad4(j0,   ip);
  vec4 p1 = grad4(j1.x, ip);
  vec4 p2 = grad4(j1.y, ip);
  vec4 p3 = grad4(j1.z, ip);
  vec4 p4 = grad4(j1.w, ip);

// Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  p4 *= taylorInvSqrt(dot(p4,p4));

// Mix contributions from the five corners
  vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
  vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);
  m0 = m0 * m0;
  m1 = m1 * m1;
  return 49.0 * ( dot(m0*m0, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 )))
               + dot(m1*m1, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;

}



//------------------------------------------------------------------------
// Camera
//
// Move the camera. In this case it's using time and the mouse position
// to orbitate the camera around the origin of the world (0,0,0), where
// the yellow sphere is.
//------------------------------------------------------------------------
void doCamera( out vec3 camPos, out vec3 camTar, in float time, in float mouseX )
{
    float an = 10.0*mouseX+4.5;
	camPos = vec3(3.5*sin(an),1.0,3.5*cos(an));
    camTar = vec3(0.0,0.0,0.0);
}


//------------------------------------------------------------------------
// Background 
//
// The background color. In this case it's just a black color.
//------------------------------------------------------------------------
vec3 doBackground( void )
{
    return vec3(0.003,0.003,0.005);
}
    
//------------------------------------------------------------------------
// Modelling 
//
// Defines the shapes (a sphere in this case) through a distance field, in
// this case it's a sphere of radius 1.
//------------------------------------------------------------------------
float doModel( vec3 p )
{
    float r = texture(iChannel0, vec2(0.8, 0.)).r + 0.5;
    float n = max(0., texture(iChannel0, vec2(0.05, 0.)).r * 3.5 - 1.);
    
    n = n * exp(snoise(vec4(p * 2.1, iTime * 2.3)));
    
    return length(p) - (1. + n * 0.05) * .9;
}

//------------------------------------------------------------------------
// Material 
//
// Defines the material (colors, shading, pattern, texturing) of the model
// at every point based on its position and normal. In this case, it simply
// returns a constant yellow color.
//------------------------------------------------------------------------
vec3 doMaterial( in vec3 pos, in vec3 nor )
{
    return vec3(0.125,0.1,0.2)+(vec3(.6,0.9,.4)*3.*clamp(length(pos)-0.94,0.,1.));
}

//------------------------------------------------------------------------
// Lighting
//------------------------------------------------------------------------
float calcSoftshadow( in vec3 ro, in vec3 rd );

vec3 doLighting( in vec3 pos, in vec3 nor, in vec3 rd, in float dis, in vec3 mal )
{
    vec3 lin = vec3(0.0);

    // key light
    //-----------------------------
    vec3  view = normalize(-rd);
    vec3  lig1 = normalize(vec3(1.0,0.7,0.9));
    vec3  lig2 = normalize(vec3(1.0,0.9,0.9)*-1.);
    
    float spc1 = gaussianSpecular(lig1, view, nor, 0.95)*0.5;
    float dif1 = max(0., orenNayarDiffuse(lig1, view, nor, -20.1, 1.0));
    float sha1 = 0.0; if( dif1>0.01 ) sha1=calcSoftshadow( pos+0.01*nor, lig1 );
    vec3  col1 = vec3(2.,4.2,4.);
    lin += col1*spc1+dif1*col1*sha1;
    
    float spc2 = gaussianSpecular(lig2, view, nor, 0.95);
    float dif2 = max(0., orenNayarDiffuse(lig2, view, nor, -20.1, 1.0));
    float sha2 = 0.0; if( dif2>0.01 ) sha2=calcSoftshadow( pos+0.01*nor, lig2 );
    vec3  col2 = vec3(2.00,0.05,0.15);
    lin += col2*spc2+dif2*col2*sha1;

    // ambient light
    //-----------------------------
    lin += vec3(0.05);

    
    // surface-light interacion
    //-----------------------------
    vec3 col = mal*lin;

    return col;
}

float calcIntersection( in vec3 ro, in vec3 rd )
{
	const float maxd = 20.0;           // max trace distance
	const float precis = 0.001;        // precission of the intersection
    float h = precis*2.0;
    float t = 0.0;
	float res = -1.0;
    for( int i=0; i<90; i++ )          // max number of raymarching iterations is 90
    {
        if( h<precis||t>maxd ) break;
	    h = doModel( ro+rd*t );
        t += h;
    }

    if( t<maxd ) res = t;
    return res;
}

vec3 calcNormal( in vec3 pos )
{
    const float eps = 0.002;             // precision of the normal computation

    const vec3 v1 = vec3( 1.0,-1.0,-1.0);
    const vec3 v2 = vec3(-1.0,-1.0, 1.0);
    const vec3 v3 = vec3(-1.0, 1.0,-1.0);
    const vec3 v4 = vec3( 1.0, 1.0, 1.0);

	return normalize( v1*doModel( pos + v1*eps ) + 
					  v2*doModel( pos + v2*eps ) + 
					  v3*doModel( pos + v3*eps ) + 
					  v4*doModel( pos + v4*eps ) );
}

float calcSoftshadow( in vec3 ro, in vec3 rd )
{
    float res = 1.0;
    float t = 0.0001;                 // selfintersection avoidance distance
	float h = 1.0;
    for( int i=0; i<5; i++ )         // 40 is the max numnber of raymarching steps
    {
        h = doModel(ro + rd*t);
        res = min( res, 4.0*h/t );   // 64 is the hardness of the shadows
		t += clamp( h, 0.02, 2.0 );   // limit the max and min stepping distances
    }
    return clamp(res,0.0,1.0);
}

mat3 calcLookAtMatrix( in vec3 ro, in vec3 ta, in float roll )
{
    vec3 ww = normalize( ta - ro );
    vec3 uu = normalize( cross(ww,vec3(sin(roll),cos(roll),0.0) ) );
    vec3 vv = normalize( cross(uu,ww));
    return mat3( uu, vv, ww );
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 p = (-iResolution.xy + 2.0*fragCoord.xy)/iResolution.y;
    vec2 m = iMouse.xy/iResolution.xy;

    //-----------------------------------------------------
    // camera
    //-----------------------------------------------------
    
    // camera movement
    vec3 ro, ta;
    doCamera( ro, ta, iTime, m.x );

    // camera matrix
    mat3 camMat = calcLookAtMatrix( ro, ta, 0.0 );  // 0.0 is the camera roll
    
	// create view ray
	vec3 rd = normalize( camMat * vec3(p.xy,2.0) ); // 2.0 is the lens length

    //-----------------------------------------------------
	// render
    //-----------------------------------------------------

	vec3 col = doBackground();

	// raymarch
    float t = calcIntersection( ro, rd );
    if( t>-0.5 )
    {
        // geometry
        vec3 pos = ro + t*rd;
        vec3 nor = calcNormal(pos);

        // materials
        vec3 mal = doMaterial( pos, nor );
        vec3 lcl = doLighting( pos, nor, rd, t, mal );

        col = mix(lcl, col, fogFactorExp2(t, 0.1));
	}

	//-----------------------------------------------------
	// postprocessing
    //-----------------------------------------------------
    // gamma
	col = pow( clamp(col,0.0,1.0), vec3(0.4545) );
    col += dot(p,p*0.035);
    col.r = smoothstep(0.1,1.1,col.r);
    col.g = pow(col.g, 1.1);
	   
    fragColor = vec4( col, 1.0 );
}`,
    "Tessellation Tiles": `
const int K = 2;
const float H = sqrt( 3. );
const mat2 MAT = mat2( 2., 0., -1., H ) * .5;
const vec2 CORNER = vec2( float( K ) - .5 + H * .5, float( K ) + .5 + H * .5 );
const vec2 REP_U = vec2( float( K ) * 2. + H, 1. );
const vec2 REP_V = vec2( REP_U.y, -REP_U.x );

const vec3 SQUARE0_COLOR = vec3( 0., .5, 1. );
const vec3 SQUARE1_COLOR = vec3( 1., 1., 1. );
const vec3 TRIANGLE_COLOR = vec3( 1., .9, .1 );
const vec3 VERTEX_COLOR = vec3( .9, .1, .1 );

mat2 rot30() { return mat2( H, -1, 1, H ) * .5; }
mat2 rot120() { return mat2( -1, -H, H, -1 ) * .5; }

float sdBox( in vec2 p, in vec2 b )
{
    vec2 d = abs(p) - b;
    return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

float sdEquilateralTriangle( in vec2 p, in float r ) {
    const float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r/k;
    if( p.x+k*p.y > 0.0 ) p = vec2(p.x-k*p.y, -k*p.x-p.y)/2.0;
    p.x -= clamp( p.x, -2.0*r, 0.0 );
    return -length(p) * sign(p.y);
}

const int SQUARE0 = 0;
const int SQUARE1 = 1;
const int TRIANGLE = 2;

vec2 mapToBasicShape( vec2 p, out int shape )
{
    p -= round( dot( p, REP_U ) / dot( REP_U, REP_U ) ) * REP_U;
    p -= round( dot( p, REP_V ) / dot( REP_V, REP_V ) ) * REP_V;
    if ( p.x < 0. ) p = -p;
    if ( p.y < 0. ) p = vec2( -p.y, p.x );

    shape = SQUARE0;
    if ( p.x < float( K ) && p.y < float( K ) )
        return (p - float(K+1)) - round( (p - float(K+1)) * .5 ) * 2.;

    if ( p.x > p.y ) p = CORNER + vec2( p.y - CORNER.y, CORNER.x - p.x );

    shape = SQUARE1;
    vec2 sqP = abs( rot30() * (p - CORNER) );
    if ( sqP.x <= 1. && sqP.y <= 1. )
        return sqP;

    shape = TRIANGLE;
    vec2 q = inverse( MAT ) * ( p - vec2( float( K ) ) );
    q.x = fract( q.x * .5 ) * 2.;
    if ( q.y > q.x )
        q = 2. - q;
    return MAT * q + vec2( -1, -H/3. );
}

float distToColor( float d, float margin )
{
    return 1. - smoothstep( -margin, 0., d );
}

void checkVertexBeyond( vec2 p0, vec2 p2, inout vec2 delta0 )
{
    if ( p0.y > float( K ) && p0.x < float( K ) )
        delta0 = p2 - p0 + vec2( sign( p0.x-p2.x ), H );
}

vec2 nearestVertex( vec2 origP )
{
    vec2 p = origP;
    vec2 cell = vec2( round( dot( p, REP_U ) / dot( REP_U, REP_U ) )
                    , round( dot( p, REP_V ) / dot( REP_V, REP_V ) ) );
    vec2 p0 = p - cell.x * REP_U - cell.y * REP_V;
    vec2 p1 = ( round( ( p0 - float( K ) ) * .5 ) ) * 2. + float( K );
    vec2 p2 = clamp( p1, -float( K ), float( K ) );
    vec2 p3 = p2 + cell.x * REP_U + cell.y * REP_V;

    vec2 delta0 = p2 - p0;

    for ( int i = 0;i < 4; i++ )
    {
      p0 = vec2( -p0.y, p0.x );
      p2 = vec2( -p2.y, p2.x );
      delta0 = vec2( -delta0.y, delta0.x );
      checkVertexBeyond( p0, p2, delta0 );
    }

    return distance( origP, p3 ) < length( delta0 ) ? p3 : origP + delta0;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = (fragCoord*2.-iResolution.xy)/iResolution.y;
    float scaling = .06 + sin( iTime * .2 ) * .02;
    float margin = 30. / iResolution.y * scaling;
    uv /= scaling;

    uv.x += cos( iTime * .17 ) * 15.;
    uv.y += cos( iTime * .15 ) * 10.;

    vec2 nearestVertexPos = nearestVertex( uv );

    int shape = -1;
    vec2 p = mapToBasicShape( uv, shape );

    float vertexSize = pow( .4 + sin( -distance( nearestVertexPos, vec2( -15, 10 ) ) * .1 + iTime ) * .4, 2. );

    if ( shape == SQUARE0 || shape == SQUARE1 )
    {
        p = abs( p );
        vec3 shapeColor = shape == SQUARE0 ? SQUARE0_COLOR : SQUARE1_COLOR;
        float dist = sdBox( p, vec2( 1. ) );

        float distToVertex = (2. - (p.x + p.y)) * sqrt( .5 ) - vertexSize;
        fragColor = vec4( distToColor( max( dist, -distToVertex ), margin ) * shapeColor, 1. );
        if ( distToVertex < 0. )
            fragColor = vec4( distToColor( distToVertex, margin ) * VERTEX_COLOR, 1. );
    }
    if ( shape == TRIANGLE )
    {
        float dist = sdEquilateralTriangle( p, 1. );

        float distToVertex = (H*2./3. - max( p.y, max( (rot120() * p).y, (rot120() * rot120() * p).y ) )) / sqrt( 1.5 ) - vertexSize;

        fragColor = vec4( distToColor( max( dist, -distToVertex ), margin ) * TRIANGLE_COLOR, 1. );
        if ( distToVertex < 0. )
            fragColor = vec4( distToColor( distToVertex, margin ) * VERTEX_COLOR, 1. );
    }
}
`
};
