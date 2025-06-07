
// Sample Shadertoy shaders to quickly test
export const SHADERS = {
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

    O = vec4(1.0 - burn, 0, exp(-t), 1.0);
}`,
"Fractal Leaves": `vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

vec3 hsv(float h, float s, float v) {
	vec4 K = vec4(1.0, cos(iTime) / 3.0, 1.0 / 3.0, 3.0);
	vec3 p = abs(fract(vec3(h) + K.xyz) * 6.0 - K.www);
	return v * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 surfacePosition = 0.5 * (2.0 * fragCoord - iResolution.xy) / min(iResolution.x, iResolution.y);
    
    float mouseY = iMouse.y == 0.0 ? 0.0 : iMouse.y / iResolution.y - 0.5;
	float zoom = exp(mouseY * 8.0);
    
    //vec2 p = zoom * 2.0 * surfacePosition - vec2(0.7, 0.0);
	vec2 p = zoom * 0.016 * surfacePosition - vec2(0.805, -0.176);
	//vec2 p = zoom * 0.001 * surfacePosition - vec2(1.924, 0.0);

	vec2 z = p;
	vec2 c = p;
	vec2 dz = vec2(1.0, 0.0);
	float it = 0.0;
	for(float i = 0.0; i<1024.0; i+=1.0) {
		dz = 2.0 * cmul(z, dz) + vec2(1.0, 0.0);
		z = cmul(z, z) + c;

        float a = sin(iTime * 1.5 + i * 2.0) * 0.3 + i * 1.3;
		vec2 t = mat2(cos(a), sin(a), -sin(a), cos(a)) * z;
		if(abs(t.x) > 2.0 && abs(t.y) > 2.0) { it = i; break; }
	}

	if (it == 0.0) {
		fragColor = vec4(vec3(0.0), 1.0);
	} else {
		float z2 = z.x * z.x + z.y * z.y;
		float dist = log(z2) * sqrt(z2) / length(dz);
		float r = sqrt(z2);

		float pixelsize = fwidth(p.x);
		float diagonal = length(iResolution.xy);
		float glowsize = pixelsize * diagonal / 400.0;
		float shadowsize = pixelsize * diagonal / 80.0;

		float fadeout = 0.0, glow = 0.0;
		if(dist < pixelsize) {
			fadeout = dist / pixelsize;
			glow = 1.0;
 		} else {
			fadeout = min(shadowsize / (dist + shadowsize - pixelsize) + 1.0 / (r + 1.0), 1.0);
			glow = min(glowsize / (dist + glowsize - pixelsize), 1.0);
		}

		fragColor = vec4(hsv(
			it / 32.0 + 0.4,
			1.0 - glow,
			fadeout
		), 1.0);		
	}
}
`,
"Fractal Leaves w Sound":`vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

// Superior palette function for gorgeous color blending
vec3 palette(float t, float audio) {
    // Create dynamic palette coefficients that respond to audio
    float audioShift = audio * 0.4; // Audio influence
    
    // Color palette parameters - these create the "mood" of your visualization
    vec3 a = vec3(0.5, 0.5, 0.5);  // Brightness and contrast
    vec3 b = vec3(0.5, 0.5, 0.5);  // Color balance
    vec3 c = vec3(1.0, 1.0, 1.0);  // Phase shifts (3.0, 2.0, 1.0 for rainbow)
    vec3 d = vec3(0.30, 0.20, 0.020);  // Color density
    
    // Make palette dynamic with time
    float timeScale = 0.1; // How fast colors cycle
    
    // Shift colors based on audio for bass reactivity
    c.x += sin(audioShift * 2.0) * 0.2; // Red phase shift
    c.y += audioShift * 0.3;           // Green phase shift
    c.z -= audioShift * 0.1;           // Blue phase shift
    
    // Add richness to color mixing
    a.x += sin(iTime * timeScale) * 0.1;
    b.y += cos(iTime * timeScale * 0.7) * 0.1;
    
    // The magic formula that creates beautiful color gradients
    return a + b * cos(6.28318 * (c * t + d + iTime * timeScale));
}

// A smoother glow function
float smoothGlow(float dist, float radius, float intensity) {
    return intensity * exp(-dist * dist / (radius * radius));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 surfacePosition = 0.5 * (2.0 * fragCoord - iResolution.xy) / min(iResolution.x, iResolution.y);
    
    // --- AUDIO SAMPLING WITH FREQUENCY SEPARATION ---
    const int N = 128;
    float bassSum = 0.0;
    float midSum = 0.0;
    float highSum = 0.0;
    
    // Frequency-based audio sampling
    for(int i = 0; i < N; ++i) {
        float audioValue = texelFetch(iChannel0, ivec2(i, 0), 0).r;
        if(i < 32) { // Bass frequencies
            bassSum += audioValue;
        } else if(i < 96) { // Mid frequencies
            midSum += audioValue;
        } else { // High frequencies
            highSum += audioValue;
        }
    }
    
    float bassAmp = bassSum / 32.0;
    float midAmp = midSum / 64.0;
    float highAmp = highSum / 32.0;
    
    // Smoother responses
    float smoothBass = mix(bassAmp, 0.5, 0.7);
    float smoothMid = mix(midAmp, 0.5, 0.6);
    float smoothHigh = mix(highAmp, 0.5, 0.5);
    
    // Calculate zoom based primarily on bass
    float dampFactor = 0.3;
    float baseZoom = 0.5;
    float maxZoomMultiplier = 2.5;
    float zoomDelta = smoothBass * dampFactor;
    float zoom = baseZoom + zoomDelta * (maxZoomMultiplier - 1.0);
    
    // --- ORBITAL CAMERA MOVEMENT ---
    float orbitSpeed = 0.05;
    float orbitRadius = 0.01;
    vec2 basePosition = vec2(0.805, -0.176);
    vec2 orbitOffset = vec2(
        cos(iTime * orbitSpeed) * orbitRadius,
        sin(iTime * orbitSpeed) * orbitRadius
    );
    
    vec2 p = zoom * 0.016 * surfacePosition - (basePosition + orbitOffset);
    vec2 z = p;
    vec2 c = p;
    vec2 dz = vec2(1.0, 0.0);
    float it = 0.0;
    
    // Fractal iteration
    for(float i = 0.0; i < 1024.0; i += 1.0) {
        dz = 2.0 * cmul(z, dz) + vec2(1.0, 0.0);
        z = cmul(z, z) + c;
        float a = sin(iTime * 1.5 + i * 2.0) * 0.3 + i * 1.3;
        vec2 t = mat2(cos(a), sin(a), -sin(a), cos(a)) * z;
        if(abs(t.x) > 2.0 && abs(t.y) > 2.0) { it = i; break; }
    }
    
    // Rendering with enhanced colors
    if (it == 0.0) {
        fragColor = vec4(vec3(0.0), 1.0);
    } else {
        float z2 = z.x * z.x + z.y * z.y;
        float dist = log(z2) * sqrt(z2) / length(dz);
        float r = sqrt(z2);
        float pixelsize = fwidth(p.x);
        float diagonal = length(iResolution.xy);
        float glowsize = pixelsize * diagonal / 400.0;
        float shadowsize = pixelsize * diagonal / 80.0;
        
        float fadeout = 0.0, glow = 0.0;
        if(dist < pixelsize) {
            fadeout = dist / pixelsize;
            glow = 1.0;
        } else {
            fadeout = min(shadowsize / (dist + shadowsize - pixelsize) + 1.0 / (r + 1.0), 1.0);
            glow = min(glowsize / (dist + glowsize - pixelsize), 1.0);
        }
        
        // Dynamic coloring based on iteration count, audio, and glow
        float colorIndex = it / 128.0; // Normalized iteration count
        
        // Create phase shifts based on different frequency bands
        colorIndex = fract(colorIndex + smoothBass * 0.2 + smoothMid * 0.1);
        
        // Get beautiful color from our palette function
        vec3 color = palette(colorIndex, smoothBass);
        
        // Apply glow effect (use mid frequencies to control glow intensity)
        float glowIntensity = 1.0 + smoothMid * 2.0;
        vec3 glowColor = palette(fract(colorIndex + 0.5), smoothHigh);
        color = mix(color, glowColor, glow * glowIntensity);
        
        // Apply fadeout and additional high-frequency modulation
        color *= fadeout;
        color += glowColor * smoothGlow(dist, glowsize * 4.0, smoothHigh * 0.3);
        
        fragColor = vec4(color, 1.0);
    }
}`,
"Fractal leaves Jumper":`vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

// Superior palette function with brightness control and complementary colors
vec3 palette(float t, float audio) {
    // Enhanced audio response curve
    float audioReactive = audio * 0.5 + pow(audio, 3.0) * 0.5;
    
    // Higher baseline brightness to prevent darkness
    vec3 a = vec3(0.65, 0.60, 0.65);  // Increased minimum brightness
    
    // More controlled amplitude for better color harmony
    vec3 b = vec3(0.35, 0.30, 0.35);  // Smaller variations for harmony
    
    // Carefully tuned frequency ratios for complementary colors
    // Using golden ratio (1.618) relationships creates natural harmony
    float phi = 1.618;
    vec3 c = vec3(1.0, 1.0/phi, 1.0/(phi*phi));
    
    // Phase shifts designed for complementary color schemes
    vec3 d = vec3(0.2, 0.4, 0.6);  // Spaced for better color distribution
    
    // Gentle time evolution that preserves color harmony
    float timeFlow = iTime * 0.05;
    d.x += timeFlow;
    d.y += timeFlow * 0.7;
    d.z += timeFlow * 0.3;
    
    // Audio influences color temperature rather than arbitrary shifts
    // This maintains color harmony while still being reactive
    float warmth = audio * 0.3;
    a += vec3(warmth, warmth*0.5, 0.0);  // Warm colors with audio
    
    // Controlled audio reactivity that preserves harmony
    b += vec3(0.05, 0.05, 0.15) * audioReactive;
    
    // Add subtle pulsing for electronic music feel without breaking harmony
    float pulse = sin(iTime * 0.75) * 0.5 + 0.5;
    b *= 1.0 + pulse * 0.2 * audioReactive;
    
    // The core palette calculation
    vec3 color = a + b * cos(6.28318 * (c * t + d));
    
    // Brightness safeguard - ensure no colors are too dark
    color = max(color, vec3(0.15, 0.15, 0.2)); 
    
    // Subtle saturation boost for vibrant but harmonious colors
    float luminance = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(luminance), color, 1.2);
    
    return color;
}

// A smoother glow function
float smoothGlow(float dist, float radius, float intensity) {
    return intensity * exp(-dist * dist / (radius * radius));
}

// Smoother zoom calculation with temporal smoothing
float getSmoothedZoom(float bassAudio) {
    // Create a smooth base zoom oscillation
    float slowOsc = sin(iTime * 0.07);
    float baseZoom = 1.0 + 0.4 * slowOsc;
    
    // Apply cubic easing to the bass response for gentler acceleration
    float bassResponse = bassAudio * bassAudio * (3.0 - 2.0 * bassAudio);
    
    // Create a memory effect by blending with delayed signals
    float delay1 = sin(iTime * 0.11 - 0.3) * 0.5 + 0.5;
    float delay2 = sin(iTime * 0.05 - 0.7) * 0.5 + 0.5;
    
    // Blend for temporal smoothness (using multiple different phases)
    float blendFactor = 0.1;
    float smoothedBass = mix(
        mix(delay1, delay2, 0.3),
        bassResponse,
        blendFactor
    );
    
    // Apply smoothstep for more organic transitions
    smoothedBass = smoothstep(0.0, 1.0, smoothedBass);
    
    // Calculate final zoom with gentler multipliers
    float zoomVariation = 1.2 + 0.5 * sin(iTime * 0.03);
    
    // Final smooth zoom value
    return baseZoom + smoothedBass * zoomVariation;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 surfacePosition = 0.5 * (2.0 * fragCoord - iResolution.xy) / min(iResolution.x, iResolution.y);
    
    // --- AUDIO SAMPLING WITH FREQUENCY SEPARATION ---
    const int N = 128;
    float bassSum = 0.0;
    float midSum = 0.0;
    float highSum = 0.0;
    
    // Frequency-based audio sampling
    for(int i = 0; i < N; ++i) {
        float audioValue = texelFetch(iChannel0, ivec2(i, 0), 0).r;
        if(i < 32) { // Bass frequencies
            bassSum += audioValue;
        } else if(i < 96) { // Mid frequencies
            midSum += audioValue;
        } else { // High frequencies
            highSum += audioValue;
        }
    }
    
    float bassAmp = bassSum / 32.0;
    float midAmp = midSum / 64.0;
    float highAmp = highSum / 32.0;
    
    // Smoother responses
    float smoothBass = mix(bassAmp, 0.5, 0.7);
    float smoothMid = mix(midAmp, 0.5, 0.6);
    float smoothHigh = mix(highAmp, 0.5, 0.5);
    
    // Calculate zoom with our new smooth function
    float zoom = getSmoothedZoom(smoothBass);
    
    // --- ORBITAL CAMERA MOVEMENT ---
    float orbitSpeed = 0.05;
    float orbitRadius = 0.01;
    vec2 basePosition = vec2(0.805, -0.176);
    vec2 orbitOffset = vec2(
        cos(iTime * orbitSpeed) * orbitRadius,
        sin(iTime * orbitSpeed) * orbitRadius
    );
    
    vec2 p = zoom * 0.016 * surfacePosition - (basePosition + orbitOffset);
    vec2 z = p;
    vec2 c = p;
    vec2 dz = vec2(1.0, 0.0);
    float it = 0.0;
    
    // Fractal iteration
    for(float i = 0.0; i < 1024.0; i += 1.0) {
        dz = 2.0 * cmul(z, dz) + vec2(1.0, 0.0);
        z = cmul(z, z) + c;
        float a = sin(iTime * 1.5 + i * 2.0) * 0.3 + i * 1.3;
        vec2 t = mat2(cos(a), sin(a), -sin(a), cos(a)) * z;
        if(abs(t.x) > 2.0 && abs(t.y) > 2.0) { it = i; break; }
    }
    
    // Rendering with enhanced colors
    if (it == 0.0) {
        fragColor = vec4(vec3(0.0), 1.0);
    } else {
        float z2 = z.x * z.x + z.y * z.y;
        float dist = log(z2) * sqrt(z2) / length(dz);
        float r = sqrt(z2);
        float pixelsize = fwidth(p.x);
        float diagonal = length(iResolution.xy);
        float glowsize = pixelsize * diagonal / 400.0;
        float shadowsize = pixelsize * diagonal / 80.0;
        
        float fadeout = 0.0, glow = 0.0;
        if(dist < pixelsize) {
            fadeout = dist / pixelsize;
            glow = 1.0;
        } else {
            fadeout = min(shadowsize / (dist + shadowsize - pixelsize) + 1.0 / (r + 1.0), 1.0);
            glow = min(glowsize / (dist + glowsize - pixelsize), 1.0);
        }
        
        // Dynamic coloring based on iteration count, audio, and glow
        float colorIndex = it / 128.0; // Normalized iteration count
        
        // Create phase shifts based on different frequency bands
        colorIndex = fract(colorIndex + smoothBass * 0.2 + smoothMid * 0.1);
        
        // Get beautiful color from our palette function
        vec3 color = palette(colorIndex, smoothBass);
        
        // Apply glow effect (use mid frequencies to control glow intensity)
        float glowIntensity = 1.0 + smoothMid * 2.0;
        vec3 glowColor = palette(fract(colorIndex + 0.5), smoothHigh);
        color = mix(color, glowColor, glow * glowIntensity);
        
        // Apply fadeout and additional high-frequency modulation
        color *= fadeout;
        color += glowColor * smoothGlow(dist, glowsize * 4.0, smoothHigh * 0.3);
        
        fragColor = vec4(color, 1.0);
    }
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
};

