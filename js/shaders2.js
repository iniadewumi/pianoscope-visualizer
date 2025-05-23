export const SHADERS2 = {
    "Blueprint (needs work)": `#define SPEED .1
#define FOV 3.

#define MAX_STEPS 80
#define EPS .001
#define RENDER_DIST 5.
#define AO_SAMPLES 4.
#define AO_RANGE 100.

#define PI 3.14159265359
#define saturate(x) clamp(x, 0., 1.)

// Precomputed globals
float _house = 0.;
float _boat = 0.;
float _spaceship = 0.;
float _atmosphere = 0.;
mat3 _kifsRot = mat3(1, 0, 0, 0, 1, 0, 0, 0, 1);
float _kifsOffset = 0.;

// Rotate 2d space with given angle
void tRotate(inout vec2 p, float angel) {
    float s = sin(angel), c = cos(angel);
    p *= mat2(c, -s, s, c);
}

// Divide 2d space into s chunks around the center
void tFan(inout vec2 p, float s) {
    float k = s / PI / 2.;
    tRotate(p, -floor((atan(p.y, p.x)) * k + .5) / k);
}

// Rectangle distance
float sdRect(vec2 p, vec2 r) {
    p = abs(p) - r;
    return min(max(p.x, p.y), 0.) + length(max(p, 0.));
}

// Box distance
float sdBox(vec3 p, vec3 r) {
    p = abs(p) - r;
    return min(max(p.x, max(p.y, p.z)), 0.) + length(max(p, 0.));
}

// Sphere distance
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

// 3d cross distance
float sdCross(vec3 p, vec3 r) {
    p = abs(p) - r;
    p.xy = p.x < p.y ? p.xy : p.yx;
    p.yz = p.y < p.z ? p.yz : p.zy;
    p.xy = p.x < p.y ? p.xy : p.yx;
    return length(min(p.yz, 0.)) - max(p.y, 0.);
}

// Union
float opU(float a, float b) {
    return min(a, b);
}

// Intersection
float opI(float a, float b) {
    return max(a, b);
}

// Subtraction
float opS(float a, float b) {
    return max(a, -b);
}

// Smooth union
float opSU(float a, float b, float k) {
    float h = clamp(.5 + .5 * (b - a) / k, 0., 1.);
    return mix(b, a, h) - k * h * (1. - h);
}

// House distance
float sdHouse(vec3 p) {
    p.y += .075;
    vec3 boxDim = vec3(.2, .15, .2);

    // Add the walls
    float d = sdBox(p, boxDim);

    // Add the windows
    vec3 q = abs(p);
    vec3 windSize = vec3(.04, .04, .06);
    q -= windSize + vec3(.005);
    d = opI(d, opU(sdCross(q, windSize), .11 - abs(p.y)));

    // Add the roof
    q = p;
    q.y -= .38;
    tFan(q.xz, 4.);
    tRotate(q.xy, PI / 4.);
    d = opU(d, sdBox(q, vec3(.35, .01, .35)));

    // Make it hollow
    d = opS(d, sdBox(p, boxDim - vec3(.02)));
    return d;
}

// Boat distance
float sdBoat(vec3 p) {

    // Add the mast
    float d = sdBox(p + vec3(0, .05, 0), vec3(.01, .2, .01));

    // Add the sail
    vec3 q = p + vec3(0, -.05, .12);
    float a = sdSphere(q, .2);
    a = opS(a, sdSphere(q, .195));
    q.x = abs(q.x);
    tRotate(q.yx, .1);
    a = opI(a, sdBox(q - vec3(0, 0, .1), vec3(.1)));
    d = opU(d, a);

    // Add the body of the boat
    p.x = abs(p.x);
    p.x += .1;
    a = sdSphere(p, .3);
    a = opS(a, sdSphere(p, .29));
    a = opI(a, p.y + .15);
    d = opU(d, a);
    return d;
}

// Spaceship distance
float sdSpaceship(vec3 p) {
    tFan(p.xz, 6.);
    p.x += .3;

    // Add the cap
    float d = sdSphere(p, .4);
    d = opS(d, p.y - .12);

    // Add the body
    d = opU(d, sdSphere(p, .39));

    // Add the fins
    d = opU(d, opI(sdSphere(p + vec3(0, .24, 0), .41), sdRect(p.zx, vec2(.005, .5))));
    d = opS(d, sdSphere(p + vec3(0, .3, 0), .37));
    d = opS(d, p.y + .25);
    return d;
}

// Atmosphere distance
float sdAtmosphere(vec3 p) {
    float time = iTime;
    tRotate(p.yz, time);
    vec3 q = p;
    tFan(q.xz, 12.);
    float d = sdBox(q - vec3(.3, 0, 0), vec3(.01));
    tRotate(p.yx, time);
    q = p;
    tFan(q.yz, 12.);
    d = opU(d, sdBox(q - vec3(0, .23, 0), vec3(.01)));
    tRotate(p.xz, time);
    q = p;
    tFan(q.yx, 12.);
    d = opU(d, sdBox(q - vec3(0, .16, 0), vec3(.01)));

    return d;
}

// Distance estimation of everything together
float map(vec3 p) {
    float d = _house <= 0. ? 5. : sdHouse(p) + .1 - _house * .1;
    if (_boat > 0.) d = opU(d, sdBoat(p) + .1 - _boat * .1);
    if (_spaceship > 0.) d = opU(d, sdSpaceship(p) + .1 - _spaceship * .1);
    if (_atmosphere > 0.) d = opU(d, sdAtmosphere(p) + .1 - _atmosphere * .1);

    float s = 1.;
    for (int i = 0; i < 4; ++i) {
        tFan(p.xz, 10.);
        p = abs(p);
        p -= _kifsOffset;
        p *= _kifsRot;
        s *= 2.;
    }

    return opSU(d, sdBox(p * s, vec3(s / 17.)) / s, .1);
}

// Trace the scene from ro (origin) to rd (direction, normalized)
float trace(vec3 ro, vec3 rd, float maxDist, out float steps, out float nt) {
    float total = 0.;
    steps = 0.;
    nt = 100.;

    for (int i = 0; i < MAX_STEPS; ++i) {
        ++steps;
        float d = map(ro + rd * total);
        nt = min(d, nt);
        total += d;
        if (d < EPS || maxDist < total) break;
    }

    return total;
}

// Calculate the normal vector
vec3 getNormal(vec3 p) {
    vec2 e = vec2(.0001, 0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    ));
}

// Ambient occlusion
float calculateAO(vec3 p, vec3 n) {
    float r = 0., w = 1., d;

    for (float i = 1.; i <= AO_SAMPLES; i++) {
        d = i / AO_SAMPLES / AO_RANGE;
        r += w * (d - map(p + n * d));
        w *= .5;
    }

    return 1. - saturate(r * AO_RANGE);
}

// A lovely function that goes up and down periodically between 0 and 1
float pausingWave(float x, float a, float b) {
    x = abs(fract(x) - .5) * 1. - .5 + a;
    return smoothstep(0., a - b, x);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Transform screen coordinates
    vec2 uv = fragCoord.xy / iResolution.xy;
    uv = uv * 2. - 1.;
    uv.x *= iResolution.x / iResolution.y;

    // Transform mouse coordinates
    vec2 mouse = iMouse.xy / iResolution.xy * 2. - 1.;
    mouse.x *= iResolution.x / iResolution.y;
    mouse *= 2.;

    // FIX 1: Replace iChannel1 with fallback or audio-reactive value
    // If you want audio reactivity, use this:
    float smoothedFFTValue = texture2D(iChannel0, vec2(0.1, 0.0)).x;
    // Or use this for static version:
    // float smoothedFFTValue = 0.8;

    // Set time-dependent constants
    float speed = .25 / 10.5;
    float time = mod(iTime, 290.);
    time -= 10.5;
    if (time > 167.) time -= 167.;
    else if (time > 63.) time -= 63.;
    time -= 5.25;
    time *= speed;

    // Determine which object to show
    _house = pausingWave(time, .15, .125);
    _boat = pausingWave(time - .125 / .1, .15, .125);
    _spaceship = pausingWave(time - .25 / .1, .15, .125);
    _atmosphere = pausingWave(time - .375 / .1, .15, .125) * step(10., iTime);

    // Set up kifs rotation matrix
    float a = -texture2D(iChannel0, vec2(.5, .25)).x + sin(iTime) * .2 + .9;
    float s = sin(a), c = cos(a);
    _kifsRot *= mat3(c, -s, 0, s, c, 0, 0, 0, 1);
    _kifsRot *= mat3(1, 0, 0, 0, c, -s, 0, s, c);
    _kifsRot *= mat3(c, 0, s, 0, 1, 0, -s, 0, c);

    // Set up kifs offset
    _kifsOffset = .07 + texture2D(iChannel0, vec2(.1, .25)).x * .06;

    // Set up camera position
    vec3 rd = normalize(vec3(uv, FOV));
    vec3 ro = vec3(0, 0, -2);

    // Light is relative to the camera
    vec3 light = vec3(-1., .5, 0);

    vec2 rot = vec2(0);
    if (iMouse.z > 0. && iMouse.x > 0. && iMouse.y > 0.) {
        // Rotate the scene using the mouse
        rot = -mouse;
    } else {
        // Otherwise rotate constantly as time passes
        rot = vec2(
            iTime * SPEED * PI,
            mix(sin(iTime * SPEED) * PI / 8., PI / 2. - 1e-5, saturate(exp(-iTime + 10.5))));
    }

    tRotate(rd.yz, rot.y);
    tRotate(rd.xz, rot.x);
    tRotate(light.xz, rot.x);
    tRotate(ro.yz, rot.y);
    tRotate(ro.xz, rot.x);

    // March
    float steps, outline, dist = trace(ro, rd, RENDER_DIST, steps, outline);

    // Calculate hit point coordinates
    vec3 p = ro + rd * dist;

    // Calculate normal
    vec3 normal = getNormal(p);

    // Light direction
    vec3 l = normalize(light - p);

    // Ambient light
    float ambient = .1;

    // Diffuse light
    float diffuse = max(0., dot(l, normal));

    // Specular light
    float specular = pow(max(0., dot(reflect(-l, normal), -rd)), 4.);

    // "Ambient occlusion"
    float ao = calculateAO(p, normal);

    // Create the background grid
    vec2 gridUv = fragCoord.xy - iResolution.xy / 2.;
    float grid = dot(step(mod(gridUv.xyxy, vec4(20, 20, 100, 100)), vec4(1)), vec4(.1, .1, .2, .2));

    // Create blue background
    vec3 bg = vec3(0, .1, .3) * saturate(1.5 - length(uv) * .5);

    // Find the edges in the geometry
    float edgeWidth = .0015;
    float edge = smoothstep(1., .0, dot(normal, getNormal(p - normal * edgeWidth))) * step(length(p), 1.);

    // Get the outline of the shapes
    outline = smoothstep(.005, .0, outline) * step(1., length(p));

    // Diagonal strokes used for shading
    vec2 strokes = sin(vec2(uv.x + uv.y, uv.x - uv.y) * iResolution.y * PI / 4.) * .5 - .5;

    // First part of the shading: ao + marching steps
    float highlights = (steps / float(MAX_STEPS) + sqrt(1. - ao)) * step(length(p), 1.) * .5;
    highlights = floor(highlights * 5.) / 10.;

    // Second part of the shading: ambient + diffuse + specular light
    float fog = saturate(length(ro) - dist * dist * .25);
    float lightValue = (ambient + diffuse + specular) * fog;
    lightValue = floor(lightValue * 5.) / 10.;

    // FIX 2: Ensure colorFactor is always valid
    float colorFactor = clamp(0.5 + 0.5 * smoothedFFTValue, 0.3, 1.5);

    // FIX 3: Simplify final color calculation to avoid potential issues
    vec3 finalColor = mix(bg, vec3(1., .9, .7) * colorFactor,
        max(max(max(saturate(highlights + strokes.x), saturate(lightValue + strokes.y)) * fog,
            (edge + outline) * 2. + strokes.y), grid));

    // FIX 4: Remove problematic step function at the end
    fragColor = vec4(pow(saturate(finalColor), vec3(1. / 2.2)), 1.0);
}`,
"Cosmic Knot": `// Original DE from Knighty's Fragmentarium frag.
// PostFX from jmpep.
// Whatever is left (not much) by Syntopia.

#define MaxSteps 30
#define MinimumDistance 0.0009
#define normalDistance     0.0002

#define PI 3.141592
#define Scale 3.0
#define FieldOfView 0.5
#define Jitter 0.06
#define FudgeFactor 1.0

#define Ambient 0.32184
#define Diffuse 0.5
#define LightDir vec3(1.0)
#define LightColor vec3(0.6,1.0,0.158824)
#define LightDir2 vec3(1.0,-1.0,1.0)
#define LightColor2 vec3(1.0,0.933333,1.0)
#define Offset vec3(0.92858,0.92858,0.32858)

// Return rotation matrix for rotating around vector v by angle
mat3  rotationMatrix3(vec3 v, float angle)
{
	float c = cos(radians(angle));
	float s = sin(radians(angle));
	
	return mat3(c + (1.0 - c) * v.x * v.x, (1.0 - c) * v.x * v.y - s * v.z, (1.0 - c) * v.x * v.z + s * v.y,
		(1.0 - c) * v.x * v.y + s * v.z, c + (1.0 - c) * v.y * v.y, (1.0 - c) * v.y * v.z - s * v.x,
		(1.0 - c) * v.x * v.z - s * v.y, (1.0 - c) * v.y * v.z + s * v.x, c + (1.0 - c) * v.z * v.z
		);
}

vec2 rotate(vec2 v, float a) {
	return vec2(cos(a)*v.x + sin(a)*v.y, -sin(a)*v.x + cos(a)*v.y);
}

#define Type 5
float U;
float V ;
float W ;
float T =  1.0;

float VRadius = 0.05048;
float SRadius = 0.05476;
vec3 RotVector = vec3(0.0,1.0,1.0);
float RotAngle = 0.0;


mat3 rot;
vec4 nc,nd,p;
void init() {
     U = 0.0*cos(iTime)*0.5+0.1;
    V =  0.2*sin(iTime*0.1)*0.5+0.2;
     W =  1.0*cos(iTime*1.2)*0.5+0.5;

	if (iMouse.z>0.0) {
		U = iMouse.x/iResolution.x;
		W = 1.0-U;
		V = iMouse.y/iResolution.y;
		T = 1.0-V;
	}
	float cospin=cos(PI/float(Type)), isinpin=1./sin(PI/float(Type));
	float scospin=sqrt(2./3.-cospin*cospin), issinpin=1./sqrt(3.-4.*cospin*cospin);

	nc=0.5*vec4(0,-1,sqrt(3.),0.);
	nd=vec4(-cospin,-0.5,-0.5/sqrt(3.),scospin);

	vec4 pabc,pbdc,pcda,pdba;
	pabc=vec4(0.,0.,0.,1.);
	pbdc=0.5*sqrt(3.)*vec4(scospin,0.,0.,cospin);
	pcda=isinpin*vec4(0.,0.5*sqrt(3.)*scospin,0.5*scospin,1./sqrt(3.));
	pdba=issinpin*vec4(0.,0.,2.*scospin,1./sqrt(3.));
	
	p=normalize(U*pabc+V*pbdc+W*pcda+T*pdba);

	rot = rotationMatrix3(normalize(RotVector), RotAngle);//in reality we need a 4D rotation
}

vec4 fold(vec4 pos) {
	for(int i=0;i<Type*(Type-2);i++){
		pos.xy=abs(pos.xy);
		float t=-2.*min(0.,dot(pos,nc)); pos+=t*nc;
		t=-2.*min(0.,dot(pos,nd)); pos+=t*nd;
	}
	return pos;
}

float DD(float ca, float sa, float r){
	//magic formula to convert from spherical distance to planar distance.
	//involves transforming from 3-plane to 3-sphere, getting the distance
	//on the sphere then going back to the 3-plane.
	return r-(2.*r*ca-(1.-r*r)*sa)/((1.-r*r)*ca+2.*r*sa+1.+r*r);
}

float dist2Vertex(vec4 z, float r){
	float ca=dot(z,p), sa=0.5*length(p-z)*length(p+z);//sqrt(1.-ca*ca);//
	return DD(ca,sa,r)-VRadius;
}

float dist2Segment(vec4 z, vec4 n, float r){
	//pmin is the orthogonal projection of z onto the plane defined by p and n
	//then pmin is projected onto the unit sphere
	float zn=dot(z,n),zp=dot(z,p),np=dot(n,p);
	float alpha=zp-zn*np, beta=zn-zp*np;
	vec4 pmin=normalize(alpha*p+min(0.,beta)*n);
	//ca and sa are the cosine and sine of the angle between z and pmin. This is the spherical distance.
	float ca=dot(z,pmin), sa=0.5*length(pmin-z)*length(pmin+z);//sqrt(1.-ca*ca);//
	return DD(ca,sa,r)-SRadius;
}
//it is possible to compute the distance to a face just as for segments: pmin will be the orthogonal projection
// of z onto the 3-plane defined by p and two n's (na and nb, na and nc, na and and, nb and nd... and so on).
//that involves solving a system of 3 linear equations.
//it's not implemented here because it is better with transparency

float dist2Segments(vec4 z, float r){
	float da=dist2Segment(z, vec4(1.,0.,0.,0.), r);
	float db=dist2Segment(z, vec4(0.,1.,0.,0.), r);
	float dc=dist2Segment(z, nc, r);
	float dd=dist2Segment(z, nd, r);
	
	return min(min(da,db),min(dc,dd));
}

float DE(vec3 pos) {
	float r=length(pos);
	vec4 z4=vec4(2.*pos,1.-r*r)*1./(1.+r*r);//Inverse stereographic projection of pos: z4 lies onto the unit 3-sphere centered at 0.
	z4.xyw=rot*z4.xyw;
	z4=fold(z4);//fold it

	return min(dist2Vertex(z4,r),dist2Segments(z4, r));
}

vec3 lightDir;
vec3 lightDir2;


// Two light sources. No specular 
vec3 getLight(in vec3 color, in vec3 normal, in vec3 dir) {
	float diffuse = max(0.0,dot(-normal, lightDir)); // Lambertian
	
	float diffuse2 = max(0.0,dot(-normal, lightDir2)); // Lambertian
	
	return
	(diffuse*Diffuse)*(LightColor*color) +
	(diffuse2*Diffuse)*(LightColor2*color);
}



// Finite difference normal
vec3 getNormal(in vec3 pos) {
	vec3 e = vec3(0.0,normalDistance,0.0);
	
	return normalize(vec3(
			DE(pos+e.yxx)-DE(pos-e.yxx),
			DE(pos+e.xyx)-DE(pos-e.xyx),
			DE(pos+e.xxy)-DE(pos-e.xxy)
			)
		);
}

// Solid color 
vec3 getColor(vec3 normal, vec3 pos) {
	return vec3(1.0,1.0,1.0);
}


// Pseudo-random number
// From: lumina.sourceforge.net/Tutorials/Noise.html
float rand(vec2 co){
	return fract(cos(dot(co,vec2(4.898,7.23))) * 23421.631);
}

vec4 rayMarch(in vec3 from, in vec3 dir, in vec2 fragCoord) {
	// Add some noise to prevent banding
	float totalDistance = Jitter*rand(fragCoord.xy+vec2(iTime));
	vec3 dir2 = dir;
	float distance;
	int steps = 0;
	vec3 pos;
	for (int i=0; i <= MaxSteps; i++) {
		pos = from + totalDistance * dir;
		distance = DE(pos)*FudgeFactor;
		totalDistance += distance;
		if (distance < MinimumDistance) break;
		steps = i;
	}
	
	// 'AO' is based on number of steps.
	// Try to smooth the count, to combat banding.
	float smoothStep =   float(steps) ;
		float ao = 1.0-smoothStep/float(MaxSteps);
	
	// Since our distance field is not signed,
	// backstep when calc'ing normal
	vec3 normal = getNormal(pos-dir*normalDistance*3.0);
	vec3 bg = vec3(0.2);
	if (steps == MaxSteps) {
		return vec4(bg,1.0);
	}
	vec3 color = getColor(normal, pos);
	vec3 light = getLight(color, normal, dir);
	
	color = mix(color*Ambient+light,bg,1.0-ao);
	return vec4(color,1.0);
}

#define BLACK_AND_WHITE
#define LINES_AND_FLICKER
#define BLOTCHES
#define GRAIN

#define FREQUENCY 10.0

vec2 uv;
float rand(float c){
	return rand(vec2(c,1.0));
}

float randomLine(float seed)
{
	float b = 0.01 * rand(seed);
	float a = rand(seed+1.0);
	float c = rand(seed+2.0) - 0.5;
	float mu = rand(seed+3.0);
	
	float l = 1.0;
	
	if ( mu > 0.2)
		l = pow(  abs(a * uv.x + b * uv.y + c ), 1.0/8.0 );
	else
		l = 2.0 - pow( abs(a * uv.x + b * uv.y + c), 1.0/8.0 );				
	
	return mix(0.5, 1.0, l);
}

// Generate some blotches.
float randomBlotch(float seed)
{
	float x = rand(seed);
	float y = rand(seed+1.0);
	float s = 0.01 * rand(seed+2.0);
	
	vec2 p = vec2(x,y) - uv;
	p.x *= iResolution.x / iResolution.y;
	float a = atan(p.y,p.x);
	float v = 1.0;
	float ss = s*s * (sin(6.2831*a*x)*0.1 + 1.0);
	
	if ( dot(p,p) < ss ) v = 0.2;
	else
		v = pow(dot(p,p) - ss, 1.0/16.0);
	
	return mix(0.3 + 0.2 * (1.0 - (s / 0.02)), 1.0, v);
}



vec3 degrade(vec3 image)
{
		// Set frequency of global effect to 20 variations per second
		float t = float(int(iTime * FREQUENCY));
		
		// Get some image movement
		vec2 suv = uv + 0.002 * vec2( rand(t), rand(t + 23.0));
		
		#ifdef BLACK_AND_WHITE
		// Pass it to B/W
		float luma = dot( vec3(0.2126, 0.7152, 0.0722), image );
		vec3 oldImage = luma * vec3(0.7, 0.7, 0.7);
		#else
		vec3 oldImage = image;
		#endif
		// Create a time-varyting vignetting effect
		float vI = 16.0 * (uv.x * (1.0-uv.x) * uv.y * (1.0-uv.y));
		vI *= mix( 0.7, 1.0, rand(t + 0.5));
		
		// Add additive flicker
		vI += 1.0 + 0.4 * rand(t+8.);
		
		// Add a fixed vignetting (independent of the flicker)
		vI *= pow(16.0 * uv.x * (1.0-uv.x) * uv.y * (1.0-uv.y), 0.4);
		
		// Add some random lines (and some multiplicative flicker. Oh well.)
		#ifdef LINES_AND_FLICKER
		int l = int(8.0 * rand(t+7.0));
		
		if ( 0 < l ) vI *= randomLine( t+6.0+17.* float(0));
		if ( 1 < l ) vI *= randomLine( t+6.0+17.* float(1));
		if ( 2 < l ) vI *= randomLine( t+6.0+17.* float(2));		
		if ( 3 < l ) vI *= randomLine( t+6.0+17.* float(3));
		if ( 4 < l ) vI *= randomLine( t+6.0+17.* float(4));
		if ( 5 < l ) vI *= randomLine( t+6.0+17.* float(5));
		if ( 6 < l ) vI *= randomLine( t+6.0+17.* float(6));
		if ( 7 < l ) vI *= randomLine( t+6.0+17.* float(7));
		
		#endif
		
		// Add some random blotches.
		#ifdef BLOTCHES
		int s = int( max(8.0 * rand(t+18.0) -2.0, 0.0 ));

		if ( 0 < s ) vI *= randomBlotch( t+6.0+19.* float(0));
		if ( 1 < s ) vI *= randomBlotch( t+6.0+19.* float(1));
		if ( 2 < s ) vI *= randomBlotch( t+6.0+19.* float(2));
		if ( 3 < s ) vI *= randomBlotch( t+6.0+19.* float(3));
		if ( 4 < s ) vI *= randomBlotch( t+6.0+19.* float(4));
		if ( 5 < s ) vI *= randomBlotch( t+6.0+19.* float(5));
	
		#endif
	
		vec3 outv = oldImage * vI;
		
		// Add some grain (thanks, Jose!)
		#ifdef GRAIN
        outv *= (1.0+(rand(uv+t*.01)-.2)*.15);		
        #endif
		return outv;	
}		


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	uv = fragCoord.xy / iResolution.xy;
	
	init();
	
	// Camera position //(eye), and camera target
	vec3 camPos = (12.0+2.0*sin(iTime*0.6))*vec3(cos(iTime*0.3),0.0,sin(iTime*0.3));
	vec3 target = vec3(0.0,0.0,0.0);
	vec3 camUp  = vec3(0.0,1.0,0.0);
	
	
	
	// Calculate orthonormal camera reference system
	vec3 camDir   = normalize(target-camPos); // direction for center ray
	camUp = normalize(camUp-dot(camDir,camUp)*camDir); // orthogonalize
	vec3 camRight = normalize(cross(camDir,camUp));
	
	lightDir= -normalize(camPos+7.5*camUp);
	lightDir2=-normalize( camPos- 6.5*camRight);

	vec2 coord =-1.0+2.0*fragCoord.xy/iResolution.xy;
	float vignette = 0.4+(1.0-coord.x*coord.x)
		*(1.0-coord.y*coord.y);
	coord.x *= iResolution.x/iResolution.y;
	

	// Get direction for this pixel
	vec3 rayDir = normalize(camDir + (coord.x*camRight + coord.y*camUp)*FieldOfView);
	
	vec3 col = rayMarch(camPos, rayDir, fragCoord).xyz;   

    // vignetting 
    // col *= clamp(vignette,0.0,1.0);
	col = degrade(col);
	fragColor = vec4(col,1.0);
}
`,
"Million Eyes": `
float udBox( vec4 p, vec4 b )
{
  return length(max(abs(p)-b,0.0));
}

#define gridSize 200
vec4 c = vec4(gridSize,gridSize,gridSize,gridSize /2);
float sdSphere( vec4 p, float s )
{
    
  return abs(length(mod(p, c) - 0.5f*c)-s);
}
float getDistance(vec4 point){
    	return sdSphere(point, 25.);
}

struct HitData{
	float distance;
    float normal;
};
HitData rayMarch(vec4 point, vec4 dir){
    HitData hd;
	float marched = 0.;
    float epsilon = 0.1;
    float lastDistance = 0.;
    while(marched < 10000.){
    	float distance = getDistance(point);
        marched += distance;
        point += dir*distance;
        if(distance < epsilon){
            return HitData(marched, 1.-distance/lastDistance);
        }
        lastDistance = distance;
        
    }return HitData(1000.,1.);
}
vec4 eye = vec4(0,0,0,-1);
float layers = 24.;
vec4 getPosition(vec2 coord, float layer){
    return vec4(coord.x / iResolution.x * 2. - 1., (coord.y/iResolution.y * 2. - 1.) * iResolution.y / iResolution.x, layer/layers * 2. - 1.,0.);
}

float color(float d){
    return d / 1000.;
}
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    eye = vec4(0, 0 ,0, - 1) * 2.;
    //spheres[0] = vec3(sin(iTime)*20.,0,20.);
    //radii[0] = abs(sin(iTime * 1.3f + 20.)) * 10.;
    float l = (fract(iTime/50.)*2.-1.)*layers;
    vec4 pos = getPosition(fragCoord, l);
    vec4 dir = normalize(pos - eye);
    HitData hd = rayMarch(pos, dir);
	float amount = color(hd.distance) * hd.normal;
    amount = hd.normal;
    fragColor = vec4(amount,amount,amount,1);
}`,
    "Piano Optic Cables": `struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Light {
    vec3 position;
    float strongth;
    vec3 color;
};

#define PI 3.1415926
    
mat4 euler(float x, float y, float z) {
    mat4 xmat = mat4(vec4(1.0,  0.0,    0.0,    0.0),
                     vec4(0.0,  cos(x), sin(x), 0.0),
                     vec4(0.0, -sin(x), cos(x), 0.0),
                     vec4(0.0,  0.0,    0.0,    1.0));
    mat4 ymat = mat4(vec4( cos(y), 0.0, sin(y), 0.0),
                     vec4( 0.0,    1.0, 0.0,    0.0),
                     vec4(-sin(y), 0.0, cos(y), 0.0),
                     vec4( 0.0,    0.0, 0.0,    1.0));
    mat4 zmat = mat4(vec4( cos(z),  sin(z), 0.0, 0.0),
                     vec4(-sin(z),  cos(z), 0.0, 0.0),
                     vec4( 0.0,     0.0,    1.0, 0.0),
                     vec4( 0.0,     0.0,    0.0, 1.0));
    
    return xmat*ymat*zmat;
}

mat4 transform(float x, float y, float z) {
    return mat4(vec4(1.0, 0.0, 0.0, 0.0),
                vec4(0.0, 1.0, 0.0, 0.0),
                vec4(0.0, 0.0, 1.0, 0.0),
                vec4(x,   y,   z,   1.0));
}

float sphereSDF(vec3 center, float radius, vec3 point) {
    return length(point - center) - radius;
}

float planeSDF(vec3 origin, vec3 normal, vec3 point) {
    return dot(point - origin, normal);
}

float ropeSDF(float coiledness, uint n, vec3 point) {
    for (uint i = 0u; i < n; ++i) {
        float r = length(point.xz);
    	float t = atan(-point.x, -point.z) + PI;
        
        t -= 2.0*PI*coiledness;
        t = mod(t, 2.0*PI/3.0) + 2.0*PI/3.0;
        
        point.x = r*sin(t);
        point.z = r*cos(t);
        
        point.z += 1.0;
        point.xz *= 1.0 + 1.0/sin(PI/3.0);
        //point.z *= -1.0;
    }
    
    point.xz /= 1.0 + sin(PI/3.0);
    
    float lpxz = length(point.xz);
    
    vec2 d = vec2(lpxz, abs(point.y + 0.5)) - vec2(1.0,0.5);
    
    for (uint i = 0u; i < n; ++i) d.x /= 1.0 + 1.0/sin(PI/3.0);
    
    d.x *= 1.0 + sin(PI/3.0);
    
    return min(max(d.x,d.y), 0.0) + length(max(d, 0.0));
}

float sceneSDF(vec3 point) {
    point.y /= 20.0;
    return ropeSDF(1.0-(0.5*sin(iTime*0.2)+0.5)*(point.y+1.0), 6u, point);
   /*return min(
       min(
           min(
               sphereSDF(vec3(-0.7, 0.7, 0.0), 0.5, point),
               sphereSDF(vec3(0.7, 0.7, 0.0), 0.5, point)
           ),
           sphereSDF(vec3(0.0), 1.0, point)
       ),
       planeSDF(vec3(0.0), vec3(0.0, 1.0, 0.0), point)
     );
   */
}

vec3 sceneSDFGradient(vec3 point, float epsilon) {
    vec3 xe = vec3(epsilon, 0.0, 0.0)/2.0;
    vec3 ye = vec3(0.0, epsilon, 0.0)/2.0;
    vec3 ze = vec3(0.0, 0.0, epsilon)/2.0;
    
    return vec3(
        (sceneSDF(point + xe) - sceneSDF(point - xe)) / epsilon,
        (sceneSDF(point + ye) - sceneSDF(point - ye)) / epsilon,
        (sceneSDF(point + ze) - sceneSDF(point - ze)) / epsilon
      );
}

vec3 sceneSDFNormal(vec3 point) {
    return normalize(sceneSDFGradient(point, 0.01));
}

vec3 rayPoint(Ray ray, float dist) {
    return ray.origin + dist * ray.direction;
}

vec3 screen(vec3 a, vec3 b) {
    return vec3(1.0) - (vec3(1.0) - a)*(vec3(1.0) - b);
}

vec3 lightPoint(Light light, vec3 point, vec3 normal, vec3 camera, vec3 diffuse, vec3 bounce, vec3 current) {
    vec3 lightchord = light.position - point;
    
    vec3 lightcolor = light.color * 1.0 / pow(length(lightchord/3.0)/light.strongth+1.0, 2.0);
    
    vec3 colour = diffuse * lightcolor * max(dot(normal, normalize(lightchord)), 0.0);
    colour = screen(colour, bounce * lightcolor * max(vec3(1.0) - 5.0*(vec3(1.0) - dot(normalize(lightchord), reflect(normalize(point - camera), normal))), 0.0));
    
    return screen(current, colour);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float lightangle = iTime;
    
    Light light1 = Light(vec3(2.0*cos(lightangle), 2.0, 2.0*sin(lightangle)), 10.0, vec3(1.0, 0.0, 0.0));
    
    lightangle += PI*2./3.;
    
    Light light2 = Light(vec3(2.0*cos(lightangle), 2.0, 2.0*sin(lightangle)), 10.0, vec3(0.0, 1.0, 0.0));
    
    lightangle += PI*2./3.;
    
    Light light3 = Light(vec3(2.0*cos(lightangle), 2.0, 2.0*sin(lightangle)), 10.0, vec3(0.0, 0.0, 1.0));
    
    float disttoscreen = 1.0;
    
    vec2 uv = fragCoord/iResolution.xy - vec2(0.5);
    uv.y *= iResolution.y/iResolution.x;
    
    vec3 camorigin = vec3(-6.0, 6.0, 0.0);
    
    mat4 camtoscene = transform(camorigin.x, camorigin.y, camorigin.z)*euler(PI*0.5, -PI*0.18, 0.0);
    
    Ray ray = Ray((camtoscene*vec4(vec3(0.0),1.0)).xyz,
                  normalize(camtoscene*vec4(uv.x, uv.y, disttoscreen, 0.0)).xyz);
    
    vec3 point = camorigin;
    
    float scenedist = sceneSDF(point);
    float raydist = 0.0;
    
    float epsilon = 0.001;
    float end = 100.0;
    
    while (scenedist > epsilon) {
        if (raydist > end) {
            fragColor = vec4(0.0, 0.0, 0.0, texture(iChannel0, vec2(0.1, 0.5)).r / 1. );
            return;
        }
        
        point = rayPoint(ray, raydist);
        
        scenedist = sceneSDF(point);
        
        raydist += scenedist;
    }
    
    vec3 normal = sceneSDFNormal(point);
    vec3 diffuse = vec3(1.0);
    vec3 bounce = vec3(1.0);
        
    vec3 colour = lightPoint(light1, point, normal, camorigin, diffuse, bounce, vec3(0.0));
    colour = lightPoint(light2, point, normal, camorigin, diffuse, bounce, colour);
    colour = lightPoint(light3, point, normal, camorigin, diffuse, bounce, colour);

    // Output to screen
    fragColor = vec4(colour,1.0);
}`,
    "Checker Rau": `void mainImage( out vec4 fragColor, in vec2 fragCoord )
{

    // Sample audio data
    float audioL = texture(iChannel0, vec2(0.1, 0.5)).r;
    float audioH = texture(iChannel0, vec2(0.9, 0.5)).r;
    float audio = mix(audioL, audioH, 0.5);
    float speed = 0.25 / (iResolution.x / 120.0 + texture(iChannel0, vec2(0.1, 0.5)).r * 0.5);
    float colorResolution = audio;
    
    int x = int(floor(fragCoord.x));
    int y = int(floor(fragCoord.y));
    int positionFactor = (x - y) & (y + x);
    float speedFactor = iTime * speed;
    float outColor = 
    mod(
        mod(
            float(positionFactor) * (speedFactor), 
            float(iResolution.x)
        ), 
        colorResolution
    ) / colorResolution;
    fragColor = vec4(sin(audioL), outColor * sin(audioH), tan(outColor), 1.0);
}`,
    "10 Eyes - Needs sound effect": `vec2 cMul(vec2 a, vec2 b) {
	return vec2(a.x*b.x -  a.y*b.y,a.x*b.y + a.y * b.x);
}

vec2 cInverse(vec2 a) {
	return	vec2(a.x,-a.y)/dot(a,a);
}


vec2 cDiv(vec2 a, vec2 b) {
	return cMul( a,cInverse(b));
}


vec2 cPower(vec2 z, float n) {
	float r2 = dot(z,z);
	return pow(r2,n/2.0)*vec2(cos(n*atan(z.y/z.x)),sin(n*atan(z.y/z.x)));
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 p = fragCoord.xy / iResolution.xy;
	
	float t = iTime;

	float zPower = 1.16578;
	float aa = -0.15000;
	float bb = 0.89400;
	float cc = -0.05172;
	float dd = 0.10074;
	int i = 0;

	vec2 A = vec2(aa, aa);
	vec2 B = vec2(bb, bb);
	vec2 C = vec2(cc, cc);
	vec2 D = vec2(dd, dd);
	
	float speed = 0.25;
	vec2 c = vec2(cos(t*speed), sin(t*speed));
	float s = 2.5;
    vec2 z = s*((-1.0 + 2.0*p)*vec2(iResolution.x/(iResolution.y),1.0));
	const int iter = 96;
	float e = 128.0;

    for( int j=0; j<iter; j++ )
    {
		z = cPower(z, zPower);
		z = abs(z);
		z = cMul(z,z) + c;		
		
		z = cDiv((cMul(A, z) + B), (cMul(z,C) + D));
		
    	if (dot(z,z) > e) break;
		i++;
	}
	
	float ci = float(i) + 1.0 - log2(0.5*log2(dot(z,z)));

	float red = 0.5 + 0.5*cos(6.0*ci+0.0);
	float green = 0.5+0.5*cos(6.0*ci+0.4);
	float blue = 0.5+0.5*cos(6.0*ci+0.8);

	fragColor = vec4(red, green, blue, 1.0);
}
`,
"Drive Home": `
// "The Drive Home" by Martijn Steinrucken aka BigWings - 2017
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// Email:countfrolic@gmail.com Twitter:@The_ArtOfCode
//
// I was looking for something 3d, that can be made just with a point-line distance function.
// Then I saw the cover graphic of the song I'm using here on soundcloud, which is a bokeh traffic
// shot which is a perfect for for what I was looking for.
//
// It took me a while to get to a satisfying rain effect. Most other people use a render buffer for
// this so that is how I started. In the end though, I got a better effect without. Uncomment the
// DROP_DEBUG define to get a better idea of what is going on.
//
// If you are watching this on a weaker device, you can uncomment the HIGH_QUALITY define
//
// Music:
// Mr. Bill - Cheyah (Zefora's digital rain remix) 
// https://soundcloud.com/zefora/cheyah
//
// Video can be found here:
// https://www.youtube.com/watch?v=WrxZ4AZPdOQ
//
// Making of tutorial:
// https://www.youtube.com/watch?v=eKtsY7hYTPg
//

#define S(x, y, z) smoothstep(x, y, z)
#define B(a, b, edge, t) S(a-edge, a+edge, t)*S(b+edge, b-edge, t)
#define sat(x) clamp(x,0.,1.)

#define streetLightCol vec3(1., .7, .3)
#define headLightCol vec3(.8, .8, 1.)
#define tailLightCol vec3(1., .1, .1)

#define HIGH_QUALITY
#define CAM_SHAKE 1.
#define LANE_BIAS .5
#define RAIN
//#define DROP_DEBUG

vec3 ro, rd;

float N(float t) {
	return fract(sin(t*10234.324)*123423.23512);
}
vec3 N31(float p) {
    //  3 out, 1 in... DAVE HOSKINS
   vec3 p3 = fract(vec3(p) * vec3(.1031,.11369,.13787));
   p3 += dot(p3, p3.yzx + 19.19);
   return fract(vec3((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y, (p3.y+p3.z)*p3.x));
}
float N2(vec2 p)
{	// Dave Hoskins - https://www.shadertoy.com/view/4djSRW
	vec3 p3  = fract(vec3(p.xyx) * vec3(443.897, 441.423, 437.195));
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}


float DistLine(vec3 ro, vec3 rd, vec3 p) {
	return length(cross(p-ro, rd));
}
 
vec3 ClosestPoint(vec3 ro, vec3 rd, vec3 p) {
    // returns the closest point on ray r to point p
    return ro + max(0., dot(p-ro, rd))*rd;
}

float Remap(float a, float b, float c, float d, float t) {
	return ((t-a)/(b-a))*(d-c)+c;
}

float BokehMask(vec3 ro, vec3 rd, vec3 p, float size, float blur) {
	float d = DistLine(ro, rd, p);
    float m = S(size, size*(1.-blur), d);
    
    #ifdef HIGH_QUALITY
    m *= mix(.7, 1., S(.8*size, size, d));
    #endif
    
    return m;
}



float SawTooth(float t) {
    return cos(t+cos(t))+sin(2.*t)*.2+sin(4.*t)*.02;
}

float DeltaSawTooth(float t) {
    return 0.4*cos(2.*t)+0.08*cos(4.*t) - (1.-sin(t))*sin(t+cos(t));
}  

vec2 GetDrops(vec2 uv, float seed, float m) {
    
    float t = iTime+m*30.;
    vec2 o = vec2(0.);
    
    #ifndef DROP_DEBUG
    uv.y += t*.05;
    #endif
    
    uv *= vec2(10., 2.5)*2.;
    vec2 id = floor(uv);
    vec3 n = N31(id.x + (id.y+seed)*546.3524);
    vec2 bd = fract(uv);
    
    vec2 uv2 = bd;
    
    bd -= .5;
    
    bd.y*=4.;
    
    bd.x += (n.x-.5)*.6;
    
    t += n.z * 6.28;
    float slide = SawTooth(t);
    
    float ts = 1.5;
    vec2 trailPos = vec2(bd.x*ts, (fract(bd.y*ts*2.-t*2.)-.5)*.5);
    
    bd.y += slide*2.;								// make drops slide down
    
    #ifdef HIGH_QUALITY
    float dropShape = bd.x*bd.x;
    dropShape *= DeltaSawTooth(t);
    bd.y += dropShape;								// change shape of drop when it is falling
    #endif
    
    float d = length(bd);							// distance to main drop
    
    float trailMask = S(-.2, .2, bd.y);				// mask out drops that are below the main
    trailMask *= bd.y;								// fade dropsize
    float td = length(trailPos*max(.5, trailMask));	// distance to trail drops
    
    float mainDrop = S(.2, .1, d);
    float dropTrail = S(.1, .02, td);
    
    dropTrail *= trailMask;
    o = mix(bd*mainDrop, trailPos, dropTrail);		// mix main drop and drop trail
    
    #ifdef DROP_DEBUG
    if(uv2.x<.02 || uv2.y<.01) o = vec2(1.);
    #endif
    
    return o;
}

void CameraSetup(vec2 uv, vec3 pos, vec3 lookat, float zoom, float m) {
	ro = pos;
    vec3 f = normalize(lookat-ro);
    vec3 r = cross(vec3(0., 1., 0.), f);
    vec3 u = cross(f, r);
    float t = iTime;
    
    vec2 offs = vec2(0.);
    #ifdef RAIN
    vec2 dropUv = uv; 
    
    #ifdef HIGH_QUALITY
    float x = (sin(t*.1)*.5+.5)*.5;
    x = -x*x;
    float s = sin(x);
    float c = cos(x);
    
    mat2 rot = mat2(c, -s, s, c);
   
    #ifndef DROP_DEBUG
    dropUv = uv*rot;
    dropUv.x += -sin(t*.1)*.5;
    #endif
    #endif
    
    offs = GetDrops(dropUv, 1., m);
    
    #ifndef DROP_DEBUG
    offs += GetDrops(dropUv*1.4, 10., m);
    #ifdef HIGH_QUALITY
    offs += GetDrops(dropUv*2.4, 25., m);
    //offs += GetDrops(dropUv*3.4, 11.);
    //offs += GetDrops(dropUv*3., 2.);
    #endif
    
    float ripple = sin(t+uv.y*3.1415*30.+uv.x*124.)*.5+.5;
    ripple *= .005;
    offs += vec2(ripple*ripple, ripple);
    #endif
    #endif
    vec3 center = ro + f*zoom;
    vec3 i = center + (uv.x-offs.x)*r + (uv.y-offs.y)*u;
    
    rd = normalize(i-ro);
}

vec3 HeadLights(float i, float t) {
    float z = fract(-t*2.+i);
    vec3 p = vec3(-.3, .1, z*40.);
    float d = length(p-ro);
    
    float size = mix(.03, .05, S(.02, .07, z))*d;
    float m = 0.;
    float blur = .1;
    m += BokehMask(ro, rd, p-vec3(.08, 0., 0.), size, blur);
    m += BokehMask(ro, rd, p+vec3(.08, 0., 0.), size, blur);
    
    #ifdef HIGH_QUALITY
    m += BokehMask(ro, rd, p+vec3(.1, 0., 0.), size, blur);
    m += BokehMask(ro, rd, p-vec3(.1, 0., 0.), size, blur);
    #endif
    
    float distFade = max(.01, pow(1.-z, 9.));
    
    blur = .8;
    size *= 2.5;
    float r = 0.;
    r += BokehMask(ro, rd, p+vec3(-.09, -.2, 0.), size, blur);
    r += BokehMask(ro, rd, p+vec3(.09, -.2, 0.), size, blur);
    r *= distFade*distFade;
    
    return headLightCol*(m+r)*distFade;
}


vec3 TailLights(float i, float t) {
    t = t*1.5+i;
    
    float id = floor(t)+i;
    vec3 n = N31(id);
    
    float laneId = S(LANE_BIAS, LANE_BIAS+.01, n.y);
    
    float ft = fract(t);
    
    float z = 3.-ft*3.;						// distance ahead
    
    laneId *= S(.2, 1.5, z);				// get out of the way!
    float lane = mix(.6, .3, laneId);
    vec3 p = vec3(lane, .1, z);
    float d = length(p-ro);
    
    float size = .05*d;
    float blur = .1;
    float m = BokehMask(ro, rd, p-vec3(.08, 0., 0.), size, blur) +
    			BokehMask(ro, rd, p+vec3(.08, 0., 0.), size, blur);
    
    #ifdef HIGH_QUALITY
    float bs = n.z*3.;						// start braking at random distance		
    float brake = S(bs, bs+.01, z);
    brake *= S(bs+.01, bs, z-.5*n.y);		// n.y = random brake duration
    
    m += (BokehMask(ro, rd, p+vec3(.1, 0., 0.), size, blur) +
    	BokehMask(ro, rd, p-vec3(.1, 0., 0.), size, blur))*brake;
    #endif
    
    float refSize = size*2.5;
    m += BokehMask(ro, rd, p+vec3(-.09, -.2, 0.), refSize, .8);
    m += BokehMask(ro, rd, p+vec3(.09, -.2, 0.), refSize, .8);
    vec3 col = tailLightCol*m*ft; 
    
    float b = BokehMask(ro, rd, p+vec3(.12, 0., 0.), size, blur);
    b += BokehMask(ro, rd, p+vec3(.12, -.2, 0.), refSize, .8)*.2;
    
    vec3 blinker = vec3(1., .7, .2);
    blinker *= S(1.5, 1.4, z)*S(.2, .3, z);
    blinker *= sat(sin(t*200.)*100.);
    blinker *= laneId;
    col += blinker*b;
    
    return col;
}

vec3 StreetLights(float i, float t) {
	 float side = sign(rd.x);
    float offset = max(side, 0.)*(1./16.);
    float z = fract(i-t+offset); 
    vec3 p = vec3(2.*side, 2., z*60.);
    float d = length(p-ro);
	float blur = .1;
    vec3 rp = ClosestPoint(ro, rd, p);
    float distFade = Remap(1., .7, .1, 1.5, 1.-pow(1.-z,6.));
    distFade *= (1.-z);
    float m = BokehMask(ro, rd, p, .05*d, blur)*distFade;
    
    return m*streetLightCol;
}

vec3 EnvironmentLights(float i, float t) {
	float n = N(i+floor(t));
    
    float side = sign(rd.x);
    float offset = max(side, 0.)*(1./16.);
    float z = fract(i-t+offset+fract(n*234.));
    float n2 = fract(n*100.);
    vec3 p = vec3((3.+n)*side, n2*n2*n2*1., z*60.);
    float d = length(p-ro);
	float blur = .1;
    vec3 rp = ClosestPoint(ro, rd, p);
    float distFade = Remap(1., .7, .1, 1.5, 1.-pow(1.-z,6.));
    float m = BokehMask(ro, rd, p, .05*d, blur);
    m *= distFade*distFade*.5;
    
    m *= 1.-pow(sin(z*6.28*20.*n)*.5+.5, 20.);
    vec3 randomCol = vec3(fract(n*-34.5), fract(n*4572.), fract(n*1264.));
    vec3 col = mix(tailLightCol, streetLightCol, fract(n*-65.42));
    col = mix(col, randomCol, n);
    return m*col*.2;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	float t = iTime;
    vec3 col = vec3(0.);
    vec2 uv = fragCoord.xy / iResolution.xy; // 0 <> 1
    
    uv -= .5;
    uv.x *= iResolution.x/iResolution.y;
    
    vec2 mouse = iMouse.xy/iResolution.xy;
    
    vec3 pos = vec3(.3, .15, 0.);
    
    float bt = t * 5.;
    float h1 = N(floor(bt));
    float h2 = N(floor(bt+1.));
    float bumps = mix(h1, h2, fract(bt))*.1;
    bumps = bumps*bumps*bumps*CAM_SHAKE;
    
    pos.y += bumps;
    float lookatY = pos.y+bumps;
    vec3 lookat = vec3(0.3, lookatY, 1.);
    vec3 lookat2 = vec3(0., lookatY, .7);
    lookat = mix(lookat, lookat2, sin(t*.1)*.5+.5);
    
    uv.y += bumps*4.;
    CameraSetup(uv, pos, lookat, 2., mouse.x);
   
    t *= .03;
    t += mouse.x;
    
    // fix for GLES devices by MacroMachines
    #ifdef GL_ES
	const float stp = 1./8.;
	#else
	float stp = 1./8.;
	#endif
    
    for(float i=0.; i<1.; i+=stp) {
       col += StreetLights(i, t);
    }
    
    for(float i=0.; i<1.; i+=stp) {
        float n = N(i+floor(t));
    	col += HeadLights(i+n*stp*.7, t);
    }
    
    #ifndef GL_ES
    #ifdef HIGH_QUALITY
    stp = 1./32.;
    #else
    stp = 1./16.;
    #endif
    #endif
    
    for(float i=0.; i<1.; i+=stp) {
       col += EnvironmentLights(i, t);
    }
    
    col += TailLights(0., t);
    col += TailLights(.5, t);
    
    col += sat(rd.y)*vec3(.6, .5, .9);
    
	fragColor = vec4(col, 1.);
}`,
"Opaque Door - colored": `// Enhanced Kaleidoscope with Vibrant Colors for Audio Visualization
// Added dynamic color palettes and subtle audio reactivity

#define C .03  // emphasize parent tile
#define H 1.  // hold between transitions
#define PI 3.14159265
#define R iResolution

// The angles are defined by divisors of PI in each row
const mat3 D = mat3(
    3,3,3,
    2,4,4,
    2,3,6);

// Enhanced hue shift with saturation control
vec3 hs(vec3 c, float s) {
    vec3 m = vec3(cos(s), s=sin(s)*.5774, -s);
    return c*mat3(m+=(1.-m.x)/3., m.zxy, m.yzx);
}

// Vibrant color palette generator
vec3 colorPalette(float t, int mode) {
    vec3 a, b, c, d;
    
    if (mode == 0) {
        // Electric rave colors
        a = vec3(0.5, 0.5, 0.5);
        b = vec3(0.5, 0.5, 0.5);
        c = vec3(1.0, 1.0, 0.5);
        d = vec3(0.8, 0.9, 0.3);
    } else if (mode == 1) {
        // Neon cyber colors
        a = vec3(0.5, 0.5, 0.5);
        b = vec3(0.5, 0.5, 0.5);
        c = vec3(1.0, 0.7, 0.4);
        d = vec3(0.0, 0.15, 0.20);
    } else {
        // Psychedelic colors
        a = vec3(0.5, 0.5, 0.5);
        b = vec3(0.5, 0.5, 0.5);
        c = vec3(2.0, 1.0, 0.0);
        d = vec3(0.5, 0.20, 0.25);
    }
    
    return a + b * cos(6.28318 * (c * t + d));
}

// Audio reactivity (subtle)
float getAudioLevel() {
    // Sample from audio texture if available
    return texture(iChannel0, vec2(0.1, 0.0)).x * 0.3; // Subtle multiplier for loud music
}

// Iterative folding/reflection
vec4 fold(vec2 p, float sc, vec3 ang, out int nf) {
    vec3 c = cos(ang), s = sin(ang);
    mat3 N = mat3(1, 0, C,
                 -c.x, s.x, 0,
                 -c.z, -(c.y+c.x*c.z)/s.x, 1);
    sc *= s.z;
    vec3 u, q = vec3(p,sc);
    nf = 0;
    for(int i=0; i<9999; i++) {
        for(int j=0; j<3; j++) {
            u[j] = dot(q, N[j]);
            if(u[j] < 0.) {q -= 2.*u[j]*N[j]*vec3(1,1,0); nf++;}
        }
        if(i >= nf) break;
    }
    return vec4(q.xy, u.yz);
}

void mainImage(out vec4 O, vec2 U) {
    const float sc = 5.;
    float t = iTime*.2;
    float audioLevel = getAudioLevel();
    
    int m = 2, mm = m;
    m = int(t)%3, mm = int(t+1.)%3;
    
    U = sc*(U*2. - R.xy)/R.y;
    
    int nf;    
    float f = smoothstep(0.,1.,mod(t,1.)*(H+1.)-H);
    O = fold(U, 1., mix(PI/D[m],PI/D[mm],f), nf);
    U = O.xy;
    vec3 u = O.xzw;
    
    if(iMouse.z>0.) U -= (iMouse.xy*2.-R.xy)/R.y;
    else U -= t*.5;
    
    O = texture(iChannel0,U);
    
    // Enhanced color system
    float colorTime = t + audioLevel * 2.0; // Subtle audio influence on color timing
    int colorMode = int(colorTime * 0.5) % 3; // Cycle through color modes
    
    // Base color transformation
    vec3 baseColor = (O.bgr-.25)*2.;
    
    // Apply vibrant color palette
    vec3 paletteColor = colorPalette(length(U) * 0.1 + colorTime, colorMode);
    
    // Mix original hue shift with palette
    float hueShift = PI*(1.-.5*mix(float(m),float(mm),f)) + audioLevel * 0.5;
    vec3 hueShifted = hs(baseColor, hueShift);
    
    // Blend hue-shifted colors with palette
    O.rgb = mix(hueShifted, paletteColor, 0.6 + audioLevel * 0.2);
    
    // Add color intensity boost
    O.rgb = pow(O.rgb, vec3(0.8)); // Gamma adjust for more vibrant colors
    O.rgb *= 1.2 + audioLevel * 0.3; // Brightness boost with subtle audio response
    
    // Color saturation enhancement
    float luminance = dot(O.rgb, vec3(0.299, 0.587, 0.114));
    O.rgb = mix(vec3(luminance), O.rgb, 1.5 + audioLevel * 0.2);
    
    O *= exp(-float(nf)/sc*.4)*1.5;
    O *= smoothstep(-1.,1.,(min(u.x,min(u.y,u.z))-.01)*R.y/(sc*2.));
    
    O.a = 1.;
}`,
"Bottles": `mat2 rot(float a){return mat2(cos(a),-sin(a),sin(a),cos(a));}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{

    vec3 col;
    float t;
    
    for(int c=0;c<3;c++){
	    vec2 uv = (fragCoord*20.0-iResolution.xy)/iResolution.y;
        t = iTime;
        for(int i=0;i<5;i++)
        {
            uv += sin(col.yx);
        	uv += float(i) + (sin(iTime+uv.x)+cos(iTime+uv.y));
        }
     col[c] = (sin(uv.x)+cos(uv.y));
	}
    
    fragColor = vec4(col,1.0);
    
}
`,
"Radio active Shrooms": `#define T (iTime/clamp(40.-iTime, -15.,40.)+165.)
#define HPI 1.5707963267948966
#define PI 3.141592653589793
const int Iterations = 29;
const float Eps = 0.0002;
float C3, S3, C2, S2, C1, S1;

float map(vec3 p) {
    float len, t = 0.6;
    for (int i = 0; i <= Iterations; i++) {
   // p.y -= (p.y*p.x*p.x*p.x)/(4300./float(i*6));
        p.xz = S3 * p.xz + C3 * vec2(-1.0, 1.0) * p.zx;
        
        p = p.yzx;
        p.xz = S2 * p.xz + C2 * vec2(-1.0, 1.0) * p.zx;
        p = p.yzx;
        
        p.xz = S1 * p.xz + C1 * vec2(-1.0, 1.0) * p.zx;
         p = p.yzx;
        p.xy = -abs(p.xy);
      
        p.xy += vec2(t, t*1.2 +0.0);
        t *= .618;
    }
     len = length(p)-.001;
    return len ;
}

vec3 GetColorAmount(vec3 p) {
    float amount = 0.54 * clamp(2.1 - (pow(length(p), abs(sin(T * 0.1)) * 2.8) / 6.0), 0.0, 1.1);
    vec3 col = 0.2 + (cos(T * 0.33) * 0.130) * cos(6.28319 * (vec3(0.00021, 0.0009, cos(T * 0.23) * 0.330) + (amount / 1.101) * vec3(0.50, 0.5, 0.0995)));
    return col * amount;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float time = pow(abs((sin(T * .35))) * 1., 1.4) + T*.633 ;
    float t = time; S1 = sin(t); C1 = cos(t);
    t = time * PI; S2 = sin(t); C2 = cos(t);
    t = t * PI; S3 = sin(t); C3 = cos(t);
    float angle = T * 0.5;
    vec3 camPos = vec3(3.0 * cos(angle), 1.5, 3.0 * sin(angle));
    vec3 target = vec3(0.0);
    vec3 camDir = normalize(target - camPos);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 camRight = normalize(cross(camDir, up));
    vec3 camUp = cross(camRight, camDir);
    vec3 rayDir = normalize(camDir + uv.x * camRight + uv.y * camUp);
    vec3 rayPos = camPos;
    float totalDist = 0.0;
    vec3 color = vec3(0.0);
    for (int i = 0; i < 35; i++) {
        float d = map(rayPos);
        vec3 glow = GetColorAmount(rayPos);
        color += glow * exp(-0.61 * totalDist);
        if (abs(d) < Eps) break;
        rayPos += rayDir * d;
        totalDist += d;
    }
    fragColor = vec4(pow(color.grb*1.5, color.bbb*12.), 2.0 - pow(totalDist, 0.5));
}
//https://www.shadertoy.com/view/sldGRS
//https://www.shadertoy.com/view/Ns2Gz3
//iq

//thanks
`,
"Howww???": `/*

	Raymarched 2D Sierpinski
	------------------------

	Raymarching a 2D Sierpinski Carpet pattern. The raymarching process is pretty straight
	forward. Basically, Sierpinski height values are added to a plane. Height maps with 
	sharp edges don't raymarch particularly well, so a little edge smoothing was necessary,
	but that's about it.

	The rest is just lighting. Most of it is made up. A bit of diffuse, specular, fake 
	environment mapping, etc.
	

*/

#define FAR 5.

// Tri-Planar blending function. Based on an old Nvidia writeup:
// GPU Gems 3 - Ryan Geiss: https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch01.html
vec3 tex3D( sampler2D tex, in vec3 p, in vec3 n ){
   
    n = max((abs(n) - .2)*7., .001);
    n /= (n.x + n.y + n.z );  
    
	p = (texture(tex, p.yz)*n.x + texture(tex, p.zx)*n.y + texture(tex, p.xy)*n.z).xyz;
    
    return p*p;
}

// Compact, self-contained version of IQ's 3D value noise function.
float n3D(vec3 p){
    
	const vec3 s = vec3(7, 157, 113);
	vec3 ip = floor(p); p -= ip; 
    vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
    p = p*p*(3. - 2.*p); //p *= p*p*(p*(p * 6. - 15.) + 10.);
    h = mix(fract(sin(h)*43758.5453), fract(sin(h + s.x)*43758.5453), p.x);
    h.xy = mix(h.xz, h.yw, p.y);
    return mix(h.x, h.y, p.z); // Range: [0, 1].
}

// Sierpinski Carpet heightmap - Essentially, space is divided into 3 each iteration, 
// and a shape of some kind is rendered. In this case, it's a smooth edged rectangle (w)
// with a bit of curvature (l) around the sides.
//
// There are some opportunites to optimize, but I'll leave it partly readable for now.
//
float heightMap(vec2 p){
    
    p /= 2.; // Extra scaling.
    
    float  h = 0., a = 1., sum = 0.; // Height, amplitude, sum.
    
    for(int i=0; i<4; i++){
    
        p = fract(p)*3.; // Subdividing space.
        // Far more interesting, mutated subdivision, courtesy of Aiekick.
        //p = fract(p+sin(p.yx*9.)*0.025 + cos(p.yx*9.)*0.025)*3.; 
        // Another one with a time component.
        //p = fract(p + sin(p*9. + cos(p.yx*13. + iTime*2.))*0.02)*3.;
        
        vec2 w = .5 - abs(p - 1.5); // Prepare to make a square. Other shapes are also possible.
        float l = sqrt( max(16.0*w.x*w.y*(1.0-w.x)*(1.0-w.y), 0.))*.5+.5; // Edge shaping.
        w = smoothstep(0., .05, w); // Smooth edge stepping.
        h = max(h, w.x*w.y*a*l); // Producing the smooth edged, shaped square.
        //h += w.x*w.y*a*l;
        //h = max(h, abs(abs(w.x)-abs(w.y))*a*l);
        sum += a; // Keep a total... This could be hardcoded to save cycles.
        a *= .4; // Lower the amplitude for the next subdivision, just because it looks tidier.
        //if(i==2)a*=.75;
    }
    
    return h/sum;
    
}

// Raymarching a heightmap on an XY-plane. Pretty standard.
float map(vec3 p){

    // Cheap, lame distortion, if you wanted it.
    //p.xy += sin(p.xy*7. + cos(p.yx*13. + iTime))*.01;
    
    // Back plane, placed at vec3(0, 0, 1), with plane normal vec3(0., 0., -1).
    // Adding some height to the plane from the heightmap. Not much else to it.
    return 1. - p.z - heightMap(p.xy)*.125;
    
}

// Texture bump mapping. Four tri-planar lookups, or 12 texture lookups in total. I tried to 
// make it as concise as possible. Whether that translates to speed, or not, I couldn't say.
vec3 doBumpMap( sampler2D tx, in vec3 p, in vec3 n, float bf){
   
    const vec2 e = vec2(0.001, 0);
    
    // Three gradient vectors rolled into a matrix, constructed with offset greyscale texture values.    
    mat3 m = mat3( tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));
    
    vec3 g = vec3(0.299, 0.587, 0.114)*m; // Converting to greyscale.
    g = (g - dot(tex3D(tx,  p , n), vec3(0.299, 0.587, 0.114)) )/e.x; g -= n*dot(n, g);
                      
    return normalize( n + g*bf ); // Bumped normal. "bf" - bump factor.
    
}


// Standard normal function.
vec3 getNormal(in vec3 p) {
	const vec2 e = vec2(0.0025, 0);
	return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
}

// I keep a collection of occlusion routines... OK, that sounded really nerdy. :)
// Anyway, I like this one. I'm assuming it's based on IQ's original.
float calculateAO(in vec3 pos, in vec3 nor)
{
	float sca = 3., occ = 0.;
    for(int i=0; i<5; i++){
    
        float hr = .01 + float(i)*.5/4.;        
        float dd = map(nor * hr + pos);
        occ += (hr - dd)*sca;
        sca *= 0.7;
    }
    return clamp(1.0 - occ, 0., 1.);    
}

// Basic raymarcher.
float trace(in vec3 ro, in vec3 rd){
    
    // Note that the ray is starting just above the raised plane, since nothing is
    // in the way. It's normal practice to start at zero.
    float d, t = 0.75; 
    for(int j=0;j<32;j++){
      
        d = map(ro + rd*t); // distance to the function.
        // The plane "is" the far plane, so no far=plane break is needed.
        if(abs(d)<0.001*(t*.125 + 1.) || t>FAR) break;

        t += d*.7; // Total distance from the camera to the surface.
    
    }

    return min(t, FAR);
    
}

// Cool curve function, by Shadertoy user, Nimitz.
//
// It gives you a scalar curvature value for an object's signed distance function, which 
// is pretty handy for all kinds of things. Here's it's used to darken the crevices.
//
// From an intuitive sense, the function returns a weighted difference between a surface 
// value and some surrounding values - arranged in a simplex tetrahedral fashion for minimal
// calculations, I'm assuming. Almost common sense... almost. :)
//
// Original usage (I think?) - Cheap curvature: https://www.shadertoy.com/view/Xts3WM
// Other usage: Xyptonjtroz: https://www.shadertoy.com/view/4ts3z2
float curve(in vec3 p){

    const float eps = 0.02, amp = 8., ampInit = 0.6;

    vec2 e = vec2(-1., 1.)*eps; //0.05->3.5 - 0.04->5.5 - 0.03->10.->0.1->1.
    
    float t1 = map(p + e.yxx), t2 = map(p + e.xxy);
    float t3 = map(p + e.xyx), t4 = map(p + e.yyy);
    
    return clamp((t1 + t2 + t3 + t4 - 4.*map(p))*amp + ampInit, 0., 1.);
}


// Simple environment mapping. Pass the reflected vector in and create some
// colored noise with it. The normal is redundant here, but it can be used
// to pass into a 3D texture mapping function to produce some interesting
// environmental reflections.
vec3 envMap(vec3 rd, vec3 sn){
    
    vec3 sRd = rd; // Save rd, just for some mixing at the end.
    
    // Add a time component, scale, then pass into the noise function.
    rd.xy -= iTime*.25;
    rd *= 3.;
    
    float c = n3D(rd)*.57 + n3D(rd*2.)*.28 + n3D(rd*4.)*.15; // Noise value.
    c = smoothstep(0.4, 1., c); // Darken and add contast for more of a spotlight look.
    
    vec3 col = vec3(c, c*c, c*c*c*c); // Simple, warm coloring.
    //vec3 col = vec3(min(c*1.5, 1.), pow(c, 2.5), pow(c, 12.)); // More color.
    
    // Mix in some more red to tone it down and return.
    return mix(col, col.yzx, sRd*.25+.25); 
    
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ){
    
    
    // Unit directional ray with no divide, courtesy of Coyote.
    vec3 rd = normalize(vec3(2.*fragCoord - iResolution.xy, iResolution.y));
    
    // Rotating the XY-plane back and forth, for a bit of variance.
    // 2D rotation with fewer instructions, courtesy of Fabrice Neyret.
    vec2 a = sin(vec2(1.570796, 0) - sin(iTime/4.)*.3);
    rd.xy = rd.xy*mat2(a, -a.y, a.x);
    
    
    // Ray origin. Moving in the X-direction to the right.
    vec3 ro = vec3(iTime, cos(iTime/4.), 0.);
    
    
    // Light position, hovering around camera.
    vec3 lp = ro + vec3(cos(iTime/2.)*.5, sin(iTime/2.)*.5, -.5);
    
    // Standard raymarching segment. Because of the straight forward setup, not many 
    // iterations are needed.
 	float t = trace(ro, rd);
    
   
    // Surface postion, surface normal and light direction.
    vec3 sp = ro + rd*t;
    vec3 sn = getNormal(sp);
    
    
	// Texture scale factor.
    const float tSize0 = 1./2.;
    // Texture-based bump mapping.
	sn = doBumpMap(iChannel0, sp*tSize0, sn, 0.002);    
    
    
    // Point light.
    vec3 ld = lp - sp; // Light direction vector.
    float lDist = max(length(ld), 0.001); // Light distance.
    float atten = 1./(1. + lDist*lDist*.125); // Light attenuation.
    ld /= lDist; // Normalizing the light direction vector.
    
   // Obtaining the surface texel, then ramping up the contrast a bit.
    vec3 oC = smoothstep(0., 1., tex3D(iChannel0, sp*tSize0, sn));
    // Using the height map to highlight the raised squares. Not for any particular reason.
    oC *= smoothstep(0., .125, heightMap(sp.xy))*1.5 + .5;

    
    float diff = max(dot(ld, sn), 0.); // Diffuse.
    float spec = pow(max( dot( reflect(-ld, sn), -rd ), 0.0 ), 32.); // Specular.
    float fre = clamp(dot(sn, rd) + 1., .0, 1.); // Fake fresnel, for the glow.
    
    // Shading. Note, there are no actual shadows. The camera is front on, so the following
    // two functions are enough to give a shadowy appearance.
    float crv = curve(sp); // Curve value, to darken the crevices.
    float ao = calculateAO(sp, sn); // Ambient occlusion, for self shadowing.
 
    
    // Combining the terms above to light the texel.
    vec3 col = (oC*(diff + .25) + vec3(1, .7, .3)*spec) + vec3(.1, .3, 1)*pow(fre, 4.)*4.;
    
    col += (oC*.5+.5)*envMap(reflect(rd, sn), sn)*6.; // Fake environment mapping.
    //col += envMap(reflect(rd, sn), sn)*4.;
    
    // Applying the shades.
    col *= (atten*crv*ao);
    
    // Vignette.
    vec2 uv = fragCoord/iResolution.xy;
    col *= pow(16.*uv.x*uv.y*(1.-uv.x)*(1.-uv.y), 0.125);

    
    // Presenting to the screen.
	fragColor = vec4(sqrt(clamp(col, 0., 1.)), 1.);
}
`,
"Voxel Corridor": `/*

	Voxel Corridor
	--------------

	I love the voxel aesthetic, so after looking at some of Akohdr's examples, I went on a bit 
	of a voxel trip and put this simple scene together... Although, "scene" would  be putting 
	it loosely. :)

	Quasi-discreet distance calculations sound simple enough to perform in theory, but are just 
	plain fiddly to code, so I was very thankful to have fb39ca4's, IQ's, Reinder's, and everyone 
	elses voxel examples to refer to.

	The code is pretty straight forward. I tried my best to write it in such way that enables
	someone to plug in any normal distance function and have it render the voxelized version.

	Mainly based on the following:

	Voxel Ambient Occlusion - fb39ca4
    https://www.shadertoy.com/view/ldl3DS

	Minecraft - Reinder
    https://www.shadertoy.com/view/4ds3WS

	Other examples:
	Rounded Voxels - IQ
    https://www.shadertoy.com/view/4djGWR

	Sampler - w23
	https://www.shadertoy.com/view/MlfGRM

	Text In Space - akohdr
	https://www.shadertoy.com/view/4d3SWB

*/

#define PI 3.14159265
#define FAR 60.

// 2x2 matrix rotation. Note the absence of "cos." It's there, but in disguise, and comes courtesy
// of Fabrice Neyret's "ouside the box" thinking. :)
mat2 rot2( float a ){ vec2 v = sin(vec2(1.570796, 0) + a);	return mat2(v, -v.y, v.x); }

// Tri-Planar blending function. Based on an old Nvidia tutorial.
vec3 tex3D( sampler2D tex, in vec3 p, in vec3 n ){
  
    n = max(abs(n), 0.001);//n = max((abs(n) - 0.2)*7., 0.001); //  etc.
    n /= (n.x + n.y + n.z ); 
	p = (texture(tex, p.yz)*n.x + texture(tex, p.zx)*n.y + texture(tex, p.xy)*n.z).xyz;
    return p*p;
}

// The path is a 2D sinusoid that varies over time, depending upon the frequencies, and amplitudes.
vec2 path(in float z){ 
    //return vec2(0); // Straight.
    float a = sin(z * 0.11);
    float b = cos(z * 0.14);
    return vec2(a*4. -b*1.5, b*1.7 + a*1.5); 
    //return vec2(a*4. -b*1.5, 0.); // Just X.
    //return vec2(0, b*1.7 + a*1.5); // Just Y.
}

/*
// Alternate distance field -- Twisted planes. 
float map(vec3 p){
    
     // You may need to reposition the light to work in with the shadows, but for
     // now, I'm repositioning the scene up a bit.
     p.y -= .75;
     p.xy -= path(p.z); // Move the scene around a sinusoidal path.
     p.xy = rot2(p.z/8.)*p.xy; // Twist it about XY with respect to distance.
    
     float n = dot(sin(p*1. + sin(p.yzx*.5 + iTime*.0)), vec3(.25)); // Sinusoidal layer.
     
     return 4. - abs(p.y) + n; // Warped double planes, "abs(p.y)," plus surface layers.
 
}
*/

// Standard perturbed tunnel function.
//
float map(vec3 p){
     
     // Offset the tunnel about the XY plane as we traverse Z.
     p.xy -= path(p.z);
    
     // Standard tunnel.
     float n = 5. - length(p.xy*vec2(1, .8));
     // Square tunnel. Almost redundant in a voxel renderer. :)
     //float n = 5. - max(abs(p.x), abs(p.y*.8));
     
     // Tunnel with a floor.
     return min(p.y + 3., n); //n = min(-abs(p.y) + 3., n);
 
}

/*
float brickShade(vec2 p){
    
    p.x -= step(p.y, 1.)*.5;
    
    p = fract(p);
    
    return pow(16.*p.x*p.y*(1.-p.x)*(1.-p.y), 0.25);
    
}
*/

// The brick groove pattern. Thrown together too quickly.
// Needs some tidy up, but it's quick enough for now.
//
const float w2h = 2.; // Width to height ratio.
const float mortW = 0.05; // Morter width.

float brickMorter(vec2 p){
	
    p.x -= step(1., p.y)*.5;
    
    p = abs(fract(p + vec2(0, .5)) - .5)*2.;
    
    // Smooth grooves. Better for bump mapping.
    return smoothstep(0., mortW, p.x)*smoothstep(0., mortW*w2h, p.y);
    
}

float brick(vec2 p){
    
	p = fract(p*vec2(0.5/w2h, 0.5))*2.;

    return brickMorter(p);//*(brickShade(p)*.25 + .75);
}


// Surface bump function. Cheap, but with decent visual impact.
float bumpSurf3D( in vec3 p, in vec3 n){

    n = abs(n);
    
    if (n.x>0.5) p.xy = p.zy;
    else if (n.y>0.5) p.xy = p.xz;
    
    return brick(p.xy);
    
}

// Standard function-based bump mapping function.
vec3 doBumpMap(in vec3 p, in vec3 nor, float bumpfactor){
    
    const vec2 e = vec2(0.001, 0);
    float ref = bumpSurf3D(p, nor);                 
    vec3 grad = (vec3(bumpSurf3D(p - e.xyy, nor),
                      bumpSurf3D(p - e.yxy, nor),
                      bumpSurf3D(p - e.yyx, nor) )-ref)/e.x;                     
          
    grad -= nor*dot(nor, grad);          
                      
    return normalize( nor + grad*bumpfactor );
	
}

// Texture bump mapping. Four tri-planar lookups, or 12 texture lookups in total. I tried to 
// make it as concise as possible. Whether that translates to speed, or not, I couldn't say.
vec3 doBumpMap( sampler2D tx, in vec3 p, in vec3 n, float bf){
   
    const vec2 e = vec2(0.001, 0);
    
    // Three gradient vectors rolled into a matrix, constructed with offset greyscale texture values.    
    mat3 m = mat3( tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));
    
    vec3 g = vec3(0.299, 0.587, 0.114)*m; // Converting to greyscale.
    g = (g - dot(tex3D(tx,  p , n), vec3(0.299, 0.587, 0.114)) )/e.x; g -= n*dot(n, g);
                      
    return normalize( n + g*bf ); // Bumped normal. "bf" - bump factor.
    
}


// This is just a slightly modified version of fb39ca4's code, with some
// elements from IQ and Reinder's examples. They all work the same way:
// Obtain the current voxel, then test the distance field for a hit. If
// the ray has moved into the voxelized isosurface, break. Otherwise, move
// to the next voxel. That involves a bit of decision making - due to the
// nature of voxel boundaries - and the "mask," "side," etc, variable are
// an evolution of that. If you're not familiar with the process, it's 
// pretty straight forward, and there are a lot of examples on Shadertoy, 
// plus a lot more articles online.
//
vec3 voxelTrace(vec3 ro, vec3 rd, out vec3 mask){
    
    vec3 p = floor(ro) + .5;

	vec3 dRd = 1./abs(rd); // 1./max(abs(rd), vec3(.0001));
	rd = sign(rd);
    vec3 side = dRd*(rd*(p - ro) + .5);
    
    mask = vec3(0);
	
	for (int i = 0; i < 64; i++) {
		
        if (map(p)<0.) break;
        
        // Note that I've put in the messy reverse step to accomodate
        // the "less than or equals" logic, rather than just the "less than."
        // Without it, annoying seam lines can appear... Feel free to correct
        // me on that, if my logic isn't up to par. It often isn't. :)
        mask = step(side, side.yzx)*(1. - step(side.zxy, side));
		side += mask*dRd;
		p += mask*rd;
	}
    
    return p;    
}


// Voxel shadows. They kind of work like regular hard-edged shadows. They
// didn't present too many problems, but it was still nice to have Reinder's
// Minecraft shadow example as a reference. Fantastic example, if you've
// never seen it:
//
// Minecraft - Reinder
// https://www.shadertoy.com/view/4ds3WS
//
float voxShadow(vec3 ro, vec3 rd, float end){

    float shade = 1.;
    vec3 p = floor(ro) + .5;

	vec3 dRd = 1./abs(rd);//1./max(abs(rd), vec3(.0001));
	rd = sign(rd);
    vec3 side = dRd*(rd*(p - ro) + .5);
    
    vec3 mask = vec3(0);
    
    float d = 1.;
	
	for (int i = 0; i < 16; i++) {
		
        d = map(p);
        
        if (d<0. || length(p - ro)>end) break;
        
        mask = step(side, side.yzx)*(1. - step(side.zxy, side));
		side += mask*dRd;
		p += mask*rd;                
	}

    // Shadow value. If in shadow, return a dark value.
    return shade = step(0., d)*.7 + .3;
    
}

///////////
//
// This is a trimmed down version of fb39ca4's voxel ambient occlusion code with some 
// minor tweaks and adjustments here and there. The idea behind voxelized AO is simple. 
// The execution, not so much. :) So damn fiddly. Thankfully, fb39ca4, IQ, and a few 
// others have done all the hard work, so it's just a case of convincing yourself that 
// it works and using it.
//
// Refer to: Voxel Ambient Occlusion - fb39ca4
// https://www.shadertoy.com/view/ldl3DS
//
vec4 voxelAO(vec3 p, vec3 d1, vec3 d2) {
   
    // Take the four side and corner readings... at the correct positions...
    // That's the annoying bit that I'm glad others have worked out. :)
	vec4 side = vec4(map(p + d1), map(p + d2), map(p - d1), map(p - d2));
	vec4 corner = vec4(map(p + d1 + d2), map(p - d1 + d2), map(p - d1 - d2), map(p + d1 - d2));
	
    // Quantize them. It's either occluded, or it's not, so to speak.
    side = step(side, vec4(0));
    corner = step(corner, vec4(0));
    
    // Use the side and corner values to produce a more honed in value... kind of.
    return 1. - (side + side.yzwx + max(corner, side*side.yzwx))/3.;    
	
}

float calcVoxAO(vec3 vp, vec3 sp, vec3 rd, vec3 mask) {
    
    // Obtain four AO values at the appropriate quantized positions.
	vec4 vAO = voxelAO(vp - sign(rd)*mask, mask.zxy, mask.yzx);
    
    // Use the fractional voxel postion and and the proximate AO values
    // to return the interpolated AO value for the surface position.
    sp = fract(sp);
    vec2 uv = sp.yz*mask.x + sp.zx*mask.y + sp.xy*mask.z;
    return mix(mix(vAO.z, vAO.w, uv.x), mix(vAO.y, vAO.x, uv.x), uv.y);

}
///////////

void mainImage( out vec4 fragColor, in vec2 fragCoord ){
	
	// Screen coordinates.
	vec2 uv = (fragCoord - iResolution.xy*0.5)/iResolution.y;
	
	// Camera Setup.
	vec3 camPos = vec3(0., 0.5, iTime*8.); // Camera position, doubling as the ray origin.
	vec3 lookAt = camPos + vec3(0.0, 0.0, 0.25);  // "Look At" position.

 
    // Light positioning. 
 	vec3 lightPos = camPos + vec3(0, 2.5, 8);// Put it a bit in front of the camera.

	// Using the Z-value to perturb the XY-plane.
	// Sending the camera, "look at," and two light vectors down the tunnel. The "path" function is 
	// synchronized with the distance function. Change to "path2" to traverse the other tunnel.
	lookAt.xy += path(lookAt.z);
	camPos.xy += path(camPos.z);
	lightPos.xy += path(lightPos.z);

    // Using the above to produce the unit ray-direction vector.
    float FOV = PI/2.; // FOV - Field of view.
    vec3 forward = normalize(lookAt-camPos);
    vec3 right = normalize(vec3(forward.z, 0., -forward.x )); 
    vec3 up = cross(forward, right);

    // rd - Ray direction.
    vec3 rd = normalize(forward + FOV*uv.x*right + FOV*uv.y*up);
    
    //vec3 rd = normalize(forward + FOV*uv.x*right + FOV*uv.y*up);
    //rd = normalize(vec3(rd.xy, rd.z - dot(rd.xy, rd.xy)*.25));    
    
    // Swiveling the camera about the XY-plane (from left to right) when turning corners.
    // Naturally, it's synchronized with the path in some kind of way.
	rd.xy = rot2( path(lookAt.z).x/24. )*rd.xy;

    // Raymarch the voxel grid.
    vec3 mask;
	vec3 vPos = voxelTrace(camPos, rd, mask);
	
    // Using the voxel position to determine the distance from the camera to the hit point.
    // I'm assuming IQ is responsible for this clean piece of logic.
	vec3 tCube = (vPos - camPos - .5*sign(rd))/rd;
    float t = max(max(tCube.x, tCube.y), tCube.z);
    
	
    // Initialize the scene color.
    vec3 sceneCol = vec3(0);
	
	// The ray has effectively hit the surface, so light it up.
	if(t<FAR){
	
   	
    	// Surface position and surface normal.
	    vec3 sp = camPos + rd*t;
        
        // Voxel normal.
        vec3 sn = -(mask*sign( rd ));
        
        // Sometimes, it's necessary to save a copy of the unbumped normal.
        vec3 snNoBump = sn;
        
        // I try to avoid it, but it's possible to do a texture bump and a function-based
        // bump in succession. It's also possible to roll them into one, but I wanted
        // the separation... Can't remember why, but it's more readable anyway.
        //
        // Texture scale factor.
        const float tSize0 = 1./4.;
        // Texture-based bump mapping.
	    sn = doBumpMap(iChannel0, sp*tSize0, sn, 0.02);

        // Function based bump mapping. Comment it out to see the under layer. It's pretty
        // comparable to regular beveled Voronoi... Close enough, anyway.
        sn = doBumpMap(sp, sn, .15);
        
       
	    // Ambient occlusion.
	    float ao = calcVoxAO(vPos, sp, rd, mask) ;//calculateAO(sp, sn);//*.75 + .25;

        
    	// Light direction vectors.
	    vec3 ld = lightPos-sp;

        // Distance from respective lights to the surface point.
	    float lDist = max(length(ld), 0.001);
    	
    	// Normalize the light direction vectors.
	    ld /= lDist;
	    
	    // Light attenuation, based on the distances above.
	    float atten = 1./(1. + lDist*.2 + lDist*0.1); // + distlpsp*distlpsp*0.025
    	
    	// Ambient light.
	    float ambience = 0.25;
    	
    	// Diffuse lighting.
	    float diff = max( dot(sn, ld), 0.0);
   	
    	// Specular lighting.
	    float spec = pow(max( dot( reflect(-ld, sn), -rd ), 0.0 ), 32.);

	    
	    // Fresnel term. Good for giving a surface a bit of a reflective glow.
        //float fre = pow( clamp(dot(sn, rd) + 1., .0, 1.), 1.);
        
 
        // Object texturing.
        //
        // Obfuscated way to tinge the floor and ceiling with a bit of brown.
	    vec3 texCol = vec3(1, .6, .4) + step(abs(snNoBump.y), .5)*vec3(0,.4, .6);
	    
        // Multiplying by the texture color.
	    texCol *= tex3D(iChannel0, sp*tSize0, sn);
        
        //texCol *= bumpSurf3D( sp, sn)*.25 + .75; // Darken the grout, if you wanted.

        
        // Shadows... I was having all sorts of trouble trying the move the ray off off the
        // block. Thanks to Reinder's "Minecraft" example for showing me the ray needs to 
        // be bumped off by the normal, not the unit direction ray. :)
        float shading = voxShadow(sp + snNoBump*.01, ld, lDist);
    	
    	// Combining the above terms to produce the final color. It was based more on acheiving a
        // certain aesthetic than science.
        sceneCol = texCol*(diff + ambience) + vec3(.7, .9, 1.)*spec;// + vec3(.5, .8, 1)*spec2;
        //sceneCol += texCol*vec3(.8, .95, 1)*pow(fre, 4.)*2.; // White mortar... not really.
        


	    // Shading.
        sceneCol *= atten*shading*ao;
        
        // "fb39ca4" did such a good job with the AO, that it's worth a look on its own. :)
        //sceneCol = vec3(ao); 

	   
	
	}
       
    // Blend in a bit of logic-defying fog for atmospheric effect. :)
    sceneCol = mix(sceneCol, vec3(.08, .16, .34), smoothstep(0., .95, t/FAR)); // exp(-.002*t*t), etc.

    // Clamp and present the badly gamma corrected pixel to the screen.
	fragColor = vec4(sqrt(clamp(sceneCol, 0., 1.)), 1.0);
	
}`,
    "Mandelbrot Fract": `// ----------------CAUTION!!!--- FLASHING BRIGHT LIGHTS!!!-------------------------



// Credits - fractal zoom with smooth iter count adapted from - iq (Inigo quilez) - https://iquilezles.org/articles/msetsmooth
// Koch Snowflake symmetry from tutorial by Martijn Steinrucken aka The Art of Code/BigWings - 2020 - https://www.youtube.com/watch?v=il_Qg9AqQkE&ab_channel=TheArtofCode
//music - Sajanka (Official) Sajanka - Sun Is Coming

//Some notes - color is determined by date and not time - hour of day dependent.
//Move the mouse on the Y axis to change the symmetry.

//----------------------------------------------------------------------------------

//uncomment to sample audio input from MIC instead of SoundCloud.
//#define MIC_INPUT

//comment to make it less trippy and noisy.
#define EXTRA_DMT


#define PI 3.14159265359



#define date iDate
#define time iTime
#define resolution iResolution

float freqs[4];


vec2 rot(vec2 p,float a){
    
    float c = cos(a);
    float s = sin(a);
    
    mat2 m = mat2(c,-s,s,c);
    
    p*=m;
    return p  ;
}

float localTime(){

float d = date.w / date.x;
return d;

}

vec3 randomCol(float sc){

 float d = localTime();
	float r = sin(sc * 1. * d)*.5+.5;
	float g = sin(sc * 2. * d)*.5+.5;
	float b = sin(sc * 4. * d)*.5+.5;

	vec3 col = vec3(r,g,b);
	col = clamp(col,0.,1.);

	return col;
	}


//--------------------------------------------------mandelbrot generator-----------https://iquilezles.org/articles/msetsmooth

	float mandelbrot(vec2 c )
{
    #if 1
    {
        float c2 = dot(c, c);
        // skip computation inside M1 - https://iquilezles.org/articles/mset1bulb
        if( 256.0*c2*c2 - 96.0*c2 + 32.0*c.x - 3.0 < 0.0 ) return 0.0;
        // skip computation inside M2 - https://iquilezles.org/articles/mset2bulb
        if( 16.0*(c2+2.0*c.x+1.0) - 1.0 < 0.0 ) return 0.0;
    }
    #endif


    const float B = 128.0;
    float l = 0.0;
    vec2 z  = vec2(0.0);
    for( int i=0; i<256; i++ )
    {
        z = vec2( z.x*z.x - z.y*z.y, 2.0*z.x*z.y ) + c;
        if( dot(z,z)>(B*B) ) break;
        l += 1.0;
    }

    if( l>255.0 ) return 0.0;


    // equivalent optimized smooth interation count
    float sl = l - log2(log2(dot(z,z))) + 4.0;



     return sl;
 }


vec3 mandelbrotImg(vec2 p)
{

    //uncomment to see unmaped set
	//p = (-resolution.xy + 2.0*gl_FragCoord.xy)/resolution.y;
    float mtime =  time;
    mtime -= freqs[3];
    float zoo = 0.62 + 0.38*cos(.1*mtime);
   float coa = cos( 0.015*(1.0-zoo)*mtime );
   float sia = sin( 0.015*(1.0-zoo)*mtime );
   zoo = pow( zoo,6.0);
   vec2 xy = vec2( p.x*coa-p.y*sia, p.x*sia+p.y*coa);
   vec2 c = vec2(-.745,.186) + xy*zoo;

        float l = mandelbrot(c);
        
        
	vec3 col1 = 0.5 + 0.5*cos( 3.0 + l*.15 + randomCol(.1));
    #ifdef EXTRA_DMT
    vec3 col2 = 0.5 + 0.5*cos( 3.0 + l*.15 / randomCol(.1));
    #else
    vec3 col2 = 0.5 + 0.5*cos( 3.0 + l*.15 * randomCol(.1));
    #endif
    vec3 col = mix(col1,col2,sin(mtime)*.5+.5);




return col;
}

//-----------------functions-----------

float remap(float a1, float a2 ,float b1, float b2, float t)
{
	return b1+(t-a1)*(b2-b1)/(a2-a1);
}


vec2 remap(float a1, float a2 ,float b1, float b2, vec2 t)
{
	return b1+(t-a1)*(b2-b1)/(a2-a1);
}


vec4 remap(float a1, float a2 ,float b1, float b2, vec4 t)
{
	return b1+(t-a1)*(b2-b1)/(a2-a1);
}





vec2 N(float angle) {
    return vec2(sin(angle), cos(angle));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
//--------get soundtrack frequencies----


    #ifdef MIC_INPUT
	freqs[0] = texture( iChannel1, vec2( 0.01, 0.25 ) ).x;
	freqs[1] = texture( iChannel1, vec2( 0.07, 0.25 ) ).x;
	freqs[2] = texture( iChannel1, vec2( 0.15, 0.25 ) ).x;
	freqs[3] = texture( iChannel1, vec2( 0.30, 0.25 ) ).x;
    #else 
    freqs[0] = texture( iChannel0, vec2( 0.01, 0.25 ) ).x;
	freqs[1] = texture( iChannel0, vec2( 0.07, 0.25 ) ).x;
	freqs[2] = texture( iChannel0, vec2( 0.15, 0.25 ) ).x;
	freqs[3] = texture( iChannel0, vec2( 0.30, 0.25 ) ).x;
    #endif
    float avgFreq = (freqs[0] +freqs[1] +freqs[2] +freqs[3])/4.;

//--------image part---------
    vec2 uv = (fragCoord.xy-.5*resolution.xy)/resolution.y;
	vec2 mouse = 1. - iMouse.xy/resolution.xy; // 0 1
	vec2 ouv = uv;
    //uv.y -= .05;
    
    uv = rot(uv,(sin(iTime*.1) / freqs[0]*.1  ) * PI  ) ;  
	uv *= 4.0 - (avgFreq * 1.5  );

    uv.x = abs(uv.x);
    
    vec3 col = vec3(0);
    float d;

    float angle = 0.;
    vec2 n = N((5./6.)*3.1415);

    uv.y += tan((5./6.)*3.1415)*.5;
   	d = dot(uv-vec2(.5, 0), n);
    uv -= max(0.,d)*n*2.;

    float scale = 1.;

    n = N( freqs[0]*(2./3.)*3.1415);
    uv.x += .5;
    for(int i=0; i<10; i++) {
        uv *= 3.;
        scale *= 3.;
        uv.x -= 1.5;

        uv.x = abs(uv.x);
        uv.x -= .5;
        d = dot(uv, n);
        uv -= min(0.,d)*n*2.;
    }

    d = length(uv/ clamp(freqs[2],0.1,.9 )- vec2(clamp(uv.x,-1., 1.), 0));
    col += smoothstep(10./resolution.y, .0, d/scale);
    uv /= scale;	// normalization

   
	vec3 manCol = mandelbrotImg(uv);
	 col += manCol;


 		// vignette effect
	  col *= 1.0 - 0.5*length(uv *0.5) * freqs[1];

	 
    fragColor = vec4( col,1.0);
}`,
"Crystalline": `#define DTR 0.01745329
#define rot(a) mat2(cos(a),sin(a),-sin(a),cos(a))

vec2 uv;
vec3 cp,cn,cr,ro,rd,ss,oc,cc,gl,vb;
vec4 fc;
float tt,cd,sd,io,oa,td;
int es=0,ec;

float bx(vec3 p,vec3 s){vec3 q=abs(p)-s;return min(max(q.x,max(q.y,q.z)),0.)+length(max(q,0.));}
float smin(float a, float b, float k){float h=clamp(0.5+0.5*(b-a)/k,0.,1.);return mix(b,a,h)-k*h*(1.-h);}

vec3 lattice(vec3 p, int iter, float an)
{
		for(int i = 0; i < iter; i++)
		{
			p.xy *= rot(an*DTR);
			p.yz=abs(p.yz)-1.;
			p.xz *= rot(-an*DTR);
		}
		return p;
}

float mp(vec3 p)
{
//now with mouse control
if(iMouse.z>0.){
    p.yz*=rot(2.0*(iMouse.y/iResolution.y-0.5));
    p.zx*=rot(-7.0*(iMouse.x/iResolution.x-0.5));
}
		vec3 pp=p;
		
		p.xz*=rot(tt*0.1);
		p.xy*=rot(tt*0.1);

		p=lattice(p,9,45.+cos(tt*0.1)*5.);
	

		sd = bx(p,vec3(1)) - 0.01;
	
		sd = smin(sd, sd, 0.8);

		gl += exp(-sd*0.001) * normalize(p*p) * 0.003;
	
		sd=abs(sd)-0.001;

		if(sd<0.001)
		{
			oc=vec3(1);
			io=1.2;
			oa=0.0;
			ss=vec3(0);
		  vb=vec3(0.,10,2.8);
			ec=2;	
		}
		return sd;
}

void tr(){vb.x=0.;cd=0.;for(float i=0.;i<256.;i++){mp(ro+rd*cd);cd+=sd;td+=sd;if(sd<0.0001||cd>128.)break;}}
void nm(){mat3 k=mat3(cp,cp,cp)-mat3(.001);cn=normalize(mp(cp)-vec3(mp(k[0]),mp(k[1]),mp(k[2])));}

void px()
{
  cc=vec3(0.35,0.25,0.45)+length(pow(abs(rd+vec3(0,0.5,0)),vec3(3)))*0.3+gl;
  vec3 l=vec3(0.9,0.7,0.5);
  if(cd>128.){oa=1.;return;}
  float df=clamp(length(cn*l),0.,1.);
  vec3 fr=pow(1.-df,3.)*mix(cc,vec3(0.4),0.5);
	float sp=(1.-length(cross(cr,cn*l)))*0.2;
	float ao=min(mp(cp+cn*0.3)-0.3,0.3)*0.4;
  cc=mix((oc*(df+fr+ss)+fr+sp+ao+gl),oc,vb.x);
}

void render(vec2 frag, vec2 res, float time, out vec4 col)
{
	tt=mod(time+25., 260.);
  uv=vec2(frag.x/res.x,frag.y/res.y);
  uv-=0.5;uv/=vec2(res.y/res.x,1);
	float an = (sin(tt*0.3)*0.5+0.5);
    an = 1.-pow(1.-pow(an, 5.),10.);
  ro=vec3(0,0,-5. - an*15.);rd=normalize(vec3(uv,1));
  
	for(int i=0;i<25;i++)
  {
		tr();cp=ro+rd*cd;
    nm();ro=cp-cn*0.01;
    cr=refract(rd,cn,i%2==0?1./io:io);
    if(length(cr)==0.&&es<=0){cr=reflect(rd,cn);es=ec;}
    if(max(es,0)%3==0&&cd<128.)rd=cr;es--;
		if(vb.x>0.&&i%2==1)oa=pow(clamp(cd/vb.y,0.,1.),vb.z);
		px();fc=fc+vec4(cc*oa,oa)*(1.-fc.a);	
		if((fc.a>=1.||cd>128.))break;
  }
  col = fc/fc.a;
}
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    render(fragCoord.xy,iResolution.xy,iTime,fragColor);
} `,
    "Silk (Needs work)": `float colormap_red(float x) {
    if (x < 0.0) {
        return 54.0 / 255.0;
    } else if (x < 20049.0 / 82979.0) {
        return (829.79 * x + 54.51) / 255.0;
    } else {
        return 1.0;
    }
}

float colormap_green(float x) {
    if (x < 20049.0 / 82979.0) {
        return 0.0;
    } else if (x < 327013.0 / 810990.0) {
        return (8546482679670.0 / 10875673217.0 * x - 2064961390770.0 / 10875673217.0) / 255.0;
    } else if (x <= 1.0) {
        return (103806720.0 / 483977.0 * x + 19607415.0 / 483977.0) / 255.0;
    } else {
        return 1.0;
    }
}

float colormap_blue(float x) {
    if (x < 0.0) {
        return 54.0 / 255.0;
    } else if (x < 7249.0 / 82979.0) {
        return (829.79 * x + 54.51) / 255.0;
    } else if (x < 20049.0 / 82979.0) {
        return 127.0 / 255.0;
    } else if (x < 327013.0 / 810990.0) {
        return (792.02249341361393720147485376583 * x - 64.364790735602331034989206222672) / 255.0;
    } else {
        return 1.0;
    }
}

vec4 colormap(float x) {
    return vec4(colormap_red(x), colormap_green(x), colormap_blue(x), 1.0);
}

// https://iquilezles.org/articles/warp
/*float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float a = textureLod(iChannel0,(p+vec2(0.5,0.5))/256.0,0.0).x;
	float b = textureLod(iChannel0,(p+vec2(1.5,0.5))/256.0,0.0).x;
	float c = textureLod(iChannel0,(p+vec2(0.5,1.5))/256.0,0.0).x;
	float d = textureLod(iChannel0,(p+vec2(1.5,1.5))/256.0,0.0).x;
    return mix(mix( a, b,f.x), mix( c, d,f.x),f.y);
}*/


float rand(vec2 n) { 
    return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

float noise(vec2 p){
    vec2 ip = floor(p);
    vec2 u = fract(p);
    u = u*u*(3.0-2.0*u);

    float res = mix(
        mix(rand(ip),rand(ip+vec2(1.0,0.0)),u.x),
        mix(rand(ip+vec2(0.0,1.0)),rand(ip+vec2(1.0,1.0)),u.x),u.y);
    return res*res;
}

const mat2 mtx = mat2( 0.80,  0.60, -0.60,  0.80 );

float fbm( vec2 p )
{
    float f = 0.0;

    f += 0.500000*noise( p + iTime  ); p = mtx*p*2.02;
    f += 0.031250*noise( p ); p = mtx*p*2.01;
    f += 0.250000*noise( p ); p = mtx*p*2.03;
    f += 0.125000*noise( p ); p = mtx*p*2.01;
    f += 0.062500*noise( p ); p = mtx*p*2.04;
    f += 0.015625*noise( p + sin(iTime) );

    return f/0.96875;
}

float pattern( in vec2 p )
{
	return fbm( p + fbm( p + fbm( p ) ) );
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.x;
	float shade = pattern(uv);
    fragColor = vec4(colormap(shade).rgb, shade);
}

/** SHADERDATA
{
	"title": "Base warp fBM",
	"description": "Noise but Pink",
	"model": "person"
}
*/`,
"Sunny Portal": `void mainImage(out vec4 o, vec2 u) {
    // Initialize variables properly
    float i = 0.0;
    float d = 0.0;
    float s = 0.0;
    
    // Normalize coordinates
    vec2 uv = (u - iResolution.xy * 0.5) / iResolution.y;
    
    // Get audio data with fallback values
    float bass = max(texture(iChannel0, vec2(0.05, 0.0)).x, 0.1) * 0.5;
    float mid = max(texture(iChannel0, vec2(0.3, 0.0)).x, 0.1) * 0.3;
    
    // Initialize output color
    o = vec4(0.0);
    
    // Main loop
    for(i = 0.0; i < 100.0; i++) {
        vec3 p = vec3(uv * d, d);
        
        s = 0.15;
        while(s < 1.0) {
            // Subtle audio reactivity
            float intensity = 16.0 + mid * 5.0;
            p += abs(dot(cos(iTime + p.z + p * s * intensity), vec3(0.01))) / s;
            s *= 1.4;
        }
        
        // Add some baseline value to ensure visibility
        s = 0.05 + abs(5.0 - length(p.xy)) * (0.3 + bass * 0.04);
        d += s;
        
        // Accumulate color with guaranteed non-zero values
        o += vec4(4.0, 2.0, 1.0, 0.0) / (s + 0.001);
    }
    
    // Ensure we don't end up with all black
    float brightness = 6.0 + bass * 1.5;
    o = tanh(o*o / (2e7 / brightness) / (dot(uv,uv) + 0.1));
    
    // Add minimum brightness
    o = max(o, 0.05);
}`,

"Venom Mint": `// fabrice -> https://www.shadertoy.com/view/4dKSDV

#define H(n)  fract( 1e4 * sin( n.x+n.y/.7 +vec2(1,12.34)  ) )
float rand(vec2 p) {                  // cheap 2‑D hash → [0,1)
    return fract(sin(dot(p, vec2(27.13, 91.17))) * 43758.5453);
}

vec3 voronoiVarRadius(vec2 p, float dt) {
    vec2  base   = floor(p);         
    vec3  dists  = vec3(9.0); 
    float differ;       

    for(int j = -3; j <= 3; ++j)
    for(int i = -3; i <= 3; ++i) {
        vec2 cell  = base + vec2(i, j);         
        vec2 site  = cell + H(cell) + 0.08 * sin(dt + 6.28 * H(cell));            
        float r    = mix(0.5, 3.0, rand(cell) ); 

        differ = length(p - site) - r;         
        differ *= differ;                                 

        differ < dists.x ? (dists.yz = dists.xy, dists.x = differ) :
        differ < dists.y ? (dists.z  = dists.y , dists.y = differ) :
        differ < dists.z ?                dists.z = differ  :
                       differ;
    }
    return dists;   // x = F1², y = F2², z = F3²
}

vec3 voronoiDistSq(vec2 p, float dt)
{
    vec2 cellId, diff;
    float d;
    vec3 dists = vec3(9.0);       

    for (int k = 0; k < 9; ++k)
    {
        cellId = ceil(p) + vec2(k - (k/3)*3, k/3) - 2.0;
        // diff   = H(cellId) + cellId - p;
        diff = H(cellId) + cellId + 0.08 * cos(dt + 6.28 * H(cellId)) - p;

        d = dot(diff, diff);      

        d < dists.x ? (dists.yz = dists.xy, dists.x = d) :
        d < dists.y ? (dists.z  = dists.y , dists.y = d) :
        d < dists.z ?               dists.z = d        :
                       d;
    }
    return dists;
}


vec3 pal( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d )
{
    return a + b*cos( 6.28318*(c*t+d) );
}

void colorPalette(int effectIndex, vec2 uv, out vec3 bg, out vec3 rgbColor){
    float hue = mod(uv.y / 0.6 + uv.x, 1.0);
    
    bg = vec3(0.3631,0.5847,0.6969);
    rgbColor = pal(uv.y,vec3(0.129,0.404,0.49),vec3(0.153,0.024,0.002),vec3(0.169,0.514,0.549),vec3(0.153,0.424,0.502) );
}

float heightAt(vec2 p, float dt)
{
    // vec3  d        = voronoiDistSq(p, dt);
    vec3  d        = voronoiDistSq(p, dt);
    float F1       = sqrt(d.x);
    float F2       = sqrt(d.y);
    float F3       = sqrt(d.z);
    float edgeDiff = F2 - F1;               // ≥ 0, biggest at cell centre
    float curvature = (F3 - F2) / F2;

    // use a *positive* scale so height rises inward
    return clamp( 1.5 * edgeDiff * curvature, 0.05, 1.0 );   // 0.8‒1.0 works well
}

int effectIndex = 0;

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float dt = iTime;
    vec2 uv = 1.5 * (fragCoord + fragCoord - iResolution.xy) / iResolution.y;
    uv.x += dt * 0.1;   

    float scale2 = floor(rand(iResolution.xy) * 1.0) + 2.5;
    vec3 d1 = voronoiDistSq(uv * 1.0, dt);  
    vec3 d2 = voronoiVarRadius(uv * scale2, dt);  


    float w   = 3.0;
    float edgeDist = sqrt(d1.y) - sqrt(d1.x);
    float mask = smoothstep( w, 0.0, edgeDist ) *
                 smoothstep(-w, 3.0, -edgeDist);   

    float F1 = sqrt(d2.x);
    float F2 = sqrt(d2.y);
    float F3 = sqrt(d2.z);

    float edgeWidth = F2 - F1;
    float curvature = (F3 - F2) / F2;

    // approximate the gradient to fake the  normal
    float eps = 10.0 / iResolution.x * -5.3;
    float ho = heightAt(d2.xy, dt);
    
    float dhx = heightAt(uv + vec2(eps, 0.0), dt) - ho;
    float dhy = heightAt(uv + vec2(0.0, eps), dt) - ho;
    vec3 normal = normalize( vec3(dhx, dhy, 0.02 ) );

    // per pixel lightning 
    vec3 lightDir = normalize(vec3(0.1, 0.4, 0.5));
    vec3 viewDir = vec3(0.4, 0.1, 1.0); 
    
    float diffuse = max( dot(normal, lightDir), 0.0);

    vec3 reflextDir = reflect( -lightDir, normal);
    float specular = pow( max( dot(reflextDir, viewDir), 0.0), 64.0);

    float rim = pow(1.0 - normal.z, 4.0);

    vec3 col1 = 2.4 * sqrt(d1);
    col1 -= vec3(col1.x);
    col1 += 8.0 * ( col1.y / (col1.y / col1.z + 1.0) - 0.5 ) - col1;

    vec3 col2 = 20.0 * sqrt(d2);
    col2 -= vec3(col2.x);
    col2 += 5.0 * ( col2.y / (col2.y / col2.z + 1.0) - 0.5 ) - col2;

    vec3 finalColor = mix( col1, col2, mask );
    /*–‑‑ colors –‑‑*/
    vec3 bg, rgbColor;
    effectIndex = int(mod(dt / 10.0, 6.0));
    colorPalette(effectIndex, uv, bg, rgbColor);

    vec3 colorGrad =  rgbColor * diffuse * finalColor
           + vec3(1.0) * specular 
           +   rgbColor * rim * 0.3; 
    float alpha = smoothstep(-0.003, 0.003, F1);
    vec3 color = mix(bg,colorGrad,  alpha);

    fragColor = vec4( color, 1.0 );
    // fragColor = vec4(normal * 0.5 + 0.5, 1.0); // check if normals look valid
}`,
"Celluloid": `// Created by Danil (2021+) https://cohost.org/arugl
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// self https://www.shadertoy.com/view/NslGRN


// --defines for "DESKTOP WALLPAPERS" that use this shader--
// comment or uncomment every define to make it work (add or remove "//" before #define)


// this shadertoy use ALPHA, NO_ALPHA set alpha to 1, BG_ALPHA set background as alpha
// iChannel0 used as background if alpha ignored by wallpaper-app
//#define NO_ALPHA
//#define BG_ALPHA
//#define SHADOW_ALPHA
//#define ONLY_BOX


// save PERFORMANCE by disabling shadow
//#define NO_SHADOW


// static CAMERA position, 0.49 on top, 0.001 horizontal
//#define CAMERA_POS 0.049


// speed of ROTATION
#define ROTATION_SPEED 0.8999


// static SHAPE form, default 0.5
//#define STATIC_SHAPE 0.15


// static SCALE far/close to camera, 2.0 is default, exampe 0.5 or 10.0
//#define CAMERA_FAR 0.1


// ANIMATION shape change
//#define ANIM_SHAPE


// ANIMATION color change
//#define ANIM_COLOR


// custom COLOR, and change those const values
//#define USE_COLOR
const vec3 color_blue=vec3(0.5,0.65,0.8);
const vec3 color_red=vec3(0.99,0.2,0.1);


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

#define tshift 53.

// reflect back side
//#define backside_refl

// Camera with mouse
#define MOUSE_control

// min(iFrame,0) does not speedup compilation in ANGLE
#define ANGLE_loops 0


// this shader discover Nvidia bug with arrays https://www.shadertoy.com/view/NslGR4
// use DEBUG with BUG, BUG trigger that bug and one layer will be white on Nvidia in OpenGL
//#define DEBUG
//#define BUG

#define FDIST 0.7
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
    p+=0.072*iTime;
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

void mainImage(out vec4 fragColor, in vec2 fragCoord)
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

            if ((tb <= 0.) || (internalcol.a < 1.))
            {
                float tout = box(ro_refr, rd2, BOXDIMS, no2, false);
                no2 = n.zyx * no2.x + n.xzy * no2.y + n.yxz * no2.z;
                vec3 rout = ro_refr + tout * rd2;
                vec3 rdout = refract(rd2, -no2, IOR);
                float fresnel2 = R0 + (1. - R0) * pow(1. - dot(rdout, no2), 1.3);
                rd2 = reflect(rd2, -no2);

#ifdef backside_refl
                if((dot(rdout, no2))>0.5){fresnel2=1.;}
#endif
                ro_refr = rout;
                ro_refr.z = max(ro_refr.z, -0.999);

                accum *= fresnel2;
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
}
`,
"Weedy?": `const int MAX_STEPS = 70;
const float MAX_DIST = 1000.0;
const float MIN_DIST = 0.0001;
const vec3 spPos = vec3(0, 0, 0);
const float spRad = 0.5;
float sensitivity = 3.;
float Scale = 1.5;
vec3 Offset = vec3(-1, -0.1, -0.5);
mat3 rot;
int uIters = 10;
float uRadius = 0.02;
float uZoom = 2.0;

vec4 getDist(vec3 z)
{
    float totalScale = 1.0;
    
    vec3 orbit = vec3(0);
    
    float d = 0.0;
    
    for (int i = 0; i < uIters; i++)
    {
        z = abs(z);
        z *= Scale;
        z += Offset;
        
        totalScale *= Scale;
        z = rot * z;
        
        orbit += z * 0.1;
    }
    
    d = length(z);
    
    float dist = d / totalScale - uRadius;
    
    return vec4(dist, orbit);
}

vec4 getDist2(vec3 p) {
    
    p = abs(mod(p, 2.0) - 1.0);
    
    return vec4(max(p.x, max(p.y, p.z)) - 0.3, vec3(0.3));
}

vec4 rayMarch(vec3 ro, vec3 rd, int samples, out int it)
{
    float dO = 0.0;
    
    vec3 col;
    
    bool flag = false;
    for (int i = 0; i < samples; i++)
    {
        vec3 p = ro + rd * dO;
        vec4 d_col = getDist(p);
        col = d_col.yzw;
        float ds = d_col.x;
        dO += ds;
        if (dO > MAX_DIST || ds < MIN_DIST) {
            it = i;
            flag = true;
            break;
        }
    }
    if (!flag)
        it = samples;
    return vec4(dO, col);
}

mat3 rotationY(in float angle) {
    return mat3(    cos(angle),     0,      sin(angle),
                        0,          1.0,         0,
                    -sin(angle),    0,      cos(angle));
}

mat3 rotationX(in float angle) {
    return mat3(    1.0,    0,          0,
                    0,      cos(angle), -sin(angle),
                    0,      sin(angle), cos(angle));
}

mat3 rotationZ(in float angle) {
    return mat3(    cos(angle),     -sin(angle),    0,
                    sin(angle),     cos(angle),     0,
                        0,              0,          1);
}

vec4 color()
{
    vec2 uv = (gl_FragCoord.xy-0.5*iResolution.xy)/iResolution.y;
    
    // Calculate camera position based on time
    // Create a smooth, organic camera motion path
    float time = iTime * 0.2; // Slow down the movement
    
    // Get subtle audio reactivity
    float bass = texture(iChannel0, vec2(0.05, 0.1)).x;
    float mids = texture(iChannel0, vec2(0.3, 0.1)).x;
    
    // Base camera orbit
    float radius = 5.5 + sin(time * 0.3) * 0.5 + bass * 0.2;
    float height = sin(time * 0.2) * 0.8;
    
    // Camera angles
    float cameraAngleX = sin(time * 0.5) * 0.2;
    float cameraAngleY = time * 0.7;
    
    // Apply rotations for camera position
    mat3 rotY = rotationY(cameraAngleY);
    mat3 rotX = rotationX(cameraAngleX + mids * 0.05);
    
    vec3 ro = rotY * rotX * vec3(0, height, -radius);
    vec3 lookAt = vec3(sin(time * 0.3) * 0.3, cos(time * 0.4) * 0.3, 0);
    
    // Create camera orientation
    vec3 forward = normalize(lookAt - ro);
    vec3 right = normalize(cross(vec3(0, 1, 0), forward));
    vec3 up = cross(forward, right);
    
    vec3 rd = normalize(uv.x * right + uv.y * up + uZoom * forward);
    
    // Create fractal rotation matrix that changes over time
    float fractalTimeA = time * 0.3; 
    float fractalTimeB = time * 0.23;
    float fractalTimeC = time * 0.17;
    
    rot = rotationX(fractalTimeA) * rotationY(fractalTimeB) * rotationZ(fractalTimeC);
    
    // Apply subtle audio influence to fractal parameters
    Scale = 1.5 + bass * 0.05;
    uRadius = 0.02 + mids * 0.01;
    
    int it = 0;
    vec4 d_col = rayMarch(ro, rd, MAX_STEPS, it);
    float d = d_col.x;
    
    vec3 col = vec3(0.1);
    if (d < MAX_DIST)
        col = clamp(d_col.yzw, 0.0, 1.0);
    
    float c = 1.0 - float(it) / float(MAX_STEPS);
    if (it == 0)
        c = 0.0;
    
    col *= c;
    
    return vec4(col, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    fragColor = color();
}`,
"Cells Shroom": `const int MAX_STEPS = 70;
const float MAX_DIST = 1000.0;
const float MIN_DIST = 0.0001;
const vec3 spPos = vec3(0, 0, 0);
const float spRad = 0.5;
float sensitivity = 3.;
float Scale = 1.5;
vec3 Offset = vec3(-1, -0.1, -0.5);
mat3 rot;
int uIters = 10;
float uRadius = 0.02;
float uZoom = 2.0;

vec4 getDist(vec3 z)
{
    float totalScale = 1.0;
    
    vec3 orbit = vec3(0);
    
    float d = 0.0;
    
    for (int i = 0; i < uIters; i++)
    {
        z = abs(z);
        z *= Scale;
        z += Offset;
        
        totalScale *= Scale;
        z = rot * z;
        
        orbit += z * 0.1;
    }
    
    d = length(z);
    
    float dist = d / totalScale - uRadius;
    
    return vec4(dist, orbit);
}

vec4 getDist2(vec3 p) {
    
    p = abs(mod(p, 2.0) - 1.0);
    
    return vec4(max(p.x, max(p.y, p.z)) - 0.3, vec3(0.3));
}

vec4 rayMarch(vec3 ro, vec3 rd, int samples, out int it)
{
    float dO = 0.0;
    
    vec3 col;
    
    bool flag = false;
    for (int i = 0; i < samples; i++)
    {
        vec3 p = ro + rd * dO;
        vec4 d_col = getDist(p);
        col = d_col.yzw;
        float ds = d_col.x;
        dO += ds;
        if (dO > MAX_DIST || ds < MIN_DIST) {
            it = i;
            flag = true;
            break;
        }
    }
    if (!flag)
        it = samples;
    return vec4(dO, col);
}

mat3 rotationY(in float angle) {
    return mat3(    cos(angle),        0,        sin(angle),
                            0,        1.0,             0,
                    -sin(angle),    0,        cos(angle));
}

mat3 rotationX(in float angle) {
    return mat3(    1.0,        0,            0,
                    0,     cos(angle),    -sin(angle),
                    0,     sin(angle),     cos(angle));
}

mat3 rotationZ(in float angle) {
    return mat3(    cos(angle),        -sin(angle),    0,
                    sin(angle),        cos(angle),     0,
                            0,                0,        1);
}

vec4 color()
{
    vec2 uv = (gl_FragCoord.xy-0.5*iResolution.xy)/iResolution.y;
    
    // Create smooth camera motion paths
    float time = iTime * 0.3; // Slow down the movement
    
    // Create a more interesting camera path with multiple sine/cosine functions
    float camX = sin(time * 0.5) * cos(time * 0.2) * 1.5;
    float camY = sin(time * 0.3) * 0.7;
    float camZ = cos(time * 0.4) * sin(time * 0.6) * 1.5;
    
    // Use these values to create camera rotation
    mat3 rotY = rotationY(camX);
    mat3 rotX = rotationX(camY);
    mat3 rotZ = rotationZ(camZ * 0.3);
    
    // Position the camera with some distance from the fractal
    vec3 ro = rotY * rotX * rotZ * vec3(0, 0, -5.5);
    vec3 rd = rotY * rotX * rotZ * normalize(vec3(uv.x, uv.y, uZoom));
    
    // Also animate fractal rotation parameters for more variation
    float uAlpha = sin(time * 0.4) * 0.5;
    float uBeta = cos(time * 0.5) * 0.7 + 0.3;
    float uGamma = sin(time * 0.6) * cos(time * 0.3) + 1.0;
    
    rot = mat3(1);
    rot = rotationX(uAlpha) * rot;
    rot = rotationY(uBeta) * rot;
    rot = rotationZ(uGamma) * rot;
    
    int it = 0;
    vec4 d_col = rayMarch(ro, rd, MAX_STEPS, it);
    float d = d_col.x;
    vec3 p = ro + rd * d;
    
    vec3 col = vec3(0.1);
    if (d < MAX_DIST)
        col = clamp(d_col.yzw, 0.0, 1.0);
    
    float c = 1.0 - float(it) / float(MAX_STEPS);
    if (it == 0)
        c = 0.0;
    
    col *= c;
    return vec4(col, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    fragColor = color();
}`,
    "Tree Fractal": `
// -------------------------------------------------------
// unchanged helper functions
vec2 po(vec2 v) {
    return vec2(length(v), atan(v.y, v.x));
}
vec2 ca(vec2 u) {
    return u.x * vec2(cos(u.y), sin(u.y));
}
float ln(vec2 p, vec2 a, vec2 b) { 
    float r = dot(p - a, b - a) / dot(b - a, b - a);
    r = clamp(r, 0.0, 1.0);
    p.x += (0.7 + 0.5 * sin(0.1 * iTime)) * 0.2
           * smoothstep(1.0, 0.0, abs(r * 2.0 - 1.0))
           * sin(3.14159 * (r - 4.0 * iTime));
    return (1.0 + 0.5 * r) * length(p - a - (b - a) * r);
}
// -------------------------------------------------------

// fetch a simple “overall” level from the first row of iChannel0
float getAudioLevel() {
    const int SAMPLES = 16;
    float sum = 0.0;
    for (int i = 0; i < SAMPLES; ++i) {
        // sample across the spectrum; y=0.0 is standard for Shadertoy audio
        sum += texture(iChannel0, vec2(float(i) / float(SAMPLES), 0.0)).r;
    }
    return sum / float(SAMPLES);
}

void mainImage(out vec4 fragColor, in vec2 U) {
    vec2 R = iResolution.xy;
    // normalize & center
    vec2 uv = 4.0 * (U - 0.5 * R) / R.y;
    uv.y += 1.5;

    // grab audio once per-frame
    float audio = getAudioLevel();

    float rMin = 1e9;
    vec4 Q = vec4(0.0);

    // loop with audio-driven modulation
    for (int i = 1; i < 20; ++i) {
        // rotation step – now scaled by audio
        uv = ca(po(uv) + 0.3 * audio * vec2(0.0, 1.0));

        // keep the minimum distance
        rMin = min(rMin, ln(uv, vec2(0.0), vec2(0.0, 1.0)));

        // fractal translate / scale, also audio-modulated
        uv.y -= 1.0;
        uv.x  = abs(uv.x);
        uv   *= 1.4 + 0.5 * audio * float(i);

        // polar-then-offset
        uv   = po(uv);
        uv.y += 1.0 + 0.5 * audio * sin(sin(iTime) * float(i));
        uv   = ca(uv);

        // final color build
        Q += sin(
            1.5 * exp(-100.0 * rMin * rMin) * 1.4 * vec4(1.0, -1.8, 1.9, 4.0)
            + iTime
        );
    }

    fragColor = Q / 18.0;
}
`,
    "Tiles": `/*

    Warped Extruded Skewed Grid
    ---------------------------

    This is a warped extruded skewed grid, rendered in an early 2000s demoscene 
    style. In particular, the grid cell centers have been skewed into position, 
    then two different sized nonskewed squares have been constructed around them
    to form a pinwheel arrangement.

	I've been putting up a bunch of tech-like examples lately, so to break the
	monotony for myself, I decided to code something artsy and more reminiscent 
    of a cliche 2000s demo scene. A lot of 2000s demos were rendered in a grungey 
    dystopian graphic art style, so I kind of went with that. On the technical 
    side, it's a demonstration of grid construction consisting of two different 
    kinds of tiles rendered in a staggered fashion. There's nothing in here 
    that's particularly difficult to implement.

	One of the many things I've always admired about demo coders in general is 
    their ability to identify the most visually enticing effects that can be
    produced with the least amount of effort on whatever machine they're running 
    on.

	I've attempted to emulate that philosophy by employing a few simple 
    techniques that demoscoders take for granted. These include things like
    rendering minor details outside the main loop and concentrating on the cheap 
    aspects that have the largest visual impact, like color palettes and camera 
    motion.

    Decent color palettes and textures usually account for a large part of the 
    the final image, and they cost virtually nothing. Camera movements can add 
    interest to a scene, and in the grand scheme of things are very cheap to 
    implement. Warping space can be a little fiddly, especially when positioning 
    lights and so forth, but it's also cheap and interesting, so I've made use
    of that too.

    The only thing here that I'd consider new is the grid arrangement. I like it 
    because it's a completely ordered grid that looks randomized in height map 
    form, which makes it visually interesting and relatively cheap to produce. 
    Having said that, there are still 8 taps and some skewing involved, so it's 
    probably not the kind of scene that a slow system would be happy with.

    The frame rate in window form was surprisingly better than I'd expected. 
    Having said that, I'm on a pretty quick machine, and it doesn't like running 
    in fullscreen, so it's all relative. There are at least two things that could
    improve performance. One is face and height buffer storage and the other is
    using a pinwheel (or pythagorean) arrangement for five (or possibly three) 
    taps, but that would be at the expense of readability, so I'm going to leave 
    it alone.

    By the way, for anyone interested in just the grid code, I'll put up a much
	simpler example later.



    More interesting tilings:

    // Tiling on steroids. Very cool.
    Wythoffian Tiling Generator - mla
    https://www.shadertoy.com/view/wlGSWc
    //
    Based on:
    // 
	Wythoff Uniform Tilings + Duals - fizzer
	https://www.shadertoy.com/view/3tyXWw
    //
    Tilings - knighty
    https://www.shadertoy.com/view/4sf3zX


*/


#define SKEW_GRID

// Snap the pylons to discreet height units. It looks neater, but I wanted 
// haphazored heights, so it's off by default.
//#define QUANTIZE_HEIGHTS

// Flattening the grid in order to discern the 2D face pattern more clearly.
//#define FLAT_GRID

// Grid positioning independant of the camera path.
//#define PTH_INDPNT_GRD

// Grayscale, for that artsy look. Technically, a touch of color has been 
// left in, but the palette has a gray feel.
//#define GRAYSCALE

// Reverse the color palette.
//#define REVERSE_PALETTE

// Max ray distance: If deliberately set this up close for artistic effect (and to
// save a few cycles), but you'd normally set this higher to put the horizon further away.
#define FAR 20.



// Scene object ID to separate the floor object from the pylons.
float objID;


// Standard 2D rotation formula.
mat2 rot2(in float a){ float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }


// IQ's vec2 to float hash.
float hash21(vec2 p){  return fract(sin(dot(p, vec2(27.609, 57.583)))*43758.5453); }

// vec3 to float.
float hash31(vec3 p){
    return fract(sin(dot(p, vec3(12.989, 78.233, 57.263)))*43758.5453);
}


// The path is a 2D sinusoid that varies over time, which depends upon the frequencies and amplitudes.
vec2 path(in float z){ 
    
    //return vec2(0);
    return vec2(3.*sin(z*.1) + .5*cos(z*.4), 0);
}


// 2D texture function.
//
vec3 getTex(in vec2 p){
    
    // Stretching things out so that the image fills up the window.
    //p *= vec2(iResolution.y/iResolution.x, 1);
    vec3 tx = texture(iChannel0, p/8.).xyz;
    //vec3 tx = textureLod(iChannel0, p, 0.).xyz;
    return tx*tx; // Rough sRGB to linear conversion.
}

// Height map value, which is just the pixel's greyscale value.
float hm(in vec2 p){ return dot(getTex(p), vec3(.299, .587, .114)); }


// IQ's extrusion formula.
float opExtrusion(in float sdf, in float pz, in float h, in float sf){
    
    // Slight rounding. A little nicer, but slower.
    vec2 w = vec2( sdf, abs(pz) - h) + sf;
  	return min(max(w.x, w.y), 0.) + length(max(w, 0.)) - sf;    
}

/*
// IQ's unsigned box formula.
float sBoxSU(in vec2 p, in vec2 b, in float sf){

  return length(max(abs(p) - b + sf, 0.)) - sf;
}
*/

// IQ's signed box formula.
float sBoxS(in vec2 p, in vec2 b, in float sf){

  p = abs(p) - b + sf;
  return length(max(p, 0.)) + min(max(p.x, p.y), 0.) - sf;
}

// Skewing coordinates. "s" contains the X and Y skew factors.
vec2 skewXY(vec2 p, vec2 s){
    
    return mat2(1, -s.y, -s.x, 1)*p;
}

// Unskewing coordinates. "s" contains the X and Y skew factors.
vec2 unskewXY(vec2 p, vec2 s){

    return inverse(mat2(1, -s.y, -s.x, 1))*p;
}


// A dual face extruded block grid with additional skewing. This particular one
// is rendered in a pinwheel arrangement.
//
// The idea is very simple: Produce a skewed grid full of packed objects.
// That is, use the center pixel of each object within the cell to obtain a height 
// value (read in from a height map), then render a pylon at that height.
 
// Global local coordinates. It's lazy putting them here, but I'll tidy this up later.
vec2 gP;


vec4 blocks(vec3 q){

    
    // Scale... Kind of redundant here, but sometimes there's a distinction
    // between scale and dimension.
	const vec2 scale = vec2(1./5.);

    // Block dimension: Length to height ratio with additional scaling.
	const vec2 dim = scale;
    // A helper vector, but basically, it's the size of the repeat cell.
	const vec2 s = dim*2.;
    
    
    #ifdef SKEW_GRID
    // Skewing half way along X, and Y.
    const vec2 sk = vec2(-.5, .5);
    #else
    const vec2 sk = vec2(0);
    #endif
    
    // Distance.
    float d = 1e5;
    // Cell center, local coordinates and overall cell ID.
    vec2 p, ip;
    
    // Individual brick ID.
    vec2 id = vec2(0);
    vec2 cntr = vec2(0);
    
    // Four block corner postions.
    const vec2[4] ps4 = vec2[4](vec2(-.5, .5), vec2(.5), vec2(.5, -.5), vec2(-.5)); 
    
    // Height scale.
    #ifdef FLAT_GRID
    const float hs = 0.; // Zero height pylons for the flat grid.
    #else
    const float hs = .4;
    #endif
    
    float height = 0.; // Pylon height.


    // Local cell coordinate copy.
    gP = vec2(0);
    
    for(int i = 0; i<4; i++){

        // Block center.
        cntr = ps4[i]/2. -  ps4[0]/2.;
        
        // Skewed local coordinates.
        p = skewXY(q.xz, sk);
        ip = floor(p/s - cntr) + .5; // Local tile ID.
        p -= (ip + cntr)*s; // New local position.
        
        // Unskew the local coordinates.
        p = unskewXY(p, sk);
        
        // Correct positional individual tile ID.
        vec2 idi = ip + cntr;
 
        
        // Unskewing the rectangular cell ID.
	    idi = unskewXY(idi*s, sk); 
        
 
        // The larger grid cell face.
        //
        vec2 idi1 = idi; // Block's central position, and ID.
        float h1 = hm(idi1);
        #ifdef QUANTIZE_HEIGHTS
        h1 = floor(h1*20.999)/20.; // Discreet height units.
        #endif
        h1 *= hs; // Scale the height.
        
        // Larger face and height extrusion.
        float face1 = sBoxS(p, 2./5.*dim - .02*scale.x, .015);
        //float face1 = length(p) - 2./5.*dim.x;
        float face1Ext = opExtrusion(face1, q.y + h1, h1, .006); 
    
        
        // The second, smaller face.
        //
        //vec2 offs = vec2(3./5., -1./5.)*dim;
        vec2 offs = unskewXY(dim*.5, sk);
        vec2 idi2 = idi + offs;  // Block's central position, and ID.
        float h2 = hm(idi2);
        #ifdef QUANTIZE_HEIGHTS
        h2 = floor(h2*20.999)/20.; // Discreet height units.
        #endif
        h2 *= hs; // Scale the height.
     
        // Smaller face and height extrusion.
        float face2 = sBoxS(p - offs, 1./5.*dim - .02*scale.x, .015);
        //float face2 = length(p - offs) - 1./5.*dim.x;
        float face2Ext = opExtrusion(face2, q.y + h2, h2, .006);
         
        // Pointed face tips, for an obelisque look, but I wasn't feeling it. :)
        //face1Ext += face1*.25;
        //face2Ext += face2*.25;
        
        vec4 di = face1Ext<face2Ext? vec4(face1Ext, idi1, h1) : vec4(face2Ext, idi2, h2);
   
        
        
        // If applicable, update the overall minimum distance value,
        // ID, and box ID. 
        if(di.x<d){
            d = di.x;
            id = di.yz;
            height = di.w;
            
            // Setting the local coordinates: This is hacky, but I needed a 
            // copy for the rendering portion, so put this in at the last minute.
            gP = p;
     
        }
        
    }
    
    // Return the distance, position-based ID and pylong height.
    return vec4(d, id, height);
}

float getTwist(float z){ return z*.08; }



// Block ID -- It's a bit lazy putting it here, but it works. :)
vec3 gID;

// Speaking of lazy, here's some global glow variables. :D
// Glow: XYZ is for color (unused), and W is for individual 
// blocks.
vec4 gGlow = vec4(0);

// The extruded image.
float map(vec3 p){
    
    // Wrap the scene around the path. This mutates the geometry,
    // but it's easier to implement. By the way, it's possible to
    // snap the geometry around the path, and I've done that in
    // other examples.
    p.xy -= path(p.z);
    
    // Twist the geometry along Z. It's cheap and visually effective.
    // Demosceners having been doing this for as long as I can remember.
    p.xy *= rot2(getTwist(p.z));

    
    // Turning one plane into two. It's an old trick.
    p.y = abs(p.y) - 1.25;
  
    // There are gaps between the pylons, so a floor needs to go in
    // to stop light from getting though.
    float fl = -p.y + .01;
    
    #ifdef PTH_INDPNT_GRD
    // Keep the blocks independent of the camera movement, but still 
    // twisting with warped space.
    p.xy += path(p.z);
    #endif
    
    // The extruded blocks.
    vec4 d4 = blocks(p);
    gID = d4.yzw; // Individual block ID.
    
    // Only alowing certain blocks to glow. We're including some 
    // animation in there as well.
    float rnd = hash21(gID.xy);
    //
    // Standard blinking lights animation.
    gGlow.w = smoothstep(.992, .997, sin(rnd*6.2831 + iTime/4.)*.5 + .5);
    //gGlow.w = rnd>.05? 0. : 1.; // Static version.
 
 
    // Overall object ID.
    objID = fl<d4.x? 1. : 0.;
    
    // Combining the floor with the extruded blocks.
    return min(fl, d4.x);
 
}

  
// Basic raymarcher.
float trace(in vec3 ro, in vec3 rd){

    // Overall ray distance and scene distance.
    float t = 0., d;
     
    // Zero out the glow.
    gGlow = vec4(0);
    
    // Random dithering -- This is on the hacky side, but we're trying to cheap out 
    // on the glow by calculating it inside the raymarching loop instead of it's 
    // own one. If the the jump off point was too close to the closest object in the
    // scene, you wouldn't do this.
    t = hash31(ro.zxy + rd.yzx)*.25;
    
    for(int i = 0; i<128; i++){
    
        d = map(ro + rd*t); // Distance function.
        
        // Adding in the glow. There'd be better and worse ways to do it.
        float ad = abs(d + (hash31(ro + rd) - .5)*.05);
        const float dst = .25;
        if(ad<dst){
            gGlow.xyz += gGlow.w*(dst - ad)*(dst - ad)/(1. + t);
        }
 
        // Note the "t*b + a" addition. Basically, we're putting less emphasis on accuracy, as
        // "t" increases. It's a cheap trick that works in most situations... Not all, though.
        if(abs(d)<.001*(1. + t*.05) || t>FAR) break; // Alternative: 0.001*max(t*.25, 1.), etc.
        
        t += i<32? d*.4 : d*.7; 
        //t += d*.5; 
    }

    return min(t, FAR);
}


// Standard normal function. It's not as fast as the tetrahedral calculation, but more symmetrical.
vec3 getNormal(in vec3 p){
	
    const vec2 e = vec2(.001, 0);
    
    //vec3 n = normalize(vec3(map(p + e.xyy) - map(p - e.xyy),
    //map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
    
    // This mess is an attempt to speed up compiler time by contriving a break... It's 
    // based on a suggestion by IQ. I think it works, but I really couldn't say for sure.
    float sgn = 1.;
    float mp[6];
    vec3[3] e6 = vec3[3](e.xyy, e.yxy, e.yyx);
    for(int i = 0; i<6; i++){
		mp[i] = map(p + sgn*e6[i/2]);
        sgn = -sgn;
        if(sgn>2.) break; // Fake conditional break;
    }
    
    return normalize(vec3(mp[0] - mp[1], mp[2] - mp[3], mp[4] - mp[5]));
}



// Cheap shadows are hard. In fact, I'd almost say, shadowing particular scenes with limited 
// iterations is impossible... However, I'd be very grateful if someone could prove me wrong. :)
float softShadow(vec3 ro, vec3 lp, vec3 n, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int iter = 24; 
    
    ro += n*.0015;
    vec3 rd = lp - ro; // Unnormalized direction ray.
    

    float shade = 1.;
    float t = 0.;//.0015; // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i = 0; i<iter; i++){

        float d = map(ro + rd*t);
        shade = min(shade, k*d/t);
        //shade = min(shade, smoothstep(0., 1., k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        t += clamp(d, .01, .25); 
        
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (d<0. || t>end) break; 
    }

    // Sometimes, I'll add a constant to the final shade value, which lightens the shadow a bit --
    // It's a preference thing. Really dark shadows look too brutal to me. Sometimes, I'll add 
    // AO also just for kicks. :)
    return max(shade, 0.); 
}


// I keep a collection of occlusion routines... OK, that sounded really nerdy. :)
// Anyway, I like this one. I'm assuming it's based on IQ's original.
float calcAO(in vec3 p, in vec3 n)
{
	float sca = 3., occ = 0.;
    for( int i = 0; i<5; i++ ){
    
        float hr = float(i + 1)*.15/5.;        
        float d = map(p + n*hr);
        occ += (hr - d)*sca;
        sca *= .7;
    }
    
    return clamp(1. - occ, 0., 1.);  
}

/*
// Compact, self-contained version of IQ's 3D value noise function. I have a transparent noise
// example that explains it, if you require it.
float n3D(in vec3 p){
    
	const vec3 s = vec3(7, 157, 113);
	vec3 ip = floor(p); p -= ip; 
    vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
    p = p*p*(3. - 2.*p); //p *= p*p*(p*(p * 6. - 15.) + 10.);
    h = mix(fract(sin(h)*43758.5453), fract(sin(h + s.x)*43758.5453), p.x);
    h.xy = mix(h.xz, h.yw, p.y);
    return mix(h.x, h.y, p.z); // Range: [0, 1].
}

// Very basic pseudo environment mapping... and by that, I mean it's fake. :) However, it 
// does give the impression that the surface is reflecting the surrounds in some way.
//
// More sophisticated environment mapping:
// UI easy to integrate - XT95    
// https://www.shadertoy.com/view/ldKSDm
vec3 envMap(vec3 p){
    
    p *= 6.;
    p.y += iTime;
    
    float n3D2 = n3D(p*2.);
   
    // A bit of fBm.
    float c = n3D(p)*.57 + n3D2*.28 + n3D(p*4.)*.15;
    c = smoothstep(.45, 1., c); // Putting in some dark space.
    
    p = vec3(c, c*c, c*c*c*c); // Bluish tinge.
    
    return mix(p, p.xzy, n3D2*.4); // Mixing in a bit of purple.

}
*/ 
 

void mainImage( out vec4 fragColor, in vec2 fragCoord ){

    
    // Screen coordinates.
	vec2 uv = (fragCoord - iResolution.xy*.5)/iResolution.y;
	
	// Camera Setup.
	vec3 ro = vec3(0, 0, iTime*1.5); // Camera position, doubling as the ray origin.
    ro.xy += path(ro.z); 
    vec2 roTwist = vec2(0, 0);
    roTwist *= rot2(-getTwist(ro.z));
    ro.xy += roTwist;
    
	vec3 lk = vec3(0, 0, ro.z + .25); // "Look At" position.
    lk.xy += path(lk.z); 
    vec2 lkTwist = vec2(0, -.1); // Only twist horizontal and vertcal.
    lkTwist *= rot2(-getTwist(lk.z));
    lk.xy += lkTwist;
    
	vec3 lp = vec3(0, 0, ro.z + 3.); // Light.
    lp.xy += path(lp.z);
    vec2 lpTwist = vec2(0, -.3); // Only twist horizontal and vertcal.
    lpTwist *= rot2(-getTwist(lp.z));
    lp.xy += lpTwist;
    

    
    // Using the above to produce the unit ray-direction vector.
    float FOV = 1.; // FOV - Field of view.
    float a = getTwist(ro.z);
    // Swiveling the camera about the XY-plane.
    a += (path(ro.z).x - path(lk.z).x)/(ro.z - lk.z)/4.;
	vec3 fw = normalize(lk - ro);
	//vec3 up = normalize(vec3(-fw.x, 0, -fw.z));
	vec3 up = vec3(sin(a), cos(a), 0);
	//vec3 up = vec3(0, 1, 0);
    vec3 cu = normalize(cross(up, fw));
	vec3 cv = cross(fw, cu);   
    
    // Unit direction ray.
    vec3 rd = normalize(uv.x*cu + uv.y*cv + fw/FOV);	
	 
    
    // Raymarch to the scene.
    float t = trace(ro, rd);
    
    // Save the block ID, object ID and local coordinates.
    vec3 svGID = gID;
    float svObjID = objID;
    vec2 svP = gP; 
    
    vec3 svGlow = gGlow.xyz;
   
	
    // Initiate the scene color to black.
	vec3 col = vec3(0);
	
	// The ray has effectively hit the surface, so light it up.
	if(t < FAR){
        
  	
    	// Surface position and surface normal.
	    vec3 sp = ro + rd*t;
	    //vec3 sn = getNormal(sp, edge, crv, ef, t);
        vec3 sn = getNormal(sp);
        
          
        // Texel color. 
	    vec3 texCol;   
        
        // Transforming the texture coordinates according to the camera path
        // and Z warping.
        vec3 txP = sp;
        txP.xy -= path(txP.z);
        txP.xy *= rot2(getTwist(txP.z));
        #ifdef PTH_INDPNT_GRD
        txP.xy += path(txP.z);
        #endif

        // The extruded grid.
        if(svObjID<.5){
            
            // Coloring the individual blocks with the saved ID.
            vec3 tx = getTex(svGID.xy);
            
            // Ramping the shade up a bit.
            texCol = smoothstep(-.5, 1., tx)*vec3(1, .8, 1.8);
            
            
            // Very fake, but very cheap, bump mapping. Render some equispaced horizontal
            // dark lines, and some light adjacent ones. As you can see, it gives the
            // impression of horizontally segmented grooves on the pylons.
            const float lvls = 8.;
            
            // Vertical lines... A bit too much for this example, but useful for a fake
            // voxel setup.
            //float vLn = min(abs(txP.x - svGID.x), abs(txP.z - svGID.y));
            
            // Horizontal lines (planes, technically) around the pylons.
            float yDist = (1.25 + abs(txP.y) + svGID.z*2.);
            float hLn = abs(mod(yDist  + .5/lvls, 1./lvls) - .5/lvls);
            float hLn2 = abs(mod(yDist + .5/lvls - .008, 1./lvls) - .5/lvls);
            
            // Omitting the top and bottom planes... I was in a hurry, and it seems to
            // work, but there'd be better ways to do this. 
            if(yDist - 2.5<.25/lvls) hLn = 1e5;
            if(yDist - 2.5<.25/lvls) hLn2 = 1e5;
            
            // Rendering the dark and light lines using 2D layering techniques.
            texCol = mix(texCol, texCol*2., 1. - smoothstep(0., .003, hLn2 - .0035));
       		texCol = mix(texCol, texCol/2.5, 1. - smoothstep(0., .003, hLn - .0035));
       		 
            
            // Render a dot on the face center of each extruded block for whatever reason...
            // They were there as markers to begin with, so I got used to them. :)
            float fDot = length(txP.xz - svGID.xy) - .0086;
            texCol = mix(texCol, texCol*2., 1. - smoothstep(0., .005, fDot - .0035));
            texCol = mix(texCol, vec3(0), 1. - smoothstep(0., .005, fDot));
  

 
        }
        else {
            
            // The dark floor in the background. Hidden behind the pylons, but
            // there are very slight gaps, so it's still necessary.
            texCol = vec3(0);
        }
       
    	
    	// Light direction vector.
	    vec3 ld = lp - sp;

        // Distance from respective light to the surface point.
	    float lDist = max(length(ld), .001);
    	
    	// Normalize the light direction vector.
	    ld /= lDist;
        
        
        // Shadows and ambient self shadowing.
    	float sh = softShadow(sp, lp, sn, 16.);
    	float ao = calcAO(sp, sn); // Ambient occlusion.
        sh = min(sh + ao*.25, 1.);
	    
	    // Light attenuation, based on the distances above.
	    float atten = 3./(1. + lDist*lDist*.5);

    	
    	// Diffuse lighting.
	    float diff = max( dot(sn, ld), 0.);
        diff *= diff*1.35; // Ramping up the diffuse.
    	
    	// Specular lighting.
	    float spec = pow(max(dot(reflect(ld, sn), rd ), 0.), 32.); 
	    
	    // Fresnel term. Good for giving a surface a bit of a reflective glow.
        float fre = pow(clamp(1. - abs(dot(sn, rd))*.5, 0., 1.), 4.);
        
		// Schlick approximation. I use it to tone down the specular term. It's pretty subtle,
        // so could almost be aproximated by a constant, but I prefer it. Here, it's being
        // used to give a hard clay consistency... It "kind of" works.
		//float Schlick = pow( 1. - max(dot(rd, normalize(rd + ld)), 0.), 5.);
		//float freS = mix(.15, 1., Schlick);  //F0 = .2 - Glass... or close enough.        
        
        // Combining the above terms to procude the final color.
        col = texCol*(diff + ao*.25 + vec3(1, .4, .2)*fre*.25 + vec3(1, .4, .2)*spec*4.);
        
       
        // Fake environmental lighting: Interesting, but I couldn't justify it, both
        // from a visual and logical standpoint.
        //vec3 cTex = envMap(reflect(rd, sn)); // Be sure to uncomment the function above.
        //col += col*cTex.zyx*4.;

    
        // Shading.
        col *= ao*sh*atten;
	
	}

    
    // Applying the glow -- You perform this outside the hit logic block. The reason
    // I mention this is that I make this mistake all the time and spend ages trying
    // to figure out why it's not working. :) As for how you apply it, that's up to
    // you. I made the following up, and I'd imagine there'd be nicer ways to apply 
    // it, but it'll do.
    svGlow.xyz *= mix(vec3(4, 1, 2), vec3(4, 2, 1), min(svGlow.xyz*3.5, 1.25));
    col *= .25 + svGlow.xyz*8.;
   
    // Some colorful fog: Like the above, it's been tweaked to produce something
    // colorful that, hopefully, helps the scene. The cool thing about fog is that
    // it's about as cheap an operation as you could hope for, but has virtually
    // no impact on the frame rate. With that in mind, it's definitely worth taking
    // the time to get it looking the way you'd like it to look.
    vec3 fog =  mix(vec3(4, 1, 2), vec3(4, 2, 1), rd.y*.5 + .5);
    fog = mix(fog, fog.zyx, smoothstep(0., .35, uv.y - .35));
    col = mix(col, fog/1.5, smoothstep(0., .99, t*t/FAR/FAR));
    
    
    #ifdef GRAYSCALE
    // Grayscale... or almost grayscale. :)
    col = mix(col, vec3(1)*dot(col, vec3(.299, .587, .114)), .75);
    #endif 
 
    
    #ifdef REVERSE_PALETTE
    col = col.zyx; // A more calming blue, for those who don't like fiery things.
    #endif

    
    /*
    // Uncomment this block if you'd like to see the 2D pattern on its own.
    uv = fragCoord/iResolution.y;
    vec4 d = blocks(vec3(uv*2. + iTime/4., 0.));
    //vec2 offs = inCentRad(gV[0], gV[1], gV[2]).xy;
    vec3 oCol = smoothstep(-.05, .5, getTex(d.yz));
    float quadD = sBoxS(gP, 3./5.*vec2(1./5.) - .04, .015);
    #ifdef SKEW_GRID
    // Skewing half way along X, and not skewing in the Y direction.
    const vec2 sk = vec2(-.5, .5);//;
    #else
    const vec2 sk = vec2(0);
    #endif
    gP -= unskewXY(vec2(1./2.), sk)/5.;
    quadD = min(quadD, sBoxS(gP, 3./10.*vec2(1./5.) - .04/2., .015));
    float sf = 1./iResolution.y;
    col = mix(vec3(.1), vec3(0), 1. - smoothstep(0., sf, quadD));
    col = mix(col, oCol, 1. - smoothstep(0., sf, quadD + .006));
    */      
    
    // Rought gamma correction.
	fragColor = vec4(sqrt(max(col, 0.)), 1);
	
}`,
"Candy Chips":`
// Seigaiha Mandala by Philippe Desgranges
// Email: Philippe.desgranges@gmail.com
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
//

#define S(a,b,c) smoothstep(a,b,c)


// blends a pre-multiplied src onto a dst color (without alpha)
vec3 premulMix(vec4 src, vec3 dst)
{
    return dst.rgb * (1.0 - src.a) + src.rgb;
}

// blends a pre-multiplied src onto a dst color (with alpha)
vec4 premulMix(vec4 src, vec4 dst)
{
    vec4 res;
    res.rgb = premulMix(src, dst.rgb);
    res.a = 1.0 - (1.0 - src.a) * (1.0 - dst.a);
    return res;
}

// compute the round scale pattern and its mask
// output rgb is premultiplied by alpha
vec4 roundPattern(vec2 uv)
{
    float dist = length(uv);
    
    // Resolution dependant Anti-Aliasing for a prettier thumbnail
    // Thanks Fabrice Neyret & dracusa for pointing this out.
    float aa = 8. / iResolution.x;

    // concentric circles are made by thresholding a triangle wave function
    float triangle = abs(fract(dist * 11.0 + 0.3) - 0.5);
    float circles = S(0.25 - aa * 10.0, 0.25 + aa * 10.0, triangle);

    // a light gradient is applied to the rings
    float grad = dist * 2.0;
    vec3 col = mix(vec3(0.0, 0.5, 0.6),  vec3(0.0, 0.2, 0.5), grad * grad);
    col = mix(col, vec3(1.0), circles);
    
    // border and center are red
    vec3 borderColor = vec3(0.7, 0.2, 0.2);
    col = mix(col, borderColor, S(0.44 - aa, 0.44 + aa, dist));
    col = mix(col, borderColor, S(0.05 + aa, 0.05 - aa, dist));
    
    // computes the mask with a soft shadow
    float mask = S(0.5, 0.49, dist);
    float blur = 0.3;
    float shadow = S(0.5 + blur, 0.5 - blur, dist);
   
    return vec4(col * mask, clamp(mask + shadow * 0.55, 0.0, 1.0)); 
}


//computes the scales on a ring of a given radius with a given number of scales
vec4 ring(vec2 uv, float angle, float angleOffet, float centerDist, float numcircles, float circlesRad)
{
    // polar space is cut in quadrants (one per scale)
    float quadId = floor(angle * numcircles + angleOffet);
    
    // computes the angle of the center of the quadrant
    float quadAngle = (quadId + 0.5 - angleOffet) * (6.283 / numcircles);
    
    // computes the center point of the quadrant on the circle
    vec2 quadCenter = vec2(cos(quadAngle), sin(quadAngle)) * centerDist;
    
    // return to color of the scale in the quadrant
    vec2 circleUv = (uv + quadCenter) / circlesRad;
    return roundPattern(circleUv);
}

// computes a ring with two layers of overlapping patterns
vec4 dblRing(vec2 uv, float angle, float centerDist, float numcircles, float circlesRad, float t)
{
    // Odd and even scales dance up and down
    float s = sin(t * 3.0 + centerDist * 10.0) * 0.05;
    float d1 = 1.05 + s;
    float d2 = 1.05 - s;
    
    // the whole thing spins with a sine perturbation
    float rot = t * centerDist * 0.4 + sin(t + centerDist * 5.0) * 0.2;
    
    // compute bith rings
    vec4 ring1 = ring(uv, angle, 0.0 + rot, centerDist * d1, numcircles, circlesRad);
    vec4 ring2 = ring(uv, angle, 0.5 + rot, centerDist * d2, numcircles, circlesRad);
    
    // blend the results
    vec4 col = premulMix(ring1, ring2);
    
    // add a bit of distance shading for extra depth
    col.rgb *= 1.0 - (centerDist * centerDist) * 0.4;
    
    return col;
}

// computes a double ring on a given radius with a number of scales to fill the circle evenly
vec4 autoRing(vec2 uv, float angle, float centerDist, float t)
{
    float nbCircles = 1.0 + floor(centerDist * 23.0);
    return dblRing(uv, angle, centerDist, nbCircles, 0.23, t);
}

// Computes the pixel color for the full image at a givent time
vec3 fullImage(vec2 uv, float angle, float centerDist, float t)
{
    vec3 col;
    
    // the screen is cut in concentric rings
    float space = 0.1;
    
    // determine in which ring the pixel is
    float ringRad = floor(centerDist / space) * space;
    
	// computes the scales in the previous, current and next ring
	vec4 ringCol1 = autoRing(uv, angle, ringRad - space, t);
 	vec4 ringCol2 = autoRing(uv, angle, ringRad, t);
    vec4 ringCol3 = autoRing(uv, angle, ringRad + space, t);
    
    // blends everything together except in the center
    if (ringRad > 0.0)
    {
        col.rgb = ringCol3.rgb;
        col.rgb = premulMix(ringCol2, col.rgb);
        col.rgb = premulMix(ringCol1, col.rgb);
    }
	else
    {
        col.rgb = ringCol2.rgb; 
    }

    return col;
}

// A noise function that I tried to make as gaussian-looking as possible
float noise21(vec2 uv)
{
    vec2 n = fract(uv* vec2(19.48, 139.9));
    n += sin(dot(uv, uv + 30.7)) * 47.0;
    return fract(n.x * n.y);
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = (fragCoord - .5 * iResolution.xy) / iResolution.y; 
    
    uv *= 0.9;
    
    // Computes polar cordinates
    float angle = atan(uv.y, uv.x) / 6.283 + 0.5;
    float centerDist = length(uv);
    
    vec3 col = vec3(0.0);
    
	// average 4 samples at slightly different times for motion blur
    float noise = noise21(uv + iTime);
    for (float i = 0.0; i < 4.0; i++)
    {
        col += fullImage(uv, angle, centerDist, iTime - ((i + noise) * 0.0003));
    }
    col /= 4.0;
 
    // Output to screen
    fragColor = vec4(col,1.0);
}
`,
"Candy Chips 2": `
//
// Seigaiha Mandala by Philippe Desgranges
// Email: Philippe.desgranges@gmail.com
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
//

#define S(a,b,c) smoothstep(a,b,c)


// blends a pre-multiplied src onto a dst color (without alpha)
vec3 premulMix(vec4 src, vec3 dst)
{
    return dst.rgb * (1.0 - src.a) + src.rgb;
}

// blends a pre-multiplied src onto a dst color (with alpha)
vec4 premulMix(vec4 src, vec4 dst)
{
    vec4 res;
    res.rgb = premulMix(src, dst.rgb);
    res.a = 1.0 - (1.0 - src.a) * (1.0 - dst.a);
    return res;
}

// compute the round scale pattern and its mask
// output rgb is premultiplied by alpha
vec4 roundPattern(vec2 uv)
{
    float dist = length(uv);
    
    // Resolution dependant Anti-Aliasing for a prettier thumbnail
    // Thanks Fabrice Neyret & dracusa for pointing this out.
    float aa = 8. / iResolution.x;

    // concentric circles are made by thresholding a triangle wave function
    float triangle = abs(fract(dist * 11.0 + 0.3) - 0.5);
    float circles = S(0.25 - aa * 10.0, 0.25 + aa * 10.0, triangle);

    // a light gradient is applied to the rings
    float grad = dist * 2.0;
    vec3 col = mix(vec3(0.0, 0.5, 0.6),  vec3(0.0, 0.2, 0.5), grad * grad);
    col = mix(col, vec3(1.0), circles);
    
    // border and center are red
    vec3 borderColor = vec3(0.7, 0.2, 0.2);
    col = mix(col, borderColor, S(0.44 - aa, 0.44 + aa, dist));
    col = mix(col, borderColor, S(0.05 + aa, 0.05 - aa, dist));
    
    // computes the mask with a soft shadow
    float mask = S(0.5, 0.49, dist);
    float blur = 0.3;
    float shadow = S(0.5 + blur, 0.5 - blur, dist);
   
    return vec4(col * mask, clamp(mask + shadow * 0.55, 0.0, 1.0)); 
}


//computes the scales on a ring of a given radius with a given number of scales
vec4 ring(vec2 uv, float angle, float angleOffet, float centerDist, float numcircles, float circlesRad)
{
    // polar space is cut in quadrants (one per scale)
    float quadId = floor(angle * numcircles + angleOffet);
    
    // computes the angle of the center of the quadrant
    float quadAngle = (quadId + 0.5 - angleOffet) * (6.283 / numcircles);
    
    // computes the center point of the quadrant on the circle
    vec2 quadCenter = vec2(cos(quadAngle), sin(quadAngle)) * centerDist;
    
    // return to color of the scale in the quadrant
    vec2 circleUv = (uv + quadCenter) / circlesRad;
    return roundPattern(circleUv);
}

// computes a ring with two layers of overlapping patterns
vec4 dblRing(vec2 uv, float angle, float centerDist, float numcircles, float circlesRad, float t)
{
    // Odd and even scales dance up and down
    float s = sin(t * 3.0 + centerDist * 1.0) * 0.05;
    float d1 = 1.05 + s;
    float d2 = 1.05 - s;
    
    // the whole thing spins with a sine perturbation
    float rot = t * centerDist * 0.4 + sin(t + centerDist * 5.0) * 0.2;
    
    // compute bith rings
    vec4 ring1 = ring(uv, angle, 0.0 + rot, centerDist * d1, numcircles, circlesRad);
    vec4 ring2 = ring(uv, angle, 0.5 + rot, centerDist * d2, numcircles, circlesRad);
    
    // blend the results
    vec4 col = premulMix(ring1, ring2);
    
    // add a bit of distance shading for extra depth
    col.rgb *= 1.0 - (centerDist * centerDist) * 0.4;
    
    return col;
}

// computes a double ring on a given radius with a number of scales to fill the circle evenly
vec4 autoRing(vec2 uv, float angle, float centerDist, float t)
{
    float nbCircles = 1.0 + floor(centerDist * 23.0);
    return dblRing(uv, angle, centerDist, nbCircles, 0.23, t);
}

// Computes the pixel color for the full image at a givent time
vec3 fullImage(vec2 uv, float angle, float centerDist, float t)
{
    vec3 col;
    
    // the screen is cut in concentric rings
    float space = 0.1;
    
    // determine in which ring the pixel is
    float ringRad = floor(centerDist / space) * space;
    
	// computes the scales in the previous, current and next ring
	vec4 ringCol1 = autoRing(uv, angle, ringRad - space, t);
 	vec4 ringCol2 = autoRing(uv, angle, ringRad, t);
    vec4 ringCol3 = autoRing(uv, angle, ringRad + space, t);
    
    // blends everything together except in the center
    if (ringRad > 0.0)
    {
        col.rgb = ringCol3.rgb;
        col.rgb = premulMix(ringCol2, col.rgb);
        col.rgb = premulMix(ringCol1, col.rgb);
    }
	else
    {
        col.rgb = ringCol2.rgb; 
    }

    return col;
}

// A noise function that I tried to make as gaussian-looking as possible
float noise21(vec2 uv)
{
    vec2 n = fract(uv* vec2(19.48, 139.9));
    n += sin(dot(uv, uv + 30.7)) * 47.0;
    return fract(n.x * n.y);
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = (fragCoord - .5 * iResolution.xy) / iResolution.y; 
    
    uv *= 0.9;
    
    // Computes polar cordinates
    float angle = atan(uv.y, uv.x) / 6.283 + 0.5;
    float centerDist = length(uv);
    
    vec3 col = vec3(0.0);
    
	// average 4 samples at slightly different times for motion blur
    float noise = noise21(uv + iTime);
    for (float i = 0.0; i < 4.0; i++)
    {
        col += fullImage(uv, angle, centerDist, iTime - ((i + noise) * 0.00003));
    }
    col /= 4.0;
 
    // Output to screen
    fragColor = vec4(col,1.0);
}
`,
"Angel Heart": `// @christinacoffin
// 2015-05-07: 1st version, need to optim and do some proper AA

vec3 JuliaFractal(vec2 c, vec2 c2, float animparam, float anim2 ) {	
	vec2 z = c;
    
	float ci = 0.0;
	float mean = 0.0;
    
	for(int i = 0;i < 64; i++)
    {
		vec2 a = vec2(z.x,abs(z.y));
		
        float b = atan(a.y*(0.99+animparam*9.0), a.x+.110765432+animparam);
		
        if(b > 0.0) b -= 6.303431307+(animparam*3.1513);
		
        z = vec2(log(length(a*(0.98899-(animparam*2.70*anim2)))),b) + c2;

        if (i>0) mean+=length(z/a*b);

        mean+=a.x-(b*77.0/length(a*b));

        mean = clamp(mean, 111.0, 99999.0);
	}
    
	mean/=131.21;
	ci =  1.0 - fract(log2(.5*log2(mean/(0.57891895-abs(animparam*141.0)))));

	return vec3( .5+.5*cos(6.*ci+0.0),.5+.75*cos(6.*ci + 0.14),.5+.5*cos(6.*ci +0.7) );
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float animWings = 0.004 * cos(iTime*0.5);
    float animFlap = 0.011 * sin(iTime*1.0);    
    float timeVal = 56.48-20.1601;
	vec2 uv = fragCoord.xy - iResolution.xy*.5;
	uv /= iResolution.x*1.5113*abs(sin(timeVal));
    uv.y -= animWings*5.0; 
	vec2 tuv = uv*125.0;
	float rot=3.141592654*0.5;
  
	uv.x = tuv.x*cos(rot)-tuv.y*sin(rot);
	uv.y =1.05* tuv.x*sin(rot)+tuv.y*cos(rot);
	float juliax = tan(timeVal) * 0.011 + 0.02/(fragCoord.y*0.19531*(1.0-animFlap));
	float juliay = cos(timeVal * 0.213) * (0.022+animFlap) + 5.66752-(juliax*1.5101);//+(fragCoord.y*0.0001);// or 5.7
    
 
    float tapU = (1.0/ float(iResolution.x))*25.5;//*cos(animFlap);
    float tapV = (1.0/ float(iResolution.y))*25.5;//*cos(animFlap);
    
  
	fragColor = vec4( JuliaFractal(uv+vec2(0.0,0.0), vec2(juliax, juliay), animWings, animFlap ) ,1.0);
    
    fragColor += vec4( JuliaFractal(uv+vec2(tapU,tapV), vec2(juliax, juliay), animWings, animFlap ) ,1.0);
//    fragColor += vec4( JuliaFractal(uv+vec2(tapU,-tapV), vec2(juliax, juliay), animWings, animFlap ) ,1.0);
//    fragColor += vec4( JuliaFractal(uv+vec2(-tapU,tapV), vec2(juliax, juliay), animWings, animFlap ) ,1.0);
    fragColor += vec4( JuliaFractal(uv+vec2(-tapU,-tapV), vec2(juliax, juliay), animWings, animFlap ) ,1.0);  
    fragColor *= 0.3333;
    
    fragColor.xyz = fragColor.zyx;
	fragColor.xyz = vec3(1)-fragColor.xyz;

}`,

"Cray Cray": `
vec3 palette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(cos(0.5), 0.5, 0.5);
    vec3 d = vec3(0.5, sin(0.5), 0.35);
    
    return a + b * cos(6.24535 * (c * t + d));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 finalColor = vec3(0.0);

    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;

    // Get mic input (assumed as 1D texture in iChannel0)
    // We'll sample the middle point (0.5) for simplicity
    float micValue = texture(iChannel0, vec2(0.5, 0.0)).r;

    // Scale UV with mic input (adds reactivity)
    vec2 uv0 = uv * (1.0 + micValue * 0.005); // You can tweak the multiplier

    for (float i = 0.0; i < 3.0; i++) {
        uv = fract(uv * 2.0) - 0.5;

        float d = length(uv);

        vec3 col = palette(length(uv0) + iTime + micValue); // inject mic here too

        d = sin(d * 8.0 + iTime + micValue * 10.0) / 8.0;
        d = abs(d);
        d = 0.02 / d;

        col *= d;
        finalColor += col * d;
    }

    fragColor = vec4(finalColor, 1.0);
}
`,
"Cray Cray 2":`// --- Palette Function for Vibrant Colors ---
vec3 palette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.3, 0.2, 0.5); // Dreamy colors
    return a + b * cos(6.28318 * (c * t + d));
}

// --- Function to Draw Rings ---
float drawRing(vec2 uv, vec2 center, float radius, float thickness) {
    float d = length(uv - center) - radius;
    return smoothstep(thickness, 0.0, abs(d));
}

// --- Text Drawing Function for "AJL" ---
float drawText(vec2 uv, vec2 pos, float size, vec2 offset, vec2 aspect) {
    uv = (uv - pos) * size;       // Scale and position text
    uv.x *= aspect.x / aspect.y; // Adjust for screen aspect ratio
    vec2 d = abs(uv - offset);   // Letter structure
    float bar = max(d.x, d.y) - 0.1;
    return smoothstep(0.01, 0.0, bar);
}

float drawAJL(vec2 uv, vec2 aspect) {
    float ajl = 0.0;

    // "A"
    ajl += drawText(uv, vec2(-0.4, 0.0), 10.0, vec2(0.2, 0.2), aspect); // Main shape
    ajl += drawText(uv, vec2(-0.4, 0.0), 10.0, vec2(-0.2, 0.2), aspect); // Left edge
    ajl += drawText(uv, vec2(-0.4, 0.0), 10.0, vec2(0.0, -0.2), aspect); // Center

    // "J"
    ajl += drawText(uv, vec2(0.0, 0.0), 10.0, vec2(0.0, 0.2), aspect);   // Top bar
    ajl += drawText(uv, vec2(0.0, -0.2), 10.0, vec2(0.2, -0.5), aspect); // Hook

    // "L"
    ajl += drawText(uv, vec2(0.4, -0.1), 10.0, vec2(-0.1, 0.2), aspect); // Top bar
    ajl += drawText(uv, vec2(0.4, -0.3), 10.0, vec2(-0.1, -0.2), aspect); // Base

    return ajl;
}

// --- Fractal Background Shader ---
vec3 fractalShader(vec2 uv, float audio, float time) {
    vec2 uv0 = uv;
    vec3 finalColor = vec3(0.0);

    // Apply shake effect
    float shakeIntensity = audio * 0.05;
    uv += vec2(sin(time * 10.0) * shakeIntensity, cos(time * 15.0) * shakeIntensity);

    // Apply rotation
    float angle = time * 0.2 + audio * 0.5;
    float cosA = cos(angle);
    float sinA = sin(angle);
    uv = mat2(cosA, -sinA, sinA, cosA) * uv;

    for (float i = 0.0; i < 6.0; i++) {
        uv = fract(uv * 1.5 + audio * 0.2) - 0.5;
        float d = length(uv) * exp(-length(uv0) * 1.2);
        vec3 col = palette(length(uv0) + i * 0.4 + time * 0.4 + audio * 0.5);
        d = sin(d * 8.0 + time * 2.0) / 8.0;
        d = abs(d);
        d = pow(0.01 / d, 1.4);
        finalColor += col * d;
    }

    return finalColor;
}

// --- Main Shader ---
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    vec2 aspect = iResolution.xy / min(iResolution.x, iResolution.y);

    // Sample audio data
    float audioL = texture(iChannel0, vec2(0.1, 0.5)).r;
    float audioH = texture(iChannel0, vec2(0.9, 0.5)).r;
    float audio = mix(audioL, audioH, 0.5);

    // Background fractals
    vec3 fractal = fractalShader(uv, audio, iTime);

    // Persistent text "AJL"
    float ajl = drawAJL(uv, aspect);
    vec3 textColor = palette(iTime * 0.2 + audio * 2.0);
    vec3 text = vec3(ajl) * textColor;

    // Audio-reactive rings
    vec3 rings = vec3(0.0);
    for (float i = 0.0; i < 5.0; i++) {
        // Add synchronized shake to rings
        vec2 ringShake = vec2(sin(iTime * 5.0 + i) * audio * 0.05, cos(iTime * 5.0 + i) * audio * 0.05);
        float radius = 0.2 + i * 0.1 + audio * 0.2 * sin(iTime + i);
        float thickness = 0.02 + 0.01 * audio;
        rings += palette(i + iTime * 0.3) * drawRing(uv + ringShake, vec2(0.0, 0.0), radius, thickness);
    }

    // Combine everything
    vec3 finalColor = fractal + rings;
    finalColor = mix(finalColor, text, ajl);

    fragColor = vec4(finalColor, 1.0);
}
`,

"Bleed": `/**
"Raining Blood, From a lacerated sky. Bleeding its horror, creating my
structure now I shall reign in blood." - Slayer

This is inspired by bigwings' heartfelt (https://www.shadertoy.com/view/ltffzl)
*/

#define PI 3.141592653589793

float hash21(vec2 p)
{
	uvec2 q = uvec2(ivec2(p)) * uvec2(1597334673U, 3812015801U);
	uint n = (q.x ^ q.y) * 1597334673U;
	return float(n) / float(0xffffffffU);
}

vec3 hash13(float p) {
   vec3 p3 = fract(vec3(p) * vec3(.1031,.11369,.13787));
   p3 += dot(p3, p3.yzx + 19.19);
   return fract(vec3((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y, (p3.y+p3.z)*p3.x));
}

float rainDrops(vec2 st, float time, float size)
{
    vec2 uv = st * size;
    uv.x *= iResolution.x / iResolution.y;
    vec2 gridUv = fract(uv) - .5; // grid
   	vec2 id = floor(uv);
    vec3 h = (hash13(id.x * 467.983 + id.y * 1294.387) - .5) * .8;
    vec2 dropUv = gridUv - h.xy;
    vec4 noise = textureLod(iChannel1, id * .05, 0.);
    float drop = smoothstep(.25, 0., length(dropUv)) *
        max(0., 1. - fract(time * (noise.b + .1) * .2 + noise.g) * 2.);
    return drop;
}

vec2 wigglyDrops(vec2 st, float time, float size)
{
    vec2 wigglyDropAspect = vec2(2., 1.);
    vec2 uv = st * size * wigglyDropAspect;
    uv.x *= iResolution.x / iResolution.y;
    uv.y += time * .23;

    vec2 gridUv = fract(uv) - .5; // rectangular grid
    vec2 id = floor(uv);
    
    float h = hash21(id);
    time += h * 2. * PI;
    float w = st.y * 10.;
    float dx = (h - .5) * .8;
    dx += (.3 - abs(dx)) * pow(sin(w), 2.) * sin(2. * w) *
        pow(cos(w), 3.) * 1.05; // wiggle
    float dy = -sin(time + sin(time + sin(time) * .5)) * .45; // slow down drop before continuing falling
    dy -= (gridUv.x - dx) * (gridUv.x - dx);
    
    vec2 dropUv = (gridUv - vec2(dx, dy)) / wigglyDropAspect;
    float drop = smoothstep(.06, .0, length(dropUv));
    
    vec2 trailUv = (gridUv - vec2(dx, time * .23)) / wigglyDropAspect;
    trailUv.y = (fract((trailUv.y) * 8.) - .5) / 8.;
    float trailDrop = smoothstep(.03, .0, length(trailUv));
    trailDrop *= smoothstep(-.05, .05, dropUv.y) * smoothstep(.4, dy, gridUv.y) *
        	(1.-step(.4, gridUv.y));
    
    float fogTrail = smoothstep(-.05, .05, dropUv.y) * smoothstep(.4, dy, gridUv.y) *
			smoothstep(.05, .01, abs(dropUv.x)) * (1.-step(.4, gridUv.y));
    
    return vec2(drop + trailDrop, fogTrail);
}

vec2 getDrops(vec2 st, float time)
{
    vec2 largeDrops = wigglyDrops(st, time * 2., 1.6);
    vec2 mediumDrops = wigglyDrops(st + 2.65, (time + 1296.675) * 1.4, 2.5);
    vec2 smallDrops = wigglyDrops(st - 1.67, time - 896.431, 3.6);
    float rain = rainDrops(st, time, 20.);
    
    vec2 drops;
    drops.y = max(largeDrops.y, max(mediumDrops.y, smallDrops.y));
    drops.x = smoothstep(.4, 2., (1. - drops.y) * rain + largeDrops.x +
                          mediumDrops.x + smallDrops.x); // drops kinda blend together

    return drops;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 st = fragCoord / iResolution.xy;
    float time = mod(iTime + 100., 7200.);
    
    vec2 drops = getDrops(st, time);
    vec2 offset = drops.xy;
    float lod = (1. - drops.y) * 4.8;
    
    // This is kinda expensive, would love to use a cheaper method here.
    vec2 dropsX = getDrops(st + vec2(.001, 0.), time);
    vec2 dropsY = getDrops(st + vec2(0., .001), time);
    vec3 normal = vec3(dropsX.x - drops.x, dropsY.x - drops.x, 0.);
    normal.z = sqrt(1. - normal.x * normal.x - normal.y * normal.y);
    normal = normalize(normal);
    
    float lightning = sin(time * sin(time * 30.)); // screen flicker
    float lightningTime = mod(time, 10.) / 9.9;
   	lightning *= 1. - smoothstep(.0, .1, lightningTime)
        + smoothstep(.9, 1., lightningTime); // lightning flash mask
    
	vec3 col = textureLod(iChannel0, st+normal.xy * 3., lod).rgb;
    col *= (1. + lightning);
    
    col *= vec3(1., .8, .7); // slight red-ish tint
    col += (drops.y > 0. ? vec3(.5, -.1, -.15)*drops.y : vec3(0.)); // bloody trails
    col *= (drops.x > 0. ? vec3(.8, .2, .1) * (1.-drops.x) : vec3(1.)); // blood colored drops
    
    col = mix(col, col*smoothstep(.8, .35, length(st - .5)), .6); // vignette
    
    fragColor = vec4(col, 1.0);
}`,

"Igbeaux Bud": `

//------------------------------------------------------
//
// Fractal_Vibrations.glsl
//
// original:  https://www.shadertoy.com/view/Xly3R3
//            2016-10-05  Kaleo by BlooD2oo1
//
//   v1.0  2016-10-06  first release
//   v1.1  2018-03-23  AA added, mainVR untested!!! 
//   v1.2  2018-09-02  supersampling corrected
//
// description  a koleidoscopic 3d fractal
//
// Hires B/W fractal picture:
//   https://c2.staticflickr.com/6/5609/15527309729_b2a1d5a491_o.jpg
//
//------------------------------------------------------

float g_fScale = 1.2904082537;

mat4 g_matIterator1 = mat4(-0.6081312299, -0.7035965919, 0.3675977588, 0.0000000000,
                            0.5897225142, -0.0904228687, 0.8025279045, 0.0000000000,
                           -0.5314166546, 0.7048230171, 0.4699158072, 0.0000000000,
                            0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000 );

mat4 g_matIterator2 = mat4(-0.7798885703, 0.6242666245, -0.0454343557, -0.2313748300,
                            0.0581589043, 0.0000002980, -0.9983071089, -0.2313748300,
                           -0.6232098937, -0.7812111378, -0.0363065004, -0.2313748300,
                            0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000 );

mat4 g_matReflect1 = mat4( 0.9998783469, -0.0103046382, -0.0117080826, 0.0000000000,
                          -0.0103046382, 0.1270489097, -0.9918430448, 0.0000000000,
                          -0.0117080826, -0.9918430448, -0.1269274950, 0.0000000000,
                           0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000 );

mat4 g_matReflect2 = mat4( 0.7935718298, -0.0946179554, 0.6010749936, 0.0000000000,
                          -0.0946179554, 0.9566311240, 0.2755074203, 0.0000000000,
                           0.6010749936, 0.2755074203, -0.7502027750, 0.0000000000,
                           0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000 );

mat4 g_matReflect3 = mat4(-0.7127467394, -0.5999681950, 0.3633601665, 0.0000000000,
                          -0.5999681950, 0.7898335457, 0.1272835881, 0.0000000000,
                           0.3633601665, 0.1272835881, 0.9229129553, 0.0000000000,
                           0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000 );

vec4 g_planeReflect1 = vec4( 0.0077987094, 0.6606628895, 0.7506421208, -0.0000000000 );

vec4 g_planeReflect2 = vec4( 0.3212694824, 0.1472563744, -0.9354685545, -0.0000000000 );

vec4 g_planeReflect3 = vec4( -0.9254043102, -0.3241653740, 0.1963250339, -0.0000000000 );

/////////////////////////////////////////////////////////////////////////////////////////

vec3 HSVtoRGB(float h, float s, float v) 
{
  return((clamp(abs(fract(h +vec3(0.,2./3.,1./3.))*2.-1.)*3.-1.,0.,1.)-1.)*s+1.)*v;
}

mat3 rot3xy( vec2 angle )
{
  vec2 c = cos( angle );
  vec2 s = sin( angle );
  return mat3( c.y,       -s.y,        0.0,
               s.y * c.x,  c.y * c.x, -s.x,
               s.y * s.x,  c.y * s.x,  c.x );
}

vec4 DE1( in vec4 v )
{
  float fR = dot( v.xyz, v.xyz );
  vec4 q;
  int k = 0;
  vec3 vO = vec3( 0.0, 0.0, 0.0 );

  for ( int i = 0; i < 32; i++ )
  {
    q = v*g_matIterator1;
    v.xyz = q.xyz;

    if ( dot( v, g_planeReflect1 ) < 0.0 )
    {
      q = v*g_matReflect1;
      v.xyz = q.xyz;
      vO.x += 1.0;
    }

    if ( dot( v, g_planeReflect2 ) < 0.0 )
    {
      q = v*g_matReflect2;
      v.xyz = q.xyz;
      vO.y += 1.0;
    }

    if ( dot( v, g_planeReflect3 ) < 0.0 )
    {
      q = v*g_matReflect3;
      v.xyz = q.xyz;
      vO.z += 1.0;
    }

    q = v*g_matIterator2;
    v.xyz = q.xyz;

    v.xyz = v.xyz*g_fScale;
    fR = dot( v.xyz, v.xyz );
    k = i;
  }
  return vec4( vO, ( sqrt( fR ) - 2.0 ) * pow( g_fScale, -float(k+1) ) );
}

//------------------------------------------------------

float time = 0.0;  
float fL = 1.0;

//------------------------------------------------------
vec4 renderRay (in vec3 rayOrig, in vec3 rayDir)
{
  rayDir = normalize( rayDir );

  const float fRadius = 2.0;
  float b = dot( rayDir, rayOrig ) * 2.0;
  float c = dot( rayOrig, rayOrig ) - fRadius*fRadius;
  float ac4 = 4.0 * c;
  float b2 = b*b;

  vec4 color = vec4(0,0,0,1);
  color.rgb = -rayDir*0.2+0.8;
  color.rgb = pow( color.rgb, vec3( 0.9, 0.8, 0.5 ) );
  color.rgb *= 1.0-fL;
  if ( b2 - ac4 <= 0.0 )  return color;

  float root = sqrt( b2-ac4 );
  float at1 = max(0.0, (( -b - root ) / 2.0));
  float at2 = ( -b + root ) / 2.0;

  float t = at1;
  vec4 v = vec4( rayOrig + rayDir * t, 1.0 );
  vec4 vDE = vec4( 0.0, 0.0, 0.0, 0.0 );
  float fEpsilon = 0.0;

  float fEpsilonHelper = 1.0 / iResolution.x;
    
  float count = 0.0;
  for ( int k = 0; k < 100; k++ )
  {
    vDE = DE1( v );
    t += vDE.w;
    v.xyz = rayOrig + rayDir * t;

    fEpsilon = fEpsilonHelper * t;
		
    if ( vDE.a < fEpsilon ) 
    {
        count = float(k);
        break;
    }
    if ( t > at2 )     return color;
  }
    
  // colorizing by distance of fractal
  color.rgb = HSVtoRGB(count/25., 1.0-count/50., 0.8);
    
  vec4 vOffset = vec4( fEpsilon*1.8, 0.0, 0.0, 0.0 );
  vec4 vNormal = vec4(0.0);
  vNormal.x = DE1( v + vOffset.xyzw ).w - DE1( v - vOffset.xyzw ).w;
  vNormal.y = DE1( v + vOffset.yxzw ).w - DE1( v - vOffset.yxzw ).w;
  vNormal.z = DE1( v + vOffset.zyxw ).w - DE1( v - vOffset.zyxw ).w;
  vNormal.xyz = normalize( vNormal.xyz );

  vec4 vReflect = vec4(0.7);
  vReflect.xyz = reflect( rayDir, vNormal.xyz );

  vec2 vOccRefl = vec2( 0.0, 0.4 );
  
  float fMul = 2.0;
  float fMulMul = pow( 2.0, 9.0/10.0 ) * pow( fEpsilon, 1.0/10.0 ) * 0.5;
  float fW = 0.0;
  for ( int k = 0; k < 8; k++ )
  {
    vOccRefl.x += DE1( v + vNormal * fMul ).w / fMul;
    vOccRefl.y += DE1( v + vReflect * fMul ).w / fMul;
    fMul *= fMulMul;
  }
  vOccRefl /= 6.0;
  
  color.rgb *= vec3( vOccRefl.x * vOccRefl.y );
  color.rgb *= (vNormal.xyz*0.5+0.5)*(1.0-vOccRefl.x) +vec3(1.5)* vOccRefl.y;
  color.rgb = pow( color.rgb, vec3( 0.4, 0.5, 0.6 ) );
  color.rgb *= 1.0-fL;
  return vec4(color.rgb, 1.0);
}

//------------------------------------------------------
void mainVR (out vec4 fragColor, in vec2 fragCoord
            ,in vec3 fragRayOri, in vec3 fragRayDir)
{
  vec2 uv = (fragCoord - iResolution.xy*0.5) / iResolution.x;
  fL = length( uv );
  fragColor = renderRay (fragRayOri, fragRayDir);
}

//------------------------------------------------------
vec4 render(in vec2 pos)
{
  time = iTime * 0.1;  
  vec2 mouse = iMouse.xy / iResolution.xy;
  vec3 rayOrig = vec3( -3.0 - sin( time ), 0.0, 0.0 );
  vec2 uv = (pos - iResolution.xy*0.5) / iResolution.x;
  fL = length( uv );
  uv /= fL;
  uv *= 1.0-pow( 1.0-fL, 0.7 );
  vec3 rayDir = vec3(0.45+mouse.y, uv );

  mat3 rot = rot3xy( vec2( 0.0, time + mouse.x * 4.0) );
  rayDir  = rot * rayDir;
  rayOrig = rot * rayOrig;
    
  return renderRay (rayOrig, rayDir);
}

//------------------------------------------------------

#define AAX 2   // supersampling level. Make higher for more quality.
#define AAY 1   

float AA = float(AAX * AAY);

//------------------------------------------------------
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
  float tr = texture(iChannel0,vec2(0.01, 0.01)).r;  // sound value
  g_fScale -= (tr*2.3 - 2.0) / 18.0;    // fractal scaling

  if (AAX>1 || AAY>1)
  {
    vec4 col = vec4(0,0,0,1);
    for (int xp = 0; xp < AAX; xp++)
    for (int yp = 0; yp < AAY; yp++)
    {
      vec2 pos = fragCoord + vec2(xp,yp) / vec2(AAX,AAY);
      col += render (pos);
    }
    fragColor.rgb = col.rgb / AA;
  }
  else fragColor = render (fragCoord);
}

`,
"Dancing Blood": `vec3 palette( float t)
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
    float displacement = sin(3.0 * p.x) * sin(3.0 * p.y) * sin(3.0 * p.z) * 0.25 * (sin(iTime * 2. + cos(iTime * 12.)));
    float sphere_0 = distance_from_sphere(p, vec3(0.0), 1.8);
    
    float sphere_1 = distance_from_sphere(p, vec3(0.0, 2.0,0.0), 1.8);

    return sphere_0 + displacement;
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
"Dancing Blood 2": `vec3 palette( float t)
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
    float audioAmplitude = bassLevel * 1.6 + midLevel * 0.3 + highLevel * 0.1;
    
    // Add a baseline motion so it's always animated even with no audio
    float baselineMotion = sin(iTime * 2.0 + cos(iTime * 12.0));
    
    // Combine audio amplitude with baseline motion (weighted so audio is prominent)
    float combinedAmplitude = mix(baselineMotion * 0.25, audioAmplitude * 0.9, 0.7);
    
    // Calculate displacement using the combined amplitude
    float displacement = sin(3.0 * p.x) * sin(3.0 * p.y) * sin(3.0 * p.z) * 0.25 * combinedAmplitude;
    
    // Make sphere surface slightly ripple based on higher frequencies
    float detailRipple = sin(8.0 * p.x + iTime) * sin(8.0 * p.y + iTime) * sin(8.0 * p.z) * highLevel * 0.1;
    
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
"Blue Disks": `#define SYSTEMS_BLUE

float rand(vec2 uv){

    uv = fract(uv*vec2(420.324, 730.163));
    uv += dot(uv, uv + 45.94);
    return (fract(uv.x*uv.y));
}


vec3 r13(float seed, float zones){

    seed = 5.*mod(seed, zones);
    float r1 = rand(vec2(seed+1., seed));
    float r2 = rand(vec2(seed, seed+1.));
    float r3 = rand(vec2(r1, r2));
    return vec3(r1, r2, r3);
}

//gr - grid and radius
float sector(vec2 pos, vec2 dim, vec2 gr){
    return length(max(abs(gr - pos)-dim,0.0))-.03;
}

float red(vec3 rng){
    return .2+rng.y*rng.z;
}

float blue(vec3 rng){
    return .2+rng.y*rng.z;
}

float green(vec3 rng){
    return .6+.5*rng.z;
}

vec3 getColor(vec3 rng){
    float red, green, blue;
    #ifdef SYSTEMS_BLUE
    red = rng.y*rng.z;
    blue =  .6+.5*rng.z;
    green = .2+rng.y*rng.z;
    #elif defined(SYSTEMS_RED)
    red = .6+.5*rng.z;
    green = rng.y*rng.z;
    blue = .2+rng.y*rng.z;
    #elif defined(SYSTEMS_GREEN)
    red = .3+rng.y*rng.z;
    green = .6+.5*rng.z;
    blue = .2+rng.y*rng.z;
    #elif defined(SYSTEMS_GOLD)
    red = .6+.4*rng.z;
    green = .4+.4*rng.z;
    blue = .1+rng.y*rng.z;
    
    #endif
    return vec3(red, green, blue);
}


vec3 sects(vec4 pulse, vec2 uv){
    float r = length(uv);  //radius
    vec2 rt = vec2(r, acos(uv.x/r)-2.*floor(uv.y)*acos(-uv.x/r));// - floor(uv.y)*3.14); //radius and angle
    rt += iTime * .07;
    float zones = 12.;  //number of zones
    float zone = floor(zones*rt.y/6.283);  //zone id
    float gd = fract(zones*rt.y/6.283);  //zone continuous
    
    
    vec3 rng = r13(zone, zones);
    vec3 rngccw = r13(zone - 1., zones);
    vec3 rngcw = r13(zone + 1., zones);
    float offset = 1.;///zones;
    
    vec3 col = vec3(0.);
    float swiv = .5 + rng.z*rng.x*sin(iTime + rng.x*rng.y);
    float swivccw = .5 - offset +  rngccw.z*rngccw.x*sin(iTime + rngccw.x*rngccw.y);
    float swivcw = .5 + offset +  rngcw.z*rngcw.x*sin(iTime + rngcw.x*rngcw.y);
    
    float dist = .4 + .2*rng.x;
    float distccw = .4 + .2*rngccw.x;
    float distcw = .4 + .2*rngcw.x;
    
    float len = .2 + .2*rng.y;
    float lenccw = .2 + .2*rngccw.y;
    float lencw = .2 + .2*rngcw.y;
    
    float width = .45;
    float widthccw = .45;
    float widthcw = .45;
    
    float sect = sector(vec2(swiv*r, dist), vec2(width*r, len), vec2(gd*r, r));
    float sectccw = sector(vec2(swivccw*r, distccw), vec2(widthccw*r, lenccw), vec2(gd*r, r));
    float sectcw = sector(vec2(swivcw*r, distcw), vec2(widthcw*r, lencw), vec2(gd*r, r));
    
    //colors
    
    vec3 color = getColor(rng);
    vec3 colorccw = getColor(rngccw);
    vec3 colorcw = getColor(rngcw);
    
    //sector
    vec3 base;
    #ifdef SYSTEMS_BLUE
        base = vec3(0., .3, .6);
    #elif defined(SYSTEMS_RED)
        base = vec3(.6, 0., .2);
    #elif defined(SYSTEMS_GREEN)
        base = vec3(0., .6, .3);
    #elif defined(SYSTEMS_GOLD)
        base = vec3(.6, .4, 0.);
    #endif
    float basestrength = .08+.2*pulse.x;
    col+= basestrength*smoothstep(.002, 0., sect)*base;
    col+= basestrength*smoothstep(.002, 0., sectccw)*base;
    col+= basestrength*smoothstep(.002, 0., sectcw)*base;
    //border
    float bstrength = pulse.z;
    col+= bstrength*smoothstep(-0.002, sect, 0.007)*color;
    col+= bstrength*smoothstep(-0.002, sectccw, 0.007)*colorccw;
    col+= bstrength*smoothstep(-0.002, sectcw, 0.007)*colorcw;
    //glow
    float fsize = .5*pulse.z;
    float glow = smoothstep(.2+3.*fsize,1.6*fsize,r*(.5+fsize*sin(10.*rt.y)))+smoothstep(1., .0, r);
    float furr =  + .2*smoothstep(.3, .8, r)*sin(94.2478*(gd+.02*sin((r-.5*iTime)*20.)));
    glow += furr;
    glow *= pulse.w;
    col += glow*smoothstep(0.25*r, 0.0, sect)*color;
    col += glow*smoothstep(0.25*r, 0.0, sectccw)*colorccw;
    col += glow*smoothstep(0.25*r, 0.0, sectcw)*colorcw;

    return col;
}


vec4 sampleMusic()
{
	return vec4(
		texture( iChannel0, vec2( 0.01, 0.25 ) ).x,
		texture( iChannel0, vec2( 0.07, 0.25 ) ).x,
		texture( iChannel0, vec2( 0.15, 0.25 ) ).x,
		texture( iChannel0, vec2( 0.30, 0.25 ) ).x);
}

vec4 sampleMic()
{
	return vec4(
		texture( iChannel0, vec2( 0.01, 0.25 ) ).x,
		texture( iChannel0, vec2( 0.07, 0.25 ) ).x,
		texture( iChannel0, vec2( 0.15, 0.25 ) ).x,
		texture( iChannel0, vec2( 0.30, 0.25 ) ).x);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = (2.*fragCoord-iResolution.xy)/iResolution.y;
    float r = length(uv);  //radius
    vec2 rt = vec2(r, acos(uv.x/r)-2.*floor(uv.y)*acos(-uv.x/r));// - floor(uv.y)*3.14); //radius and angle
    // Time varying pixel color
//    float circ = step(abs(r-.5+(.02-.05*sin(rt.y+iTime))*cos(10.*rt.y-sin(iTime))), .01);
    float zones = 12.;  //number of zones
    float zone = floor(zones*rt.y/6.283);  //zone id
    
    vec4 pulse = vec4(0.);
    pulse = sampleMusic();
    pulse = sampleMic();
    vec3 col = vec3(0.);
    col += sects(pulse, uv);
    //col += vec3(step(abs(rand(vec2(zone+1., zone))-r),.02)); 
    //col += vec3(step(abs(rand(vec2(zone+2., zone+1.))-r),.02)); 
    //col += vec3(step(abs(rand(vec2(zone, zone -1.))-r),.02));
    // Output to screen
    fragColor = vec4(col,1.0);
}`,
"IN PROGRESS":`// compact version of https://www.shadertoy.com/view/4sl3RX 
// --- infinite fall --- Fabrice NEYRET  august 2013


#define L  20.
#define R(a) mat2(C=cos(a),S=sin(a),-S,C)
float C,S,v, t;

float N(vec2 u) { // infinite perlin noise
	mat2 M = R(1.7);
    C = S = 0.;
	for (float i=0.; i<L; i++)
	{   float k = i-t,
		      a = 1.-cos(6.28*k/L),
		      s = exp2(mod(k,L));
		C += a/s* ( 1. - abs( 2.* texture(iChannel0, M*u*s/1e3 ).r - 1.) ); 
		S += a/s;  M *= M;
	}
    return 1.5*C/S;
}

void mainImage( out vec4 o, vec2 u ) {
	vec2 r = iResolution.xy, e=vec2(.004,0);
    t = 1.5*iTime;
 	v = N( u = (u-.5*r) / r.y * R(t) );
	o =   v*v*v/vec4(1,2,4,1) 
        * min( 1., 51.*N(u+e) + 205.*N(u+e.yx) -256.*v ) // lum
;
}`,
"Missing Head": `// Fork of "Football" by . https://shadertoy.com/view/llKcR3
// 2019-07-25 21:23:38
// Modified from https://www.shadertoy.com/view/Xds3zN by iq.
//

/*
 Some path funtion: timefly(t) returns a 2d pivot
 pasaR(t) and pasaL(t) modifies time to get 
 initial foot targets when fead to timefly()
 
*/

#define AA 1
# define PI 3.14159265359
# define PHI 1.618033988749895
# define TAU 6.283185307179586
vec3 rightFoot;
vec3 leftFoot;
vec3 rightToe;
vec3 leftToe;
vec3 rightHand;
vec3 leftHand;
vec3 rightFootT;
vec3 leftFootT;
vec3 rightHandT;
vec3 leftHandT;
vec3 rightToeT;
vec3 leftToeT;
vec3 rightE; // Elbow
vec3 leftE;
vec3 rightK;//Knee
vec3 leftK;
vec3 rightH; //Hip
vec3 leftH;
vec3 rightS;// Shoulder
vec3 leftS;
vec3 pelvis;
vec3 torso;
vec3 head;
vec3 target;

	float pasa = 1.; // steps overlaping airtime
	float legmax = .89; // max extention
	float leg = .89+0.005; // actual max length
	float armmax = .7;// max extention
	float arm = .7 +.012;// actual max length
  	float toemax = 1.1;// max extention toe from hip
    float footlift=0.19; //lift height later multiplied by speed


# define PLOTPATH 0
 

//------------------------------------------------------------------
float sdPlane(vec3 p) {
	return p.y;
}
float sdSphere(vec3 p, float s) {
	return length(p) - s;
}
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
	vec3 pa = p - a, ba = b - a;
	float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
	return length(pa - ba * h) - r;
}
float sdRoundedCylinder( vec3 p, float ra, float rb, float h )
{
    vec2 d = vec2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}

float pathterrain(float x,float z){
    // Common height function for path and terrain
    return 
        sin(x*.5 )*1.+cos(z*.3 )*0.3
        +cos(x*3.+z )*0.1+sin(x-z*.2 )*0.2
        
        ;}
 vec3 timefly(float t) {
    // main path Called from many places
    t*=.80;
	t += (.125 + sin(t * .125));
	vec3 v =
	vec3(sin(t / 50.) * 20., 0., cos(t / 25.) * 24.) +
		vec3(sin(t / 17.1) * 07., 0., cos(t / 17.1) * 05.) +
		vec3(sin(t / 8.1) * 6., 0., cos(t / 8.1) * 8.) +
		vec3(cos(t / 3.) * 3.,0., sin(t / 3.) * 2.)
        +vec3(cos(t  )*2.,0., sin(t  )*2. );
    v.y=pathterrain(v.x,v.z);
    return v        ;
} 
float pasaR(float x){
return max(x + fract(x + 0.25) * pasa - pasa, floor(x + 0.25) - 0.25) + 0.25;
    //gait function 
}
    
float pasaL(float x){
return max(x + fract(x - 0.25) * pasa - pasa, floor(x - 0.25) + 0.25) + 0.25;
   //gait function 
}



float lpnorm(vec3 p, float s) {
	return pow(
		(
			pow(abs(p.x), s) +
			pow(abs(p.y), s) +
			pow(abs(p.z), s)),
		1.0 / s);
}

 
//------------------------------------------------------------------
vec2 opU(vec2 d1, vec2 d2) {
	return (d1.x < d2.x) ? d1 : d2;
}
float smin(float a, float b, float k)
{
	float h = clamp(.5 + .5*(a-b)/k, 0., 1.);
	return mix(a, b, h) - k*h*(1.-h);
}
vec2 bodyPlan(vec3 pos) {
	float res;
	res =  sdSphere(pos - leftFoot, .07);
	res = min(res, sdSphere(pos - leftHand, .06));
	res = min(res, sdSphere(pos - leftH, .09));
	res = min(res, sdSphere(pos - leftK, .08));
	res = min(res, sdSphere(pos - leftE, .08));
	res = min(res, sdSphere(pos - leftS, .07));	

    

    res = min(res, sdSphere(pos - rightFoot, .07));
	res = min(res, sdSphere(pos - rightHand, .06));
	res = min(res, sdSphere(pos - rightH, .09));
	res = min(res, sdSphere(pos - rightS, .07));
	res = min(res, sdSphere(pos - rightK, .08));
	res = min(res, sdSphere(pos - rightE, .08));
    
    	res = min(res, sdSphere(pos - target, .2));
	//res = min(res, sdSphere(pos - head, .16));

 	
   

    

    
    res = min(res, sdCapsule(pos ,rightToe,rightFoot, .06));
    res = smin(res, sdRoundedCylinder(pos - rightToe, .04, .02, .03 ),0.06 );

    res = min(res, sdCapsule(pos ,rightK,rightFoot, .06));
    res = min(res, sdCapsule(pos ,rightK,rightH, .07));   
    res = min(res, sdCapsule(pos ,rightE,rightHand, .05));
    res = min(res, sdCapsule(pos ,rightE,rightS, .06));
    res = min(res, sdCapsule(pos ,torso,rightS, .08));
    
    res = min(res, sdCapsule(pos ,leftToe,leftFoot, .06));
    res = smin(res, sdRoundedCylinder(pos - leftToe, .04, .02, .03 ),0.06);// todo rotate to grund normal

    res = min(res, sdCapsule(pos ,leftK,leftFoot, .06));
    res = min(res, sdCapsule(pos ,leftK,leftH, .07));   
    res = min(res, sdCapsule(pos ,leftE,leftHand, .05));
    res = min(res, sdCapsule(pos ,leftE,leftS, .06));
    res = min(res, sdCapsule(pos ,torso,leftS, .08));
    
    res = smin(res, sdSphere(pos - torso, .14),0.025);
    res = smin(res, sdSphere(pos - pelvis, .16),0.025);
    
    res = smin(res, sdCapsule(pos ,pelvis,torso, .13),0.025);
	res = min(res, sdCapsule(pos ,head,torso, .02)); 
    
    
    
    if(PLOTPATH>0)for(int i=PLOTPATH;i>-PLOTPATH/2;i--)

    {
        res = min(res, sdSphere(pos- timefly(iTime+float(i)*0.5), .04));
    
       
        
}  
  
        
 
    
     //float x=iTime;
	 // res= min(res, sdCapsule( pos, timefly(x),timefly(x+1.) , .06125));
	 // res= min(res, sdCapsule( pos, timefly(x)-perpr*-0.25,timefly(x)-perpl*0.25 , .06125));
	return vec2(res, 2.0);
}
vec2 map( in vec3 pos) {
	vec2 res = vec2(pos.y-pathterrain(pos.x,pos.z), 1.0);
	res = opU(res, bodyPlan(pos));
	return res;
}
vec2 castRay( in vec3 ro, in vec3 rd) {
	float tmin = 1.0;
	float tmax = 30.0;
	float t = tmin;
	float m = -1.0;
	for (int i = 0; i < 80; i++) {
		float precis = 0.0001 * t;
		vec2 res = map(ro + rd * t);
		if (res.x < precis || t > tmax) break;
		t += res.x * .7;
		m = res.y;
	}
	if (t > tmax) m = -1.0;
	return vec2(t, m);
}
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax) {
	float res = 1.0;
	float t = mint;
	for (int i = 0; i < 32; i++) {
		float h = map(ro + rd * t).x;
		res = min(res, 8.0 * h / t);
		t += clamp(h, 0.02, 0.10);
		if (res < 0.005 || t > tmax) break;
	}
	return clamp(res, 0.0, 1.0);
}
vec3 calcNormal( in vec3 pos) {
	vec2 e = vec2(1.0, -1.0) * 0.5773 * 0.0005;
	return normalize(e.xyy * map(pos + e.xyy).x +
		e.yyx * map(pos + e.yyx).x +
		e.yxy * map(pos + e.yxy).x +
		e.xxx * map(pos + e.xxx).x);
}
float calcAO( in vec3 pos, in vec3 nor) {
	float occ = 0.0;
	float sca = 1.0;
	for (int i = 0; i < 5; i++) {
		float hr = 0.01 + 0.12 * float(i) / 4.0;
		vec3 aopos = nor * hr + pos;
		float dd = map(aopos).x;
		occ += -(dd - hr) * sca;
		sca *= 0.95;
	}
	return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}
// https://iquilezles.org/articles/checkerfiltering
float checkersGradBox2( in vec2 p) {
	// filter kernel
	vec2 w = fwidth(p) + 0.001;
	// analytical integral (box filter)
	vec2 i = 2.0 * (abs(fract((p - 0.5 * w) * 0.5) - 0.5) - abs(fract((p + 0.5 * w) * 0.5) - 0.5)) / w;
	// xor pattern
	return 0.5 - 0.5 * i.x * i.y;
}
float checkersGradBox( in vec2 p) {
 
	return  checkersGradBox2(p) -checkersGradBox2(p-0.03 )*0.4 ;
}
vec3 render( in vec3 ro, in vec3 rd) {
	vec3 col = vec3(0.7, 0.9, 1.0) + rd.y * 0.8;
	vec2 res = castRay(ro, rd);
	float t = res.x;
	float m = res.y;
	if (m > -0.5) {
		vec3 pos = ro + t * rd;
		vec3 nor = calcNormal(pos);
		vec3 ref = reflect(rd, nor);
		// material        
		col = 0.45 + 0.35 * sin(vec3(0.05, 0.08, 0.10) * (m - 1.0));
		if (m < 1.5) {
			float f = checkersGradBox(1.2 * pos.xz);
			col = 0.3 + f * vec3(0.3);
		}
		if (m >= 2.0) {
			col = vec3(0.6);
		}
		if (m >= 3.0) {
			col = vec3(0.07);
		}
		// lighting        
		float occ = calcAO(pos, nor);
		vec3 lig = normalize(vec3(0.2, 0.7, 0.6));
		vec3 hal = normalize(lig - rd);
		float amb = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
		float dif = clamp(dot(nor, lig), 0.0, 1.0);
		float bac = clamp(dot(nor, normalize(vec3(-lig.x, 0.0, -lig.z))), 0.0, 1.0) * clamp(1.0 - pos.y, 0.0, 1.0);
		float dom = smoothstep(-0.1, 0.1, ref.y);
		float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 2.0);
		dif *= calcSoftshadow(pos, lig, 0.02, 2.5);
		dom *= calcSoftshadow(pos, ref, 0.02, 2.5);
		float spe = pow(clamp(dot(nor, hal), 0.0, 1.0), 16.0) *
			dif *
			(0.04 + 0.96 * pow(clamp(1.0 + dot(hal, rd), 0.0, 1.0), 5.0));
		vec3 lin = vec3(0.0);
		lin += 1.30 * dif * vec3(1.00, 0.80, 0.55);
		lin += 0.20 * amb * vec3(0.40, 0.60, 1.00) * occ;
		lin += 0.20 * dom * vec3(0.40, 0.60, 1.00) * occ;
		lin += 0.30 * bac * vec3(0.25, 0.25, 0.25) * occ;
		lin += 0.35 * fre * vec3(1.00, 1.00, 1.00) * occ;
		col = col * lin;
		col += 10.00 * spe * vec3(1.00, 0.90, 0.70);
		col = mix(col, vec3(0.8, 0.9, 1.0), 1.0 - exp(-0.0002 * t * t * t));
	}
	return vec3(clamp(col, 0.0, 1.0));
}
mat3 setCamera( in vec3 ro, in vec3 ta, float cr) {
	vec3 cw = normalize(ta - ro);
	vec3 cp = vec3(sin(cr), cos(cr), 0.0);
	vec3 cu = normalize(cross(cw, cp));
	vec3 cv = normalize(cross(cu, cw));
	return mat3(cu, cv, cw);
}





void setup() {
	float x = iTime   ;//Time manipulations moved to timefly
      
    
    // filter gait slightly for less stabby foot placement, too much generates skating
    float filt=18.;
	float left = 0.025+ mix(pasaR(floor(x*filt)/filt) ,pasaR(ceil(x*filt)/filt), ( fract(x*filt)));
	float right =0.025+ mix(pasaL(floor(x*filt)/filt) ,pasaL(ceil(x*filt)/filt), ( fract(x*filt)));
	
    
    float ahead=1.1;
    vec3 dif = (timefly(x + ahead) - timefly(x))/ahead; //delta x+1
	float speed = length(dif); 
     ahead = clamp(0.8,1.1,1.3-speed);
     dif = (timefly(x + ahead) - timefly(x))/ahead; //delta x+1
	 speed = length(dif); 
    
    
    
    vec3 nextdif = (timefly(x + ahead+.5) - timefly(x + .5))/ahead; 
	vec3 lean = (nextdif - dif*2.); // bank into turns

    //
      ahead=speed;
      dif = (timefly(x + ahead) - timefly(x))/ahead; //delta x+1
	  nextdif = (timefly(x + ahead+.5) - timefly(x + .5))/ahead; 
	  lean = (nextdif - dif*2.); // bank into turns

     
    
	float nextSpeed = length(timefly(x + 1.2) - timefly(x + .2));
   


    vec3 dir = normalize(dif); //Path direction 
    vec3 nextdir = normalize(nextdif); //Path direction 
	vec3 dirr = normalize(timefly(right + 1.) - timefly(right)); //Path direction Foot specific
	vec3 dirl = normalize(timefly(left + 1.) - timefly(left));
    
	vec3 perp = cross(dir,vec3(0,-1,0));// perpendicular to main path
	vec3 perpl = cross(dirl,vec3(0,-1,0));// perpendicular to intervalled step path
	vec3 perpr = cross(dirr,vec3(0,-1,0));
    
    target =(timefly(x+1.5))
               
            +(vec3(0,.4,0)+lean*1.6+dir*0.25 )*(.09/clamp(speed , 0.05, 4.5));// rolling head

       
    target.y=pathterrain(target.x,target.z);// fix for rolling head collision 
    
    target +=
        +( vec3(0,0.14+abs(sin(x*7.)*0.3),0)) ;
  
      
    vec3 tfx= timefly(x)  ; // Pelvis   path
    vec3 tfr= timefly(right) ;//intervalled step path
    vec3 tfl= timefly(left) ; //intervalled step path
    
    // foot lift component
	vec3 leftlift = vec3(0, min(0., sin(x * TAU + 1.57) * footlift * clamp(speed, 0.05, 1.5)), 0);
	vec3 rightlift = vec3(0, min(0., sin(x * TAU - 1.57) * footlift * clamp(speed, 0.05, 1.5)), 0);
 
    
    // setup targets
	rightFootT = tfr + perpr * -0.16 - rightlift;
	leftFootT = tfl + perpl * 0.16 - leftlift;
    rightToeT = tfr  + perpr * -0.19  +dir*0.172 - rightlift*0.6;
	leftToeT = tfl  + perpl * 0.19  +dir*0.172- leftlift*0.7;
    // ground collision feet and toes
    rightFootT.y=max(pathterrain(rightFootT.x,rightFootT.z),    rightFootT.y);
    leftFootT.y=max(pathterrain(leftFootT.x,leftFootT.z),    leftFootT.y);
    rightToeT.y=max(pathterrain(rightToeT.x,rightToeT.z),    rightToeT.y);
    leftToeT.y=max(pathterrain(leftToeT.x,leftToeT.z),    leftToeT.y);
    
    

    
    
	pelvis = tfx 
        + (lean  ) * clamp(nextSpeed, 0.01, .5) * 0.1 // lean into turn
        + vec3(0, .9 + cos(x * TAU * 2.) * 0.02 * speed, 0) // bob u/d with step
		+ dir * 0.1 * (-0.45 + speed) // lean in to run
		+ perpr * sin(x * TAU) * 0.025 * speed // bob l/R with step
 		+ (vec3(0,-1.,0) )*(.02/clamp(speed , 0.15, 4.5))// bend when head is close
;
    // spine component
	vec3 spine = normalize(
		 (lean  ) * clamp(nextSpeed, 0.2, .5) * 0.1 // lean into turn
		+ vec3(0, 0.3 + cos(x * TAU * 2.) * 0.0125 * speed, 0)// bob u/d with step
		+ dir * 0.05 * (-0.25 + nextSpeed)  // lean in to run
        +(vec3(0,-1.,0)+dir)*(.05/clamp(speed , 0.15, 4.5))// bend when head is close
        + perpr * cos(x * TAU) * 0.025 * speed// bob l/R with step
	);
     
    torso = pelvis + spine * 0.3;
    
    

	// Hips
    rightH = pelvis + perp * -0.11 - rightlift * 0.1 - spine * 0.08 + dir * -0.025;
	leftH = pelvis + perp * 0.11 - leftlift * 0.1 - spine * 0.08 + dir * -0.025;
    
    // Feet
	rightFoot = rightH + normalize(rightFootT - rightH) * min(legmax, length(rightFootT - rightH));
	leftFoot = leftH + normalize(leftFootT - leftH) * min(legmax, length(leftFootT - leftH));
	
    rightToe = rightH + normalize(rightToeT - rightH) * min(toemax, length(rightToeT - rightH));
	leftToe = leftH + normalize(leftToeT - leftH) * min(toemax, length(leftToeT - leftH));
    
    // Shoulder
	rightS = torso + perp * -0.2   + spine * 0.05;
	leftS = torso + perp * 0.2  + spine * 0.05;
    
    // Hand Target
    rightHandT=(rightS +  normalize(
			+perpr * -0.06 
			+vec3(0, -0.4, 0) 
			+dir * 0.3 * cos(.25 + x * TAU) * (clamp(speed, 0.0, 2.) * 0.25)
 			) 
            * armmax 
			+vec3(0, 0.2, 0) * clamp(speed - 0.6, 0., 1.) )// lift alittle with speed
        	+( target -rightS)*(1.-smoothstep(0.,1.2,(1.+sin(x*1. ))))*0.3;// reach for head 
    
     leftHandT= (leftS + normalize(
			perpl * 0.06 +
			vec3(0, -0.4, 0) +
			dir * 0.3 * cos(.25 + PI + x * TAU) * (clamp(speed, 0.0, 2.) * 0.25)
 		) * armmax +
		vec3(0, 0.2, 0) * clamp(speed - 0.6, 0., 1.))
       +( target -leftS)*(1.-smoothstep(0.,1.2,(1.+sin(x*1.+PI))))*0.3;
    
       rightHand = rightS + normalize(rightHandT - rightS) * min(armmax, length(rightHandT - rightS));
       leftHand = leftS + normalize(leftHandT - leftS) * min(armmax, length(leftHandT - leftS));
 	
     
        rightHand.y=max(pathterrain(rightHand.x,rightHand.z)+.2,    rightHand.y);
    leftHand.y=max(pathterrain(leftHand.x,leftHand.z)+.2,    leftHand.y);
 
    
    
	head = torso +normalize(
		vec3(0, .27, 0) 
		+ normalize(lean) * clamp(nextSpeed, 0.2, 1.) * 0.05 // lean into torn
		+dir * 0.1 * (-0.35 + clamp(speed, 0.5, 2.)) // lean into run
		+perpr * cos(x * TAU) * 0.025 * clamp(speed, 0.5, 2.)
        +(vec3(0,-1.,0)+dir)*(.07/clamp(speed , 0.05, 4.5))// bend when head is close

       )*0.27;// sway with step
    
    // bendy lims IK
    
	rightE = mix(rightS, rightHand, 0.5) - cross(rightS - rightHand, -normalize(perp - dir * 0.5)) *
		sqrt(max(0.0001, arm * arm - length(rightS - rightHand) * length(rightS - rightHand))) * 0.5;
	leftE = mix(leftS, leftHand, 0.5) - cross(leftS - leftHand, -normalize(perp + dir * 0.5)) *
		sqrt(max(0.0001, arm * arm - length(leftS - leftHand) * length(leftS - leftHand))) * 0.5;
	rightK = mix(rightH, rightFoot, 0.5) - cross(rightH - rightFoot, normalize(perp + dir * 0.25)) *
		sqrt(max(0.0001, leg * leg - length(rightH - rightFoot) * length(rightH - rightFoot))) * 0.5;
	leftK = mix(leftH, leftFoot, 0.5) - cross(leftH - leftFoot, normalize(perp - dir * 0.25)) *
		sqrt(max(0.0001, leg * leg - length(leftH - leftFoot) * length(leftH - leftFoot))) * 0.5;
}


void mainImage(out vec4 fragColor, in vec2 fragCoord) {
	setup();
	vec2 mo = iMouse.xy / iResolution.xy;
	float time = .0 + iTime;
	vec3 tot = vec3(0.0);
 #	if AA > 1
	for (int m = 0; m < AA; m++)
		for (int n = 0; n < AA; n++) {
			// pixel coordinates
			vec2 o = vec2(float(m), float(n)) / float(AA) - 0.5;
			vec2 p = (-iResolution.xy + 2.0 * (fragCoord + o)) / iResolution.y;
 # else
				vec2 p = (-iResolution.xy + 2.0 * fragCoord) / iResolution.y;
 # endif
			// camera	
			vec3 ta = timefly(time) + vec3(0, 0.7, 0);
			vec3 ro = ta + vec3(-0.5 + 3.5 * cos(0.1 * time + 6.0 * mo.x),
				2.0 + 2.0 * mo.y,
				0.5 + 4.0 * sin(0.1 * time + 6.0 * mo.x));
			// camera-to-world transformation
			mat3 ca = setCamera(ro, ta, 0.0);
			// ray direction
			vec3 rd = ca * normalize(vec3(p.xy, 2.5));
			// render	
			vec3 col = render(ro, rd);
			// gamma
			col = pow(col, vec3(0.4545));
			tot += col;
 # if AA > 1
		}
	tot /= float(AA * AA);
 #	endif
	fragColor = vec4(tot, 1.0);
}`,
"Mushroom": `float pi = 3.14159265359;
float sdSphere( vec3 p, float r, float shift) {
    return length(vec3(p.x, p.y-shift, p.z))-r;
}

float sdCircle(vec2 p, float r) {
    float angle = atan(p.y, p.x)+sin(length(p)*pi+0.5);
    float ripple = 0.5 + 0.5 * sin(angle * 30.0);
    return length(p) - r * ripple;
}
float sdCylinder(vec3 p, float r, float h) {
    p.y+=sin(length(p.xz)*3.0)/4.0;
    p.y*=2.5;
    p.y-=0.7;
    p.y*=0.2;
    p.xz*=0.91+(p.y*3.0);
    float d2d = sdCircle(p.xz, r);
    float dz = abs(p.y) - h;
    float k = 0.1;
    return (length(max(vec2(d2d, dz), 0.0)) - k)/8.0;
}
float sdVerticalCapsule(vec3 p, float r, float h) {
  float A = 0.04; 
  p.x+=sin(p.y*2.0+0.0*2.0)*A;
  p.z+=cos(p.y*4.0+0.0*2.0)*A-A;
  p.xz*=1.0+cos(p.y*8.0+0.0)*A;
  p.y+=2.5;
  p.y -= clamp( p.y, 0.0, h );
  return length( p ) - r;
}
float sdCutHollowSphere( vec3 p, float r, float h, float t )
{
  p.y=-p.y/1.125;
  p.xz*=1.0-cos(p.y*5.0-1.0)*0.2;
  float w = sqrt(r*r-h*h);
  vec2 q = vec2( length(p.xz), p.y );  
  return ((h*q.x<w*q.y) ? length(q-vec2(w,h)) : abs(length(q)-r) ) - t;
}
float opSmoothUnion( float d1, float d2, float k )
{
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h);
}

// Mushroom
float SDF(vec3 p, float time) {
  p.y-=1.0;
  float head = opSmoothUnion(sdCutHollowSphere(p,0.5,0.2,0.004), sdSphere(p,0.065, 0.55), 0.15); 
  float spore = sdCylinder(p, 0.3, 0.01);
  float body = sdVerticalCapsule(p, 0.04, 2.5);
  float fullhead = opSmoothUnion(head,spore,0.02);
  return opSmoothUnion(fullhead, body, 0.1);
}

vec3 calculateNormal(vec3 p, float time) {
    const float eps = 0.001;
    vec3 P=p;
    float d = SDF(p, time);
    vec3 n = vec3(
        SDF(p + vec3(eps, 0, 0), time) - SDF(p - vec3(eps, 0, 0), time),
        SDF(p + vec3(0, eps, 0), time) - SDF(p - vec3(0, eps, 0), time),
        SDF(p + vec3(0, 0, eps), time) - SDF(p - vec3(0, 0, eps), time)
    );
    return normalize(n);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    
    vec3 col = vec3(0);
    float time = iTime+pi/5.0;
    
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    
    // Camera setup
    vec3 ro = vec3(0, 0, -8.0);
    vec3 rd = normalize(vec3(uv*0.25, 1.0));

    // Rotation
    vec2 mouse = iMouse.xy/iResolution.xy+0.5;
    mouse.y=-mouse.y;
    if (mouse.y<0.0)mouse.x=-mouse.x;
    if (mouse.x<0.0)mouse.y=-mouse.y;
    float rotX = -(mouse.y - 0.5) * 3.14159;
    
    float rotY = (mouse.x - 0.5) * 6.28318;
    if (length(iMouse.xy)<50.0) {
        rotY=iTime+pi/5.0;
        rotX=pi/6.0;
    }
    mat3 rx = mat3(
        1.0, 0.0, 0.0,
        0.0, cos(rotX), -sin(rotX),
        0.0, sin(rotX), cos(rotX)
    );
    mat3 ry = mat3(
        cos(rotY), 0.0, sin(rotY),
        0.0, 1.0, 0.0,
        -sin(rotY), 0.0, cos(rotY)
    );
    mat3 rotation = rx * ry;
    ro = rotation * vec3(0, 0, -8.0);
    rd = rotation * normalize(vec3(uv*0.25, 1.0));
    
    // Raymarching
    float t = 0.0;
    vec3 p, P;
    bool hit = false;
    for(int i = 0; i < 768; i++) {
        p = ro + rd * t;
        float d = SDF(p, iTime*1.0);
        if(d < 0.001) {
            hit = true;
            break;
        }
        if(t > 20.0) break;
        t += d/2.0;
    }
    P = p;
    
    if(hit) {
        // Calculate normal for lighting
        vec3 normal = calculateNormal(P, time);
        
        // Lighting setup
        vec3 lightDir = normalize(vec3(0.5, 1.0, -0.5));
        float diff = max(dot(normal, lightDir), 0.0);
        float ambient = 0.3;
        float lighting = ambient + diff;
        
        // Calculate colors
        col.x *= 3.0*(1.5-length(p.y-2.5/10.0))*(1.75-length(p.xz));
        col.xz -= 1.0*abs(p.y*0.75);
        col = clamp((normal+1.0)/2.0, 0., 1.);
        
        // Apply lighting
        col *= lighting*0.5+0.5;
        
        // Add some specular highlights
        vec3 viewDir = normalize(ro - p);
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        col += vec3(spec * 0.5);
        
        // Animation
        vec3 n=normal;
        p*=10.0; n*=10.0;
        col*=vec3(
           (pow(sin(p.x),2.0) + pow(sin(n.z),2.0) + 1.0*pow(sin(n.y+iTime*3.0),2.0) )*0.5+0.5
        );
        
        // Result
        col = clamp(col,0.,1.);
        fragColor = vec4(col, 1.0);
        
    } else {
        fragColor = vec4(0,0,0,1);
    }
}`,

"Grid Experiment": `// Textured Audio-Reactive Tiles
// Author: GitHub Copilot for iniadewumi
// Created: 2025-05-03

#define MAX_STEPS 100
#define MIN_DIST 0.001
#define MAX_DIST 40.0

// Audio frequency bands
float freqs[4];

// Hash function for randomization
float hash(float n) {
    return fract(sin(n) * 43758.5453123);
}

// Box SDF with rounded edges
float udBox(vec3 p, vec3 b, float r) {
    return length(max(abs(p) - b, 0.0)) - r;
}

// Dynamic texture based on position and time
vec3 getDynamicTexture(vec2 p, float height, float freq) {
    // Sample basic texture
    vec2 uv = p * 0.5;
    vec3 tex = texture(iChannel1, uv).rgb;
    
    // Add some animated patterns
    float time = iTime * 0.5;
    vec2 q = p * 3.0;
    q += vec2(sin(time + p.y), cos(time + p.x)) * freq;
    
    // Create moving patterns
    float pattern = sin(q.x + time) * cos(q.y + time * 0.5);
    pattern += sin(length(p * 5.0) - time * 2.0) * 0.5;
    
    // Color palette function from the second shader
    vec3 color = 0.5 + 0.5 * cos(6.28318 * (height + vec3(0.0, 0.33, 0.67)));
    
    // Combine everything
    vec3 final = mix(tex, color, pattern * 0.5);
    final += vec3(0.2, 0.3, 0.4) * freq; // Add audio-reactive glow
    
    return final;
}

// Map function for four tiles
float map(vec3 pos, out vec3 tileColor) {
    float minDist = MAX_DIST;
    tileColor = vec3(0.0);
    
    // Grid positions for 2x2 layout
    vec2 positions[4] = vec2[4](
        vec2(-0.7, -0.7),  // Bottom left
        vec2( 0.7, -0.7),  // Bottom right
        vec2(-0.7,  0.7),  // Top left
        vec2( 0.7,  0.7)   // Top right
    );
    
    // Process each tile
    for(int i = 0; i < 4; i++) {
        float height = clamp(freqs[i] * 2.5, 0.5, 2.5);
        vec3 p = pos - vec3(positions[i].x, height * 0.5, positions[i].y);
        float d = udBox(p, vec3(0.3, height * 0.5, 0.3), 0.1);
        
        if(d < minDist) {
            minDist = d;
            // Get dynamic texture for this tile
            tileColor = getDynamicTexture(positions[i], height, freqs[i]);
        }
    }
    
    return minDist;
}

// Normal calculation
vec3 getNormal(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    vec3 dummy;
    return normalize(vec3(
        map(p + e.xyy, dummy) - map(p - e.xyy, dummy),
        map(p + e.yxy, dummy) - map(p - e.yxy, dummy),
        map(p + e.yyx, dummy) - map(p - e.yyx, dummy)
    ));
}

// Ray marching
float raymarch(vec3 ro, vec3 rd, out vec3 color) {
    float d = 0.0;
    color = vec3(0.0);
    
    for(int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * d;
        vec3 tileColor;
        float dist = map(p, tileColor);
        
        if(dist < MIN_DIST) {
            color = tileColor;
            return d;
        }
        if(d > MAX_DIST) break;
        
        d += dist;
    }
    
    return MAX_DIST;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Sample audio
    freqs[0] = texture(iChannel0, vec2(0.01, 0.25)).x;
    freqs[1] = texture(iChannel0, vec2(0.07, 0.25)).x;
    freqs[2] = texture(iChannel0, vec2(0.15, 0.25)).x;
    freqs[3] = texture(iChannel0, vec2(0.30, 0.25)).x;
    
    // Screen coordinates
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    
    // Camera setup
    float time = iTime * 0.5;
    vec3 ro = vec3(4.0 * cos(time), 3.0, 4.0 * sin(time));
    vec3 ta = vec3(0.0, 0.8, 0.0);
    
    // Camera matrix
    vec3 ww = normalize(ta - ro);
    vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
    vec3 vv = normalize(cross(uu, ww));
    vec3 rd = normalize(uv.x * uu + uv.y * vv + 1.5 * ww);
    
    // Raymarch and get color
    vec3 color;
    float d = raymarch(ro, rd, color);
    
    if(d < MAX_DIST) {
        vec3 pos = ro + rd * d;
        vec3 nor = getNormal(pos);
        
        // Lighting
        vec3 light = normalize(vec3(0.5, 0.7, -0.3));
        float diff = max(dot(nor, light), 0.0);
        float amb = 0.5 + 0.5 * nor.y;
        float spec = pow(max(dot(reflect(-light, nor), -rd), 0.0), 32.0);
        
        // Final color
        color *= diff * 0.7 + amb * 0.3;
        color += vec3(1.0) * spec * 0.5;
        
        // Add pulsing glow based on audio
        float audioIntensity = (freqs[0] + freqs[1] + freqs[2] + freqs[3]) * 0.25;
        color += color * audioIntensity * 0.3;
    } else {
        color = vec3(0.1); // Background color
    }
    
    // Add some atmosphere
    color = mix(color, vec3(0.1, 0.15, 0.2), smoothstep(0.0, 1.0, d/MAX_DIST));
    
    // Gamma correction
    color = pow(color, vec3(0.4545));
    
    fragColor = vec4(color, 1.0);
}`,
"Knodding Donkey": `//CC0 1.0 Universal https://creativecommons.org/publicdomain/zero/1.0/
//To the extent possible under law, Blackle Mori has waived all copyright and related or neighboring rights to this work.

//antialising
#define AA_SAMPLES 1

float t = 0.;
float ot = 0.;
#define ro(r) mat2(cos(r),-sin(r),sin(r),cos(r))

float linedist(vec2 p, vec2 a, vec2 b) {
  float k = dot(p-a,b-a)/dot(b-a,b-a);
  return distance(p,mix(a,b,clamp(k,0.,1.)));
}

float doodad(vec3 p, vec2 a, vec2 b, float s) {
  s/=2.;
  float wire = max(min(length(p.yz-a)-.04, length(p.yz-b)-.04),abs(p.x)-s-.04);
  return min(max(linedist(p.yz,a,b)-.05,abs(abs(p.x)-s)-.02),wire);
}

vec2 poop(vec2 a, vec2 b, float d1, float d3, float side) {
  float d2 = distance(a,b);
  float p = (d1*d1+d2*d2-d3*d3)/d2/2.;
  float o = side*sqrt(d1*d1-p*p);
  return a + mat4x2(-p,-o,o,-p,p,o,-o,p)*vec4(a,b)/d2;
}

float scene(vec3 p) {
  float dist = 1e4;
  vec2 D = ro(ot*7.)*vec2(.15,0);
  p.x-=0.025;
  {
  float side = 1.;
  vec2 M = vec2(-.4*side,0);
  vec2 a = poop(M,D,.4,.6,side);
  vec2 b = poop(M,D,.4,.6,-side);
  vec2 c = poop(M,a,.4,.5,side);
  vec2 d = poop(b,c,.35,.4,side);
  vec2 e = poop(b,d,.4,.6,side);
  
  dist = min(dist, doodad(p,D,a,.0));
  dist = min(dist, doodad(p,M,a,.1));
  dist = min(dist, doodad(p,D,b,.2));
  dist = min(dist, doodad(p,M,b,.3));
  dist = min(dist, doodad(p,b,d,.0));
  dist = min(dist, doodad(p,M,c,.0));
  dist = min(dist, doodad(p,c,d,.1));
  dist = min(dist, doodad(p,b,e,.1));
  dist = min(dist, doodad(p,c,a,.2));
  dist = min(dist, doodad(p,d,e,.2));
  }
  p.x+=0.05;
  {
  float side = -1.;
  vec2 M = vec2(-.4*side,0);
  vec2 a = poop(M,D,.4,.6,side);
  vec2 b = poop(M,D,.4,.6,-side);
  vec2 c = poop(M,a,.4,.5,side);
  vec2 d = poop(b,c,.35,.4,side);
  vec2 e = poop(b,d,.4,.6,side);
  
  dist = min(dist, doodad(p,D,a,.0));
  dist = min(dist, doodad(p,M,a,.1));
  dist = min(dist, doodad(p,D,b,.2));
  dist = min(dist, doodad(p,M,b,.3));
  dist = min(dist, doodad(p,b,d,.0));
  dist = min(dist, doodad(p,M,c,.0));
  dist = min(dist, doodad(p,c,d,.1));
  dist = min(dist, doodad(p,b,e,.1));
  dist = min(dist, doodad(p,c,a,.2));
  dist = min(dist, doodad(p,d,e,.2));
  }
  return dist;
}

vec3 norm(vec3 p) {
  mat3 k = mat3(p,p,p)-mat3(0.001);
  return normalize(scene(p)-vec3(scene(k[0]),scene(k[1]),scene(k[2])));
}
float bpm = 127.;

vec3 pixel_color(vec2 uv) {
  
  uv += texture(iChannel0,uv*2.).x*0.0025;
  
  float m = 2.*60./bpm;
  float rng = floor(m*iTime)/m;
  float w = iTime - rng;
  t =rng + mix(pow( w,3.),w,.8);
  ot =t ;
  t += fract(cos(rng)*456.)*3.;
  
  vec3 cam = normalize(vec3(1.8+cos(rng*45.)*.5,uv));
  vec3 init = vec3(-3,cos(rng*445.)*.3,-.2);
  
  float ry = sin(cos(rng*64.)*100.)*.3;
  cam.xz*=ro(ry);
  init.xz*=ro(ry);
  float rz = t*.5 + cos(rng*64.)*100.;
  cam.xy*=ro(rz);
  init.xy*=ro(rz);
  
  vec3 p = init;
  bool hit = false;
  bool trig = false;
  for (int i = 0; i < 50 && !hit; i++) {
    float dist = scene(p);
    hit = dist*dist < 1e-6;
    if (!trig) trig = dist<0.005;
    p += cam*dist;
  }
  float v = 1.-dot(uv,uv)*.5;
  vec3 n = norm(p);
  vec3 r = reflect(cam,n);
  float fact = dot(cam,r);
  vec2 grid = abs(asin(sin(uv*40.)));
  float g =smoothstep(1.52,1.58,max(grid.x,grid.y));
  float f = smoothstep(.8,.85,fact) + smoothstep(.4,.45,fact)*smoothstep(.5,1.,cos(uv.y*1000.));
  vec3 fragColor = min(vec3(1),hit ? vec3(f) : vec3(trig?1.:g))*.8;
  fragColor.xyz += texture(iChannel1,clamp(ro(ot)*(uv*6.+vec2(4.2,2))+.5,0.,1.)).xyz;
  fragColor*=v;
  return fragColor*fragColor;
}

vec2 weyl_2d(int n) {
    return fract(vec2(n*12664745, n*9560333)/exp2(24.));
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = (fragCoord-.5*iResolution.xy)/iResolution.y;
    fragColor = vec4(0);
    for (int i = 0; i < AA_SAMPLES; i++) {
        vec2 uv2 = uv + weyl_2d(i)/iResolution.y*1.25;
        fragColor += vec4(pixel_color(uv2), 1.);
    }
	fragColor.xyz = sqrt(fragColor.xyz/fragColor.w);
}
`
}

