export const SHADERS2 = {
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

    // Sample the buffer texture (iChannel1) for the smoothed FFT value.
    float smoothedFFTValue = texture(iChannel1, uv).x;

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
    float a = -texture(iChannel0, vec2(.5, .25)).x + sin(iTime) * .2 + .9;
    float s = sin(a), c = cos(a);
    _kifsRot *= mat3(c, -s, 0, s, c, 0, 0, 0, 1);
    _kifsRot *= mat3(1, 0, 0, 0, c, -s, 0, s, c);
    _kifsRot *= mat3(c, 0, s, 0, 1, 0, -s, 0, c);

    // Set up kifs offset
    _kifsOffset = .07 + texture(iChannel0, vec2(.1, .25)).x * .06;

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

    // Use the smoothed FFT value to influence the final color.  For example:
    float colorFactor = 0.5 + 0.5 * smoothedFFTValue;  // Remap smoothedFFTValue (which is 0-1)

    fragColor.rgb = mix(bg, vec3(1., .9, .7) * colorFactor, // Apply the color factor here.
        max(max(max(saturate(highlights + strokes.x), saturate(lightValue + strokes.y)) * fog,
            (edge + outline) * 2. + strokes.y), grid));

    // Gamma correction
    fragColor = pow(saturate(fragColor), vec4(1. / 2.2)) * step(abs(uv.y), 1.);
}
`,
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
}`
}

