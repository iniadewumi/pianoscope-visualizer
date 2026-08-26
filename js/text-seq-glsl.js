/**
 * Shared sequencer GLSL, injected by shadertoy-converter before mainImage()
 * when a shader defines SEQ_LINE (fitted sheet) or SEQ_COVER (legacy cover).
 *
 * Opt in:
 *   #define SEQ_LINE 0.29     Word Ladder — cap-height mapping on iTextSheet
 *   #define SEQ_COVER 1.8     Kick Afterglow — original TEXT_COVER on iChannel3
 *   #define SEQ_HAS_FIELD     reveal uses the shader's own fieldValue()
 *
 * Then sample with seqSample(uv, mid) -> vec3(sharp, soft, knockout).
 */
export const SEQ_GLSL = `
#ifndef SEQ_ROLL
#define SEQ_ROLL 1.12
#endif
#ifndef SEQ_STAGGER
#define SEQ_STAGGER 0.30
#endif
#ifndef SEQ_MARGIN
#define SEQ_MARGIN 0.90
#endif
#ifndef SEQ_ASPECT
#define SEQ_ASPECT 4.0
#endif
#ifndef SEQ_LEGACY_Y
#define SEQ_LEGACY_Y 0.54
#endif

bool seqIn01(vec2 t) {
    return t.x >= 0.0 && t.x <= 1.0 && t.y >= 0.0 && t.y <= 1.0;
}

vec2 seqCover() {
#ifdef SEQ_COVER
    float asp = SEQ_ASPECT;
    if (iChannelResolution[3].x > 1.5) {
        asp = iChannelResolution[3].x / max(iChannelResolution[3].y, 1.0);
    }
    return vec2(SEQ_COVER, SEQ_COVER / max(asp, 0.001));
#else
    float texAspect = iTextSheetRes.x / max(iTextSheetRes.y, 1.0);
    float coverY = SEQ_LINE / max(iTextLineH, 0.0001);
    float coverX = coverY * texAspect;
    float screenW = iResolution.x / max(iResolution.y, 1.0);
    vec2 ext = mix(iTextExtent, iTextExtentB, clamp(iTextMorph, 0.0, 1.0));
    float maxX = (screenW * SEQ_MARGIN) / max(ext.x, 0.001);
    float maxY = SEQ_MARGIN / max(ext.y, 0.001);
    float k = min(1.0, min(maxX / max(coverX, 0.0001), maxY / max(coverY, 0.0001)));
    return vec2(coverX * k, coverY * k);
#endif
}

vec2 seqToCoverUV(vec2 uv) {
    vec2 cover = seqCover();
    return vec2(uv.x / cover.x + 0.5, uv.y / cover.y + 0.5);
}

#ifdef SEQ_COVER
vec2 seqLegacyRel(vec2 uv) {
    vec2 tL = seqToCoverUV(uv);
    vec2 leg = max(iTextLegacyExtent, vec2(0.001));
    return (tL - vec2(0.5, SEQ_LEGACY_Y)) / leg;
}

vec2 seqToFittedUV(vec2 uv) {
    return seqLegacyRel(uv) * max(iTextExtent, vec2(0.001)) + 0.5;
}
#endif

vec2 seqToSheetUV(vec2 uv) {
#ifdef SEQ_COVER
    return seqToFittedUV(uv);
#else
    return seqToCoverUV(uv);
#endif
}

vec3 seqSampleSheet(vec2 uv) {
#ifdef SEQ_COVER
    vec2 tL = seqToCoverUV(uv);
    vec2 a = vec2(0.0);
    if (iChannelResolution[3].x > 1.5 && seqIn01(tL)) {
        a = texture(iChannel3, tL).rg;
    }
    vec2 tB = seqLegacyRel(uv) * max(iTextExtentB, vec2(0.001)) + 0.5;
    float b = 0.0;
    if (iTextSheetRes.x > 1.5 && seqIn01(tB)) {
        b = texture(iTextSheet, tB).b;
    }
    return vec3(a.r, a.g, b);
#else
    if (iTextSheetRes.x < 1.5) {
        return vec3(0.0);
    }
    vec2 t = seqToCoverUV(uv);
    if (!seqIn01(t)) {
        return vec3(0.0);
    }
    return texture(iTextSheet, t).rgb;
#endif
}

float seqWordPhase(vec2 uv) {
    if (iTextWordCount < 0.5) return 0.0;
    vec2 t = seqToSheetUV(uv);
    for (int i = 0; i < 8; i++) {
        if (float(i) >= iTextWordCount) break;
        vec4 r = iTextWords[i];
        if (t.x >= r.x && t.x <= r.z && t.y >= r.y && t.y <= r.w) {
            return float(i) / max(iTextWordCount - 1.0, 1.0);
        }
    }
    return 0.0;
}

float seqField(vec2 uv, float mid) {
#ifdef SEQ_HAS_FIELD
    return fieldValue(uv * 1.15, mid);
#else
    float n = fract(sin(dot(uv, vec2(127.1, 311.7))) * 43758.5453);
    return sin((uv.y * 9.0 + n * 2.4) * 3.14159265) * 0.5 + 0.5;
#endif
}

vec3 seqSample(vec2 uv, float mid) {
    float m = clamp(iTextMorph, 0.0, 1.0);
    if (m <= 0.0) {
#ifdef SEQ_COVER
        vec2 tL = seqToCoverUV(uv);
        vec2 a = vec2(0.0);
        if (iChannelResolution[3].x > 1.5 && seqIn01(tL)) {
            a = texture(iChannel3, tL).rg;
        }
        return vec3(a.r, a.g, 0.0);
#else
        vec3 rest = seqSampleSheet(uv);
        return vec3(rest.r, rest.g, 0.0);
#endif
    }

    vec2 cover = seqCover();
    float mode = iTextTransition;
    float ph = seqWordPhase(uv);
    float ms = clamp((m - ph * SEQ_STAGGER) / max(1.0 - SEQ_STAGGER, 0.001), 0.0, 1.0);

    if (mode < 1.5 || mode > 4.5) {
        float travel = cover.y * SEQ_ROLL;
        vec3 a = seqSampleSheet(uv - vec2(0.0, ms * travel));
        vec3 b = seqSampleSheet(uv + vec2(0.0, (1.0 - ms) * travel));
        return vec3(mix(a.r, b.b, ms), mix(a.g, b.b, ms), 0.0);
    }

    if (mode < 2.5) {
        vec3 s = seqSampleSheet(uv);
        float f = seqField(uv, mid);
        float w = 0.16;
        float thresh = mix(-w, 1.0 + w, ms);
        float sel = smoothstep(thresh + w, thresh - w, f);
        return vec3(mix(s.r, s.b, sel), mix(s.g, s.b, sel), 0.0);
    }

    if (mode < 3.5) {
        vec3 a = seqSampleSheet(uv / (1.0 + ms * 0.22));
        vec3 b = seqSampleSheet(uv / max(1.0 - (1.0 - ms) * 0.16, 0.5));
        float fade = 1.0 - ms;
        return vec3(a.r * fade + b.b * ms, a.g * fade + b.b * ms, 0.0);
    }

    vec3 s = seqSampleSheet(uv);
    float band = (ms * 1.5 - 0.25) * cover.y - cover.y * 0.5;
    float sel = smoothstep(band + cover.y * 0.06, band - cover.y * 0.06, uv.y);
    float neg = sel * (1.0 - abs(ms * 2.0 - 1.0));
    return vec3(mix(s.r, s.b, sel), mix(s.g, s.b, sel), neg);
}
`;
