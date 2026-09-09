/**
 * ShaderConverter - A utility for converting Shadertoy shaders to WebGL
 *
 * This module handles the conversion of Shadertoy-style fragment shaders to
 * standard WebGL shaders, and provides a modern UI for testing shaders.
 */

import { SHADERS } from './shaders.js';
import { SHADERS2 } from './shaders2.js';
import { TEST_SHADERS } from './test-shaders.js';
import { SEQ_GLSL } from './text-seq-glsl.js';

// Combine the shader objects (TEST_SHADERS first for picker / keyboard cycle)
const SAMPLE_SHADERS = { ...TEST_SHADERS, ...SHADERS, ...SHADERS2 };

window.ShaderConverter = (function () {
    "use strict";

    // Define WebGL preambles and main function structure
    const WEBGL_PREAMBLE = `
precision highp float;

// Version compatibility layer
#if __VERSION__ >= 300
    #define attribute in
    #define varying out
    #define texture2D texture
    #define textureCube texture
    #define texture2DLod textureLod
    #define textureCubeLod textureLod
    #define texture2DProj textureProj
    #define texture2DProjLod textureProjLod
#else
    #define in varying
    #define out varying
#endif

uniform vec3 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform float iFrame;
uniform float iSampleRate;
uniform vec4 iMouse;
uniform vec4 iDate;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
uniform vec3 iChannelResolution[4];

// Live source (camera / screen / clip). Available in every shader; iVideoActive is 0 when
// nothing is attached, and the sampler then holds a 1x1 black texel.
uniform sampler2D iVideo;
uniform float iVideoActive;
uniform vec2 iVideoResolution;
uniform float iFitMode;
uniform float iVideoMirror;
uniform float iVideoBlend;
uniform float iVideoBlendAmount;

// Maps a 0..1 screen coordinate onto the live source, honouring the Fit
// dropdown, and flips Y for WebGL texture orientation. Mirrors js/video-effects.js's
// videoUV()/videoInside()/videoAt()/videoLuma(), adapted to sample iVideo instead
// of iChannel1 so any shader can use them, not just the video-effect look.
vec2 videoUV(vec2 uv) {
    float canvasAspect = iResolution.x / max(iResolution.y, 1.0);
    float videoAspect = iVideoResolution.x / max(iVideoResolution.y, 1.0);

    vec2 scale = vec2(1.0);
    if (iFitMode < 0.5) {
        scale = canvasAspect > videoAspect
            ? vec2(1.0, videoAspect / canvasAspect)
            : vec2(canvasAspect / videoAspect, 1.0);
    } else if (iFitMode < 1.5) {
        scale = canvasAspect > videoAspect
            ? vec2(canvasAspect / videoAspect, 1.0)
            : vec2(1.0, videoAspect / canvasAspect);
    }

    vec2 v = (uv - 0.5) * scale + 0.5;
    v.y = 1.0 - v.y;

    if (iVideoMirror > 0.5) v.x = 1.0 - v.x;

    return v;
}

float videoInside(vec2 uv) {
    vec2 v = videoUV(uv);
    return step(0.0, v.x) * step(v.x, 1.0) * step(0.0, v.y) * step(v.y, 1.0);
}

vec3 videoAt(vec2 uv) {
    vec2 v = videoUV(uv);
    float inside = step(0.0, v.x) * step(v.x, 1.0) * step(0.0, v.y) * step(v.y, 1.0);
    return texture2D(iVideo, clamp(v, 0.0, 1.0)).rgb * inside;
}

float videoLuma(vec2 uv) {
    return dot(videoAt(uv), vec3(0.299, 0.587, 0.114));
}

// Display text.
//   iChannel3  - legacy 2048x512 sheet, R = sharp, G = soft. Unchanged contract.
//   iTextSheet - sheet fitted to the copy: R = slot A sharp, G = slot A soft,
//                B = slot B sharp (the incoming word during a swap).
// iTextMorph runs 0 -> 1 across a swap. iTextLineH (one line as a fraction of the
// fitted sheet) lets a shader hold cap height constant while copy length changes.
// iTextExtent is the glyph block as a fraction of that sheet, for frame fitting.
// iTextLegacyExtent is that same block as a fraction of the 2048x512 strip, so a
// SEQ_COVER shader can keep Kick Afterglow's original on-screen size.
// iTextWords carries per-word rects in sheet UV so type can be animated word by
// word rather than only as one block.
uniform sampler2D iTextSheet;
uniform vec2 iTextSheetRes;
uniform float iTextMorph;
uniform float iTextLineH;
uniform vec2 iTextExtent;
uniform vec2 iTextExtentB;
uniform vec2 iTextLegacyExtent;
uniform float iTextTransition;
uniform vec4 iTextWords[8];
uniform float iTextWordCount;
`;

    // Shadertoy's image pass renders opaque and discards whatever mainImage writes
    // to alpha. Our canvas is composited with the page, so alpha must be forced to
    // 1 here or shaders that leave it at 0 (common in golfed tanh-tonemap shaders)
    // vanish into the background.
    const WEBGL_MAIN = `
// Blends the live source under/over/keyed/multiplied into whatever the shader
// drew, independent of that shader's own iVideo use. A no-op when nothing is
// attached (iVideoActive < 0.5) or Blend is set to Off (iVideoBlend < 0.5).
vec4 pianoscopeBlendVideo(vec4 col, vec2 fragCoord) {
    if (iVideoActive < 0.5 || iVideoBlend < 0.5) return col;
    vec2 uv = fragCoord / iResolution.xy;
    vec3 v = videoAt(uv);
    float inside = videoInside(uv);
    float a = clamp(iVideoBlendAmount, 0.0, 1.0);
    vec3 s = col.rgb;
    vec3 outc = s;
    if (iVideoBlend < 1.5) {            // under: shader over video, shader luma as its alpha
        float sl = clamp(dot(s, vec3(0.299, 0.587, 0.114)) * 1.5, 0.0, 1.0);
        outc = mix(v, s, max(sl, 1.0 - a * inside));
        outc = mix(s, outc, a);
    } else if (iVideoBlend < 2.5) {     // over: video over shader, video luma as its alpha
        float vl = clamp(dot(v, vec3(0.299, 0.587, 0.114)) * 1.5, 0.0, 1.0);
        outc = mix(s, v, vl * a * inside);
    } else if (iVideoBlend < 3.5) {     // luma key: video shows where the shader is bright
        float sl = smoothstep(0.25, 0.75, dot(s, vec3(0.299, 0.587, 0.114)));
        outc = mix(s, v, sl * a * inside);
    } else {                            // multiply
        outc = mix(s, s * mix(vec3(1.0), v, inside), a);
    }
    return vec4(outc, col.a);
}

#if __VERSION__ >= 300
void main() {
    mainImage(fragColor, gl_FragCoord.xy);
    fragColor = pianoscopeBlendVideo(fragColor, gl_FragCoord.xy);
    fragColor.a = 1.0;
}
#else
void main() {
    vec4 fragColor;
    mainImage(fragColor, gl_FragCoord.xy);
    fragColor = pianoscopeBlendVideo(fragColor, gl_FragCoord.xy);
    gl_FragColor = vec4(fragColor.rgb, 1.0);
}
#endif
`;

    /**
     * Converts a Shadertoy fragment shader to WebGL format
     * @param {string} shadertoyCode - The Shadertoy shader code
     * @return {string} - The converted WebGL shader code
     */
    function convertShaderToyToWebGL(shadertoyCode) {
        if (!shadertoyCode) {
            throw new Error("No shader code provided");
        }

        // Check if code already contains a main function
        if (shadertoyCode.includes("void main()")) {
            // Already has a main function, could be a regular WebGL shader
            return shadertoyCode;
        }

        // Check if it's a Shadertoy shader (should have mainImage function)
        if (!shadertoyCode.includes("void mainImage(")) {
            throw new Error("Not a valid Shadertoy shader (missing mainImage function)");
        }

        // Shared sequencer helpers. A shader opts in with #define SEQ_LINE or
        // #define SEQ_COVER; the GLSL is injected before mainImage so the
        // shader's own fieldValue (if any) is already declared.
        if (/#define\s+SEQ_(LINE|COVER)\b/.test(shadertoyCode) &&
            !/vec3\s+seqSample\s*\(/.test(shadertoyCode)) {
            const idx = shadertoyCode.search(/void\s+mainImage\s*\(/);
            if (idx >= 0) {
                shadertoyCode = shadertoyCode.slice(0, idx) + SEQ_GLSL + '\n' + shadertoyCode.slice(idx);
            }
        }

        // Replace texture with texture2D which is needed for WebGL 1.0
        let convertedCode = shadertoyCode.replace(/\btexture\(/g, 'texture2D(');

        // iResolution is vec3 in preamble (Shadertoy). vec3(vec2, float) ports used vec3(iResolution, z).
        convertedCode = convertedCode
            .replace(/vec3\s*\(\s*iResolution\s*,\s*1\.0\s*\)/g, 'iResolution')
            .replace(/vec3\s*\(\s*iResolution\s*,\s*0\.0\s*\)/g, 'vec3(iResolution.xy, 0.0)');

        // Combine the shader parts
        return WEBGL_PREAMBLE + convertedCode + WEBGL_MAIN;
    }

    /**
     * Creates a shader test UI with sample shaders and editor
     * @param {HTMLElement} container - Container to append the UI to
     * @param {Function} onShaderApply - Callback when a shader is applied
     * @return {Object} - The UI elements
     */
    function createShaderTestUI(container, onShaderApply) {

        // Create UI elements
        const uiContainer = document.createElement('div');
        uiContainer.className = 'shader-test-ui';

        // Create header
        const header = document.createElement('div');
        header.className = 'shader-editor-header';
        const title = document.createElement('h3');
        title.textContent = 'Shader Editor';
        const closeBtn = document.createElement('button');
        closeBtn.type = 'button';
        closeBtn.className = 'shader-editor-close';
        closeBtn.title = 'Close editor';
        closeBtn.setAttribute('aria-label', 'Close editor');
        closeBtn.innerHTML = '<i class="fas fa-times"></i>';
        header.appendChild(title);
        header.appendChild(closeBtn);
        uiContainer.appendChild(header);

        // Create tab system
        const tabContainer = document.createElement('div');
        tabContainer.className = 'tab-container';
        const shaderTabBtn = document.createElement('button');
        shaderTabBtn.type = 'button';
        shaderTabBtn.className = 'tab-button active';
        shaderTabBtn.dataset.tab = 'shader';
        shaderTabBtn.textContent = 'Shader';
        const videoTabBtn = document.createElement('button');
        videoTabBtn.type = 'button';
        videoTabBtn.className = 'tab-button';
        videoTabBtn.dataset.tab = 'video';
        videoTabBtn.textContent = 'Media';
        tabContainer.appendChild(shaderTabBtn);
        tabContainer.appendChild(videoTabBtn);
        uiContainer.appendChild(tabContainer);

        // Create tab content containers
        const shaderTabContent = document.createElement('div');
        shaderTabContent.className = 'tab-content';
        shaderTabContent.id = 'shader-tab';

        const videoTabContent = document.createElement('div');
        videoTabContent.className = 'tab-content hidden';
        videoTabContent.id = 'video-tab';

        // --- Elements to be placed inside shaderTabContent ---
        const controls = document.createElement('div');
        controls.className = 'shader-editor-controls';
        const applyButton = document.createElement('button');
        applyButton.type = 'button';
        applyButton.innerHTML = '<i class="fas fa-check"></i><span>Apply Shader</span>';
        applyButton.className = 'shader-editor-button apply-button';
        const resetButton = document.createElement('button');
        resetButton.type = 'button';
        resetButton.innerHTML = '<i class="fas fa-undo"></i><span>Reset</span>';
        resetButton.className = 'shader-editor-button reset-button';
        controls.appendChild(applyButton);
        controls.appendChild(resetButton);

        const dropdownContainer = document.createElement('div');
        dropdownContainer.className = 'custom-dropdown shader-search';

        const searchIcon = document.createElement('i');
        searchIcon.className = 'fas fa-search';
        searchIcon.setAttribute('aria-hidden', 'true');

        // Create the input field that will also act as the dropdown trigger
        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.className = 'dropdown-input';
        searchInput.placeholder = 'Search shaders...';
        searchInput.setAttribute('aria-label', 'Search shaders');

        // Create the options container
        const optionsContainer = document.createElement('div');
        optionsContainer.className = 'dropdown-options';

        // Track the currently selected option
        let selectedOption = '';

        // Populate options from SAMPLE_SHADERS
        for (const name of Object.keys(SAMPLE_SHADERS)) {
            const option = document.createElement('div');
            option.className = 'dropdown-option';
            option.textContent = name;
            option.dataset.value = name;

            option.addEventListener('click', () => {
                selectedOption = name;
                searchInput.value = name;
                optionsContainer.style.display = 'none';

                // Handle shader loading
                shaderTextarea.value = SAMPLE_SHADERS[name];
                showMessage(`Loaded shader: ${name}`, 'info');
            });

            optionsContainer.appendChild(option);
        }

        // Filter options based on input
        searchInput.addEventListener('input', () => {
            const query = searchInput.value.toLowerCase();
            const options = optionsContainer.children;
            let hasVisibleOptions = false;

            Array.from(options).forEach(option => {
                const matches = option.textContent.toLowerCase().includes(query);
                option.style.display = matches ? 'block' : 'none';
                if (matches) hasVisibleOptions = true;
            });

            optionsContainer.style.display = hasVisibleOptions ? 'block' : 'none';
        });

        // Show options when input is focused
        searchInput.addEventListener('focus', () => {
            optionsContainer.style.display = 'block';
        });

        // Handle clicks outside the dropdown
        document.addEventListener('click', (e) => {
            if (!dropdownContainer.contains(e.target)) {
                optionsContainer.style.display = 'none';
            }
        });

        // Prevent hiding options when clicking inside the dropdown
        dropdownContainer.addEventListener('click', (e) => {
            e.stopPropagation();
        });

        // Assemble the dropdown
        const displayTextPanel = document.createElement('div');
        displayTextPanel.className = 'display-text-panel ui-section';
        const displayTextLabel = document.createElement('label');
        displayTextLabel.className = 'display-text-label';
        displayTextLabel.htmlFor = 'display-text-input';
        displayTextLabel.textContent = 'Display text';
        const displayTextRow = document.createElement('div');
        displayTextRow.className = 'display-text-row';
        const displayTextInput = document.createElement('textarea');
        displayTextInput.id = 'display-text-input';
        displayTextInput.className = 'display-text-input';
        displayTextInput.rows = 2;
        displayTextInput.placeholder = 'LET THE MUSIC\nDO THE TALKING';
        displayTextInput.autocomplete = 'off';
        displayTextInput.spellcheck = false;
        const displayTextActions = document.createElement('div');
        displayTextActions.className = 'display-text-actions';
        const displayTextApply = document.createElement('button');
        displayTextApply.type = 'button';
        displayTextApply.className = 'shader-editor-button apply-button display-text-apply';
        displayTextApply.innerHTML = '<i class="fas fa-check"></i><span>Apply</span>';
        const displayTextClear = document.createElement('button');
        displayTextClear.type = 'button';
        displayTextClear.className = 'shader-editor-button reset-button display-text-clear';
        displayTextClear.innerHTML = '<i class="fas fa-times"></i><span>Clear</span>';
        displayTextClear.title = 'Remove display text from the wall';
        displayTextActions.appendChild(displayTextApply);
        displayTextActions.appendChild(displayTextClear);
        displayTextRow.appendChild(displayTextInput);
        displayTextRow.appendChild(displayTextActions);
        displayTextPanel.appendChild(displayTextLabel);
        displayTextPanel.appendChild(displayTextRow);
        const displayTextMeta = document.createElement('div');
        displayTextMeta.className = 'display-text-meta';
        const displayStyleSelect = document.createElement('select');
        displayStyleSelect.id = 'display-text-style';
        displayStyleSelect.className = 'display-text-font';
        displayStyleSelect.title = 'Text style';
        displayStyleSelect.setAttribute('aria-label', 'Text style');
        const styles = (window.PianoscopeDisplayText && window.PianoscopeDisplayText.STYLES) || [];
        for (let i = 0; i < styles.length; i++) {
            const opt = document.createElement('option');
            opt.value = styles[i].id;
            opt.textContent = styles[i].name;
            displayStyleSelect.appendChild(opt);
        }
        const displayFontSelect = document.createElement('select');
        displayFontSelect.id = 'display-text-font';
        displayFontSelect.className = 'display-text-font';
        displayFontSelect.title = 'Overlay font';
        displayFontSelect.setAttribute('aria-label', 'Overlay font');
        const fonts = (window.PianoscopeDisplayText && window.PianoscopeDisplayText.FONTS) || [];
        for (let i = 0; i < fonts.length; i++) {
            // meta:true faces are caption cuts a style reaches for on its own
            // (the band's mono data line). They are not headline faces and have
            // no business in the operator's dropdown.
            if (fonts[i].meta) continue;
            const opt = document.createElement('option');
            opt.value = fonts[i].id;
            opt.textContent = fonts[i].name;
            displayFontSelect.appendChild(opt);
        }
        // The band style carries a second slot under the headline - artist,
        // tempo, room. Only that style has one, so the field only appears for it.
        const displayMetaRow = document.createElement('div');
        displayMetaRow.className = 'display-text-meta display-text-meta-row';
        const displayMetaInput = document.createElement('input');
        displayMetaInput.type = 'text';
        displayMetaInput.id = 'display-text-meta-line';
        displayMetaInput.className = 'display-text-font';
        displayMetaInput.placeholder = 'PIANOSCOPE \u00b7 124 BPM \u00b7 LISTENING ROOM';
        displayMetaInput.title = 'Data line under the band. Leave empty for the band alone.';
        displayMetaInput.setAttribute('aria-label', 'Band data line');
        displayMetaInput.autocomplete = 'off';
        displayMetaInput.spellcheck = false;
        displayMetaRow.appendChild(displayMetaInput);

        // A style with a non-null `font` pins its own typeface, so the font
        // dropdown is moot while it's selected - disable it rather than let it
        // silently do nothing.
        function syncFontSelectForStyle() {
            const allStyles = (window.PianoscopeDisplayText && window.PianoscopeDisplayText.STYLES) || [];
            const style = allStyles.find(function (s) { return s.id === displayStyleSelect.value; });
            const pinned = !!(style && style.font);
            displayFontSelect.disabled = pinned;
            displayFontSelect.title = pinned
                ? 'This style pins its own typeface - switch to Plain to pick a font'
                : 'Overlay font';
            displayMetaRow.hidden = !(style && style.meta);
        }
        const overlayToggle = document.createElement('label');
        overlayToggle.className = 'display-text-overlay-toggle';
        const overlayCheck = document.createElement('input');
        overlayCheck.type = 'checkbox';
        overlayCheck.id = 'display-text-overlay-enabled';
        overlayToggle.appendChild(overlayCheck);
        overlayToggle.appendChild(document.createTextNode('JS overlay'));
        displayTextMeta.appendChild(displayStyleSelect);
        displayTextMeta.appendChild(displayFontSelect);
        displayTextMeta.appendChild(overlayToggle);
        displayTextPanel.appendChild(displayTextMeta);
        displayTextPanel.appendChild(displayMetaRow);

        // ---- transition picker ----
        const transitionRow = document.createElement('div');
        transitionRow.className = 'display-text-meta';
        const transitionSelect = document.createElement('select');
        transitionSelect.id = 'display-text-transition';
        transitionSelect.className = 'display-text-font';
        transitionSelect.title = 'Transition used for every text change, including sequencer playback';
        transitionSelect.setAttribute('aria-label', 'Text transition');
        const transitions = (window.PianoscopeDisplayText && window.PianoscopeDisplayText.TRANSITIONS) || [];
        for (let i = 0; i < transitions.length; i++) {
            const opt = document.createElement('option');
            opt.value = transitions[i].id;
            opt.textContent = transitions[i].name;
            transitionSelect.appendChild(opt);
        }
        const previewBtn = document.createElement('button');
        previewBtn.type = 'button';
        previewBtn.className = 'shader-editor-button seq-btn';
        previewBtn.innerHTML = '<i class="fas fa-eye"></i><span>Preview</span>';
        previewBtn.title = 'Re-apply the current text using the selected transition';
        transitionRow.appendChild(transitionSelect);
        transitionRow.appendChild(previewBtn);
        const displayTextExtras = document.createElement('div');
        displayTextExtras.className = 'display-text-extras ui-section';
        const extrasLabel = document.createElement('label');
        extrasLabel.className = 'display-text-label';
        extrasLabel.textContent = 'Transition';
        extrasLabel.htmlFor = 'display-text-transition';
        displayTextExtras.appendChild(extrasLabel);
        displayTextExtras.appendChild(transitionRow);

        // ---- sequencer transport ----
        const seqSection = document.createElement('div');
        seqSection.className = 'seq-section';
        const seqLabel = document.createElement('label');
        seqLabel.className = 'display-text-label';
        seqLabel.textContent = 'Sequence';
        seqSection.appendChild(seqLabel);

        const seqRow = document.createElement('div');
        seqRow.className = 'display-text-meta';
        const seqSelect = document.createElement('select');
        seqSelect.id = 'seq-name';
        seqSelect.className = 'display-text-font';
        seqSelect.setAttribute('aria-label', 'Sequence');
        const seqNames = (window.PianoscopeTextSequencer && window.PianoscopeTextSequencer.names()) || [];
        for (let i = 0; i < seqNames.length; i++) {
            const opt = document.createElement('option');
            opt.value = seqNames[i];
            opt.textContent = seqNames[i];
            seqSelect.appendChild(opt);
        }
        const artistInput = document.createElement('input');
        artistInput.type = 'text';
        artistInput.id = 'seq-artist';
        artistInput.className = 'display-text-font';
        artistInput.placeholder = 'Artist name';
        artistInput.autocomplete = 'off';
        artistInput.setAttribute('aria-label', 'Artist name');
        seqRow.appendChild(seqSelect);
        seqRow.appendChild(artistInput);
        seqSection.appendChild(seqRow);

        const transportRow = document.createElement('div');
        transportRow.className = 'seq-transport';
        const seqBtns = document.createElement('div');
        seqBtns.className = 'seq-btns';
        const playBtn = document.createElement('button');
        playBtn.type = 'button';
        playBtn.className = 'shader-editor-button seq-btn';
        playBtn.innerHTML = '<i class="fas fa-play"></i><span>Play</span>';
        playBtn.title = 'Run the sequence (P)';
        const stepBtn = document.createElement('button');
        stepBtn.type = 'button';
        stepBtn.className = 'shader-editor-button seq-btn';
        stepBtn.innerHTML = '<i class="fas fa-forward"></i><span>Step</span>';
        stepBtn.title = 'Advance one state (Space)';
        const tapBtn = document.createElement('button');
        tapBtn.type = 'button';
        tapBtn.className = 'shader-editor-button seq-btn';
        tapBtn.innerHTML = '<i class="fas fa-drum"></i><span>Tap</span>';
        tapBtn.title = 'Tap four beats to lock tempo (T)';
        seqBtns.appendChild(playBtn);
        seqBtns.appendChild(stepBtn);
        seqBtns.appendChild(tapBtn);

        const seqFields = document.createElement('div');
        seqFields.className = 'seq-fields';
        const bpmInput = document.createElement('input');
        bpmInput.type = 'number';
        bpmInput.min = '20';
        bpmInput.max = '300';
        bpmInput.className = 'seq-num';
        bpmInput.title = 'BPM';
        bpmInput.setAttribute('aria-label', 'BPM');
        const barsInput = document.createElement('input');
        barsInput.type = 'number';
        barsInput.min = '1';
        barsInput.max = '32';
        barsInput.className = 'seq-num';
        barsInput.title = 'Bars per state';
        barsInput.setAttribute('aria-label', 'Bars per state');

        function seqField(input, labelText) {
            const wrap = document.createElement('label');
            wrap.className = 'seq-field';
            const cap = document.createElement('span');
            cap.textContent = labelText;
            wrap.appendChild(cap);
            wrap.appendChild(input);
            return wrap;
        }
        seqFields.appendChild(seqField(bpmInput, 'BPM'));
        seqFields.appendChild(seqField(barsInput, 'Bars'));

        transportRow.appendChild(seqBtns);
        transportRow.appendChild(seqFields);
        seqSection.appendChild(transportRow);

        const seqReadout = document.createElement('div');
        seqReadout.className = 'seq-readout';
        seqSection.appendChild(seqReadout);

        const displayTextHelp = document.createElement('details');
        displayTextHelp.className = 'display-text-help';
        const helpSummary = document.createElement('summary');
        helpSummary.textContent = 'Shortcuts & notes';
        const helpBody = document.createElement('p');
        helpBody.textContent = 'Pipe or newline stacks a line. Space steps, P plays, T taps tempo. The transition above applies to every state. Word Ladder and Kick Afterglow honour transitions; Kente Loom always hard-cuts.';
        displayTextHelp.appendChild(helpSummary);
        displayTextHelp.appendChild(helpBody);
        seqSection.appendChild(displayTextHelp);
        displayTextExtras.appendChild(seqSection);
        if (window.PianoscopeDisplayText) {
            displayTextInput.value = window.PianoscopeDisplayText.toDisplayValue(
                window.PianoscopeDisplayText.getText()
            );
            displayStyleSelect.value = window.PianoscopeDisplayText.getStyle();
            displayFontSelect.value = window.PianoscopeDisplayText.getFont();
            displayMetaInput.value = window.PianoscopeDisplayText.getMeta();
            syncFontSelectForStyle();
            overlayCheck.checked = window.PianoscopeDisplayText.isOverlayEnabled();
            window.PianoscopeDisplayText.onChange(function (text) {
                if (document.activeElement !== displayTextInput) {
                    displayTextInput.value = window.PianoscopeDisplayText.toDisplayValue(text);
                }
                syncClearEnabled();
            });
        }
        dropdownContainer.appendChild(searchIcon);
        dropdownContainer.appendChild(searchInput);
        dropdownContainer.appendChild(optionsContainer);

        const shaderSection = document.createElement('div');
        shaderSection.className = 'ui-section shader-code-section';
        const shaderSectionLabel = document.createElement('label');
        shaderSectionLabel.className = 'display-text-label';
        shaderSectionLabel.textContent = 'Shader';
        shaderSection.appendChild(shaderSectionLabel);
        shaderSection.appendChild(dropdownContainer);
        shaderSection.appendChild(controls);

        const shaderTextarea = document.createElement('textarea');
        shaderTextarea.className = 'shader-editor-textarea';
        shaderTextarea.placeholder = 'Paste Shadertoy shader code here...';
        shaderTextarea.setAttribute('aria-label', 'Shader source');
        // Use a consistent default shader name, handle potential missing key
        const defaultShaderName = "Pianoscope Text";
        const defaultShader = SAMPLE_SHADERS[defaultShaderName] || Object.values(SAMPLE_SHADERS)[0] || ''; // Added fallback
        shaderTextarea.value = defaultShader;

        const errorDisplay = document.createElement('div');
        errorDisplay.className = 'error-display';
        shaderSection.appendChild(shaderTextarea);
        shaderSection.appendChild(errorDisplay);

        shaderTabContent.appendChild(displayTextExtras);
        shaderTabContent.appendChild(shaderSection);

        // Media tab is populated by VideoController.mountUI() once the catalog
        // has been probed, so the markup lives with the code that drives it.

        // Add the content divs to the main UI container
        uiContainer.appendChild(displayTextPanel);
        uiContainer.appendChild(shaderTabContent);
        uiContainer.appendChild(videoTabContent);

        // Create toggle button — hidden while the sidebar is open so it cannot
        // overlay the header or the video toolbar.
        const toggleButton = document.createElement('button');
        toggleButton.type = 'button';
        toggleButton.innerHTML = '<i class="fas fa-code"></i>';
        toggleButton.className = 'toggle-shader-ui';
        toggleButton.title = 'Open Shader Editor';
        toggleButton.setAttribute('aria-label', 'Open Shader Editor');
        toggleButton.setAttribute('aria-expanded', 'false');

        // Add the toggle button and the main UI container to the document body (or specified container)
        container.appendChild(toggleButton);
        container.appendChild(uiContainer);

        // --- Tab Switching Functionality ---
        const tabButtons = uiContainer.querySelectorAll('.tab-button');
        const tabContents = uiContainer.querySelectorAll('.tab-content');
        const EDITOR_SESSION_KEY = 'pianoscope.editorSession';

        function persistEditorSession() {
            try {
                const active = uiContainer.querySelector('.tab-button.active');
                sessionStorage.setItem(EDITOR_SESSION_KEY, JSON.stringify({
                    open: uiContainer.classList.contains('open'),
                    tab: (active && active.dataset.tab) || 'shader'
                }));
            } catch (e) {}
        }

        function setActiveTab(tab) {
            tabButtons.forEach((btn) => {
                btn.classList.toggle('active', btn.dataset.tab === tab);
            });
            tabContents.forEach((content) => {
                content.classList.toggle('hidden', content.id !== tab + '-tab');
            });
            uiContainer.classList.toggle('is-video-tab', tab === 'video');
            persistEditorSession();
        }

        tabButtons.forEach(button => {
            button.addEventListener('click', () => setActiveTab(button.dataset.tab));
        });

        // Function to show status messages with animation
        function showMessage(message, type = 'success') {
            // Ensure errorDisplay exists before manipulating it
            if (!errorDisplay) return;

            errorDisplay.textContent = message;

            // Set color based on message type
            if (type === 'error') {
                errorDisplay.style.borderLeftColor = '#E83A5F'; errorDisplay.style.color = '#FF4D6D';
            } else if (type === 'success') {
                errorDisplay.style.borderLeftColor = '#4CAF50'; errorDisplay.style.color = '#8BC34A';
            } else if (type === 'info') {
                errorDisplay.style.borderLeftColor = '#2196F3'; errorDisplay.style.color = '#64B5F6';
            }

            // Animate in
            errorDisplay.style.display = 'block';
            errorDisplay.style.opacity = '0';
            errorDisplay.style.transform = 'translateY(-10px)';
            // Use requestAnimationFrame for smoother animation start
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    errorDisplay.style.opacity = '1'; errorDisplay.style.transform = 'translateY(0)';
                });
            });


            // Auto hide after delay
            setTimeout(() => {
                errorDisplay.style.opacity = '0'; errorDisplay.style.transform = 'translateY(10px)';
                setTimeout(() => { errorDisplay.style.display = 'none'; }, 300); // Wait for fade out transition
            }, 3000);
        }

        function syncClearEnabled() {
            const hasText = !!(window.PianoscopeDisplayText && window.PianoscopeDisplayText.getText());
            displayTextClear.disabled = !hasText;
        }

        function commitDisplayText() {
            if (!window.PianoscopeDisplayText) return;
            const next = window.PianoscopeDisplayText.applyText(displayTextInput.value);
            displayTextInput.value = window.PianoscopeDisplayText.toDisplayValue(next);
            syncClearEnabled();
            showMessage(next ? ('Display text: ' + next.replace(/\|/g, ' / ')) : 'Display text cleared', 'success');
        }

        function clearDisplayText() {
            if (window.PianoscopeTextSequencer) {
                const st = window.PianoscopeTextSequencer.status();
                if (st && st.running) window.PianoscopeTextSequencer.stop();
            }
            if (!window.PianoscopeDisplayText) return;
            window.PianoscopeDisplayText.applyText('', { transition: 'cut' });
            displayTextInput.value = '';
            syncClearEnabled();
            showMessage('Display text cleared', 'success');
        }

        displayTextApply.addEventListener('click', commitDisplayText);
        displayTextClear.addEventListener('click', clearDisplayText);
        syncClearEnabled();
        displayTextInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                commitDisplayText();
            }
        });
        displayFontSelect.addEventListener('change', () => {
            if (!window.PianoscopeDisplayText) return;
            window.PianoscopeDisplayText.setFont(displayFontSelect.value);
            const label = displayFontSelect.options[displayFontSelect.selectedIndex].text;
            showMessage('Font: ' + label, 'success');
        });
        displayStyleSelect.addEventListener('change', () => {
            if (!window.PianoscopeDisplayText) return;
            window.PianoscopeDisplayText.setStyle(displayStyleSelect.value);
            syncFontSelectForStyle();
            const label = displayStyleSelect.options[displayStyleSelect.selectedIndex].text;
            showMessage('Style: ' + label, 'success');
        });
        // Repaint as it is typed - the band's data line is the kind of thing an
        // operator adjusts against the frame, not blind and then applied.
        displayMetaInput.addEventListener('input', () => {
            if (!window.PianoscopeDisplayText) return;
            window.PianoscopeDisplayText.setMeta(displayMetaInput.value);
        });
        const SEQ = window.PianoscopeTextSequencer || null;

        if (window.PianoscopeDisplayText) {
            transitionSelect.value = window.PianoscopeDisplayText.getTransition();
        }
        // A cut has no duration, so there is nothing for Preview to play. Left
        // enabled it reads as a dead button - the click genuinely does nothing.
        function syncPreviewEnabled() {
            const isCut = transitionSelect.value === 'cut';
            previewBtn.disabled = isCut;
            previewBtn.title = isCut
                ? 'A cut is instant - pick a timed transition to preview it'
                : 'Re-apply the current text using the selected transition';
        }
        transitionSelect.addEventListener('change', () => {
            if (!window.PianoscopeDisplayText) return;
            window.PianoscopeDisplayText.setTransition(transitionSelect.value);
            syncPreviewEnabled();
            showMessage('Transition: ' + transitionSelect.options[transitionSelect.selectedIndex].text, 'success');
        });
        syncPreviewEnabled();
        previewBtn.addEventListener('click', () => {
            const d = window.PianoscopeDisplayText;
            if (!d) return;
            // Re-applying the same string DOES play the transition - both slots just
            // hold the same words. The old bounce through a blank state defeated it,
            // because applyText hard-cuts whenever the outgoing text is empty, and it
            // briefly persisted an empty string to storage.
            d.applyText(d.getText(), { transition: transitionSelect.value });
        });

        function renderSeqStatus(st) {
            if (!st) return;
            if (seqSelect.value !== st.sequence) seqSelect.value = st.sequence;
            if (document.activeElement !== bpmInput) bpmInput.value = st.bpm;
            if (document.activeElement !== barsInput) barsInput.value = st.bars;
            if (document.activeElement !== artistInput && st.artist !== '[ARTIST]') {
                artistInput.value = st.artist;
            }
            playBtn.innerHTML = st.running
                ? '<i class="fas fa-pause"></i><span>Pause</span>'
                : '<i class="fas fa-play"></i><span>Play</span>';
            playBtn.classList.toggle('seq-live', st.running);
            seqReadout.textContent =
                'State ' + (st.index + 1) + ' of ' + st.total +
                '  ·  ' + st.bpm + ' bpm  ·  ' + st.bars + ' bar' + (st.bars === 1 ? '' : 's') +
                '  ·  ' + (st.stepMs / 1000).toFixed(1) + 's per state';
        }

        if (SEQ) {
            SEQ.onChange(renderSeqStatus);
            seqSelect.addEventListener('change', () => { SEQ.setSequence(seqSelect.value); });
            artistInput.addEventListener('input', () => { SEQ.setArtist(artistInput.value); });
            playBtn.addEventListener('click', () => { SEQ.toggle(); });
            stepBtn.addEventListener('click', () => { SEQ.step(1); });
            tapBtn.addEventListener('click', () => {
                const r = SEQ.tap();
                showMessage(r.taps < 2 ? 'Keep tapping…' : 'Tempo: ' + r.bpm + ' bpm', 'info');
            });
            bpmInput.addEventListener('change', () => { SEQ.setBpm(parseInt(bpmInput.value, 10)); });
            barsInput.addEventListener('change', () => { SEQ.setBars(parseInt(barsInput.value, 10)); });
            SEQ.init();
            renderSeqStatus(SEQ.status());
        } else {
            seqReadout.textContent = 'Sequencer not loaded.';
        }

        overlayCheck.addEventListener('change', () => {
            if (!window.PianoscopeDisplayText) return;
            window.PianoscopeDisplayText.setOverlayEnabled(overlayCheck.checked);
            showMessage(overlayCheck.checked ? 'JS overlay on — shader will not chrome the type' : 'Shader chromes the JS font', 'info');
        });

        function setEditorOpen(open) {
            uiContainer.classList.toggle('open', open);
            toggleButton.classList.toggle('active', open);
            toggleButton.hidden = open;
            toggleButton.setAttribute('aria-expanded', open ? 'true' : 'false');
            toggleButton.title = open ? 'Close Shader Editor' : 'Open Shader Editor';
            persistEditorSession();
        }

        toggleButton.addEventListener('click', () => {
            setEditorOpen(!uiContainer.classList.contains('open'));
        });
        closeBtn.addEventListener('click', () => setEditorOpen(false));
        document.addEventListener('keydown', (e) => {
            if (e.key !== 'Escape' || !uiContainer.classList.contains('open')) return;
            const active = document.activeElement;
            if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.tagName === 'SELECT')) {
                active.blur();
                return;
            }
            setEditorOpen(false);
        });

        // Add event listener for apply button
        applyButton.addEventListener('click', () => {
            try {
                const shadertoyCode = shaderTextarea.value;
                const convertedShader = convertShaderToyToWebGL(shadertoyCode);
                applyButton.style.transform = 'scale(0.95)';
                setTimeout(() => { applyButton.style.transform = ''; }, 150);

                if (convertedShader && onShaderApply) {
                    const success = onShaderApply(convertedShader);
                    showMessage(success ? 'Shader applied successfully!' : 'Error applying shader', success ? 'success' : 'error');
                }
            } catch (error) {
                showMessage('Error: ' + error.message, 'error');
                console.error('Shader error:', error);
            }
        });

        // Add event listener for reset button
        resetButton.addEventListener('click', () => {
            resetButton.style.transform = 'scale(0.95)';
            setTimeout(() => { resetButton.style.transform = ''; }, 150);

            let resetSuccess = false;
            // Ensure window.visualizer and resetToDefault exist before calling
            if (typeof window.visualizer !== 'undefined' && window.visualizer && typeof window.visualizer.resetToDefault === 'function') {
                resetSuccess = window.visualizer.resetToDefault();
            }

            showMessage(resetSuccess ? 'Reset to default shader' : 'Error resetting shader', resetSuccess ? 'success' : 'error');

            // Update textarea content after attempting reset
            // Check if getDefaultFragmentShaderSource exists, otherwise use fallback
            let defaultFragSource = '';
            if (typeof window.visualizer !== 'undefined' && window.visualizer && typeof window.visualizer.getDefaultFragmentShaderSource === 'function') {
                defaultFragSource = window.visualizer.getDefaultFragmentShaderSource();
            }
            // If function doesn't exist or returns empty, use the hardcoded fallback
            if (!defaultFragSource) {
                defaultFragSource = SAMPLE_SHADERS[defaultShaderName] || Object.values(SAMPLE_SHADERS)[0] || '';
            }
            shaderTextarea.value = defaultFragSource;

        });

        // Add keyboard shortcut support for textarea
        shaderTextarea.addEventListener('keydown', (e) => {
            // Use `code` property for modern browsers, fallback to `key` or `keyCode`
            const key = e.code || e.key;
            if (e.ctrlKey && (key === 'Enter' || key === 'NumpadEnter')) {
                e.preventDefault();
                applyButton.click();
            }
        });

        function restoreEditorSession() {
            let session = null;
            try {
                session = JSON.parse(sessionStorage.getItem(EDITOR_SESSION_KEY) || 'null');
            } catch (e) {}
            if (!session) return;
            if (session.tab === 'shader' || session.tab === 'video') {
                setActiveTab(session.tab);
            }
            if (session.open) setEditorOpen(true);
        }

        restoreEditorSession();
        window.addEventListener('pagehide', persistEditorSession);

        // Return UI elements for reference
        return {
            container: uiContainer,
            textarea: shaderTextarea,
            applyButton: applyButton,
            resetButton: resetButton,
            errorDisplay: errorDisplay,
            toggle: toggleButton,
            showMessage: showMessage,
            toggleVisibility: () => {
                setEditorOpen(!uiContainer.classList.contains('open'));
            }
        };
    }

    // Public API
    return {
        convertShaderToyToWebGL: convertShaderToyToWebGL,
        createShaderTestUI: createShaderTestUI,
        SAMPLE_SHADERS: SAMPLE_SHADERS // Expose sample shaders if needed elsewhere
    };
})();