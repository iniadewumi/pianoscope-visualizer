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
#if __VERSION__ >= 300
void main() {
    mainImage(fragColor, gl_FragCoord.xy);
    fragColor.a = 1.0;
}
#else
void main() {
    vec4 fragColor;
    mainImage(fragColor, gl_FragCoord.xy);
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
        header.appendChild(title);
        uiContainer.appendChild(header); // Header still belongs to main container

        // Create tab system
        const tabContainer = document.createElement('div');
        tabContainer.className = 'tab-container';
        const shaderTabBtn = document.createElement('button');
        shaderTabBtn.className = 'tab-button active';
        shaderTabBtn.dataset.tab = 'shader';
        shaderTabBtn.textContent = 'Shader';
        const videoTabBtn = document.createElement('button');
        videoTabBtn.className = 'tab-button';
        videoTabBtn.dataset.tab = 'video';
        videoTabBtn.textContent = 'Video';
        tabContainer.appendChild(shaderTabBtn);
        tabContainer.appendChild(videoTabBtn);
        uiContainer.appendChild(tabContainer); // Tabs belong to main container

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
        applyButton.innerHTML = '<i class="fa fa-check"></i> Apply Shader';
        applyButton.className = 'shader-editor-button apply-button';
        const resetButton = document.createElement('button');
        resetButton.innerHTML = '<i class="fa fa-refresh"></i> Reset';
        resetButton.className = 'shader-editor-button reset-button';
        controls.appendChild(applyButton);
        controls.appendChild(resetButton);

        const sampleContainer = document.createElement('div');
        sampleContainer.className = 'sample-container';
        // Create a custom dropdown container
        const dropdownContainer = document.createElement('div');
        dropdownContainer.className = 'custom-dropdown';

        // Create the input field that will also act as the dropdown trigger
        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.className = 'dropdown-input';
        searchInput.placeholder = 'Search shaders...';

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
        displayTextPanel.className = 'display-text-panel';
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
        const displayTextApply = document.createElement('button');
        displayTextApply.type = 'button';
        displayTextApply.className = 'shader-editor-button apply-button display-text-apply';
        displayTextApply.innerHTML = '<i class="fa fa-check"></i> Apply';
        displayTextRow.appendChild(displayTextInput);
        displayTextRow.appendChild(displayTextApply);
        displayTextPanel.appendChild(displayTextLabel);
        displayTextPanel.appendChild(displayTextRow);
        const displayTextMeta = document.createElement('div');
        displayTextMeta.className = 'display-text-meta';
        const displayFontSelect = document.createElement('select');
        displayFontSelect.id = 'display-text-font';
        displayFontSelect.className = 'display-text-font';
        displayFontSelect.title = 'Overlay font';
        const fonts = (window.PianoscopeDisplayText && window.PianoscopeDisplayText.FONTS) || [];
        for (let i = 0; i < fonts.length; i++) {
            const opt = document.createElement('option');
            opt.value = fonts[i].id;
            opt.textContent = fonts[i].name;
            displayFontSelect.appendChild(opt);
        }
        const overlayToggle = document.createElement('label');
        overlayToggle.className = 'display-text-overlay-toggle';
        const overlayCheck = document.createElement('input');
        overlayCheck.type = 'checkbox';
        overlayCheck.id = 'display-text-overlay-enabled';
        overlayToggle.appendChild(overlayCheck);
        overlayToggle.appendChild(document.createTextNode(' JS overlay'));
        displayTextMeta.appendChild(displayFontSelect);
        displayTextMeta.appendChild(overlayToggle);
        displayTextPanel.appendChild(displayTextMeta);

        // ---- transition picker ----
        const transitionRow = document.createElement('div');
        transitionRow.className = 'display-text-meta';
        const transitionSelect = document.createElement('select');
        transitionSelect.id = 'display-text-transition';
        transitionSelect.className = 'display-text-font';
        transitionSelect.title = 'Transition used for every text change, including sequencer playback';
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
        previewBtn.textContent = 'Preview';
        previewBtn.title = 'Re-apply the current text using the selected transition';
        transitionRow.appendChild(transitionSelect);
        transitionRow.appendChild(previewBtn);
        displayTextPanel.appendChild(transitionRow);

        // ---- sequencer transport ----
        const seqLabel = document.createElement('label');
        seqLabel.className = 'display-text-label';
        seqLabel.textContent = 'Sequence';
        displayTextPanel.appendChild(seqLabel);

        const seqRow = document.createElement('div');
        seqRow.className = 'display-text-meta';
        const seqSelect = document.createElement('select');
        seqSelect.id = 'seq-name';
        seqSelect.className = 'display-text-font';
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
        seqRow.appendChild(seqSelect);
        seqRow.appendChild(artistInput);
        displayTextPanel.appendChild(seqRow);

        const transportRow = document.createElement('div');
        transportRow.className = 'seq-transport';
        const playBtn = document.createElement('button');
        playBtn.type = 'button';
        playBtn.className = 'shader-editor-button seq-btn';
        playBtn.textContent = 'Play';
        playBtn.title = 'Run the sequence (P)';
        const stepBtn = document.createElement('button');
        stepBtn.type = 'button';
        stepBtn.className = 'shader-editor-button seq-btn';
        stepBtn.textContent = 'Step';
        stepBtn.title = 'Advance one state (Space)';
        const tapBtn = document.createElement('button');
        tapBtn.type = 'button';
        tapBtn.className = 'shader-editor-button seq-btn';
        tapBtn.textContent = 'Tap';
        tapBtn.title = 'Tap four beats to lock tempo (T)';
        const bpmInput = document.createElement('input');
        bpmInput.type = 'number';
        bpmInput.min = '20';
        bpmInput.max = '300';
        bpmInput.className = 'seq-num';
        bpmInput.title = 'BPM';
        const barsInput = document.createElement('input');
        barsInput.type = 'number';
        barsInput.min = '1';
        barsInput.max = '32';
        barsInput.className = 'seq-num';
        barsInput.title = 'Bars per state';
        transportRow.appendChild(playBtn);
        transportRow.appendChild(stepBtn);
        transportRow.appendChild(tapBtn);
        transportRow.appendChild(bpmInput);
        transportRow.appendChild(barsInput);
        displayTextPanel.appendChild(transportRow);

        const seqReadout = document.createElement('div');
        seqReadout.className = 'display-text-hint seq-readout';
        displayTextPanel.appendChild(seqReadout);
        if (window.PianoscopeDisplayText) {
            displayTextInput.value = window.PianoscopeDisplayText.toDisplayValue(
                window.PianoscopeDisplayText.getText()
            );
            displayFontSelect.value = window.PianoscopeDisplayText.getFont();
            overlayCheck.checked = window.PianoscopeDisplayText.isOverlayEnabled();
            window.PianoscopeDisplayText.onChange(function (text) {
                if (document.activeElement !== displayTextInput) {
                    displayTextInput.value = window.PianoscopeDisplayText.toDisplayValue(text);
                }
            });
        }
        const displayTextHint = document.createElement('div');
        displayTextHint.className = 'display-text-hint';
        displayTextHint.textContent = 'Pipe or newline stacks a line. Space steps, P plays, T taps tempo. The transition above applies to every state in the sequence. Word Ladder and Kick Afterglow honour transitions (Kick Afterglow keeps its original size). Kente Loom uses the legacy sheet and always hard-cuts.';
        displayTextPanel.appendChild(displayTextHint);

        dropdownContainer.appendChild(searchInput);
        dropdownContainer.appendChild(optionsContainer);
        shaderTabContent.appendChild(displayTextPanel);
        shaderTabContent.appendChild(dropdownContainer);
        const shaderTextarea = document.createElement('textarea');
        shaderTextarea.className = 'shader-editor-textarea';
        shaderTextarea.placeholder = 'Paste Shadertoy shader code here...';
        // Use a consistent default shader name, handle potential missing key
        const defaultShaderName = "Pianoscope Text";
        const defaultShader = SAMPLE_SHADERS[defaultShaderName] || Object.values(SAMPLE_SHADERS)[0] || ''; // Added fallback
        shaderTextarea.value = defaultShader;

        const errorDisplay = document.createElement('div');
        errorDisplay.className = 'error-display';
        // --- End elements for shaderTabContent ---

        // Append shader-specific elements to shaderTabContent (THIS IS THE FIX)
        shaderTabContent.appendChild(controls);
        shaderTabContent.appendChild(sampleContainer);
        shaderTabContent.appendChild(shaderTextarea);
        shaderTabContent.appendChild(errorDisplay);

        // Video tab is populated by VideoController.mountUI() once the catalog
        // has been probed, so the markup lives with the code that drives it.

        // Add the content divs to the main UI container
        uiContainer.appendChild(shaderTabContent);
        uiContainer.appendChild(videoTabContent);

        // Create toggle button
        const toggleButton = document.createElement('button');
        toggleButton.innerHTML = '<i class="fas fa-code"></i>';
        toggleButton.className = 'toggle-shader-ui';
        toggleButton.title = "Toggle Shader Editor";

        // Add the toggle button and the main UI container to the document body (or specified container)
        container.appendChild(toggleButton);
        container.appendChild(uiContainer);

        // --- Tab Switching Functionality ---
        const tabButtons = uiContainer.querySelectorAll('.tab-button');
        const tabContents = uiContainer.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                // Deactivate all tabs
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.add('hidden'));

                // Activate selected tab
                button.classList.add('active');
                const tabId = button.dataset.tab + '-tab';
                const tabContent = document.getElementById(tabId);
                if (tabContent) {
                    tabContent.classList.remove('hidden');
                }
            });
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

        function commitDisplayText() {
            if (!window.PianoscopeDisplayText) return;
            const next = window.PianoscopeDisplayText.applyText(displayTextInput.value);
            displayTextInput.value = window.PianoscopeDisplayText.toDisplayValue(next);
            showMessage(next ? ('Display text: ' + next.replace(/\|/g, ' / ')) : 'Display text cleared', 'success');
        }

        displayTextApply.addEventListener('click', commitDisplayText);
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
            playBtn.textContent = st.running ? 'Pause' : 'Play';
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

        // Add event listener for toggle button
        toggleButton.addEventListener('click', () => {
            uiContainer.classList.toggle('open');
            toggleButton.classList.toggle('active');
            toggleButton.innerHTML = uiContainer.classList.contains('open') ? '<i class="fas fa-times"></i>' : '<i class="fas fa-code"></i>';
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

        // Return UI elements for reference
        return {
            container: uiContainer,
            textarea: shaderTextarea,
            applyButton: applyButton,
            resetButton: resetButton,
            errorDisplay: errorDisplay,
            toggle: toggleButton,
            showMessage: showMessage,
            toggleVisibility: () => { // Expose toggle function if needed externally
                uiContainer.classList.toggle('open');
                toggleButton.classList.toggle('active');
                toggleButton.innerHTML = uiContainer.classList.contains('open') ? '<i class="fas fa-times"></i>' : '<i class="fas fa-code"></i>';
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