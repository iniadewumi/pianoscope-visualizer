/**
 * Pianoscope Visualizer — the operator's side of kiosk mode.
 *
 * Opens kiosk.html as the projector window and hands it shader source. That
 * window (js/kiosk-window.js) draws with the same GL code as this one
 * (js/gl-shared.js) and pulls audio, video, display text and the pointer from
 * here every frame, so nothing else is sent to it.
 */

// A real page rather than a generated one, so reloading the projector window
// (Cmd+R, or Chrome recovering from a GPU crash) simply loads it again.
const KIOSK_URL = 'kiosk.html';

function handleKioskMessage(event) {
    // The kiosk is a page of this site, so it reports our origin. Anything
    // else could be a stray window trying to become kioskWindow and receive
    // shader sources.
    if (!event.source || event.origin !== window.location.origin) return;

    const message = event.data;
    if (!message || typeof message !== 'object') return;

    if (message.type === 'kiosk-loaded') {
        window.kioskWindow = event.source;
        event.source.postMessage({ type: 'init' }, window.location.origin);
    } else if (message.type === 'kiosk-ready') {
        if (window.kioskWindow && event.source !== window.kioskWindow) return;
        sendCurrentShaderToKiosk();
    }
}

// Function to open the kiosk mode window
function openKioskMode() {
    if (window.kioskWindow && !window.kioskWindow.closed) {
        window.kioskWindow.focus();
        sendCurrentShaderToKiosk();
        return;
    }

    const kioskWindow = window.open(KIOSK_URL, 'PianoscopeKiosk', 'width=800,height=600');

    if (!kioskWindow) {
        alert('Please allow popups to open the kiosk mode');
        return;
    }

    // Store a reference to the kiosk window
    window.kioskWindow = kioskWindow;

    moveKioskToOtherScreen(kioskWindow);
}

// window.screen only describes the screen this page is on, so landing the
// popup elsewhere needs the Window Management API. Unsupported, denied or
// single-screen all leave the popup where it opened.
async function moveKioskToOtherScreen(kioskWindow) {
    if (typeof window.getScreenDetails !== 'function') return;
    // A single screen has nowhere to go; don't raise a permission prompt for it.
    if (window.screen && window.screen.isExtended === false) return;

    try {
        // Called before any await so the click's user activation still counts.
        const details = await window.getScreenDetails();
        const current = details.currentScreen;
        const others = details.screens.filter(function(s) {
            return s !== current && !(current && s.left === current.left && s.top === current.top);
        });
        const target = others.find(function(s) { return !s.isPrimary; }) || others[0];
        if (!target || kioskWindow.closed) return;

        kioskWindow.moveTo(target.availLeft, target.availTop);
        kioskWindow.resizeTo(target.availWidth, target.availHeight);
    } catch (e) {
        console.log('Could not position on second screen', e);
    }
}

// Function to send the current shader to the kiosk window
function sendCurrentShaderToKiosk() {
    if (!window.kioskWindow || window.kioskWindow.closed) return;
    
    // Raw sources: the kiosk adds the same GLSL prefix this window does.
    let vertexSource = window.currentVertexShaderSource || null;
    let fragmentSource = window.currentFragmentShaderSource || null;
    
    // If we couldn't get the shader sources, use default fallback shaders
    if (!vertexSource) {
        vertexSource = `
            attribute vec2 position;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        `;
    }
    
    if (!fragmentSource) {
        fragmentSource = `
            precision highp float;
            
            uniform vec3 iResolution;
            uniform float iTime;
            uniform sampler2D iChannel0;
            
            void main() {
                vec2 uv = gl_FragCoord.xy / iResolution.xy;
                
                // Simple color gradient
                vec3 color = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0,2,4));
                
                // Add some audio reactivity if available
                float audio = texture2D(iChannel0, vec2(uv.x, 0.0)).x;
                color += audio * 0.2;
                
                gl_FragColor = vec4(color, 1.0);
            }
        `;
    }
    
    // Send the shaders to the kiosk window
    window.kioskWindow.postMessage({
        type: 'shader-update',
        vertexShader: vertexSource,
        fragmentShader: fragmentSource
    }, window.location.origin);
}

// Add button to open kiosk mode
document.addEventListener('DOMContentLoaded', function() {
    const buttonContainer = document.querySelector('.button-container');
    if (buttonContainer) {
        const kioskButton = document.createElement('button');
        kioskButton.id = 'kiosk-mode-button';
        kioskButton.className = 'gold-button';
        kioskButton.innerHTML = '<i class="fas fa-desktop"></i><span>&nbsp;Kiosk Mode</span>';
        kioskButton.title = 'Open in kiosk mode (fullscreen on second display)';
        kioskButton.addEventListener('click', openKioskMode);
        buttonContainer.appendChild(kioskButton);
    }

    // Audio is not pushed from here: the kiosk pulls it each frame through
    // window.pianoscopeReadAudio (see visualizer-with-shadertoy.js).
    window.addEventListener('message', handleKioskMessage);
});