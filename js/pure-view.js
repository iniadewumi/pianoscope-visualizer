// Add this code to the end of your DOMContentLoaded event listener in index.html
// or as a separate script file

document.addEventListener('DOMContentLoaded', function() {
    // Pure visualization view functionality
    const pureViewToggle = document.getElementById('pure-view-toggle');
    const miniMicToggle = document.getElementById('mini-mic-toggle');
    const miniFullscreen = document.getElementById('mini-fullscreen');
    const miniShaderToggle = document.getElementById('mini-shader-toggle');
    const mainMicToggle = document.getElementById('mic-toggle');
    const mainFullscreen = document.getElementById('fullscreen');
    const body = document.body;
    
    // Function to toggle pure view mode
    function togglePureViewMode() {
        body.classList.toggle('pure-view-mode');
        
        // Update toggle button icon
        if (body.classList.contains('pure-view-mode')) {
            pureViewToggle.innerHTML = '<i class="fas fa-eye-slash"></i>';
            pureViewToggle.title = "Exit Pure Visualization Mode";
        } else {
            pureViewToggle.innerHTML = '<i class="fas fa-eye"></i>';
            pureViewToggle.title = "Enter Pure Visualization Mode";
        }
        
        // Trigger resize to ensure canvas adjusts correctly
        window.dispatchEvent(new Event('resize'));
    }
    
    // Add event listeners for pure view controls
    if (pureViewToggle) {
        pureViewToggle.addEventListener('click', togglePureViewMode);
    }
    
    // Mini mic toggle syncs with main mic toggle
    if (miniMicToggle && mainMicToggle) {
        miniMicToggle.addEventListener('click', function() {
            // Forward click to main mic toggle
            mainMicToggle.click();
            
            // Update mini button appearance based on mic state
            setTimeout(() => {
                if (mainMicToggle.classList.contains('red-button')) {
                    miniMicToggle.innerHTML = '<i class="fas fa-microphone-slash"></i>';
                    miniMicToggle.classList.add('mic-active');
                } else {
                    miniMicToggle.innerHTML = '<i class="fas fa-microphone"></i>';
                    miniMicToggle.classList.remove('mic-active');
                }
            }, 50);
        });
    }
    
    // Mini fullscreen button syncs with main fullscreen button
    if (miniFullscreen && mainFullscreen) {
        miniFullscreen.addEventListener('click', function() {
            mainFullscreen.click();
        });
    }
    
    // Mini shader toggle syncs with main shader toggle
    if (miniShaderToggle) {
        miniShaderToggle.addEventListener('click', function() {
            // Find the main shader toggle button (might be created dynamically)
            const mainShaderToggle = document.querySelector('.toggle-shader-ui');
            if (mainShaderToggle) {
                mainShaderToggle.click();
                
                // Update mini button appearance based on shader editor state
                setTimeout(() => {
                    const shaderUI = document.querySelector('.shader-test-ui');
                    if (shaderUI && shaderUI.classList.contains('open')) {
                        miniShaderToggle.innerHTML = '<i class="fas fa-times"></i>';
                    } else {
                        miniShaderToggle.innerHTML = '<i class="fas fa-code"></i>';
                    }
                }, 50);
            }
        });
    }
    
    // Update mini controls based on main controls initial state
    function updateMiniControlStates() {
        if (mainMicToggle && miniMicToggle) {
            if (mainMicToggle.classList.contains('red-button')) {
                miniMicToggle.innerHTML = '<i class="fas fa-microphone-slash"></i>';
                miniMicToggle.classList.add('mic-active');
            }
        }
    }
    
    // Run initial update after a short delay to ensure everything is initialized
    setTimeout(updateMiniControlStates, 500);
    
    // Keyboard shortcut for pure view mode (Escape key)
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && body.classList.contains('pure-view-mode')) {
            togglePureViewMode();
        } else if (e.key === 'p' && e.ctrlKey) {
            // Ctrl+P for pure view toggle
            togglePureViewMode();
            e.preventDefault(); // Prevent browser print dialog
        }
    });
    
    // Auto-hide mini controls after inactivity in pure view mode
    let inactivityTimer;
    function startInactivityTimer() {
        clearTimeout(inactivityTimer);
        
        if (body.classList.contains('pure-view-mode')) {
            // Show controls on mouse move
            document.querySelector('.mini-controls').style.opacity = '1';
            document.querySelector('.pure-view-toggle').style.opacity = '0.7';
            
            // Hide after 3 seconds of inactivity
            inactivityTimer = setTimeout(() => {
                if (body.classList.contains('pure-view-mode')) {
                    document.querySelector('.mini-controls').style.opacity = '0';
                    document.querySelector('.pure-view-toggle').style.opacity = '0.3';
                }
            }, 3000);
        }
    }
    
    // Track mouse movement to show/hide controls
    document.addEventListener('mousemove', startInactivityTimer);
    
    // Double click on canvas to toggle pure view mode
    document.getElementById('visualizer').addEventListener('dblclick', function(e) {
        togglePureViewMode();
    });
});