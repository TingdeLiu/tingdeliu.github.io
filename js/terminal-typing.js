// Terminal typing effect for geek-style homepage
(function() {
    'use strict';
    
    // Configuration
    const config = {
        typingSpeed: 50,        // milliseconds per character
        deletingSpeed: 30,      // milliseconds per character when deleting
        pauseAfterTyping: 2000, // pause after completing a phrase
    };
    
    // Phrases to type
    const phrases = [
        "Embodied AI Engineer",
        "VLN Specialist",
        "Vision-Language Navigation",
        "探索具身智能的无限可能"
    ];
    
    let phraseIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    let timeoutId = null;
    
    // Get elements
    const terminalText = document.querySelector('.terminal-typing');
    const cursor = document.querySelector('.terminal-cursor');
    
    if (!terminalText || !cursor) {
        return; // Elements not found, exit
    }
    
    function type() {
        const currentPhrase = phrases[phraseIndex];
        
        if (!isDeleting && charIndex <= currentPhrase.length) {
            // Typing forward
            terminalText.textContent = currentPhrase.substring(0, charIndex);
            charIndex++;
            
            if (charIndex > currentPhrase.length) {
                // Finished typing, pause before deleting
                timeoutId = setTimeout(() => {
                    isDeleting = true;
                    type();
                }, config.pauseAfterTyping);
                return;
            }
            
            timeoutId = setTimeout(type, config.typingSpeed);
        } else if (isDeleting && charIndex >= 0) {
            // Deleting backward
            terminalText.textContent = currentPhrase.substring(0, charIndex);
            charIndex--;
            
            if (charIndex < 0) {
                // Finished deleting, move to next phrase
                isDeleting = false;
                phraseIndex = (phraseIndex + 1) % phrases.length;
                charIndex = 0;
                timeoutId = setTimeout(type, 500);
                return;
            }
            
            timeoutId = setTimeout(type, config.deletingSpeed);
        }
    }
    
    // Start typing effect when page loads
    document.addEventListener('DOMContentLoaded', function() {
        timeoutId = setTimeout(type, 1000); // Delay before starting
    });
    
    // Cleanup function to clear timeout when page unloads
    window.addEventListener('beforeunload', function() {
        if (timeoutId) {
            clearTimeout(timeoutId);
        }
    });
})();
