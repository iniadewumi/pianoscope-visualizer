/**
 * Text sequencer - the two-slot word ladder.
 *
 * The device is anadiplosis: the last term of one line becomes the first term of
 * the next, so a chain of two-line states reads as one argument assembled over
 * time. The visual mechanic falls straight out of it - the bottom slot promotes to
 * the top, a new line enters the bottom - which is also how a house loop is built:
 * something held, something changing.
 *
 * A step is either explicit lines, or { promote, enter } shorthand meaning "move
 * the current bottom line up, and put this in the bottom". Authored steps win, so
 * a chain is free to break the promotion rule where the copy needs it.
 *
 * Timing is either a wall-clock interval or a bar count against a tapped tempo.
 * There is no automatic beat detection here on purpose: the analyser this project
 * feeds runs at smoothingTimeConstant 0.8, which deliberately flattens the
 * transients onset detection needs. Tap is reliable; inference is not.
 */
window.PianoscopeTextSequencer = (function () {
    'use strict';

    const STORE_SEQ = 'pianoscope-seq-name';
    const STORE_BPM = 'pianoscope-seq-bpm';
    const STORE_BARS = 'pianoscope-seq-bars';

    // Every state reads as a complete thought, because people look up at random
    // moments and a frame like "MUSIC | IS" asks the room to have been watching.
    //
    // `in:` records the transition each state was authored for. It is reference
    // only - the transition picker in the sidebar drives playback. Wire it back
    // through show() if per-state choreography is ever wanted again.
    const SEQUENCES = {
        'Thesis': {
            advance: 'bars',
            steps: [
                { lines: ['CRESCENDO', '[ARTIST]'], in: 'cut' },
                { lines: ['CRESCENDO', 'IS HOUSE'], in: 'roll' },
                { lines: ['HOUSE IS', 'BLACK MUSIC'], in: 'roll' },
                { lines: ['HOUSE MUSIC', 'IS OUR CULTURE'], in: 'knockout' },
                { lines: ['OUR CULTURE', 'IS OURS'], in: 'roll' },
                { lines: ['OURS', 'TONIGHT'], in: 'reveal' },
                { lines: ['CRESCENDO', 'FOR CULTURE'], in: 'cut' }
            ]
        },
        'Roll call': {
            advance: 'bars',
            steps: [
                { lines: ['CHICAGO', '1977'], in: 'cut' },
                { lines: ['DETROIT', '1981'], in: 'cut' },
                { lines: ['JOHANNESBURG', '2012'], in: 'cut' },
                { lines: ['DALLAS', 'TONIGHT'], in: 'cut' }
            ]
        },
        'Ticker': {
            advance: 'bars',
            steps: [
                { lines: ['LOUDER'], in: 'cut' },
                { lines: ['FASTER'], in: 'cut' },
                { lines: ['CRESCENDO'], in: 'cut' }
            ]
        }
    };

    let seqName = 'Thesis';
    let index = 0;
    let running = false;
    let timer = 0;
    let bpm = 124;
    let barsPerStep = 4;
    let artist = '[ARTIST]';
    let lastLines = ['CRESCENDO', '[ARTIST]'];
    let hasShown = false;
    const listeners = [];

    // Tap tempo. Four taps is enough to lock; stale taps are dropped so a operator
    // returning to the pad after a while does not average across a gap.
    let taps = [];
    const TAP_RESET_MS = 2400;

    function dt() { return window.PianoscopeDisplayText || null; }

    function sequence() {
        return SEQUENCES[seqName] || SEQUENCES.Thesis;
    }

    function names() { return Object.keys(SEQUENCES); }

    function stepMs() {
        // 4 beats to a bar.
        return (60000 / Math.max(20, bpm)) * 4 * Math.max(1, barsPerStep);
    }

    function substitute(line) {
        return String(line).replace(/\[ARTIST\]/g, artist);
    }

    // Anadiplosis promotes the last TERM of a clause, not the whole clause:
    // "CRESCENDO IS HOUSE" -> "HOUSE IS BLACK", never "IS HOUSE IS BLACK". So a
    // promoted line drops its leading copula. A step can override with promoteAs
    // when the copy needs something the rule would not produce.
    const COPULAS = ['IS', 'ARE', 'WAS', 'WERE', 'AINT', "AIN'T"];

    function promoteTail(line) {
        const words = String(line).trim().split(/\s+/).filter(Boolean);
        while (words.length > 1 && COPULAS.indexOf(words[0].toUpperCase()) !== -1) {
            words.shift();
        }
        return words.join(' ');
    }

    /** Resolve a step against the previous state, applying promotion. */
    function resolve(step) {
        let lines;
        if (step.lines) {
            lines = step.lines.slice();
        } else if (step.promote) {
            const bottom = lastLines[lastLines.length - 1] || '';
            const top = step.promoteAs || promoteTail(bottom);
            lines = [top, step.enter || ''];
        } else {
            lines = [step.enter || ''];
        }
        lines = lines.filter(function (l) { return l && l.length; });
        lastLines = lines.slice();
        return lines.map(substitute).join('|');
    }

    function emit() {
        for (let i = 0; i < listeners.length; i++) {
            try { listeners[i](status()); } catch (e) {}
        }
    }

    function show(i) {
        const seq = sequence();
        const steps = seq.steps;
        if (!steps.length) return;
        index = ((i % steps.length) + steps.length) % steps.length;
        hasShown = true;
        const step = steps[index];
        const text = resolve(step);
        const d = dt();
        // The operator's transition picker governs every state in the chain. Steps
        // still carry an authored `in:` as a record of the intended choreography,
        // but it is deliberately NOT applied here: a picker that silently lost to
        // the step data made the sequence look like it was choosing at random.
        if (d) d.applyText(text);
        emit();
    }

    function step(delta) {
        show(index + (typeof delta === 'number' ? delta : 1));
        // A manual cue restarts the bar clock. Without this the pending timeout
        // survives, so stepping late in a bar fires the next state milliseconds
        // later and truncates the transition that just started.
        if (running) schedule();
    }

    function schedule() {
        if (timer) { clearTimeout(timer); timer = 0; }
        if (!running) return;
        timer = setTimeout(function () {
            step(1);
            schedule();
        }, stepMs());
    }

    /**
     * Resume where the operator left it. Play used to jump back to the first state,
     * which loses the place of anyone who stepped to a cue by hand before starting.
     * Only a cold start - or a sequence change - rebuilds the chain from the top.
     */
    function play() {
        if (running) return status();
        running = true;
        if (!hasShown) {
            lastLines = ['CRESCENDO', artist];
            show(0);
        }
        schedule();
        // show() emits, but only on a cold start. Resuming an already-shown chain
        // reached here without ever notifying, so the transport button kept
        // reading "Play" while the sequencer was in fact running.
        emit();
        return status();
    }

    /** Explicitly go back to the first state and run from there. */
    function restart() {
        lastLines = ['CRESCENDO', artist];
        running = true;
        show(0);
        schedule();
        return status();
    }

    function stop() {
        running = false;
        if (timer) { clearTimeout(timer); timer = 0; }
        emit();
        return status();
    }

    function toggle() { return running ? stop() : play(); }

    function tap() {
        const now = performance.now();
        if (taps.length && now - taps[taps.length - 1] > TAP_RESET_MS) taps = [];
        taps.push(now);
        if (taps.length > 5) taps.shift();
        if (taps.length >= 2) {
            let sum = 0;
            for (let i = 1; i < taps.length; i++) sum += taps[i] - taps[i - 1];
            const avg = sum / (taps.length - 1);
            if (avg > 200 && avg < 2000) {
                bpm = Math.round(60000 / avg);
                try { localStorage.setItem(STORE_BPM, String(bpm)); } catch (e) {}
                if (running) schedule();
            }
        }
        emit();
        return { bpm: bpm, taps: taps.length };
    }

    function setSequence(name) {
        if (!SEQUENCES[name]) return seqName;
        seqName = name;
        index = 0;
        hasShown = false;
        lastLines = ['CRESCENDO', artist];
        try { localStorage.setItem(STORE_SEQ, seqName); } catch (e) {}
        if (running) { show(0); schedule(); } else { emit(); }
        return seqName;
    }

    function setBpm(v) {
        bpm = Math.max(20, Math.min(300, Math.round(v) || 124));
        try { localStorage.setItem(STORE_BPM, String(bpm)); } catch (e) {}
        if (running) schedule();
        emit();
        return bpm;
    }

    function setBars(v) {
        barsPerStep = Math.max(1, Math.min(32, Math.round(v) || 4));
        try { localStorage.setItem(STORE_BARS, String(barsPerStep)); } catch (e) {}
        if (running) schedule();
        emit();
        return barsPerStep;
    }

    function setArtist(name) {
        artist = String(name || '').trim() || '[ARTIST]';
        emit();
        return artist;
    }

    function status() {
        return {
            sequence: seqName,
            names: names(),
            index: index,
            total: sequence().steps.length,
            running: running,
            bpm: bpm,
            bars: barsPerStep,
            artist: artist,
            stepMs: Math.round(stepMs())
        };
    }

    function init() {
        try {
            const s = localStorage.getItem(STORE_SEQ);
            if (s && SEQUENCES[s]) seqName = s;
            const b = parseInt(localStorage.getItem(STORE_BPM), 10);
            if (b) bpm = Math.max(20, Math.min(300, b));
            const n = parseInt(localStorage.getItem(STORE_BARS), 10);
            if (n) barsPerStep = Math.max(1, Math.min(32, n));
        } catch (e) {}
        emit();
    }

    document.addEventListener('keydown', function (e) {
        // Modifier combinations belong to the browser and to other handlers:
        // Ctrl+P is pure view, Cmd+P is print, Cmd+T is a new tab. Without this
        // guard those all drove the sequencer as well.
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        if (e.defaultPrevented) return;
        const t = e.target;
        if (t && (/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) || t.isContentEditable)) return;
        if (e.code === 'Space') { e.preventDefault(); step(1); }
        else if (e.key === 't' || e.key === 'T') { tap(); }
        else if (e.key === 'p' || e.key === 'P') { toggle(); }
    });

    return {
        SEQUENCES: SEQUENCES,
        init: init,
        names: names,
        play: play,
        restart: restart,
        stop: stop,
        toggle: toggle,
        step: step,
        show: show,
        tap: tap,
        setSequence: setSequence,
        setBpm: setBpm,
        setBars: setBars,
        setArtist: setArtist,
        status: status,
        onChange: function (fn) { if (typeof fn === 'function') listeners.push(fn); }
    };
})();
