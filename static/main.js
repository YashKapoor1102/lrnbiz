// LrnBiz — main.js

// ══════════════════════════════════════════════════════════
//  C7-26 — INSIGHT XP
//  Bonus XP for long, thoughtful answers and returning to
//  improve a chapter. Called from chapter pages after
//  analysis completes.
// ══════════════════════════════════════════════════════════
function awardInsightXP(chapterId, answers) {
    const key = 'lrnbiz_xp_' + chapterId;
    let xp = parseInt(localStorage.getItem(key) || '0');
    let bonus = 0;
    const bonusLog = [];

    // Long answer bonus: any textarea ≥ 80 chars = +5 XP each (max 2)
    let longCount = 0;
    Object.values(answers || {}).forEach(v => {
        if (typeof v === 'string' && v.trim().length >= 80 && longCount < 2) {
            bonus += 5; longCount++; bonusLog.push('💬 Thoughtful answer +5 XP');
        }
    });

    // Iteration bonus: re-running analysis = +10 XP
    const runKey = 'lrnbiz_runs_' + chapterId;
    const runs = parseInt(localStorage.getItem(runKey) || '0') + 1;
    localStorage.setItem(runKey, runs);
    if (runs > 1) { bonus += 10; bonusLog.push('🔁 Improved your work +10 XP'); }

    // Diversity bonus: 5+ distinct answers filled = +5 XP
    const filled = Object.values(answers || {}).filter(v => v && String(v).trim()).length;
    if (filled >= 5) { bonus += 5; bonusLog.push('📝 Complete answers +5 XP'); }

    if (bonus > 0) {
        xp += bonus;
        localStorage.setItem(key, xp);
        // Show XP toast
        const toast = document.createElement('div');
        toast.style.cssText = 'position:fixed;top:72px;right:16px;z-index:99100;background:rgba(255,217,61,0.18);border:1px solid rgba(255,217,61,0.4);border-radius:12px;padding:10px 16px;font-family:Nunito,sans-serif;font-size:0.82rem;font-weight:800;color:#FFD93D;opacity:0;transition:all 0.35s;max-width:220px';
        toast.innerHTML = `⭐ +${bonus} Insight XP!<div style="font-size:0.68rem;font-weight:600;color:rgba(255,217,61,0.75);margin-top:4px">${bonusLog.join('<br>')}</div>`;
        document.body.appendChild(toast);
        setTimeout(() => { toast.style.opacity = '1'; toast.style.transform = 'translateY(4px)'; }, 10);
        setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 400); }, 3500);
    }
    return xp;
}

// ══════════════════════════════════════════════════════════
//  SOUND FX  (Web Audio API — no external files needed)
//  Respects user preference stored in localStorage.
// ══════════════════════════════════════════════════════════
const SoundFX = (() => {
    let _ctx   = null;
    let _ready = null; // Promise that resolves to a running AudioContext
    let _enabled = localStorage.getItem('lrnbiz_sound') !== 'off';

    // Returns a Promise that resolves with the AudioContext once it is running.
    // Cached after first call — all subsequent calls reuse the same Promise.
    function _ensure() {
        if (_ready) return _ready;
        try {
            _ctx = new (window.AudioContext || window.webkitAudioContext)();
        } catch(e) {
            return (_ready = Promise.reject(e));
        }
        _ready = (_ctx.state === 'running')
            ? Promise.resolve(_ctx)
            : _ctx.resume().then(() => _ctx);
        return _ready;
    }

    function _beep(freq, dur, type = 'sine', vol = 0.18, delay = 0) {
        if (!_enabled) return;
        _ensure().then(ctx => {
            const osc  = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.type = type;
            osc.frequency.setValueAtTime(freq, ctx.currentTime + delay);
            gain.gain.setValueAtTime(vol, ctx.currentTime + delay);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + delay + dur);
            osc.start(ctx.currentTime + delay);
            osc.stop(ctx.currentTime + delay + dur + 0.05);
        }).catch(() => {});
    }

    return {
        isEnabled: () => _enabled,
        toggle: () => {
            _enabled = !_enabled;
            localStorage.setItem('lrnbiz_sound', _enabled ? 'on' : 'off');
            return _enabled;
        },
        // Called on first user gesture — creates and unlocks the AudioContext early
        // so all subsequent sounds play without delay.
        _warmup: () => { _ensure().catch(() => {}); },
        // Short pop when selecting an answer
        pick:  () => { _beep(880, 0.08, 'sine', 0.12); },
        // Cheerful ding when a new field is filled
        tick:  () => { _beep(523, 0.1, 'sine', 0.15); _beep(659, 0.12, 'sine', 0.15, 0.08); },
        // Ascending arpeggio on rank/badge unlock
        level: () => { [523, 659, 784, 1047].forEach((f, i) => _beep(f, 0.18, 'sine', 0.16, i * 0.09)); },
        // Success fanfare on module complete
        win:   () => { [784, 784, 1047].forEach((f, i) => _beep(f, 0.22, 'sine', 0.2, i * 0.14)); },
        // Gentle buzz for a conflict/warning
        warn:  () => { _beep(220, 0.15, 'sawtooth', 0.06); },
    };
})();

// Warm up AudioContext on first user click so all subsequent sounds work reliably
document.addEventListener('click', () => { SoundFX._warmup(); }, { once: true });

// Scroll to top on every page load
window.addEventListener('load', () => window.scrollTo({ top: 0, behavior: 'instant' }));

// ══════════════════════════════════════════════════════════
//  C7-47 — OFFLINE FETCH HELPER
//  Wraps fetch with a friendly offline message on failure.
// ══════════════════════════════════════════════════════════
function showOfflineToast(msg) {
    if (typeof clearAnalysisProgress === 'function') clearAnalysisProgress();
    const existing = document.getElementById('offlineToast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.id = 'offlineToast';
    toast.style.cssText = 'position:fixed;bottom:24px;left:50%;transform:translateX(-50%);z-index:9300;background:rgba(248,113,113,0.18);border:1px solid rgba(248,113,113,0.4);border-radius:12px;padding:10px 18px;font-family:Nunito,sans-serif;font-size:0.82rem;font-weight:700;color:#F87171;text-align:center;max-width:320px';
    toast.innerHTML = msg || '📶 Connection issue — your answers are saved. Check your internet and try again.';
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 7000);
}

// Prevent double-submit (event delegation — works regardless of when main.js loads)
document.addEventListener('submit', (e) => {
    const form = e.target.closest('form');
    if (!form) return;
    const btn = form.querySelector('[type="submit"]:not([style*="display:none"])');
    if (btn) { btn.disabled = true; btn.style.opacity = '0.65'; }
});

// ══════════════════════════════════════════════════════════
//  C6-38 — BADGES
//  Awarded for: completing a chapter, score ≥ 80, 3+ analysis
//  runs on same chapter, AI Honesty Score gap < 5
// ══════════════════════════════════════════════════════════
const BADGES = {
    chapter_done:  { emoji: '📘', name: 'Chapter Complete!',   desc: 'You finished a full chapter' },
    high_scorer:   { emoji: '🌟', name: 'High Scorer!',        desc: 'Mentor score ≥ 80' },
    iterative:     { emoji: '🔁', name: 'Iteration Badge!',    desc: '3+ analyses on the same chapter' },
    aligned_ai:    { emoji: '🤝', name: 'Aligned AI Badge!',   desc: 'AI Honesty gap was under 5 pts — AI genuinely agreed with your plan!' },
};

function _awardBadge(badgeKey) {
    const badge = BADGES[badgeKey];
    if (!badge) return;
    const earnedKey = 'lrnbiz_badge_' + badgeKey;
    if (localStorage.getItem(earnedKey)) return; // only once
    localStorage.setItem(earnedKey, '1');
    SoundFX.level();
    // Show badge toast
    const toast = document.createElement('div');
    toast.style.cssText = 'position:fixed;bottom:140px;left:50%;transform:translateX(-50%) scale(0.8);z-index:99000;background:linear-gradient(135deg,rgba(167,139,250,0.25),rgba(52,211,153,0.15));border:1px solid rgba(167,139,250,0.4);border-radius:14px;padding:14px 20px;text-align:center;font-family:Nunito,sans-serif;opacity:0;transition:all 0.35s';
    toast.innerHTML = `<div style="font-size:2rem">${badge.emoji}</div><div style="font-weight:900;font-size:0.9rem;color:var(--purple,#A78BFA);margin:4px 0">${badge.name}</div><div style="font-size:0.72rem;color:rgba(240,239,244,0.7)">${badge.desc}</div>`;
    document.body.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '1'; toast.style.transform = 'translateX(-50%) scale(1)'; }, 10);
    setTimeout(() => { toast.style.opacity = '0'; toast.style.transform = 'translateX(-50%) scale(0.8)'; setTimeout(() => toast.remove(), 400); }, 3500);
}

// ══════════════════════════════════════════════════════════
//  C7-33 — SHARE RESULT MOMENT
//  Shows a shareable text after successful chapter analysis
// ══════════════════════════════════════════════════════════
function showShareMoment(chapterName, mentorScore) {
    if (!mentorScore) return;
    const existing = document.getElementById('shareMomentBar');
    if (existing) existing.remove();
    const bar = document.createElement('div');
    bar.id = 'shareMomentBar';
    bar.style.cssText = 'position:fixed;bottom:0;left:0;right:0;z-index:9500;background:linear-gradient(90deg,rgba(52,211,153,0.18),rgba(167,139,250,0.14));border-top:1px solid rgba(52,211,153,0.3);padding:10px 16px;display:flex;align-items:center;justify-content:space-between;font-family:Nunito,sans-serif;font-size:0.82rem;gap:10px;flex-wrap:wrap';
    const shareText = `I just scored ${mentorScore}/100 on the ${chapterName} chapter in LrnBiz! 🚀 #LrnBiz #StudentEntrepreneur`;
    const copyBtn = `<button onclick="navigator.clipboard.writeText('${shareText.replace(/'/g,"\\'")}').then(()=>{this.textContent='✅ Copied!';setTimeout(()=>{this.textContent='📋 Copy to share'},1500)})" style="background:rgba(52,211,153,0.2);border:1px solid rgba(52,211,153,0.4);color:#34D399;font-size:0.75rem;font-weight:800;padding:5px 12px;border-radius:99px;cursor:pointer">📋 Copy to share</button>`;
    bar.innerHTML = `<span>🎉 <strong>${chapterName}</strong> done! Mentor score: <strong style="color:#34D399">${mentorScore}/100</strong> — share your progress!</span><div style="display:flex;gap:8px;align-items:center">${copyBtn}<button onclick="document.getElementById('shareMomentBar').remove()" style="background:none;border:none;color:var(--muted,rgba(240,239,244,0.4));cursor:pointer;font-size:1rem">✕</button></div>`;
    document.body.appendChild(bar);
    setTimeout(() => { if (document.getElementById('shareMomentBar')) bar.remove(); }, 12000);
}

function checkBadges(score, gap, _chap, analysisCount) {
    if (score !== undefined) {
        _awardBadge('chapter_done');
        if (score >= 80) _awardBadge('high_scorer');
    }
    if (analysisCount >= 3) _awardBadge('iterative');
    if (gap !== undefined && Math.abs(gap) < 5) _awardBadge('aligned_ai');
}

// ══════════════════════════════════════════════════════════
//  BIG CONFETTI — fires for scores ≥ 90 (more dots + trophy toast)
// ══════════════════════════════════════════════════════════
function launchBigConfetti(score) {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const _cols = ['#A78BFA','#34D399','#FFD93D','#F87171','#60A5FA','#FB923C'];
    for (let _i = 0; _i < 140; _i++) {
        const _el = document.createElement('div');
        const _sz = (8 + Math.random() * 8) + 'px';
        _el.style.cssText = `position:fixed;top:-10px;left:${Math.random()*100}vw;`
            + `width:${_sz};height:${_sz};`
            + `border-radius:${Math.random() > 0.5 ? '50%' : '3px'};`
            + `background:${_cols[Math.floor(Math.random() * _cols.length)]};`
            + `pointer-events:none;z-index:99999;`
            + `animation:confettiFall ${1.2 + Math.random() * 2}s linear forwards;`
            + `animation-delay:${Math.random() * 0.8}s`;
        document.body.appendChild(_el);
        setTimeout(() => _el.remove(), 3500);
    }
    const toast = document.createElement('div');
    toast.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%) scale(0.8);z-index:99001;'
        + 'background:linear-gradient(135deg,rgba(52,211,153,0.3),rgba(167,139,250,0.25));'
        + 'border:2px solid rgba(52,211,153,0.5);border-radius:20px;padding:20px 36px;'
        + 'text-align:center;font-family:Nunito,sans-serif;opacity:0;transition:all 0.3s;pointer-events:none';
    toast.innerHTML = `<div style="font-size:2.8rem">🏆</div>`
        + `<div style="font-size:1.2rem;font-weight:900;color:#34D399;margin:6px 0">Outstanding!</div>`
        + `<div style="font-size:0.88rem;color:rgba(240,239,244,0.85)">Mentor Score: <strong>${score}/100</strong></div>`;
    document.body.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '1'; toast.style.transform = 'translate(-50%,-50%) scale(1)'; }, 10);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 400); }, 3200);
}

// ══════════════════════════════════════════════════════════
//  GAP EXPLAINER — injects a <details> block explaining where
//  the AI was too generous vs the Mentor. Call after analysis.
//  containerId: id of an empty div placed near rsGap
// ══════════════════════════════════════════════════════════
function renderGapExplainer(containerId, pureTruth, hybridTruth, violations) {
    const el = document.getElementById(containerId);
    if (!el) return;
    const gap = (pureTruth || 0) - (hybridTruth || 0);
    if (gap < 10) { el.style.display = 'none'; el.innerHTML = ''; return; }
    // Show errors first, then warnings — context chapter rules are all warnings so
    // filtering to errors-only produced an empty list and a misleading fallback message.
    const allViols = (violations || []);
    const errors   = allViols.filter(v => v.severity === 'error');
    const warnings = allViols.filter(v => v.severity !== 'error');
    const topViols = (errors.length ? errors : warnings).slice(0, 3);
    const ruleList = topViols.map(v => {
        const icon = v.severity === 'error' ? '📌' : '🟡';
        return `<li style="margin:4px 0;line-height:1.55">${icon} ${v.message}</li>`;
    }).join('');
    const bodyText = topViols.length
        ? `The Mentor scored these areas lower than the AI:<ul style="margin:6px 0 0;padding-left:16px">${ruleList}</ul>`
        : `No specific rule violations fired — the gap is because the Mentor model is naturally more measured than the AI Optimist. A gap under 15 points with no rule violations is normal and means your plan is broadly sound.`;
    el.innerHTML = `<details style="margin-top:8px">
        <summary style="cursor:pointer;font-size:0.72rem;font-weight:800;color:#FB923C;list-style:none;
            padding:6px 10px;background:rgba(251,146,60,0.08);border:1px solid rgba(251,146,60,0.2);
            border-radius:8px;display:flex;align-items:center;gap:6px">
            🤖 Where the AI was too kind (+${gap} pts) ▾
        </summary>
        <div style="margin-top:6px;padding:8px 12px;background:rgba(251,146,60,0.05);
            border:1px solid rgba(251,146,60,0.15);border-radius:8px;font-size:0.72rem;
            color:rgba(240,239,244,0.82);line-height:1.6">
            The AI Optimist gave you <strong style="color:#FB923C">${pureTruth}</strong>,
            but your Mentor gave <strong style="color:#A78BFA">${hybridTruth}</strong> — a ${gap}-point gap.
            ${bodyText}
        </div>
    </details>`;
    el.style.display = '';
}

// ══════════════════════════════════════════════════════════
//  C7-45 — ANALYSIS WAIT PROGRESS SEQUENCE
//  Shows animated step progress while fetch is in flight.
//  Call showAnalysisProgress(buttonEl) before fetch,
//  call clearAnalysisProgress(buttonEl) when done.
// ══════════════════════════════════════════════════════════
const _ANALYSIS_STEPS = ['🐍 Checking rules…', '🤖 Asking AI Optimist…', '🧠 Getting Mentor score…'];
let _analysisProgressTimer = null;
let _analysisProgressEl = null;

function showAnalysisProgress(_anchorEl) {
    clearAnalysisProgress();
    _analysisProgressEl = document.createElement('div');
    _analysisProgressEl.id = 'analysisProgressSeq';
    _analysisProgressEl.style.cssText = 'position:fixed;bottom:24px;left:50%;transform:translateX(-50%);z-index:9200;background:var(--s2,#1C1A30);border:1px solid rgba(167,139,250,0.3);border-radius:12px;padding:10px 18px;font-family:Nunito,sans-serif;font-size:0.82rem;font-weight:700;color:#A78BFA;display:flex;align-items:center;gap:10px;box-shadow:0 6px 20px rgba(0,0,0,0.4);min-width:220px;justify-content:center';
    _analysisProgressEl.innerHTML = `<span id="apSpinner" style="display:inline-block;animation:spin 0.8s linear infinite">⚙️</span><span id="apStepText">${_ANALYSIS_STEPS[0]}</span>`;
    if (!document.getElementById('apSpinStyle')) {
        const st = document.createElement('style');
        st.id = 'apSpinStyle';
        st.textContent = '@keyframes spin{to{transform:rotate(360deg)}}';
        document.head.appendChild(st);
    }
    document.body.appendChild(_analysisProgressEl);
    let idx = 0;
    _analysisProgressTimer = setInterval(() => {
        idx = (idx + 1) % _ANALYSIS_STEPS.length;
        const t = document.getElementById('apStepText');
        if (t) t.textContent = _ANALYSIS_STEPS[idx];
    }, 2200);
}

function clearAnalysisProgress() {
    if (_analysisProgressTimer) { clearInterval(_analysisProgressTimer); _analysisProgressTimer = null; }
    if (_analysisProgressEl) { _analysisProgressEl.remove(); _analysisProgressEl = null; }
    const existing = document.getElementById('analysisProgressSeq');
    if (existing) existing.remove();
}

// ══════════════════════════════════════════════════════════
//  C6-39 — IMPROVEMENT CELEBRATION
//  Celebrates score jumps ≥ 15 pts more than confetti for
//  a 72 on first try
// ══════════════════════════════════════════════════════════
function checkImprovementCelebration(module, newScore) {
    const prevKey = 'lrnbiz_prev_score_' + module;
    const prevScore = parseInt(localStorage.getItem(prevKey) || '0');
    const jump = newScore - prevScore;
    localStorage.setItem(prevKey, String(newScore));
    if (jump >= 15 && prevScore > 0) {
        // Big improvement — show special banner
        const banner = document.createElement('div');
        banner.style.cssText = 'position:fixed;top:72px;left:50%;transform:translateX(-50%);z-index:99001;background:linear-gradient(135deg,#FFD93D,#A78BFA);color:#09090F;border-radius:12px;padding:10px 20px;font-family:Nunito,sans-serif;font-weight:900;font-size:0.9rem;opacity:0;transition:opacity 0.4s;text-align:center;max-width:320px;width:90%';
        banner.textContent = `🚀 +${jump} points! Amazing improvement!`;
        document.body.appendChild(banner);
        setTimeout(() => { banner.style.opacity = '1'; }, 10);
        setTimeout(() => { banner.style.opacity = '0'; setTimeout(() => banner.remove(), 500); }, 3500);
        SoundFX.win();
    }
}

// ══════════════════════════════════════════════════════════
//  S21-23 — WHY THIS SCORE?
//  Renders the top-issues summary above the mentor text.
//  Called from every chapter's analysis response handler.
// ══════════════════════════════════════════════════════════
function renderWhyScore(topIssues) {
    const el = document.getElementById('whyScoreBox');
    if (!el) return;
    if (!topIssues || topIssues.length === 0) {
        el.innerHTML = '<div style="font-size:0.76rem;color:#34D399">✅ No major issues — your plan looks solid!</div>';
    } else {
        el.innerHTML =
            '<div style="font-size:0.68rem;font-weight:900;text-transform:uppercase;letter-spacing:0.07em;color:rgba(240,239,244,0.45);margin-bottom:5px">Why this score?</div>' +
            topIssues.map(i =>
                `<div style="font-size:0.76rem;line-height:1.5;color:${i.severity === 'error' ? '#F87171' : '#FCD34D'};margin-bottom:3px">${i.emoji} ${i.title}</div>`
            ).join('');
    }
    el.style.display = '';
}

// ══════════════════════════════════════════════════════════
//  PULSE RADAR
//  Animates a Chart.js radar from zero → real values so
//  the spider always visibly moves after each analysis,
//  even when values haven't changed.
//
//  chartInstance — the Chart.js radar object
//  dataArray     — numeric array aligned to chart labels
//  colorFn       — optional function(dataArray) → colors array
// ══════════════════════════════════════════════════════════
function pulseRadar(chartInstance, dataArray, colorFn) {
    if (!chartInstance || !dataArray) return;
    const zeros = dataArray.map(() => 0);
    // Step 1 — collapse to zero instantly (no animation)
    chartInstance.data.datasets[0].data = zeros;
    chartInstance.update('none');
    // Step 2 — expand to real values with the chart's built-in animation
    setTimeout(() => {
        chartInstance.data.datasets[0].data = [...dataArray];
        if (colorFn) chartInstance.data.datasets[0].pointBackgroundColor = colorFn(dataArray);
        chartInstance.update('active');
    }, 80);
}

