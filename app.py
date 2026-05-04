"""
lrnbiz/app.py
=============
Flask backend for LrnBiz — an AI-powered Intelligent Tutoring System (ITS) for young
entrepreneurs. Students work through five sequential chapters, each building on the last:

  Chapter 1  /context    Personal context — budget, grade, location, business type
  Chapter 2  /idea       Business idea — niche validation with rules + AI
  Chapter 3  /customer   Target customer — persona builder with alignment checks
  Chapter 4  /money      Money Math — pricing, costs, break-even, profit modelling
  Chapter 5  /discovery  Customer discovery — interview evidence quality check

─────────────────────────────────────────────────────────────────────────────────────────
TRIPLE TRUTH SCORING  (the core academic contribution)
─────────────────────────────────────────────────────────────────────────────────────────
Each chapter produces three simultaneous scores:

  1. Symbolic Score (Rules Check)
     A deterministic rules engine reads JSON rule files and fires matching rules against
     the student's answers. Each rule has a severity (error/warning) and a score_impact.
     Score starts at 100; each violation subtracts points. Always consistent — no AI.

  2. Pure LLM Score (AI Optimist)
     llama-3.1-8b-instant with a blindly positive system prompt. Ignores all problems,
     only praises. Self-reports its own score (SCORE: N line in response). This score
     is the "sycophancy baseline" — what a naive AI would tell the student.

  3. Hybrid Score (Mentor)
     llama-3.1-8b-instant given the triggered rule violations as context. Uses the
     Socratic method: asks ONE question targeting the most critical issue. Self-reports
     MENTOR_SCORE: N. This score represents honest, rule-grounded feedback.

  Sycophancy Gap = AI Optimist − Mentor Score
  A positive gap means the AI was over-enthusiastic relative to honest rules-based feedback.
  Students can see this gap — it teaches them that AI encouragement ≠ reliable advice.

─────────────────────────────────────────────────────────────────────────────────────────
KEY DESIGN DECISIONS
─────────────────────────────────────────────────────────────────────────────────────────
- Filesystem sessions (flask_sessions/) rather than cookies: a completed student session
  contains ~8–12 KB of data (LLM text + violations + score history), which exceeds the
  4 KB browser cookie limit and would be silently truncated.

- Rules loaded once at startup via functools.lru_cache: JSON files are read once per
  server process and cached. Editing a rule file requires a server restart.

- Both LLMs use temperature=0: consistency is more important than variety for a scoring
  system. A student re-running analysis on the same answers should get the same score.

- The Hybrid Mentor is explicitly told to ONLY address issues in the triggered rules list
  and NOT to invent new concerns. This prevents the LLM from hallucinating problems that
  the deterministic rules did not detect.

- Two different LLM model sizes: the small model (8b) simulates a naive AI reviewer;
  the large model (70b) produces more nuanced Socratic questions. The performance gap
  between them is the pedagogical point of the sycophancy measurement.

─────────────────────────────────────────────────────────────────────────────────────────
FILE STRUCTURE  (key sections by approximate line number)
─────────────────────────────────────────────────────────────────────────────────────────
  Lines   1–110   Imports, .env loader, Groq client init, Flask app setup, session config
  Lines 110–165   Rule engine: _eval_cond(), compute_health_scores()
  Lines 165–230   Scoring helpers: sentiment scorer, LLM score parser, fallback formula
  Lines 230–320   ModuleAuditor: triple_truth() — combines all three scores
  Lines 320–490   Utility functions: rate limiter, radar updater, session save, research log
  Lines 490–730   Module registry, context processor, LLM call functions
  Lines 730–790   Card lock rules (client-side UI locking), chapter guard
  Lines 790+      Flask routes: one per chapter + teacher + admin + API endpoints
"""

import json
import os
import datetime
import functools
import threading
import concurrent.futures
import secrets
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, flash, Response

# ── .env loader ───────────────────────────────────────────────
def _load_dotenv():
    """Load all KEY=VALUE pairs from .env next to this file into os.environ.
    Does not overwrite variables already set in the environment."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key = key.strip()
            val = val.strip()
            if key and key not in os.environ:
                os.environ[key] = val

_load_dotenv()


def _load_env_key():
    """Return GROQ_API_KEY from environment (already populated by _load_dotenv)."""
    return os.environ.get("GROQ_API_KEY", "")

try:
    from groq import Groq
    _groq_key = _load_env_key()
    _groq_client = Groq(api_key=_groq_key) if _groq_key else None
    GROQ_AVAILABLE = _groq_client is not None
except Exception as e:
    _groq_client = None
    GROQ_AVAILABLE = False
    _groq_init_error = e  # stored for post-app-init logging


def _groq_create(**kwargs):
    """Wrapper around groq chat completions with one automatic 429 retry.
    Groq free tier allows 30 req/min — eval runs 10 calls per persona, so
    rapid consecutive runs hit the cap. One 10-second sleep + retry recovers."""
    import time
    try:
        return _groq_client.chat.completions.create(**kwargs)
    except Exception as e:
        if '429' in str(e) or 'rate_limit' in str(e).lower() or 'rate limit' in str(e).lower():
            time.sleep(10)
            return _groq_client.chat.completions.create(**kwargs)
        raise


app = Flask(__name__)
_secret = os.environ.get('LRNBIZ_SECRET_KEY')
if not _secret:
    raise RuntimeError(
        "LRNBIZ_SECRET_KEY environment variable is not set. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\" "
        "and add it to your .env file."
    )
app.secret_key = _secret

# Server-side filesystem sessions — avoids the ~4KB cookie limit that silently
# drops audit_scores when a student completes all 5 chapters (LLM text + violations
# + score history fills the cookie and the browser truncates it).
from flask_session import Session as FlaskSession
_SESSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flask_sessions')
os.makedirs(_SESSION_DIR, exist_ok=True)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = _SESSION_DIR
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
FlaskSession(app)

# C6-40: Share tokens persisted to disk so they survive server restarts
_teacher_shares_lock = threading.Lock()
_SHARE_TOKENS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'share_tokens.json')

def _load_share_tokens():
    try:
        if os.path.exists(_SHARE_TOKENS_PATH):
            with open(_SHARE_TOKENS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_share_tokens(tokens: dict):
    try:
        with open(_SHARE_TOKENS_PATH, 'w', encoding='utf-8') as f:
            json.dump(tokens, f, indent=2)
    except Exception:
        pass

_teacher_shares = _load_share_tokens()

# Log Groq initialisation outcome now that app.logger is available
with app.app_context():
    app.logger.debug(f"[INIT] app file: {__file__}")
    app.logger.debug(f"[INIT] Groq key present: {'YES' if (_groq_client is not None) else 'NO'}")
    app.logger.debug(f"[INIT] GROQ_AVAILABLE={GROQ_AVAILABLE}")
    if not GROQ_AVAILABLE and '_groq_init_error' in dir():
        app.logger.warning(f"[INIT] Groq client failed to initialise: {_groq_init_error}")


# ══════════════════════════════════════════════════════════════
#  JSON DATA LOADERS  (cached for the server process lifetime
#  via lru_cache — editing a JSON rule file requires a server
#  restart for changes to take effect)
# ══════════════════════════════════════════════════════════════
@functools.lru_cache(maxsize=None)
def _load_json(filename: str) -> dict:
    path = os.path.join(app.root_path, filename)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
#  RULE ENGINE
#  Mirrors the client-side evalCond() in JS for server-side use.
# ══════════════════════════════════════════════════════════════
def _eval_cond(condition: dict, answers: dict) -> bool:
    if 'all' in condition:
        return all(_eval_cond(c, answers) for c in condition['all'])
    if 'any' in condition:
        return any(_eval_cond(c, answers) for c in condition['any'])
    field = condition.get('field', '')
    v = answers.get(field)
    if 'eq'  in condition: return v == condition['eq']
    if 'neq' in condition: return v != condition['neq'] and v is not None
    if 'lt'  in condition: return float(v or 0) < condition['lt']
    if 'gt'  in condition: return float(v or 0) > condition['gt']
    if 'lte' in condition: return float(v or 0) <= condition['lte']
    if 'gte' in condition: return float(v or 0) >= condition['gte']
    if 'in'  in condition: return v in condition['in']
    return False


def _extract_fields(condition: dict) -> list:
    """Recursively extract every field name referenced in a condition."""
    if 'all' in condition:
        return [f for c in condition['all'] for f in _extract_fields(c)]
    if 'any' in condition:
        return [f for c in condition['any'] for f in _extract_fields(c)]
    return [condition['field']] if 'field' in condition else []


def compute_health_scores(answers: dict, rules: list) -> tuple:
    """
    Apply all rules whose required fields are present in answers.
    Returns:
        scores   – dict of {axis: 0-100}
        triggered – list of rules that fired
    Health axes: Gold (budget), Energy (hustle), Influence (reach),
                 Knowledge, Target (focus), Passion
    """
    scores = {k: 100 for k in ('Gold', 'Energy', 'Influence', 'Knowledge', 'Target', 'Passion')}
    triggered = []

    for rule in rules:
        required_fields = list(set(_extract_fields(rule.get('condition', {}))))
        # Only fire if every required field has a non-empty answer
        if not all(answers.get(f) not in (None, '') for f in required_fields):
            continue
        if _eval_cond(rule['condition'], answers):
            triggered.append(rule)
            for axis, delta in rule.get('score_impact', {}).items():
                if axis in scores:
                    scores[axis] = max(0, min(100, scores[axis] + delta))

    return scores, triggered


# ══════════════════════════════════════════════════════════════
#  SENTIMENT SCORING  (keyword heuristic, 0–100)
# ══════════════════════════════════════════════════════════════
_POS_WORDS = [
    'great','excellent','amazing','wonderful','fantastic','perfect','love','awesome',
    'brilliant','outstanding','incredible','superb','best','good','positive','strong',
    'success','win','thrive','grow','achieve','excited','passionate','confident',
    'powerful','innovative','creative','opportunity','potential','congratulations',
    'impressive','exceptional','remarkable','magnificent','phenomenal',
]
_NEG_WORDS = [
    # Genuine negative signals — rules violations, serious gaps
    'risk','problem','issue','difficult','conflict','warn',
    'doubt','worry','struggle','fail','lose','bad',
    'wrong','mistake','error','gap','missing','lack','poor',
    'unfortunately','unclear','inconsistent','not viable','not clear',
    # Note: 'consider','question','challenge','careful','caution','however','but'
    # are intentionally excluded — a mentor asking good Socratic questions
    # should not be penalised as "negative". Those are coaching, not criticism.
]

def compute_sentiment_score(text: str) -> int:
    """Returns 0–100 representing how positive/approving the text is.
    Used only for hybrid (Mentor) score calibration — NOT for pure_truth."""
    t = text.lower()
    pos = sum(1 for w in _POS_WORDS if w in t)
    neg = sum(1 for w in _NEG_WORDS if w in t)
    total = pos + neg
    if total == 0:
        return 50
    return min(100, max(0, int(pos / total * 100)))


def _parse_llm_score(text: str) -> tuple:
    """Extract the SCORE line from a pure LLM response.
    Returns (clean_text, score_int).
    The pure LLM is asked to self-report its score — this gives us the real
    sycophancy measurement rather than a simulated formula.
    Falls back to None if no valid score line found.
    """
    import re as _re_s
    match = _re_s.search(r'\bSCORE:\s*(\d{1,3})', text, _re_s.IGNORECASE)
    if match:
        score = min(100, max(0, int(match.group(1))))
        clean = _re_s.sub(r'\n*\bSCORE:\s*\d{1,3}', '', text, flags=_re_s.IGNORECASE).strip()
        return clean, score
    return text.strip(), None


def _strip_mentor_score_line(text: str) -> str:
    """Remove the MENTOR_SCORE line from hybrid text before showing to students."""
    import re as _re_m
    return _re_m.sub(r'\n*MENTOR_SCORE:\s*\d{1,3}', '', text, flags=_re_m.IGNORECASE).strip()


def _ai_optimist_score(symbolic_score: int) -> int:
    """Fallback-only formula used when the LLM score cannot be parsed.
    Should rarely trigger in production — the pure LLM functions now ask
    the LLM to self-report its score directly.
    """
    if symbolic_score < 25:
        return min(92, max(35, round(35 + symbolic_score * 0.8)))
    return min(92, max(55, round(55 + symbolic_score * 0.4)))


# ══════════════════════════════════════════════════════════════
#  MODULE AUDITOR  — universal, applies to every ITS stage
#
#  Each module has three pillars:
#    1. Symbolic  — deterministic rules engine (ground truth, 0–100)
#    2. Pure LLM  — blind optimist, no rule context
#    3. Hybrid    — LLM + rules context (Socratic mentor)
#
#  Truth score = 100 – |symbolic_score – llm_sentiment|
#  Sycophancy gap = pure_truth – hybrid_truth
#    (negative = LLM over-validates relative to hybrid)
# ══════════════════════════════════════════════════════════════
class ModuleAuditor:
    """Drop-in auditor for any module. Call triple_truth() to get all scores."""

    def symbolic_from_violations(self, violations: list, total_rules: int = 0,  # total_rules reserved for future normalisation
                                  error_weight: int = 25,
                                  warning_weight: int = 10) -> int:
        """Convert a violations list into a 0–100 symbolic health score."""
        errors   = sum(1 for v in violations if v.get('severity') == 'error')
        warnings = sum(1 for v in violations if v.get('severity') == 'warning')
        score    = 100 - errors * error_weight - warnings * warning_weight
        return max(0, min(100, score))

    def triple_truth(self, symbolic_score: int,
                     pure_text: str, hybrid_text: str,
                     module_id: str) -> dict:
        """Compute all three truth scores and sycophancy gap.
        pure_truth is the score the LLM self-reported inside pure_text (SCORE: N line).
        This makes the sycophancy gap a real measurement, not a simulated formula.
        Falls back to _ai_optimist_score() only if the LLM omitted the score line.
        """
        # Extract LLM-reported scores — both LLMs now self-report a score line
        _clean_pure, _llm_score = _parse_llm_score(pure_text)
        pure_truth = _llm_score if _llm_score is not None else _ai_optimist_score(symbolic_score)

        # Parse MENTOR_SCORE from hybrid text
        import re as _re_h
        _mentor_match = _re_h.search(r'MENTOR_SCORE:\s*(\d{1,3})', hybrid_text, _re_h.IGNORECASE)
        if _mentor_match:
            _parsed_hybrid = max(0, min(100, int(_mentor_match.group(1))))
        else:
            # Fallback to sentiment if LLM didn't emit the score line
            hybrid_sentiment = compute_sentiment_score(hybrid_text)
            _raw_hybrid = round(100 - abs(symbolic_score - hybrid_sentiment))
            _parsed_hybrid = max(0, min(symbolic_score + 20, min(100, _raw_hybrid)))

        # hybrid_truth uses the raw LLM score so the sycophancy gap
        # remains a genuine LLM-vs-LLM measurement, not LLM-vs-rules.
        hybrid_truth = _parsed_hybrid

        # ── Score banding ────────────────────────────────────────────
        # Band labels are computed from a rules-anchored version of the
        # score (clamped to symbolic ±15) so small LLM drift between
        # runs (e.g. 71→65) doesn't visibly change the student's band.
        # The raw numeric scores and gap are untouched — only the
        # display label is stabilised.
        _ANCHOR_WINDOW  = 15
        _hybrid_anchored = max(symbolic_score - _ANCHOR_WINDOW,
                               min(symbolic_score + _ANCHOR_WINDOW, hybrid_truth))

        def _score_band(n: int) -> dict:
            if n >= 81: return {'label': 'Strong',     'color': '#34D399'}
            if n >= 61: return {'label': 'Good',        'color': '#A78BFA'}
            if n >= 41: return {'label': 'Developing',  'color': '#FFD93D'}
            if n >= 21: return {'label': 'Fair',        'color': '#FB923C'}
            return             {'label': 'Poor',        'color': '#F87171'}

        # Band from anchored score (stable label); pure band from raw score
        mentor_band = _score_band(_hybrid_anchored)
        pure_band   = _score_band(pure_truth)

        mentor_score_note = None
        gap = symbolic_score - hybrid_truth
        if hybrid_truth < 50:
            mentor_score_note = (
                "Mentor score is low because the feedback contains significant concerns "
                "about your plan — review the mentor comments above and address the gaps."
            )
        elif gap > 20:
            mentor_score_note = (
                f"Your rules score ({symbolic_score}) is higher than your Mentor score ({hybrid_truth}). "
                "The mentor found issues in your plan that the rules alone don't fully capture — "
                "read the mentor feedback carefully."
            )

        return {
            'module_id':           module_id,
            'symbolic_score':      symbolic_score,
            'pure_truth':          pure_truth,
            'pure_text_clean':     _clean_pure,
            'hybrid_truth':        hybrid_truth,
            'hybrid_text_clean':   _strip_mentor_score_line(hybrid_text),
            'sycophancy_gap':      pure_truth - hybrid_truth,
            'pure_score_source':   'llm' if _llm_score is not None else 'fallback',
            'mentor_score_source': 'llm' if _mentor_match else 'fallback',
            'mentor_score_note':   mentor_score_note,
            'mentor_band':         mentor_band,
            'pure_band':           pure_band,
        }


_auditor = ModuleAuditor()

_research_log_lock = threading.Lock()


# ── S21-23: top-issues helper ─────────────────────────────────
def _top_issues_from_violations(violations: list, n: int = 3) -> list:
    """Return the n highest-priority violations for the 'Why this score?' summary.
    Errors before warnings; within each tier, preserve original order."""
    errors   = [v for v in violations if v.get('severity') == 'error']
    warnings = [v for v in violations if v.get('severity') != 'error']
    top = (errors + warnings)[:n]
    return [
        {
            'title':    v.get('message', v.get('id', ''))[:120],
            'emoji':    '📌' if v.get('severity') == 'error' else '🟡',
            'severity': v.get('severity', 'warning'),
        }
        for v in top
    ]


# ── Admin HTTP Basic Auth ─────────────────────────────────────
def _require_admin(f):
    """Decorator that enforces HTTP Basic Auth using the ADMIN_PASSWORD env var.
    Parses the Authorization header directly for Flask 2/3 compatibility."""
    import base64 as _b64
    @functools.wraps(f)
    def _wrapped(*args, **kwargs):
        admin_pw = os.environ.get('LRNBIZ_ADMIN_PASSWORD', '')
        if not admin_pw:
            return Response('Admin access disabled — set LRNBIZ_ADMIN_PASSWORD env var.', 503)
        _401 = Response('Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="LrnBiz Admin"'})
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Basic '):
            return _401
        try:
            decoded = _b64.b64decode(auth_header[6:]).decode('utf-8', errors='replace')
            _, _, password = decoded.partition(':')
        except Exception:
            return _401
        if not secrets.compare_digest(password, admin_pw):
            return _401
        return f(*args, **kwargs)
    return _wrapped


def _check_llm_rate_limit(max_calls: int = 10) -> int:
    """C6-41: Per-session rate limit — max_calls per minute across all modules.
    Returns 0 if the call is allowed, or the seconds remaining until the window
    resets (>0) if the limit is already reached. Callers show the wait time."""
    now = datetime.datetime.now(datetime.timezone.utc)
    window_key = '_llm_window_start'
    calls_key  = '_llm_calls_total'
    window_start = session.get(window_key)
    elapsed = 0.0
    if window_start:
        try:
            elapsed = (now - datetime.datetime.fromisoformat(window_start)).total_seconds()
        except (ValueError, TypeError):
            elapsed = 61.0
    # Reset counter if more than 60s have passed
    if not window_start or elapsed > 60:
        session[window_key] = now.isoformat()
        session[calls_key] = 0
        elapsed = 0.0
    count = session.get(calls_key, 0)
    if count >= max_calls:
        return max(1, int(60 - elapsed))
    session[calls_key] = count + 1
    session.modified = True
    return 0


def _safe_float(val, default=0):
    """Convert a value to float safely, returning default on failure."""
    try:
        return float(val) if val not in (None, '') else default
    except (ValueError, TypeError):
        return default


def _update_chapter_radar(axis: str, hybrid_truth: int) -> dict:
    """Boost a chapter-specific radar axis based on the chapter hybrid score.
    Returns the updated radar dict (also saved to session)."""
    _default_radar = {'Passion': 0, 'Energy': 0, 'Gold': 0, 'Influence': 0, 'Knowledge': 0, 'Target': 0}
    radar = dict(session.get('radar_scores', _default_radar) or _default_radar)
    boost = int(hybrid_truth * 0.85)
    radar[axis] = max(radar.get(axis, 0), boost)
    session['radar_scores'] = radar
    session.modified = True
    return radar


def _save_audit_with_delta(module_id: str, triple: dict, violations: list = None):
    """Save triple truth to session. On first run sets hybrid_truth_initial;
    subsequent runs preserve it so we can track improvement from Socratic hints.
    Optionally saves the list of violated rule messages for tooltip display.
    Maintains score_history (up to 10 entries) for compare-attempts view."""
    triple['analysed_at'] = datetime.datetime.now().strftime('%d %b %Y, %H:%M')
    triple['scored_at']   = datetime.datetime.now(datetime.timezone.utc).isoformat()
    existing = session.get('audit_scores', {}).get(module_id, {})
    triple['hybrid_truth_initial'] = existing.get('hybrid_truth_initial', triple['hybrid_truth'])
    # Reset baseline only if score dropped dramatically — threshold 50 prevents accidental
    # form submissions from permanently overwriting a student's progress record.
    if triple['hybrid_truth'] < triple['hybrid_truth_initial'] - 50:
        triple['hybrid_truth_initial'] = triple['hybrid_truth']
        triple['baseline_was_reset'] = True
    triple['hybrid_truth_final']   = triple['hybrid_truth']
    triple['hint_delta']           = triple['hybrid_truth_final'] - triple['hybrid_truth_initial']
    if violations is not None:
        triple['violated_rules'] = [
            {'message': v.get('message', v.get('id', '')), 'severity': v.get('severity', 'warning')}
            for v in violations if v.get('message') or v.get('id')
        ]
    # O15-16: Maintain score history for compare-attempts view (keep last 10 attempts)
    history = existing.get('score_history', [])
    history.append({'score': triple['hybrid_truth'], 'at': triple['analysed_at']})
    triple['score_history'] = history[-10:]
    session.setdefault('audit_scores', {})[module_id] = triple
    session.modified = True


def save_research_log(module_id: str, symbolic_score: int, violations: list,
                      pure_llm_text: str, pure_truth: int,
                      hybrid_llm_text: str, hybrid_truth: int,
                      sycophancy_gap: int, post_discovery: bool = False):
    """Append a structured research entry to researchlogs.json for analysis."""
    log_path = os.path.join(app.root_path, 'researchlogs.json')
    # C8-15: separate CC flags from NR flags for research analysis
    cc_flags = [v['id'] for v in violations if v.get('id', '').startswith('CC')]
    nr_flags = [v['id'] for v in violations if not v.get('id', '').startswith('CC')]
    entry = {
        'timestamp':       datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'module_id':       module_id,
        'symbolic_score':  symbolic_score,
        'violations':      violations,
        'cc_flags':        cc_flags,
        'nr_flags':        nr_flags,
        'pure_llm_text':   pure_llm_text,
        'pure_truth':      pure_truth,
        'hybrid_llm_text': hybrid_llm_text,
        'hybrid_truth':    hybrid_truth,
        'sycophancy_gap':  sycophancy_gap,
        'post_discovery':  post_discovery,
    }
    try:
        with _research_log_lock:
            logs = []
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            logs.append(entry)
            # C6-42: Cap at 500 records to prevent unbounded growth
            if len(logs) > 500:
                logs = logs[-500:]
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2)
    except Exception as _log_err:
        app.logger.warning(f'[save_research_log] Failed to write to researchlogs.json: {_log_err}')

    # Keep only last 3 log entries per module to prevent session bloat
    try:
        log = session.get('research_log', [])
        module_entries = [e for e in log if e.get('module') == module_id]
        other_entries  = [e for e in log if e.get('module') != module_id]
        if len(module_entries) > 3:
            module_entries = module_entries[-3:]
        session['research_log'] = other_entries + module_entries
        session.modified = True
    except Exception:
        pass  # Never crash on session trimming


# Module metadata — 5-chapter ITS journey
MODULE_REGISTRY = [
    {'id': 'context',   'name': 'Context',             'emoji': '👤', 'route': '/context',   'chapter': 1},
    {'id': 'idea',      'name': 'Business Idea',       'emoji': '💡', 'route': '/idea',      'chapter': 2},
    {'id': 'customer',  'name': 'Target Customer',     'emoji': '🎯', 'route': '/customer',  'chapter': 3},
    {'id': 'money',     'name': 'Money Math',          'emoji': '💰', 'route': '/money',     'chapter': 4},
    {'id': 'discovery', 'name': 'Customer Discovery',  'emoji': '🔍', 'route': '/discovery', 'chapter': 5},
]

# Injected into every template via context processor
CHAPTERS = MODULE_REGISTRY


@app.context_processor
def inject_globals():
    """Make chapters + audit_scores + radar available in every template."""
    audit_scores = session.get('audit_scores', {})
    _default_radar = {'Passion': 0, 'Energy': 0, 'Gold': 0, 'Influence': 0, 'Knowledge': 0, 'Target': 0}
    base_radar   = session.get('radar_scores', _default_radar) or _default_radar
    # Boost radar axes as student completes each module
    boosted = dict(base_radar)
    if True:
        if 'idea'      in audit_scores: boosted['Target']    = min(100, boosted.get('Target',    0) + 15)
        if 'customer'  in audit_scores: boosted['Influence'] = min(100, boosted.get('Influence', 0) + 15)
        if 'money'     in audit_scores: boosted['Gold']      = min(100, boosted.get('Gold',      0) + 15)
        if 'discovery' in audit_scores: boosted['Knowledge'] = min(100, boosted.get('Knowledge', 0) + 15)
    return {
        'chapters':          CHAPTERS,
        'audit_scores':      audit_scores,
        'radar_scores_json': json.dumps(boosted),
        'groq_available':    GROQ_AVAILABLE,
    }


# ══════════════════════════════════════════════════════════════
#  LLM CALLS via Groq
# ══════════════════════════════════════════════════════════════
# C6-44: User-friendly offline message when Groq API key is missing/expired
_NO_KEY_MSG = (
    "AI features are currently offline — your Rules Check score is still accurate! 🐍 "
    "(To enable AI feedback, add a valid GROQ_API_KEY to your .env file.)"
    "\nSCORE: 70"
)


def call_context_pure_llm(answers: dict, scores: dict) -> str:
    """Encouraging LLM for the context/background chapter.
    The context chapter contains the student's personal profile (grade, location, experience),
    NOT a business idea — so the prompt evaluates their entrepreneurial ambition and readiness,
    not an idea. This prevents the AI from scoring low because 'there's no idea here'.
    """
    if not GROQ_AVAILABLE:
        return _NO_KEY_MSG
    # Exclude who_talked_to from the AI profile — it's early validation context,
    # not customer data. Sending it raw causes the AI to misread it as "friends = customers."
    _CONTEXT_EXCLUDE = {'who_talked_to', 'biggest_challenge', 'strongest_part'}
    profile = '\n'.join(
        f"  {k}: {v}" for k, v in answers.items()
        if v not in (None, '') and k not in _CONTEXT_EXCLUDE
    )
    try:
        rsp = _groq_create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. A student has shared their entrepreneurial "
                        "background and ambition with you. Respond encouragingly in exactly 2 sentences — "
                        "celebrate their ambition and the exciting possibilities ahead. "
                        "Then on a new line write exactly: SCORE: [0-100] reflecting your excitement "
                        "about this student's entrepreneurial spirit and ambition. "
                        "Any student showing ambition and effort deserves at least 70. "
                        "Reserve 80-90 for students with clear passion, real experience, or a strong idea direction."
                    )
                },
                {
                    "role": "user",
                    "content": f"Student background:\n{profile}\n\nHow excited are you about this student's entrepreneurial journey?"
                }
            ],
            max_tokens=180,
            temperature=0.0
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}\nSCORE: 80"


def call_hybrid_llm(answers: dict, scores: dict, triggered_rules: list) -> str:
    """Hybrid mentor — praises strong plans; asks ONE Socratic question for weak ones.
    This ensures hybrid_truth >= pure_truth for strong plans (sycophancy gap near zero)
    and hybrid_truth > pure_truth for weak plans (gap shows sycophancy detection working)."""
    if not GROQ_AVAILABLE:
        return _NO_KEY_MSG

    # Exclude raw numeric fields so the LLM cannot do its own feasibility math.
    # All feasibility checks belong to the rules engine; the LLM must only address
    # issues explicitly listed in triggered_rules.
    # unit_price always mirrors sell_price (hidden field) — including both makes the LLM
    # think "price == price → no profit", which is wrong. Exclude unit_price.
    _PROFILE_EXCLUDE = {'hours_per_week', 'budget', 'unit_cost', 'startup_cost',
                        'monthly_units', 'unit_price', 'unit_price_gt_unit_cost',
                        'monthly_profit_lt_threshold', 'months_to_breakeven_gt_threshold'}
    profile = '\n'.join(
        f"  {k}: {v}" for k, v in answers.items()
        if v not in (None, '') and k not in _PROFILE_EXCLUDE
    )

    # No conflicts → strong plan. Give specific genuine praise (high sentiment → high hybrid_truth).
    # C7-34: Consistent mentor persona — Sage
    # C7-38: Reference the actual business idea in feedback
    idea_name = answers.get('sell_what') or answers.get('business_type') or answers.get('why_this_idea') or 'your business idea'
    customer_name = answers.get('sell_to') or answers.get('who_talked_to') or 'your customers'

    if not triggered_rules:
        try:
            rsp = _groq_create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Sage, a friendly but honest business mentor who helps school students "
                            "build real businesses. You are warm, direct, and specific — never vague. "
                            "The student's plan has passed all rule checks — no violations were found. "
                            "Write 2-3 sentences of SPECIFIC praise that mentions their actual business idea and customer by name. "
                            "End with one forward-looking tip. End with a period.\n"
                            "Then on a new line write exactly: MENTOR_SCORE: [number 0-100] "
                            "where the number is your honest assessment of how strong this plan is. "
                            "Judge on specificity, clarity, and real-world viability — not just rule compliance. "
                            "A thin but rule-passing plan scores 50-65. A specific, detailed, convincing plan scores 75-90."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Student's business idea: {idea_name}\n"
                            f"Target customer: {customer_name}\n\n"
                            f"Full profile:\n{profile}\n\n"
                            "Give specific praise referencing their actual idea and customer:"
                        )
                    }
                ],
                max_tokens=180,
                temperature=0.0
            )
            return rsp.choices[0].message.content.strip()
        except Exception:
            _fb_score = min(90, max(60, round(scores.get('symbolic', 70) if isinstance(scores.get('symbolic'), (int, float)) else 70)))
            return f"Your idea — {idea_name} — shows real promise! The choices you've made work well together and give you a solid foundation. Keep refining your customer focus and you'll be ready to launch!\nMENTOR_SCORE: {_fb_score}"

    # Has conflicts → ask ONE Socratic question targeting the most severe issue.
    severity_order = {'error': 0, 'warning': 1}
    sorted_rules = sorted(triggered_rules, key=lambda r: severity_order.get(r.get('severity', 'warning'), 2))
    hints_text = '\n'.join(
        f"- [{r['id']}] {r.get('socratic_hint', r.get('message', ''))}"
        for r in sorted_rules
    )
    scores_text = ', '.join(f"{k}: {v}/100" for k, v in scores.items())
    n_errors = sum(1 for r in triggered_rules if r.get('severity') == 'error')

    try:
        rsp = _groq_create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Sage, a warm but intellectually honest Socratic mentor for school-age entrepreneurs. "
                        "CRITICAL RULE: You may ONLY address issues that appear in the 'Rule violations' list below. "
                        "Do NOT invent new concerns. Do NOT calculate time, hours, or profitability yourself — "
                        "the rules engine has already done that. If an issue is not in the violations list, do NOT mention it.\n\n"
                        "When there are violations, use this structure:\n"
                        "1. If there are MULTIPLE violations: name each one in a single short phrase, separated by ' · ' "
                        "(e.g. '400 videos/month is unrealistic · your cost field shows $400/video which looks like a typo'). "
                        "If there is only ONE violation: one sentence naming it with the student's real numbers.\n"
                        "NEVER show formulas — write '$6 profit' not '$(2-0.50)*4=$6'.\n"
                        "2. ONE powerful Socratic question about the most critical issue. End with a question mark.\n"
                        "Total response: 2-4 sentences.\n"
                        "Then on a new line write exactly: MENTOR_SCORE: [number 10-65] "
                        "where the number reflects how ready the plan is (lower = more problems; "
                        "give 10-30 for multiple critical errors; give 35-50 for 2-3 moderate issues; "
                        "give 50-65 for just 1 minor issue)."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Student's business idea: {idea_name}\n"
                        f"Target customer: {customer_name}\n\n"
                        f"Full profile:\n{profile}\n\n"
                        f"Symbolic health scores: {scores_text}\n"
                        f"Rule violations: {len(triggered_rules)} total, {n_errors} critical errors\n\n"
                        f"Most critical issues to address (use the specific numbers/details below):\n{hints_text}\n\n"
                        "Name the most critical issue specifically, then ask your Socratic question:"
                    )
                }
            ],
            max_tokens=220,
            temperature=0.0
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        return f"There are some issues to address with your plan.\nMENTOR_SCORE: 40"


# ══════════════════════════════════════════════════════════════
#  CARD LOCK RULES  (client-side UI locking — stays in Python)
#  These are separate from business_rules.json: they control
#  which vcard options are greyed-out based on slider values,
#  before the student has even submitted anything.
# ══════════════════════════════════════════════════════════════
CARD_LOCK_RULES = [
    {
        "id": "city_needs_budget",
        "condition": {"field": "budget", "lt": 5000},
        "locks": [{"field": "location", "value": "City"}],
        "tooltip": "City markets need permits, stall fees & transport — typically $5,000+ to get started. Raise your budget to unlock City."
    },
    {
        "id": "food_needs_budget",
        "condition": {"field": "budget", "lt": 2000},
        "locks": [{"field": "business_type", "value": "food-beverage"}],
        "tooltip": "Food businesses need ingredients, packaging, hygiene certs & equipment. Set your budget to $2,000+ to unlock Food & Drink."
    },
    {
        "id": "product_needs_budget",
        "condition": {"field": "budget", "lt": 1000},
        "locks": [{"field": "business_type", "value": "physical-product"}],
        "tooltip": "Physical products need materials, manufacturing & packaging. Set budget to $1,000+ to unlock Physical Product."
    },
    {
        "id": "maker_needs_budget",
        "condition": {"field": "budget", "lt": 500},
        "locks": [{"field": "business_type", "value": "maker"}],
        "tooltip": "Custom / Maker businesses need tools, materials & equipment. Set your budget to $500+ to unlock Custom / Maker."
    },
    {
        "id": "resale_needs_budget",
        "condition": {"field": "budget", "lt": 500},
        "locks": [{"field": "business_type", "value": "resale-retail"}],
        "tooltip": "Resale businesses need money to buy initial stock. Set your budget to $500+ to unlock Resale."
    },
    {
        "id": "travel_grade6_locked",
        "condition": {"field": "grade_level", "eq": "Grade 6-8"},
        "locks": [{"field": "delivery_model", "value": "Requires Travel"}],
        "tooltip": "Grade 6-8 students rely on adults for travel. Pick a delivery model you can do independently."
    },
    {
        "id": "online_no_budget",
        "condition": {"field": "budget", "lt": 500},
        "locks": [{"field": "delivery_model", "value": "Online"}],
        "tooltip": "Online selling needs a website, tools & marketing budget. Set budget to $500+ to unlock Online."
    },
    {
        "id": "rural_travel_no_budget",
        "condition": {"all": [
            {"field": "location", "eq": "Rural"},
            {"field": "budget", "lt": 5000}
        ]},
        "locks": [{"field": "delivery_model", "value": "Requires Travel"}],
        "tooltip": "Rural travel-based businesses need a vehicle, fuel budget & logistics. Increase your budget to $5,000+ to unlock."
    },
]


# ══════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════
@app.route('/')
def index():
    radar = dict(session.get('radar_scores', {}))
    audit = dict(session.get('audit_scores', {}))
    weakest = None
    weakest_route = None
    _axis_chapter = {
        'Gold': '/money', 'Energy': '/context', 'Passion': '/context',
        'Influence': '/customer', 'Knowledge': '/idea', 'Target': '/idea'
    }
    if radar and audit:
        weakest = min(radar, key=radar.get)
        weakest_route = _axis_chapter.get(weakest, '/progress')
    return render_template('index.html', weakest=weakest, weakest_route=weakest_route)


@app.route('/reset')
def reset():
    session.clear()
    # Serve a tiny JS page that clears browser sessionStorage + lrnbiz localStorage keys,
    # then redirects to index.  The Flask session cookie is already cleared above.
    _keys = [
        'lrnbiz_draft_idea_text', 'lrnbiz_idea_score', 'lrnbiz_idea_history',
        'lrnbiz_niche_def_seen', 'lrnbiz_idea_rs_open', 'lrnbiz_why_idea',
        'lrnbiz_scores_updated',
    ]
    _ls_clear = ''.join(f"localStorage.removeItem('{k}');" for k in _keys)
    return (
        f'<!doctype html><html><body><script>'
        f'sessionStorage.clear();{_ls_clear}'
        f'window.location="/";</script>'
        f'<noscript><meta http-equiv="refresh" content="0;url=/"></noscript>'
        f'Resetting…</body></html>'
    ), 200


# Chapter order — each chapter requires the previous one to be completed
_CHAPTER_ORDER = ['context', 'idea', 'customer', 'money', 'discovery']

def _chapter_guard(chapter_id: str):
    """Redirect to the first incomplete chapter if prerequisites aren't done.
    Returns a redirect response if blocked, or None if the chapter is accessible."""
    audit_scores = session.get('audit_scores', {})
    idx = _CHAPTER_ORDER.index(chapter_id)
    chapter_names = {m['id']: m['name'] for m in MODULE_REGISTRY}
    for i in range(idx):
        if _CHAPTER_ORDER[i] not in audit_scores:
            first_incomplete = _CHAPTER_ORDER[i]
            needed_name = chapter_names.get(first_incomplete, f'Chapter {i+1}')
            flash(f'Complete "{needed_name}" first before moving on.', 'locked')
            return redirect(url_for(first_incomplete))
    return None


@app.route('/context', methods=['GET', 'POST'])
def context():
    if request.method == 'POST':
        session['grade_level']       = request.form.get('grade_level')
        session['location']          = request.form.get('location')
        session['delivery_model']    = request.form.get('delivery_model')
        session['budget']            = request.form.get('budget')
        session['hours_per_week']    = request.form.get('hours_per_week')
        session['prior_experience']  = request.form.get('prior_experience')
        session['why_this_idea']     = request.form.get('why_this_idea')
        session['business_type']     = _normalise_business_type(request.form.get('business_type', ''))
        session['who_talked_to']     = request.form.get('who_talked_to')
        session['biggest_challenge'] = request.form.get('biggest_challenge')
        session['strongest_part']    = request.form.get('strongest_part')
        session['interest_level']    = request.form.get('interest_level')

        # Ensure context audit scores are always saved even if AJAX analysis was skipped
        if 'context' not in session.get('audit_scores', {}):
            _ctx_answers = {
                'grade_level':      request.form.get('grade_level', ''),
                'location':         request.form.get('location', ''),
                'delivery_model':   request.form.get('delivery_model', ''),
                'budget':           _safe_float(request.form.get('budget')),
                'hours_per_week':   _safe_float(request.form.get('hours_per_week')),
                'prior_experience': request.form.get('prior_experience', ''),
                'why_this_idea':    request.form.get('why_this_idea', ''),
                'business_type':    request.form.get('business_type', ''),
                'who_talked_to':    request.form.get('who_talked_to', ''),
                'biggest_challenge':request.form.get('biggest_challenge', ''),
                'strongest_part':   request.form.get('strongest_part', ''),
                'interest_level':   request.form.get('interest_level', ''),
            }
            try:
                rules = _load_json('business_rules.json')
                scores, triggered = compute_health_scores(_ctx_answers, rules)
                symbolic_avg   = sum(scores.values()) / len(scores) if scores else 50
                symbolic_score = round(symbolic_avg)
                pure_text   = call_context_pure_llm(_ctx_answers, scores)
                hybrid_text = call_hybrid_llm(_ctx_answers, scores, triggered)
                triple = _auditor.triple_truth(symbolic_score, pure_text, hybrid_text, 'context')
                fallback_violations = [
                    {'id': r['id'], 'message': r.get('title', r['id']), 'severity': r.get('severity', 'warning')}
                    for r in triggered
                ]
                _save_audit_with_delta('context', triple, fallback_violations)
                # Also save radar scores so other pages can show the radar
                session['radar_scores'] = {k: min(100, max(0, round(v))) for k, v in scores.items()}
                session.modified = True
            except Exception:
                pass  # Don't block navigation if analysis fails

        # O17-01: If downstream chapters already have scores, mark them as stale so
        # the student knows their earlier analysis may no longer match the new context.
        _existing_audit = session.get('audit_scores', {})
        _downstream = [m for m in ('idea', 'customer', 'money', 'discovery') if m in _existing_audit]
        if _downstream:
            for _m in _downstream:
                _existing_audit[_m]['stale'] = True
            session['audit_scores'] = _existing_audit
            session['context_stale_warning'] = True
            session.modified = True

        return redirect(url_for('idea'))

    # Item 6: Reset form if ?reset=1
    if request.args.get('reset'):
        for k in ('grade_level', 'location', 'delivery_model', 'budget', 'hours_per_week',
                  'prior_experience', 'why_this_idea', 'business_type', 'who_talked_to',
                  'biggest_challenge', 'strongest_part', 'interest_level'):
            session.pop(k, None)
        session.modified = True
        return redirect(url_for('context'))

    # Normalise any legacy business_type value that may have arrived via old session/restore
    if session.get('business_type'):
        session['business_type'] = _normalise_business_type(session['business_type'])

    rules = _load_json('business_rules.json')
    saved_answers = {k: session.get(k, '') for k in (
        'grade_level', 'location', 'delivery_model', 'budget', 'hours_per_week',
        'prior_experience', 'why_this_idea', 'business_type', 'who_talked_to',
        'biggest_challenge', 'strongest_part', 'interest_level',
    )}
    return render_template(
        'context.html',
        conflict_rules_json=json.dumps(rules),
        card_lock_rules_json=json.dumps(CARD_LOCK_RULES),
        saved_answers_json=json.dumps({k: v for k, v in saved_answers.items() if v}),
    )


@app.route('/idea', methods=['GET', 'POST'])
def idea():
    if request.method == 'GET':
        _g = _chapter_guard('idea')
        if _g: return _g
    if request.method == 'POST':
        session['sell_what']          = request.form.get('sell_what')
        session['sell_to']            = request.form.get('sell_to')
        session['sell_where']         = request.form.get('sell_where')
        session['sell_price']         = request.form.get('sell_price')
        session['niche_description']  = request.form.get('niche_description')
        session['problem_solved']     = request.form.get('problem_solved')
        session['similar_sellers']    = request.form.get('similar_sellers')
        return redirect(url_for('customer'))

    niche_rules_data = _load_json('niche_rules.json')
    stage1_keys = [
        'grade_level', 'location', 'delivery_model', 'budget',
        'hours_per_week', 'prior_experience', 'why_this_idea',
        'business_type', 'who_talked_to', 'biggest_challenge',
        'strongest_part', 'interest_level',
    ]
    stage1 = {k: session.get(k) for k in stage1_keys if session.get(k)}
    saved_idea = session.get('problem_solved') or session.get('niche_description') or ''
    # Item 25: Server-side saved analysis for injection
    saved_analysis = session.get('audit_scores', {}).get('idea')
    # O17-01: Consume stale warning — show once, then clear
    show_stale_warning = session.pop('context_stale_warning', False)
    session.modified = True

    return render_template(
        'idea.html',
        niche_rules_json=json.dumps(niche_rules_data.get('rules', [])),
        stage1_json=json.dumps(stage1),
        saved_idea=saved_idea,
        saved_sell_what=session.get('sell_what', ''),
        saved_sell_to=session.get('sell_to', ''),
        saved_sell_where=session.get('sell_where', ''),
        saved_sell_price=session.get('sell_price', ''),
        saved_analysis_json=json.dumps(saved_analysis) if saved_analysis else 'null',
        show_stale_warning=show_stale_warning,
        cc020_mismatch=_check_cc020(session),
    )


@app.route('/customer', methods=['GET', 'POST'])
def customer():
    if request.method == 'GET':
        _g = _chapter_guard('customer')
        if _g: return _g
    if request.method == 'POST':
        session['customer_age_range']  = request.form.get('customer_age_range')
        session['customer_role']       = request.form.get('customer_role')
        session['customer_location']   = request.form.get('customer_location')
        session['customer_problem']    = request.form.get('customer_problem')
        session['willingness_to_pay']  = request.form.get('willingness_to_pay')
        session['competitor_name']     = request.form.get('competitor_name')
        session['competitor_gap']      = request.form.get('competitor_gap')
        session['problem_confirmed']   = request.form.get('problem_confirmed', 'Yes')
        session['differentiation']     = request.form.get('differentiation')
        session['how_to_reach']        = request.form.get('how_to_reach')
        session['customer_quote']      = request.form.get('customer_quote')
        return redirect(url_for('money'))
    rules_data = _load_json('customer_rules.json')
    saved_idea = (session.get('problem_solved') or session.get('niche_description') or
                  session.get('sell_what') or '')
    return render_template('customer.html',
                           customer_rules_json=json.dumps(rules_data.get('rules', [])),
                           saved_idea=saved_idea,
                           archetype=session.get('idea_archetype', 'other'),
                           ch1_location=session.get('location', ''),
                           ch1_budget=session.get('budget', ''),
                           sell_price=session.get('sell_price') or session.get('unit_price') or '',
                           saved_customer_location=session.get('customer_location', ''),
                           saved_customer_problem=session.get('customer_problem', ''),
                           saved_willingness_to_pay=session.get('willingness_to_pay', ''),
                           saved_differentiation=session.get('differentiation', ''),
                           saved_how_to_reach=session.get('how_to_reach', ''),
                           customer_passed=session.get('customer_passed', False),
                           cc020_mismatch=_check_cc020(session))


@app.route('/money', methods=['GET', 'POST'])
def money():
    if request.method == 'GET':
        _g = _chapter_guard('money')
        if _g: return _g
    if request.method == 'POST':
        session['startup_cost']    = request.form.get('startup_cost')
        session['unit_cost']       = request.form.get('unit_cost')
        session['unit_price']      = request.form.get('unit_price')
        session['sell_price']      = request.form.get('sell_price') or request.form.get('unit_price')
        session['monthly_units']   = request.form.get('monthly_units')
        session['profit_plan']     = request.form.get('profit_plan')
        session['has_contingency'] = request.form.get('has_contingency')
        return redirect(url_for('discovery'))
    rules_data = _load_json('money_rules.json')
    saved_sell_price = session.get('sell_price') or session.get('unit_price') or ''
    archetype = session.get('idea_archetype', 'other')
    arch_params = ARCHETYPE_PARAMS.get(archetype, ARCHETYPE_PARAMS['other'])
    # Cross-chapter context for the money page
    ch1_budget = ''
    try:
        ch1_budget = float(str(session.get('budget', '') or '0').split('.')[0])
    except (ValueError, TypeError):
        ch1_budget = ''
    return render_template('money.html',
                           money_rules_json=json.dumps(rules_data.get('rules', [])),
                           saved_sell_price=saved_sell_price,
                           saved_unit_cost=session.get('unit_cost', ''),
                           saved_startup_cost=session.get('startup_cost', ''),
                           saved_monthly_units=session.get('monthly_units', ''),
                           saved_profit_plan=session.get('profit_plan', ''),
                           saved_has_contingency=session.get('has_contingency', ''),
                           archetype=archetype,
                           cost_floor=arch_params.get('startup_floor', 0),
                           ch1_budget=ch1_budget,
                           sell_price=saved_sell_price,
                           cc020_mismatch=_check_cc020(session))


@app.route('/discovery', methods=['GET', 'POST'])
def discovery():
    if request.method == 'GET':
        _g = _chapter_guard('discovery')
        if _g: return _g
    if request.method == 'POST':
        session['interviews_completed']  = request.form.get('interviews_completed')
        session['interview_sources']     = request.form.get('interview_sources')
        session['question_type']         = request.form.get('question_type')
        session['discovery_changed_plan']= request.form.get('discovery_changed_plan')
        session['insight_applied']       = request.form.get('insight_applied')
        name = request.form.get('student_name', '').strip()
        if name:
            session['student_name'] = name
        return redirect(url_for('certificate'))
    rules_data = _load_json('discovery_rules.json')
    archetype = session.get('idea_archetype', 'other')
    arch_params = ARCHETYPE_PARAMS.get(archetype, ARCHETYPE_PARAMS['other'])
    return render_template('discovery.html',
                           discovery_rules_json=json.dumps(rules_data.get('rules', [])),
                           archetype=archetype,
                           interview_floor=arch_params.get('interview_floor', 3),
                           cc020_mismatch=_check_cc020(session),
                           saved_interviews_completed=session.get('interviews_completed', ''),
                           saved_interview_sources=session.get('interview_sources', ''),
                           saved_question_type=session.get('question_type', ''),
                           saved_discovery_changed_plan=session.get('discovery_changed_plan', ''),
                           saved_insight_applied=session.get('insight_applied', ''))


@app.route('/final')
def final():
    # Guard: redirect to first chapter that hasn't been validated yet.
    # Prevents the nav link from showing 4/5 when the student jumped here
    # before running analysis on the last chapter.
    for _mid in _CHAPTER_ORDER:
        if _mid not in session.get('audit_scores', {}):
            flash(f'Run Analysis on each chapter first — you\'re almost there!', 'locked')
            return redirect(url_for(_mid))
    audit_scores = session.get('audit_scores', {})
    name = session.get('student_name', 'Young Entrepreneur')
    # Build per-module summary
    modules = []
    module_order = ['context','idea','customer','money','discovery']
    module_labels = {
        'context':   ('👤', 'About You'),
        'idea':      ('💡', 'Business Idea'),
        'customer':  ('🎯', 'Target Customer'),
        'money':     ('💰', 'Money Math'),
        'discovery': ('🔍', 'Customer Discovery'),
    }
    for mid in module_order:
        if mid in audit_scores:
            t = audit_scores[mid]
            emoji, label = module_labels[mid]
            modules.append({
                'id': mid,
                'emoji': emoji,
                'label': label,
                'hybrid_truth': t.get('hybrid_truth', 0),
                'symbolic_score': t.get('symbolic_score', 0),
                'pure_truth': t.get('pure_truth', 0),
                'sycophancy_gap': t.get('sycophancy_gap', 0),
                'hint_delta': t.get('hint_delta', 0),
                'hybrid_truth_initial': t.get('hybrid_truth_initial', t.get('hybrid_truth', 0)),
            })

    avg_hybrid = round(sum(m['hybrid_truth'] for m in modules) / len(modules)) if modules else 0
    avg_gap = round(sum(abs(m['sycophancy_gap']) for m in modules) / len(modules)) if modules else 0
    total_delta = sum(max(0, m['hint_delta']) for m in modules)

    # O17-05: total iteration count across all chapters (from score_history)
    total_iterations = sum(
        len(audit_scores.get(mid, {}).get('score_history', []))
        for mid in ('context', 'idea', 'customer', 'money', 'discovery')
    )

    return render_template('final.html',
        modules=modules,
        student_name=name,
        avg_hybrid=avg_hybrid,
        avg_gap=avg_gap,
        total_delta=total_delta,
        modules_done=len(modules),
        total_iterations=total_iterations,
        sell_what=session.get('sell_what','your idea'),
        sell_to=session.get('sell_to','your customers'),
        sell_where=session.get('sell_where',''),
        story=session.get('story', ''),
    )


# ── API: Run full ITS analysis ─────────────────────────────────
@app.route('/api/analyse', methods=['POST'])
def api_analyse():
    _rl = _check_llm_rate_limit()
    if _rl:
        return jsonify({'error': f'Too many requests — please wait {_rl}s before re-running analysis.', 'retry_after': _rl}), 429
    try:
        answers = request.json.get('answers', {})
        # Convert numeric fields that may come in as strings from form
        for numeric_field in ('budget', 'hours_per_week'):
            if numeric_field in answers and answers[numeric_field] != '':
                try:
                    answers[numeric_field] = float(answers[numeric_field])
                except (ValueError, TypeError):
                    pass

        rules = _load_json('business_rules.json')
        scores, triggered = compute_health_scores(answers, rules)

        # Smart business type vs budget cross-check
        _btype = answers.get('business_type', '').lower()
        _budget = _safe_float(answers.get('budget', 0))
        _HIGH_COST_TYPES = ['manufacturing', 'restaurant', 'café', 'cafe', 'store', 'shop', 'retail', 'franchise', 'factory']
        _LOW_BUDGET_TYPES = ['digital', 'online', 'app', 'software', 'tutoring', 'service', 'consulting', 'freelance']
        if _budget > 0 and any(t in _btype for t in _HIGH_COST_TYPES) and _budget < 200:
            triggered.append({'id': 'budget_business_mismatch', 'title': 'Budget too low for this type', 'emoji': '💸', 'severity': 'warning'})
        elif _budget > 5000 and any(t in _btype for t in _LOW_BUDGET_TYPES):
            triggered.append({'id': 'budget_too_high_for_type', 'title': 'Budget seems high for a digital/service business', 'emoji': '🤔', 'severity': 'warning'})

        pure_response   = call_context_pure_llm(answers, scores)
        hybrid_response = call_hybrid_llm(answers, scores, triggered)
        # Floor mentor at max(70, symbolic-5) when no violations — context is profile data,
        # not a business plan; the LLM scores it conservatively even when complete.
        if not triggered and hybrid_response:
            import re as _re_ctx_live
            _ctx_sym_floor = max(70, round(sum(scores.values()) / len(scores)) - 5) if scores else 70
            _ctx_m = _re_ctx_live.search(r'MENTOR_SCORE:\s*(\d{1,3})', hybrid_response)
            if _ctx_m and int(_ctx_m.group(1)) < _ctx_sym_floor:
                hybrid_response = _re_ctx_live.sub(r'MENTOR_SCORE:\s*\d{1,3}',
                                                    f'MENTOR_SCORE: {_ctx_sym_floor}', hybrid_response)

        # Use client-side merit scores (0–100) as symbolic ground truth.
        # Server penalty scores start at 100 and are not a fair baseline.
        client_scores = request.json.get('client_scores', {})
        if client_scores and len(client_scores) == 6:
            symbolic_avg = sum(client_scores.values()) / 6
        else:
            symbolic_avg = sum(scores.values()) / len(scores)

        symbolic_score = round(symbolic_avg)
        triple = _auditor.triple_truth(symbolic_score, pure_response, hybrid_response, 'context')

        # Convert triggered rules to violations-style for sidebar
        violations = [
            {'id': r['id'], 'message': r.get('title', r['id']),
             'severity': r.get('severity', 'warning')}
            for r in triggered
        ]
        # Persist to session so progress dashboard can read it (include violations for tooltip)
        _save_audit_with_delta('context', triple, violations)
        # Store raw radar scores for the mini radar chart on all pages
        if client_scores and len(client_scores) == 6:
            session['radar_scores'] = {k: min(100, max(0, round(v))) for k, v in client_scores.items()}
            session.modified = True

        save_research_log('context', symbolic_score, violations,
                          pure_response, triple['pure_truth'],
                          hybrid_response, triple['hybrid_truth'],
                          triple['sycophancy_gap'])

        # S21-23: top_issues — top 3 triggered rules sorted by total absolute score impact
        def _total_impact(rule):
            return sum(abs(v) for v in rule.get('score_impact', {}).values())
        top_issues = sorted(triggered, key=_total_impact, reverse=True)[:3]

        return jsonify({
            'scores': scores,
            'triggered_rules': [
                {
                    'id':       r['id'],
                    'title':    r['title'],
                    'emoji':    r.get('emoji', ''),
                    'severity': r.get('severity', 'warning'),
                }
                for r in triggered
            ],
            'top_issues': [
                {
                    'title':    r['title'],
                    'emoji':    r.get('emoji', '⚠️'),
                    'severity': r.get('severity', 'warning'),
                    'message':  r.get('message', ''),
                }
                for r in top_issues
            ],
            'violations':    violations,
            'total_rules':   len(rules),
            'symbolic_score': symbolic_score,
            'pure_llm':      pure_response,
            'hybrid_llm':    _strip_mentor_score_line(hybrid_response),
            'pure_truth':    triple['pure_truth'],
            'hybrid_truth':  triple['hybrid_truth'],
            'triple':        triple,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════
#  NICHE VALIDATOR  (ITS Stage 2 — deterministic rule engine)
# ══════════════════════════════════════════════════════════════
def _llm_element_present(idea_text: str, rule_id: str) -> bool:
    """Semantic fallback for requires_any rules.
    Returns True if the LLM confirms the required element is present,
    regardless of how the student phrased it.
    Called only when keyword matching fails — keeps LLM calls minimal.
    Returns False immediately for short/incomplete text — no point calling
    the LLM when the text is too vague to evaluate at all."""
    if len(idea_text.strip()) < 50:
        return False  # too short — the LLM makes too many false positives on brief vague phrases
    # NR003/NR004/NR005: keyword-only — the 8B LLM is too permissive (passes "for people",
    # no location, no problem) and cannot be reliably constrained with prompt examples.
    # Only NR008 (product name) uses LLM fallback since product nouns vary too widely.
    _questions = {
        'NR008': (
            'Does this text explicitly name a specific product or service being sold? '
            '(e.g. "tutoring sessions", "handmade bracelets", "tire changing service", "healthy snacks") '
            'Answer yes or no only.'
        ),
    }
    question = _questions.get(rule_id)
    if not question:
        return False
    try:
        rsp = _groq_create(
            model='llama-3.1-8b-instant',
            messages=[
                {'role': 'system', 'content':
                    'You are checking whether a business idea description contains a specific element. '
                    'Reply with exactly one word: yes or no.'},
                {'role': 'user', 'content': f'Business idea: "{idea_text}"\n\n{question}'}
            ],
            max_tokens=5,
            temperature=0.0,
        )
        answer = rsp.choices[0].message.content.strip().lower()
        return answer.startswith('yes')
    except Exception:
        return False  # on error, let the keyword result stand (fail = keyword miss)


def validate_niche(text: str, rules: list) -> list:
    """Runs all niche_rules.json rules against text. Returns list of violation dicts."""
    violations = []
    t = text.strip().lower()
    words = t.split()
    word_count = max(len(words), 1)

    for rule in rules:
        rtype    = rule.get('type')
        severity = rule.get('severity', 'error')
        failed   = False

        if rtype == 'min_length':
            failed = len(text.strip()) < rule['threshold']

        elif rtype == 'blacklist':
            failed = any(term in t for term in rule['terms'])

        elif rtype == 'requires_any':
            # Stage 1: fast keyword pass — no LLM cost if an indicator matches.
            found = any(ind in t for ind in rule['indicators'])
            # Stage 2: semantic fallback — only triggered when keywords miss.
            # The LLM answers a single yes/no question so we never need to
            # enumerate every possible phrasing in the JSON.
            if not found and GROQ_AVAILABLE:
                found = _llm_element_present(text, rule['id'])
            failed = not found  # rule fires when required element is absent

        elif rtype == 'generic_density':
            generic_count = sum(1 for term in rule['terms'] if term in words)
            failed = (generic_count / word_count) >= rule.get('min_density', 0.18)

        if failed:
            violations.append({
                'id':            rule['id'],
                'flag':          rule.get('flag', rule['id']),
                'severity':      severity,
                'message':       rule['message'],
                'socratic_hint': rule.get('socratic_hint', ''),
            })

    return violations


# ══════════════════════════════════════════════════════════════
#  CROSS-CHAPTER INTEGRITY CHECKS
#  Links Chapter 1 (profile / budget) against Chapter 2 (idea).
#  Violations are injected before scoring — they lower the Mentor
#  (hybrid_truth) score for delusional or impossible combinations
#  WITHOUT touching the AI Optimist formula.
# ══════════════════════════════════════════════════════════════

def _detect_idea_category(text: str) -> set:
    """Keyword-based idea classifier. Returns a set of category labels."""
    t = text.lower()
    cats = set()
    if any(kw in t for kw in [
        'manufactur', 'factory', 'assembly line', 'mass produc',
        'make iphone', 'build iphone', 'produce iphone', 'sell iphone',
        'make samsung', 'make phone', 'build phone', 'make laptop',
        'build laptop', 'make computer', 'build computer', 'make car',
        'build car', 'produce electronic', 'hardware startup',
        'make device', 'build device', 'semiconductor', 'circuit board',
        'make airpod', 'build airpod', 'make smartwatch', 'build watch',
    ]):
        cats.add('manufacturing')
    if any(kw in t for kw in [
        'snack', 'bak', 'cake', 'cookie', 'meal', ' food', 'food ',
        'restaurant', 'catering', 'cooking', 'beverage', 'drink',
        'juice', 'lemonade', 'smoothie', 'candy', 'chocolate',
        'sandwich', 'lunch box', 'dinner', 'breakfast', 'recipe',
        'sell food', 'homemade food',
    ]):
        cats.add('food')
    if any(kw in t for kw in [
        'handmade', 'craft', 'jewelry', 'jewellery', 'bracelet', 'necklace',
        'candle', 'soap', 'knit', '3d print', 'sew', 'clothing',
        't-shirt', 'hoodie', 'pottery', 'art print', 'sticker', 'custom merch',
    ]):
        cats.add('physical_product')
    import re as _re
    if any(kw in t for kw in [
        'website', 'web site', 'software', 'platform', 'web app', 'mobile app',
        'saas', 'plugin', 'online tool', 'digital product', 'coding project',
    ]) or _re.search(r'\bapp\b', t) or _re.search(r'\bgame\b', t) or _re.search(r'\bcode\b', t) or _re.search(r'\bcoding\b', t):
        cats.add('digital')
    if any(kw in t for kw in [
        'tutor', 'teach', 'clean', 'lawn', 'mow', 'babysit',
        'dog walk', 'dog wash', 'photograph', 'coach',
        'lesson', 'repair', ' fix ', 'consult', 'organiz', 'event plan',
    ]):
        cats.add('service')
    if any(kw in t for kw in [
        'million', 'billion', 'worldwide', 'global market', 'everyone in the',
        'mass produc', 'national scale', 'international', 'millions of',
        'scale to the world', 'sell to everyone',
    ]):
        cats.add('scale_delusion')
    return cats


# ══════════════════════════════════════════════════════════════
#  ARCHETYPE TAGGER — LLM classifies idea into functional archetype
#  so CC rules and scoring thresholds are calibrated per business type.
# ══════════════════════════════════════════════════════════════

# CC020 — Ch1 business_type vs Ch2 archetype conflict map (module-level for reuse)
# Legacy business_type values → new archetype slugs.
# Applied at every entry point where session data is loaded (restore code, context GET).
_LEGACY_BUSINESS_TYPE_MAP = {
    'Food':    'food-beverage',
    'Product': 'physical-product',
    'Service': 'service',
    'Digital': 'digital-product',
    'Online':  'digital-product',
}

def _normalise_business_type(value: str) -> str:
    """Coerce any legacy business_type value to the current archetype slug."""
    return _LEGACY_BUSINESS_TYPE_MAP.get(value, value)

# Signal keywords for each selectable archetype.
# Used as a "benefit of the doubt" check: if the student selected archetype X
# and the idea text contains signals for X, trust the student even if the LLM
# returned a conflicting archetype.
_ARCHETYPE_IDEA_SIGNALS = {
    'food-beverage': [
        'food', 'eat', 'cook', 'bake', 'meal', 'snack', 'drink', 'beverage', 'kitchen',
        'wrap', 'burger', 'pizza', 'cake', 'bread', 'recipe', 'breakfast', 'lunch',
        'dinner', 'catering', 'menu', 'flavour', 'flavor', 'ingredient', 'cafe', 'coffee',
        'juice', 'sauce', 'curry', 'noodle', 'sushi', 'taco', 'sandwich', 'soup', 'salad',
        'dessert', 'chocolate', 'pastry', 'biscuit', 'cookie', 'smoothie', 'lemonade',
        'street food', 'fusion', 'homemade food', 'home kitchen',
    ],
    'service': [
        'clean', 'wash', 'deliver', 'dog walk', 'walk dogs', 'mow', 'garden', 'fix',
        'repair', 'install', 'errand', 'babysit', 'childcare', 'care for', 'ironing',
        'laundry', 'pet sit', 'house sit', 'grocery', 'moving', 'handyman', 'maintenance',
        'pressure wash', 'window clean',
    ],
    'skills-for-hire': [
        'teach', 'tutor', 'lesson', 'coach', 'train', 'design', 'graphic', 'logo',
        'code', 'program', 'develop', 'write', 'copywrite', 'edit', 'proofread',
        'translate', 'music lesson', 'guitar', 'piano', 'art class', 'photography',
        'video edit', 'social media manage', 'cv writing', 'cv help',
    ],
    'digital-product': [
        'app', 'mobile app', 'web app', 'website builder', 'software', 'game',
        'plugin', 'template', 'digital download', 'online course', 'e-book', 'ebook',
        'saas', 'bot', 'script', 'dashboard', 'api', 'browser extension',
        'notion template', 'figma', 'spreadsheet template', 'digital product',
        'software tool', 'online tool', 'digital tool',
    ],
    'physical-product': [
        'craft', 'handmade', 'handcraft', 'manufacture', 'physical product',
        'assemble', 'package', 'merchandise', 'sell product', 'sell products',
        'candle', 'jewellery', 'jewelry', 'clothing', 't-shirt', 'tote bag',
        'sticker', 'art print', 'poster', 'keyring', 'phone case', 'plushie',
        'figurine', 'wax melt', 'resin',
    ],
    'resale-retail': [
        'resell', 'resale', 'flip', 'second-hand', 'secondhand', 'vintage', 'preloved',
        'pre-loved', 'thrift', 'depop', 'vinted', 'buy and sell', 'import', 'dropship',
        'wholesale', 'stock', 'arbitrage', 'car boot', 'ebay', 'amazon resell',
    ],
    'maker': [
        'custom', 'bespoke', '3d print', 'engrave', 'laser cut', 'tailor', 'sew',
        'embroider', 'woodwork', 'metalwork', 'weld', 'forge', 'commission',
        'made to order', 'made-to-order', 'build to spec', 'one of a kind',
    ],
    'content-media': [
        'youtube', 'tiktok', 'instagram page', 'blog', 'newsletter', 'podcast',
        'stream', 'twitch', 'content creator', 'vlog', 'channel', 'audience',
        'followers', 'subscribers', 'media brand', 'influencer', 'monetise audience',
    ],
}

def _student_selection_matches_idea(selected: str, idea_text: str) -> bool:
    """Return True if the idea text contains signal words for the student's selected archetype.
    Used as benefit-of-the-doubt: if the student selected X and their idea sounds like X,
    trust them even if the LLM said something different."""
    signals = _ARCHETYPE_IDEA_SIGNALS.get(selected, [])
    if not signals:
        return False
    t = idea_text.lower()
    return any(s in t for s in signals)

# CC020 — compatible pairs: archetypes close enough that CC020 should NOT fire.
# If the student-selected archetype and LLM-detected archetype are both in the same
# compatibility group, no conflict is raised.
_ARCHETYPE_COMPATIBLE_GROUPS = [
    # Physical goods cluster — making vs. selling physical items are both "product" businesses
    frozenset(['physical-product', 'maker', 'resale-retail']),
    # Labour / knowledge cluster — service and skills-for-hire overlap heavily
    frozenset(['service', 'skills-for-hire', 'expert', 'event-experience']),
    # Digital cluster — digital product, content, and marketplace are all screen-based
    frozenset(['digital-product', 'content-media', 'marketplace']),
    # Food is its own group — food-beverage only compatible with itself
    frozenset(['food-beverage']),
]

def _archetypes_compatible(a: str, b: str) -> bool:
    """Return True if a and b belong to the same compatibility group (no CC020 needed)."""
    if a == b:
        return True
    for grp in _ARCHETYPE_COMPATIBLE_GROUPS:
        if a in grp and b in grp:
            return True
    return False

_ARCHETYPE_DISPLAY_LABELS = {
    'food-beverage':    'Food & Drink',
    'service':          'Service',
    'skills-for-hire':  'Skills & Tutoring',
    'digital-product':  'Digital Product',
    'physical-product': 'Physical Product',
    'resale-retail':    'Resale',
    'maker':            'Custom / Maker',
    'content-media':    'Content & Media',
    'event-experience': 'Events & Experience',
    'marketplace':      'Marketplace',
    'expert':           'Expert / Coaching',
}

def _check_cc020(session_obj) -> dict | None:
    """Return a mismatch dict if Ch1 business_type conflicts with Ch2 archetype, else None.
    business_type now stores an archetype value directly (e.g. 'food-beverage').
    Returns friendly display labels so templates can render them directly."""
    sa = session_obj.get('stage1_answers', {})
    business_type = sa.get('business_type', '') or session_obj.get('business_type', '')
    archetype = session_obj.get('idea_archetype', '')
    if not business_type or not archetype or archetype in ('other', 'non-viable'):
        return None
    if _archetypes_compatible(business_type, archetype):
        return None
    return {
        'business_type': _ARCHETYPE_DISPLAY_LABELS.get(business_type, business_type),
        'archetype':     _ARCHETYPE_DISPLAY_LABELS.get(archetype, archetype),
    }


_ARCHETYPES = [
    'physical-product',   # handmade goods, crafts, custom merch, 3D prints
    'food-beverage',      # snacks, drinks, meal prep, baked goods, catering
    'service',            # any service business — local, remote, in-person, online
    'skills-for-hire',    # tutoring, design, coding, music lessons, photography
    'digital-product',    # app, game, browser extension, template pack, plugin
    'resale-retail',      # thrift flip, import goods, dropshipping, arbitrage
    'event-experience',   # party planning, workshops, sports camps, escape rooms
    'content-media',      # newsletter, YouTube, podcast, blog, social content
    'marketplace',        # platform connecting buyers + sellers, takes a cut
    'maker',              # build-to-order, custom fabrication, 3D printing, repairs
    'expert',             # consulting, coaching, advice, strategy — knowledge not tasks
    'non-viable',         # joke/prank ideas, physically impossible, illegal, safety risk
    'other',              # fallback for genuinely novel or ambiguous ideas
]

# Per-archetype calibration thresholds used by CC rules.
# Universal rules (NR + CC001–CC011) run regardless of archetype.
# Archetype rules (CC012+) use these values for calibrated checks.
ARCHETYPE_PARAMS = {
    'physical-product': {
        'margin_floor':      2.0,   # sell price must be ≥ 2× unit cost
        'interview_floor':   3,     # min interviews before analysing customer chapter
        'hours_floor':       4,     # min hours/week to be viable
        'startup_floor':     30,    # min plausible startup cost ($)
        'minutes_per_unit':  30,    # expected production time per unit (mins)
        'extra_flags':       [],
    },
    'food-beverage': {
        'margin_floor':      2.5,
        'interview_floor':   5,
        'hours_floor':       6,
        'startup_floor':     80,
        'minutes_per_unit':  20,
        'extra_flags':       ['food_safety', 'perishable_risk'],
    },
    'service': {
        'margin_floor':      None,  # services: no unit COGS; skip margin check
        'interview_floor':   3,
        'hours_floor':       5,
        'startup_floor':     0,
        'minutes_per_unit':  60,    # 1 hr per service delivery
        'extra_flags':       [],
    },
    'skills-for-hire': {
        'margin_floor':      None,
        'interview_floor':   3,
        'hours_floor':       8,
        'startup_floor':     0,
        'minutes_per_unit':  60,
        'extra_flags':       [],
    },
    'digital-product': {
        'margin_floor':      5.0,   # near-zero COGS → high margin expected
        'interview_floor':   5,
        'hours_floor':       10,
        'startup_floor':     0,
        'minutes_per_unit':  0,     # zero marginal cost per digital unit
        'extra_flags':       ['platform_dependency'],
    },
    'resale-retail': {
        'margin_floor':      1.5,
        'interview_floor':   3,
        'hours_floor':       4,
        'startup_floor':     50,
        'minutes_per_unit':  10,
        'extra_flags':       ['supplier_risk', 'inventory_risk'],
    },
    'event-experience': {
        'margin_floor':      2.0,
        'interview_floor':   5,
        'hours_floor':       8,
        'startup_floor':     30,
        'minutes_per_unit':  120,   # events take hours to deliver
        'extra_flags':       [],
    },
    'content-media': {
        'margin_floor':      None,
        'interview_floor':   5,
        'hours_floor':       10,
        'startup_floor':     0,
        'minutes_per_unit':  0,
        'extra_flags':       ['monetisation_unclear'],
    },
    'marketplace': {
        'margin_floor':      None,
        'interview_floor':   8,     # two-sided: need supply AND demand interviews
        'hours_floor':       10,
        'startup_floor':     0,
        'minutes_per_unit':  0,
        'extra_flags':       ['cold_start_problem'],
    },
    'maker': {
        'margin_floor':      2.0,   # custom builds must cover materials + time
        'interview_floor':   3,
        'hours_floor':       6,
        'startup_floor':     50,    # tools, materials, supplies
        'minutes_per_unit':  90,    # custom builds take longer than mass-produced
        'extra_flags':       [],
    },
    'expert': {
        'margin_floor':      None,  # pure knowledge — no unit cost
        'interview_floor':   5,     # must validate that people value the advice
        'hours_floor':       4,
        'startup_floor':     0,
        'minutes_per_unit':  60,    # consultation session
        'extra_flags':       ['credibility_required'],
    },
    'non-viable': {
        'margin_floor':      None,
        'interview_floor':   0,
        'hours_floor':       0,
        'startup_floor':     0,
        'minutes_per_unit':  0,
        'extra_flags':       ['non_viable_concept'],
    },
    'other': {
        'margin_floor':      2.0,
        'interview_floor':   3,
        'hours_floor':       4,
        'startup_floor':     0,
        'minutes_per_unit':  30,
        'extra_flags':       [],
    },
}


def _check_solution_alignment(idea_text: str, archetype: str = 'other') -> list:
    """NR011: Verify the product/service actually addresses the stated customer problem.
    Catches ideas like 'tutoring kids because they need food' where solution ≠ problem.
    Archetype is passed as context so the LLM doesn't misfire on two-sided models (marketplace)."""
    if not GROQ_AVAILABLE or len(idea_text.strip()) < 30:
        return []
    # marketplace ideas have two-sided logic that can look incoherent — skip alignment check
    if archetype == 'marketplace':
        return []
    try:
        rsp = _groq_create(
            model='llama-3.1-8b-instant',
            messages=[
                {'role': 'system', 'content':
                    'You check if a business idea makes logical sense. '
                    'Respond with JSON only: {"coherent": true, "reason": "one sentence"}. '
                    'Set coherent=false ONLY when the service or product clearly does NOT '
                    'address the stated customer problem (e.g. selling food when the problem '
                    'is boredom, or offering tutoring when the problem is hunger). '
                    f'This idea has been classified as business type: {archetype}. '
                    'If the idea is vague but not obviously mismatched, set coherent=true.'},
                {'role': 'user', 'content':
                    f'Business idea: "{idea_text}"\n'
                    'Does the product or service actually solve the stated customer problem?'}
            ],
            max_tokens=80, temperature=0.0,
            response_format={'type': 'json_object'},
        )
        data = json.loads(rsp.choices[0].message.content)
        if not data.get('coherent', True):
            reason = data.get('reason', 'the solution does not match the problem described')
            return [{
                'id': 'NR011',
                'flag': 'solution_problem_mismatch',
                'severity': 'error',
                'message': (
                    f"📌 Solution doesn't match problem: {reason}\n\n"
                    "What to do: Make sure what you're selling actually solves the problem "
                    "you named. Ask: if a customer has this specific problem, would your "
                    "product actually help them?"
                ),
            }]
    except Exception:
        pass
    return []


def _detect_archetype(idea_text: str, stage1_answers: dict) -> str:
    """LLM-based archetype tagger. Returns one of the 10 archetype strings.
    Falls back to keyword-based heuristic if LLM is unavailable.
    Result is cached in session under 'idea_archetype'."""
    # --- LLM path ---
    if GROQ_AVAILABLE:
        try:
            business_type = stage1_answers.get('business_type', '')
            prompt_user = (
                f"Business idea: \"{idea_text}\"\n"
                f"Business type selected: {business_type}\n\n"
                f"Classify this into exactly ONE archetype from this list:\n"
                f"{', '.join(_ARCHETYPES)}\n\n"
                "Definitions:\n"
                "physical-product: handmade or manufactured goods sold individually\n"
                "food-beverage: anything edible or drinkable — meals, snacks, baked goods, drinks, catering, street food. Use this even if orders are taken in advance or cooked fresh per order.\n"
                "service: tasks done for customers in person — cleaning, dog walking, delivery, gardening, repairs, errands. The business performs an action for someone.\n"
                "skills-for-hire: selling a personal skill — tutoring, graphic design, coding, music lessons, writing, photography. The business sells expertise directly.\n"
                "digital-product: a product that exists as a file or platform — apps, games, software, templates, e-books, online courses, SaaS tools.\n"
                "physical-product: tangible goods made or sourced and sold — candles, clothing, art prints, packaged goods. The business sells an object.\n"
                "resale-retail: buying existing items and reselling at a markup — vintage clothes, second-hand goods, imported products, dropshipping.\n"
                "event-experience: organising events, workshops, camps, or in-person experiences.\n"
                "content-media: building an audience through content — YouTube channel, TikTok, newsletter, podcast, blog. Revenue comes from the audience.\n"
                "marketplace: a platform that connects buyers and sellers and takes a cut.\n"
                "maker: custom fabrication of NON-FOOD physical items — 3D printing, custom jewellery, woodwork, laser cutting, bespoke tailoring.\n"
                "expert: high-level consulting, coaching or strategic advice — selling knowledge, not performing tasks.\n"
                "non-viable: joke, illegal, physically impossible, or safety-risk ideas.\n"
                "other: genuinely does not fit any category above.\n\n"
                "RULES:\n"
                "- Food/cooking/drinks of any kind → ALWAYS food-beverage, NEVER maker or service.\n"
                "- Building websites or apps for clients → skills-for-hire, not digital-product.\n"
                "- Selling a finished digital file or tool → digital-product, not skills-for-hire.\n"
                "- YouTube/TikTok/newsletter → content-media, not digital-product or service.\n"
                "- Reselling preloved or second-hand items → resale-retail, not physical-product.\n"
                "- Custom physical items (non-food) made to order → maker, not physical-product.\n\n"
                "Reply with ONLY the archetype name, nothing else."
            )
            rsp = _groq_create(
                model='llama-3.1-8b-instant',
                messages=[
                    {'role': 'system', 'content': 'You classify business ideas into archetypes. Reply with exactly one archetype name.'},
                    {'role': 'user',   'content': prompt_user},
                ],
                max_tokens=10,
                temperature=0.0,
            )
            tag = rsp.choices[0].message.content.strip().lower().replace(' ', '-')
            if tag in _ARCHETYPES:
                t_lower = idea_text.lower()
                selected = stage1_answers.get('business_type', '')

                # Universal benefit-of-the-doubt rule:
                # If the LLM's tag would trigger CC020 (different compatible group from
                # the student's selection) BUT the idea text contains signal words for
                # the student's selected archetype — trust the student's selection.
                if selected and selected in _ARCHETYPE_IDEA_SIGNALS:
                    if not _archetypes_compatible(selected, tag) and tag not in ('other', 'non-viable'):
                        if _student_selection_matches_idea(selected, idea_text):
                            return selected

                # C9-12: tiebreaker — if LLM says physical-product but text has
                # custom/build-to-order signals, prefer 'maker'.
                # food-beverage is NEVER overridden to maker (food made to order is still food).
                _maker_signals = ['custom', 'bespoke', 'made to order', 'build to spec',
                                  'made-to-order', 'handcrafted to', 'personalised', 'personalized',
                                  'one of a kind', 'one-of-a-kind', 'commissioned']
                if tag == 'physical-product' and any(s in t_lower for s in _maker_signals):
                    return 'maker'
                return tag
        except Exception:
            pass

    # --- Keyword fallback ---
    t_lower = idea_text.lower()
    _maker_signals = ['custom', 'bespoke', 'made to order', 'build to spec', 'commissioned']
    cats = _detect_idea_category(idea_text)
    if 'manufacturing' in cats:
        return 'maker' if any(s in t_lower for s in _maker_signals) else 'physical-product'
    if 'food'          in cats:  return 'food-beverage'
    if 'digital'       in cats:  return 'digital-product'
    if 'service'       in cats:  return 'service'
    if 'physical_product' in cats:
        return 'maker' if any(s in t_lower for s in _maker_signals) else 'physical-product'
    return 'other'


def _detect_contradictions(answers_snapshot: dict) -> list:
    """Second LLM pass: checks for logical contradictions WITHIN and ACROSS chapters.
    Works on structured values only (not free text scoring) — purely a consistency check.
    Returns violations in the same format as CC rules (id, flag, severity, message).
    Called once per full-plan validate; cached in session to avoid repeated LLM calls."""
    if not GROQ_AVAILABLE:
        return []
    try:
        # Build a compact structured summary for the LLM
        lines = []
        # Only pass fields relevant to LOGICAL contradictions (channel/customer mismatch,
        # differentiation vs problem, discovery changes). Exclude financial and time fields —
        # those are handled by the dedicated rules engine and passing them here causes the
        # small contradiction-detector LLM to generate false capacity/margin violations.
        field_map = {
            'customer_age_range':     'Customer age range',
            'how_to_reach':           'Reach channel',
            'differentiation':        'Differentiation',
            'problem_confirmed':      'Problem strength',
            'customer_location':      'Customer location',
            'interviews_completed':   'Interviews done',
            'interview_sources':      'Interview sources',
            'discovery_changed_plan': 'Discovery changed plan?',
            'business_type':          'Business type',
            'delivery_model':         'Delivery model',
            'archetype':              'Business archetype',
        }
        for key, label in field_map.items():
            val = answers_snapshot.get(key, '')
            if val not in ('', None):
                lines.append(f"- {label}: {val}")
        if not lines:
            return []

        plan_summary = '\n'.join(lines)
        prompt = (
            "You are a strict business plan consistency checker for a student entrepreneurship tool. "
            "Read the following structured plan data and identify ONLY clear logical contradictions. "
            "Do NOT comment on quality or give general advice.\n\n"
            "Plan data:\n" + plan_summary + "\n\n"
            "Check ONLY for these logical contradictions:\n"
            "- Reach channel doesn't match customer age or location (e.g. TikTok for under-10s)\n"
            "- Differentiation is 'Cheaper' but problem is 'Helpless' (helpless customers are willing to pay premium)\n"
            "- 'Just a minor annoyance' problem but high willingness to pay\n"
            "- Discovery changed plan but student didn't select any change option\n\n"
            "DO NOT flag these — they are NOT contradictions:\n"
            "- Word of mouth / referrals with in-person or local customers (this is correct and expected)\n"
            "- Any channel paired with 'In-person (same area)' or 'Local' proximity\n"
            "- Time, capacity, or hours-vs-units issues — the rules engine handles that separately.\n"
            "- Price, cost, or margin issues — the rules engine handles that separately.\n\n"
            "Return a JSON array of objects. Each object has:\n"
            '  "flag": short_snake_case_name\n'
            '  "severity": "error" or "warning"\n'
            '  "message": one clear sentence stating the contradiction and what to fix\n\n'
            "Return [] if no contradictions are found. Return ONLY valid JSON, no explanation."
        )
        rsp = _groq_create(
            model='llama-3.1-8b-instant',
            messages=[
                {'role': 'system', 'content': 'You are a business plan contradiction detector. Output only valid JSON arrays.'},
                {'role': 'user',   'content': prompt},
            ],
            max_tokens=400,
            temperature=0.0,
        )
        raw = rsp.choices[0].message.content.strip()
        # Extract JSON array from response (may have surrounding text)
        import re as _re2
        match = _re2.search(r'\[.*\]', raw, _re2.DOTALL)
        if not match:
            return []
        items = json.loads(match.group(0))
        violations = []
        for i, item in enumerate(items[:5]):  # cap at 5 contradiction violations
            if isinstance(item, dict) and item.get('flag') and item.get('message'):
                violations.append({
                    'id':       f'CD{i+1:03d}',
                    'flag':     item.get('flag', 'contradiction'),
                    'severity': item.get('severity', 'warning'),
                    'message':  (
                        f"🔁 Cross-check: {item['message']}\n\n"
                        "What to do: Review the conflicting answers above and update whichever is inaccurate."
                    ),
                    'source':   'contradiction_detector',
                })
        return violations
    except Exception:
        return []


def validate_cross_chapter_conflicts(idea_text: str, stage1_answers: dict) -> list:
    """Cross-chapter integrity check: compares Chapter 1 profile/budget against
    Chapter 2 idea. Returns violation dicts in the same format as validate_niche.

    Severity 'error' → forces the FAIL path (Mentor score = symbolic_score, no LLM praise).
    Severity 'warning' → reduces symbolic_score by 10 but does not block.
    AI Optimist (pure_truth) is always computed from symbolic_score and remains inflated —
    so the gap between AI and Mentor score visibly widens for delusional ideas.
    """
    if not idea_text or not stage1_answers:
        return []

    violations = []
    cats = _detect_idea_category(idea_text)

    try:
        budget = int(str(stage1_answers.get('budget', '0') or '0').split('.')[0])
    except ValueError:
        budget = 0
    try:
        hours = int(str(stage1_answers.get('hours_per_week', '5') or '5').split('.')[0])
    except ValueError:
        hours = 5

    delivery   = stage1_answers.get('delivery_model', '')
    experience = stage1_answers.get('prior_experience', '')

    # CC001 — Manufacturing complex hardware on a student budget
    if 'manufacturing' in cats and budget < 10000:
        violations.append({
            'id': 'CC001', 'flag': 'budget_manufacturing_gap', 'severity': 'error',
            'message': (
                f"📌 Budget vs idea mismatch: manufacturing physical hardware or electronics "
                f"requires $50,000–$500,000+ for tooling, components, IP licensing, and testing. "
                f"Your stated budget of ${budget:,} covers only a fraction of a single prototype.\n\n"
                f"What to do: Redesign your idea around customising or reselling existing products — "
                f"phone cases, accessories, or branded merchandise are achievable for a student."
            ),
            'socratic_hint': (
                "Student wants to manufacture hardware on a student budget. Ask: what is the minimum "
                "cost to produce one working prototype of this product? Walk them through the gap."
            ),
        })

    # CC002 — Physical handmade product with zero budget
    if 'physical_product' in cats and budget == 0 and 'manufacturing' not in cats:
        violations.append({
            'id': 'CC002', 'flag': 'zero_budget_product', 'severity': 'warning',
            'message': (
                "🟡 Physical products need upfront material costs — even handmade jewellery "
                "or candles need supplies. With $0 budget you'd need someone to front materials.\n\n"
                "What to do: Start with a zero-cost service first (tutoring, dog walking) to save "
                "enough for your first batch of materials."
            ),
            'socratic_hint': "Ask: what materials do you need for your first 5 products, and where does that money come from?",
        })

    # CC003 — Food business under minimum viable food safety budget
    if 'food' in cats and budget < 500:
        violations.append({
            'id': 'CC003', 'flag': 'food_budget_gap', 'severity': 'warning',
            'message': (
                f"🟡 Food businesses typically need $500–$1,500 before the first sale — "
                f"kitchen access, food handling certification, packaging, and ingredients. "
                f"Your budget of ${budget:,} may leave you underfunded.\n\n"
                f"What to do: Look into cottage food laws in your area — home kitchen + "
                f"direct-to-customer sales is the lowest-cost legal entry point."
            ),
            'socratic_hint': "Ask: have you checked your local food safety rules? What would you need to spend before your first legal sale?",
        })

    # CC004 — Food at school (regulatory barrier)
    if 'food' in cats and delivery == 'School':
        violations.append({
            'id': 'CC004', 'flag': 'food_at_school', 'severity': 'warning',
            'message': (
                "🟡 Selling food on school grounds usually requires written approval from the principal "
                "and must comply with your school's canteen policy — some schools prohibit student food sales entirely.\n\n"
                "What to do: Get written permission from your principal before investing in any stock."
            ),
            'socratic_hint': "Ask: have you spoken to your teacher or principal about selling food at school? What is the school's policy?",
        })

    # CC006 — Requires travel with zero budget
    if delivery == 'Requires Travel' and budget == 0:
        violations.append({
            'id': 'CC006', 'flag': 'travel_zero_budget', 'severity': 'warning',
            'message': (
                "🟡 Your business requires travel but your stated budget is $0. "
                "Transport costs add up and will erode your margins fast.\n\n"
                "What to do: Either include transport in your startup budget, or switch your "
                "delivery model to Online or Neighbourhood to keep costs near zero."
            ),
            'socratic_hint': "Ask: how will you get to your customers, and what will each trip cost you?",
        })

    # CC007 — Scale delusion (global / mass-market language)
    if 'scale_delusion' in cats:
        violations.append({
            'id': 'CC007', 'flag': 'scale_delusion', 'severity': 'warning',
            'message': (
                "🟡 Your idea describes a global or mass-market scale. Every successful business "
                "started with 5–10 real customers, not millions.\n\n"
                "What to do: Rewrite your idea to name your FIRST 10 customers — by location, "
                "age group, or specific description. Prove it works locally before thinking big."
            ),
            'socratic_hint': "Ask: who are your first 10 customers by name or description? A plan for 'everyone' is a plan for no one.",
        })

    # CC008 — Digital product with no experience and no budget
    if 'digital' in cats and budget == 0 and experience == 'Never sold':
        violations.append({
            'id': 'CC008', 'flag': 'digital_zero_base', 'severity': 'warning',
            'message': (
                "🟡 Building a digital product with no budget and no prior sales experience "
                "is a steep climb. Even free tools take weeks to learn, and most apps need hosting.\n\n"
                "What to do: Offer the service manually first — do it by hand for your first 5 "
                "customers, then automate once you've validated people will actually pay."
            ),
            'socratic_hint': "Ask: what specific skill do you have to build this? What is the simplest non-digital version of this service you could offer first?",
        })

    # CC009 — business_type=service/skills-for-hire but idea is physical product or manufacturing
    business_type = stage1_answers.get('business_type', '')
    if business_type in ('service', 'skills-for-hire') and ('physical_product' in cats or 'manufacturing' in cats):
        violations.append({
            'id': 'CC009', 'flag': 'service_type_product_idea', 'severity': 'warning',
            'message': (
                "🟡 You described your business type as a Service in Chapter 1, but your idea "
                "sounds like a physical product — these have very different cost structures.\n\n"
                "What to do: Either update Chapter 1 to 'Physical Product' if you're making something to sell, "
                "or refocus your idea on the service component (e.g., custom orders, personalisation)."
            ),
            'socratic_hint': "Ask: is this business mainly about doing something for people (service) or making something to sell (product)?",
        })

    # CC010 — Hours too low for food business
    if 'food' in cats and hours < 4:
        violations.append({
            'id': 'CC010', 'flag': 'low_hours_food', 'severity': 'warning',
            'message': (
                f"🟡 Running a food business typically takes 8–15 hours per week once prep, "
                f"selling time, and cleanup are included. Your current commitment of {hours} hr"
                f"{'s' if hours != 1 else ''}/week may not be enough.\n\n"
                "What to do: Either increase your available hours or simplify your product range "
                "so each batch takes less time to make and sell."
            ),
            'socratic_hint': "Ask: how long does it take to prep, sell, and clean up after one batch? Multiply by how often you'd sell each week.",
        })

    # CC011 — Hours too low for manufacturing-adjacent idea
    if 'manufacturing' in cats and hours < 3:
        violations.append({
            'id': 'CC011', 'flag': 'low_hours_manufacturing', 'severity': 'warning',
            'message': (
                f"🟡 Manufacturing or assembling products typically requires significant weekly time "
                f"for sourcing, production, quality checks, and sales. Your current {hours} hr"
                f"{'s' if hours != 1 else ''}/week commitment may not be realistic.\n\n"
                "What to do: Either revise your available hours upward or choose a simpler product "
                "that can be made and sold within your time budget."
            ),
            'socratic_hint': "Ask: how many units can you realistically produce in your available hours, and is that enough to make a profit?",
        })

    # ── Archetype-aware CC rules (CC012–CC015) ──────────────────────
    # These use ARCHETYPE_PARAMS to calibrate thresholds per business type.
    # The archetype is tagged by _detect_archetype() and stored in session.
    archetype = stage1_answers.get('_archetype', '')
    params    = ARCHETYPE_PARAMS.get(archetype, {})

    # CC012 — Production capacity: monthly units target vs available hours
    # Catches "200 units/month but only 4 hrs/week" for physical businesses
    if archetype and params:
        mins_per_unit = params.get('minutes_per_unit', 0)
        monthly_units = int(str(stage1_answers.get('monthly_units', '0') or '0').split('.')[0])
        if mins_per_unit > 0 and monthly_units > 0 and hours > 0:
            available_minutes = hours * 4 * 60   # hrs/week × ~4 weeks × 60 min
            required_minutes  = monthly_units * mins_per_unit
            if required_minutes > available_minutes * 1.5:
                realistic = int(available_minutes / mins_per_unit)
                violations.append({
                    'id': 'CC012', 'flag': 'capacity_gap', 'severity': 'warning',
                    'message': (
                        f"🟡 Capacity check ({archetype}): {monthly_units} units/month × "
                        f"~{mins_per_unit} min each = {required_minutes:,} minutes needed, "
                        f"but you only have ~{available_minutes:,} minutes/month available. "
                        f"Realistic output at your hours: ~{realistic} units/month.\n\n"
                        "What to do: Either reduce your monthly target, increase your hours, "
                        "or reduce how long each unit takes to make."
                    ),
                    'socratic_hint': "Ask: how many units can you realistically produce in your available time each month?",
                })

    # CC013 — Archetype extra flags: food safety, cold start, monetisation
    if archetype and params:
        for flag in params.get('extra_flags', []):
            if flag == 'food_safety' and 'CC003' not in [v['id'] for v in violations]:
                violations.append({
                    'id': 'CC013', 'flag': 'food_safety_reminder', 'severity': 'warning',
                    'message': (
                        "🟡 Food businesses need to check local food safety regulations before "
                        "selling — even at school markets. Most places require a food handler "
                        "certificate and approved kitchen.\n\n"
                        "What to do: Search '[your state/country] cottage food law' to understand "
                        "the legal requirements for selling homemade food."
                    ),
                    'socratic_hint': "Ask: have you checked the legal requirements for selling food in your area?",
                })
            elif flag == 'cold_start_problem':
                violations.append({
                    'id': 'CC014', 'flag': 'marketplace_cold_start', 'severity': 'warning',
                    'message': (
                        "🟡 Marketplace businesses need supply AND demand before launch — "
                        "buyers won't come if there are no sellers, and vice versa. "
                        "This 'cold start' problem is why most marketplaces fail.\n\n"
                        "What to do: Identify which side (buyers or sellers) you'll recruit "
                        "first, and how you'll guarantee the other side follows."
                    ),
                    'socratic_hint': "Ask: which side of your marketplace will you recruit first, and what incentive do you offer them?",
                })
            elif flag == 'monetisation_unclear':
                violations.append({
                    'id': 'CC015', 'flag': 'content_monetisation_path', 'severity': 'warning',
                    'message': (
                        "🟡 Content and media businesses (YouTube, newsletters, podcasts) "
                        "typically take 12–24 months to generate any revenue. Your money "
                        "chapter needs a clear monetisation path (ads, sponsorship, paid tier).\n\n"
                        "What to do: State exactly how this business makes money — and when "
                        "you'd realistically earn your first dollar."
                    ),
                    'socratic_hint': "Ask: how does this business make money, and how long will that take?",
                })

    # CC016: expert archetype — flag if no prior experience or credibility signal mentioned
    if archetype == 'expert':
        prior_exp = str(stage1_answers.get('prior_experience', '')).lower()
        has_credibility = any(kw in prior_exp for kw in [
            'experience', 'year', 'worked', 'trained', 'certified', 'studied',
            'degree', 'qualification', 'professional', 'industry', 'expert',
        ])
        if not has_credibility or prior_exp in ('', 'none', 'no experience'):
            violations.append({
                'id': 'CC016', 'flag': 'expert_credibility_gap', 'severity': 'warning',
                'message': (
                    "🟡 Expert businesses (consulting, coaching, strategic advice) rely on "
                    "credibility — customers need a reason to trust your advice over a professional's.\n\n"
                    "What to do: In your context chapter, describe what relevant experience, "
                    "training, or unique perspective makes you qualified to advise on this topic."
                ),
                'socratic_hint': "Ask: why would a paying customer take this student's advice?",
            })

    # CC020: Ch1 business_type (now an archetype value) vs Ch2 detected archetype
    if business_type and archetype:
        if not _archetypes_compatible(business_type, archetype) and archetype not in ('other', 'non-viable'):
            sel_label  = _ARCHETYPE_DISPLAY_LABELS.get(business_type, business_type)
            det_label  = _ARCHETYPE_DISPLAY_LABELS.get(archetype, archetype)
            violations.append({
                'id': 'CC020', 'flag': 'business_type_archetype_mismatch', 'severity': 'error',
                'message': (
                    f"🔴 Cross-chapter conflict: You selected '{sel_label}' as your business "
                    f"type in Chapter 1, but your Chapter 2 idea was classified as '{det_label}'. "
                    f"These chapters are telling different stories.\n\n"
                    f"👉 Update Chapter 1 or revise your idea in Chapter 2 to make them consistent."
                ),
                'socratic_hint': (
                    f"Ask: does the student's Chapter 1 business type ('{sel_label}') "
                    f"actually match the idea they described in Chapter 2 ('{det_label}')? "
                    "Which chapter needs to be updated to make them consistent?"
                ),
            })

    return violations


def validate_simple_rules(answers: dict, rules: list) -> list:
    """Generic rule evaluator for customer/money/discovery modules.
    Supports: blacklist_field, field_equals, min_length_field, min_value_field, derived_check."""
    violations = []
    for rule in rules:
        rtype    = rule.get('type')
        severity = rule.get('severity', 'error')
        field    = rule.get('field', '')
        val      = answers.get(field, '')
        failed   = False

        if rtype == 'blacklist_field':
            failed = str(val) in rule.get('blocked_values', [])

        elif rtype == 'field_equals':
            failed = str(val) in rule.get('blocked_values', [])

        elif rtype == 'min_length_field':
            failed = len(str(val).strip()) < rule.get('threshold', 0)

        elif rtype == 'min_value_field':
            try:
                failed = float(val or 0) < rule.get('threshold', 0)
            except (ValueError, TypeError):
                failed = True

        elif rtype == 'derived_check':
            # derived_field is the computed key injected into answers as 'Yes'/'No'
            derived_key = rule.get('derived_field') or field
            derived_val = answers.get(derived_key, '')
            failed = str(derived_val) in rule.get('blocked_values', [])

        if failed:
            violations.append({
                'id':            rule['id'],
                'flag':          rule.get('flag', rule['id']),
                'severity':      severity,
                'message':       rule['message'],
                'socratic_hint': rule.get('socratic_hint', ''),
            })
    return violations


def call_niche_pure_llm(niche_text: str, stage1_answers: dict) -> str:
    """Neutral LLM evaluator for niche — no rule awareness, self-reports score.
    The SCORE line is parsed to give the genuine LLM assessment score."""
    if not GROQ_AVAILABLE:
        return _NO_KEY_MSG
    profile = '\n'.join(f"{k}: {v}" for k, v in stage1_answers.items() if v)[:300]
    try:
        rsp = _groq_create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. A student has shared their business niche with you. "
                        "Respond encouragingly in 1–2 sentences — focus on what could work and show enthusiasm. "
                        "Then on a new line write exactly: SCORE: [0-100] reflecting your view of the niche's potential."
                    )
                },
                {
                    "role": "user",
                    "content": f"Student niche: \"{niche_text}\"\nProfile: {profile}\nEvaluate this niche:"
                }
            ],
            max_tokens=120,
            temperature=0.0
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}\nSCORE: 80"


def call_niche_socratic_llm(niche_text: str, violations: list, stage1_answers: dict) -> str:
    """Hybrid mentor: praises strong niches; asks one Socratic question for weak ones.
    This ensures hybrid_truth >= pure_truth for both good and bad plans."""
    if not GROQ_AVAILABLE:
        return _NO_KEY_MSG

    errors   = [v for v in violations if v['severity'] == 'error']
    cc_warns = [v for v in violations if v.get('severity') == 'warning']
    snippets = [f"{k}: {v}" for k, v in stage1_answers.items() if v]
    profile_context = '\n'.join(snippets[:6]) or "No prior context."

    # CC warnings with no hard errors → surface the warning as Mentor feedback
    # (LLM praise would be misleading when there's a real-world constraint)
    if not errors and cc_warns:
        primary = cc_warns[0]
        extra = cc_warns[1:]
        msg = primary['message']
        if extra:
            msg += '\n\nAlso check:\n' + '\n'.join(f"• {v['message']}" for v in extra)
        # Append a MENTOR_SCORE so triple_truth() doesn't fall back to the sentiment formula
        _has_disrespect = any(v.get('flag') == 'disrespectful_language' for v in cc_warns)
        _warn_score = 20 if _has_disrespect else 55
        msg += f'\nMENTOR_SCORE: {_warn_score}'
        return msg

    # No errors and no CC warnings → good niche. Give specific, warm praise.
    if not errors:
        try:
            # NOTE: Uses 8b instead of 70b to avoid rate-limiting on eval runs (25 personas).
            # Main call_hybrid_llm uses 70b — this is a known model-size inconsistency
            # documented as a limitation in Section 5.7 of the paper.
            rsp = _groq_create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an honest, encouraging business mentor for young entrepreneurs. "
                            "The student's niche has passed quality checks. Write exactly 2 sentences of "
                            "SPECIFIC, genuine praise that names what makes this niche strong "
                            "(specific customer, clear problem, findable location). "
                            "Be warm and concrete. End with a period, not a question mark. "
                            "IMPORTANT: If the description mentions any physically impossible market "
                            "(Mars, space, aliens, time travel) — do NOT praise that as creative or innovative. "
                            "Focus praise only on the real, grounded parts of the idea.\n"
                            "Then on a new line write exactly: MENTOR_SCORE: [number 75-100] "
                            "reflecting how strong this niche is (90-100 = very specific customer + clear problem + "
                            "findable location; 75-85 = passes but one element is thin)."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Student niche: \"{niche_text}\"\nContext: {profile_context}\nPraise this niche specifically:"
                    }
                ],
                max_tokens=140,
                temperature=0.0
            )
            return rsp.choices[0].message.content.strip()
        except Exception:
            _sym = _auditor.symbolic_from_violations(violations)
            _fb_score = min(92, max(65, round(50 + _sym * 0.4)))
            return f"Excellent niche — you've identified a specific customer with a real, findable problem. This clarity will make your marketing and outreach much easier!\nMENTOR_SCORE: {_fb_score}"

    # Has errors → return deterministic feedback based on the primary violation flag.
    # We do NOT call the LLM here because LLMs reliably find something positive to say
    # even when given explicit violation lists — this inflates the Mentor score and
    # defeats the purpose of the rules engine.
    #
    # IMPORTANT: for inappropriate content, return immediately without passing
    # the text to the LLM at all — the LLM must never process harmful input.
    # Sort errors so the most fundamental issue surfaces first.
    _flag_priority = {
        'inappropriate_content': 0,
        'unrealistic_concept': 1,
        'intangible_product': 2,
        'solution_problem_mismatch': 3,
        'non_human_customer': 4,
        'too_short': 5,
    }
    errors.sort(key=lambda v: _flag_priority.get(v.get('flag', ''), 99))
    primary = errors[0]
    flag = primary.get('flag', '')
    if flag == 'inappropriate_content':
        return (
            "🚫 This isn't a suitable business idea for this platform. "
            "LrnBiz is a school-safe environment — please describe a real, legal, "
            "and school-appropriate business idea."
            "\nMENTOR_SCORE: 5"
        )

    _violation_templates = {
        'inappropriate_content': (
            "🚫 This isn't a suitable business idea for this platform. "
            "LrnBiz is a school-safe environment — please describe a real, legal, "
            "and school-appropriate business idea."
        ),
        'missing_what': (
            "🔴 Missing product or service: you've said who you're selling to, but not what you're selling. "
            "Name the specific thing — handmade bracelets, tutoring sessions, baked goods? "
            "What would a customer actually hand you money for?"
        ),
        'missing_who': (
            "🔴 Missing customer: your description doesn't name a specific type of person. "
            "Who exactly is your customer — what's their age, role, or life situation? "
            "A 14-year-old student is not the same customer as a 40-year-old parent."
        ),
        'missing_context': (
            "🔴 Missing real location: saying 'on Earth' or 'everywhere' isn't a usable WHERE. "
            "Name the specific setting where your customer has this problem — "
            "at school, online, in the suburbs, at the gym?"
        ),
        'missing_need': (
            "🔴 Missing customer problem: 'I want to sell' is your goal, not your customer's need. "
            "What frustration, gap, or unmet desire does your customer have right now? "
            "Why would they pay you instead of doing nothing?"
        ),
        'vague_audience': (
            "🔴 Audience too broad: a niche that includes everyone serves no one. "
            "Who would NOT be your customer? Narrowing down is how you win — "
            "which specific type of person benefits most from what you're offering?"
        ),
        'intangible_product': (
            "🔴 Non-deliverable product: a feeling or abstract concept can't be handed to a customer. "
            "What would a customer actually receive? Describe a concrete product or service — "
            "e.g. a comfort box, motivational coaching session, or wellness journal."
        ),
        'non_human_customer': (
            "🔴 Non-human customer: animals, creatures, and fictional beings can't pay for things. "
            "If your product is for pets, your real customer is the pet owner — not the pet. "
            "Rewrite your niche: who is the human who would actually buy this?"
        ),
        'unrealistic_concept': (
            "🔴 Not a viable student business: your description mentions a location or market "
            "that real customers can't reach today (e.g. Mars, outer space, another dimension). "
            "What real-world version of this idea could you run at school, online, or in your area?"
        ),
        'too_short': (
            "🔴 Too vague to evaluate: this description is too short to analyse. "
            "Expand it — who specifically is your customer, where do they have this problem, "
            "what are they selling, and what do they need?"
        ),
        'too_generic': (
            "⚠️ Too many buzzwords: 'cheap', 'easy', 'amazing' are not a business insight. "
            "Replace one adjective with a real, specific example of your customer's frustration. "
            "What does this cost them in time, money, or stress right now?"
        ),
    }

    # Flags that are so fundamental that listing other violations adds nothing —
    # the student needs to fix this one thing first before anything else matters.
    _blocking_flags = {
        'inappropriate_content',
        'unrealistic_concept',
        'intangible_product',
        'solution_problem_mismatch',
        'non_human_customer',
        'too_short',
    }

    # Promote the most fundamental violation to primary if a later error is more critical.
    for v in violations:
        if v.get('flag') in _blocking_flags and v.get('flag') != flag:
            primary = v
            flag = v['flag']
            break

    primary_msg = _violation_templates.get(flag) or primary.get('message') or f"🔴 {primary['message']}"

    # Compute rules-grounded score to anchor triple_truth — prevents sentiment fallback
    _sym = _auditor.symbolic_from_violations(violations)

    # Blocking issues: show only the single most important problem.
    # No "Also fix" list — the student fixes this first, then resubmits.
    if flag in _blocking_flags:
        return f"{primary_msg}\nMENTOR_SCORE: {_sym}"

    # Non-blocking: lead with primary, then at most 2 secondary issues (not a wall of text).
    extra_flags = [v for v in violations if v != primary and v.get('flag') not in _blocking_flags][:2]
    if extra_flags:
        extra_lines = '\n'.join(
            f"• {_violation_templates.get(v.get('flag',''), v['message'])}"
            for v in extra_flags
        )
        return f"{primary_msg}\n\nAlso fix:\n{extra_lines}\nMENTOR_SCORE: {_sym}"
    return f"{primary_msg}\nMENTOR_SCORE: {_sym}"


def call_customer_pure_llm(answers: dict) -> str:
    """Neutral LLM evaluator for customer profile — no rule awareness, self-reports score."""
    if not GROQ_AVAILABLE:
        return _NO_KEY_MSG
    _CUSTOMER_EXCLUDE = {'hours_per_week', 'budget', 'archetype', 'business_type', '_business_idea'}
    profile = '\n'.join(f"{k}: {v}" for k, v in answers.items()
                        if v not in (None, '', 'Not sure', 'Not sure yet') and k not in _CUSTOMER_EXCLUDE)
    try:
        rsp = _groq_create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful AI assistant. A student has shared their target customer profile. "
                    "Respond encouragingly in 1–2 sentences — focus on what looks promising. "
                    "Then on a new line write exactly: SCORE: [0-100] reflecting your view of this customer profile's strength."
                )},
                {"role": "user", "content":
                    f"Student's customer profile:\n{profile}\n\nEvaluate this customer target:"}
            ],
            max_tokens=130, temperature=0.0
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}\nSCORE: 78"


def call_money_pure_llm(answers: dict) -> str:
    """Encouraging LLM for the money/pricing chapter.
    Frames the data as a financial plan, not a business idea — so the AI praises
    the student's financial thinking rather than judging viability of an 'idea'."""
    if not GROQ_AVAILABLE:
        return _NO_KEY_MSG
    profile = '\n'.join(f"{k}: {v}" for k, v in answers.items() if v not in (None, ''))
    try:
        rsp = _groq_create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful AI assistant. A student has shared their pricing and financial plan. "
                    "Respond encouragingly in exactly 2 sentences — praise the fact that they are thinking "
                    "about money and pricing, and highlight what looks promising in their numbers. "
                    "Then on a new line write exactly: SCORE: [0-100] reflecting your enthusiasm "
                    "for this student's financial planning effort."
                )},
                {"role": "user", "content":
                    f"Student's financial plan:\n{profile}\n\nWhat do you think of this pricing and financial plan?"}
            ],
            max_tokens=150, temperature=0.0
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}\nSCORE: 80"


def call_discovery_pure_llm(answers: dict) -> str:
    """Encouraging LLM for the discovery/research chapter.
    Frames the data as customer research progress — so the AI encourages the student's
    effort to talk to customers rather than judging a 'business idea'."""
    if not GROQ_AVAILABLE:
        return _NO_KEY_MSG
    profile = '\n'.join(f"{k}: {v}" for k, v in answers.items() if v not in (None, ''))
    try:
        rsp = _groq_create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful AI assistant. A student has shared their customer research and discovery work. "
                    "Respond encouragingly in exactly 2 sentences — celebrate any effort to talk to real customers "
                    "and highlight what looks promising in their research approach. "
                    "Then on a new line write exactly: SCORE: [0-100] reflecting your enthusiasm "
                    "for this student's customer discovery effort."
                )},
                {"role": "user", "content":
                    f"Student's customer research:\n{profile}\n\nWhat do you think of this customer discovery work?"}
            ],
            max_tokens=150, temperature=0.0
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}\nSCORE: 80"


def call_customer_hybrid_llm(answers: dict, violations: list) -> str:
    """Honest mentor for customer profile: praises clear targets, asks Socratic question for vague ones."""
    if not GROQ_AVAILABLE:
        return _NO_KEY_MSG
    # Strip cross-chapter keys injected for contradiction detection — they are not customer data
    # and cause the LLM to invent questions about hours/week, budget, archetype, etc.
    _CUSTOMER_EXCLUDE = {'hours_per_week', 'budget', 'archetype', 'business_type', '_business_idea'}
    profile = '\n'.join(f"{k}: {v}" for k, v in answers.items()
                        if v not in (None, '', 'Not sure', 'Not sure yet') and k not in _CUSTOMER_EXCLUDE)
    errors = [v for v in violations if v.get('severity') == 'error']
    if not errors:
        try:
            # NOTE: Uses 8b to avoid rate-limiting. Known model-size inconsistency vs
            # main call_hybrid_llm (70b). Documented in paper Section 5.7.
            rsp = _groq_create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": (
                        "You are an honest, encouraging business mentor for young entrepreneurs. "
                        "The student's customer profile has passed all checks. "
                        "Write exactly 2 sentences of SPECIFIC, genuine praise that names their exact "
                        "customer age group, problem they solve, and how they'll reach them. "
                        "Be concrete and warm. End with a period. "
                        "Then on a new line write exactly: MENTOR_SCORE: [number 0-100] "
                        "where the number is your honest assessment of how well-defined this customer profile is. "
                        "Judge on specificity, clarity and real-world viability — not just rule compliance. "
                        "A thin but rule-passing profile scores 50-65. A specific, detailed, convincing profile scores 75-90."
                    )},
                    {"role": "user", "content": f"Customer profile:\n{profile}\n\nPraise this specifically:"}
                ],
                max_tokens=150, temperature=0.0
            )
            return rsp.choices[0].message.content.strip()
        except Exception:
            _sym = _auditor.symbolic_from_violations(violations)
            _fb_score = min(88, max(55, round(45 + _sym * 0.4)))
            return f"You've identified a specific customer with a real problem and a clear plan to reach them — that's the foundation of a successful business!\nMENTOR_SCORE: {_fb_score}"
    primary = errors[0]
    hint = primary.get('socratic_hint') or primary.get('message', 'Who exactly is your ideal customer, and where can you find them?')
    _sym = _auditor.symbolic_from_violations(violations)
    _fb_score = max(20, min(55, 100 - len(errors) * 25 - (len(violations) - len(errors)) * 8))
    # Use pre-written socratic_hint directly — avoids LLM inventing off-topic questions.
    return f"{hint}\nMENTOR_SCORE: {_fb_score}"


# ── API: Generate 3 Socratic hints for rough idea ──────────────
@app.route('/api/idea_hints', methods=['POST'])
def api_idea_hints():
    """Given a rough idea, return 3 Socratic hints to help the student refine it."""
    try:
        rough_idea = request.json.get('rough_idea', '').strip()
        context    = request.json.get('context', {})
        if not rough_idea:
            return jsonify({'error': 'No idea provided'}), 400
        if not GROQ_AVAILABLE:
            return jsonify({'hints': [
                {'type': 'who',     'emoji': '🧑‍🤝‍🧑', 'title': 'Who exactly?',       'question': 'Think about a specific type of person — what age, situation, or role defines your ideal customer?'},
                {'type': 'problem', 'emoji': '😤',       'title': 'What\'s the real pain?', 'question': 'What specific frustration or problem does this person experience right now that your idea solves?'},
                {'type': 'edge',    'emoji': '⭐',        'title': 'What\'s your edge?',  'question': 'Why would someone choose you over existing options — what can you offer that they can\'t get elsewhere?'},
            ]})
        ctx_summary = ', '.join(f'{k}={v}' for k, v in context.items() if v)[:200]
        rsp = _groq_create(
            model='llama-3.1-8b-instant',
            messages=[
                {'role': 'system', 'content': (
                    'You are a Socratic business mentor for school-age entrepreneurs. '
                    'Given a student\'s rough business idea, generate exactly 3 short, targeted questions '
                    'to help them think deeper — one about WHO (target customer), one about PROBLEM (specific pain), '
                    'one about EDGE (why choose you). '
                    'Format your response as JSON: {"who": "...", "problem": "...", "edge": "..."}. '
                    'Keep each question under 25 words. Be friendly, direct, not preachy.'
                )},
                {'role': 'user', 'content': f'Student idea: "{rough_idea}"\nContext: {ctx_summary}\nGenerate the 3 hint questions.'}
            ],
            max_tokens=200,
            temperature=0.0,
            response_format={'type': 'json_object'},
        )
        _FALLBACK_HINTS = [
            {'type': 'who',     'emoji': '🧑‍🤝‍🧑', 'title': 'Who exactly?',           'question': 'Who is one specific person (age, role, situation) who would pay for this — and who would NOT?'},
            {'type': 'problem', 'emoji': '😤',       'title': "What's the real pain?",   'question': 'What frustration or problem does your customer have right now that your idea actually fixes?'},
            {'type': 'edge',    'emoji': '⭐',        'title': "What's your edge?",       'question': 'Why would someone choose you over a shop or app that already exists — what can YOU offer that they cannot?'},
        ]
        try:
            raw  = json.loads(rsp.choices[0].message.content)
            hints = [
                {'type': 'who',     'emoji': '🧑‍🤝‍🧑', 'title': 'Who exactly?',           'question': raw.get('who', '')     or _FALLBACK_HINTS[0]['question']},
                {'type': 'problem', 'emoji': '😤',       'title': "What's the real pain?",   'question': raw.get('problem', '') or _FALLBACK_HINTS[1]['question']},
                {'type': 'edge',    'emoji': '⭐',        'title': "What's your edge?",       'question': raw.get('edge', '')    or _FALLBACK_HINTS[2]['question']},
            ]
        except (json.JSONDecodeError, AttributeError):
            hints = _FALLBACK_HINTS
        return jsonify({'hints': hints})
    except Exception:
        # Never crash — always return fallback hints
        return jsonify({'hints': [
            {'type': 'who',     'emoji': '🧑‍🤝‍🧑', 'title': 'Who exactly?',           'question': 'Who is one specific person (age, role, situation) who would pay for this — and who would NOT?'},
            {'type': 'problem', 'emoji': '😤',       'title': "What's the real pain?",   'question': 'What frustration or problem does your customer have right now that your idea actually fixes?'},
            {'type': 'edge',    'emoji': '⭐',        'title': "What's your edge?",       'question': 'Why would someone choose you over a shop or app that already exists — what can YOU offer that they cannot?'},
        ]})


# ── API: Validate niche + triple truth scoring ─────────────────
@app.route('/api/validate_niche', methods=['POST'])
def api_validate_niche():
    _rl = _check_llm_rate_limit()
    if _rl:
        return jsonify({'error': f'Too many requests — please wait {_rl}s before re-running analysis.', 'retry_after': _rl}), 429
    try:
        data           = request.json
        niche_text     = data.get('niche_text', '').strip()
        stage1_answers = data.get('stage1_answers', {})

        niche_rules_data = _load_json('niche_rules.json')
        rules            = niche_rules_data.get('rules', [])
        violations       = validate_niche(niche_text, rules)
        violations      += validate_cross_chapter_conflicts(niche_text, stage1_answers)

        # ── Semantic niche classification (complements keyword rules) ──
        # Catches paraphrased copycat / price-only / mass-market patterns
        # that slip through the blacklists (e.g. "inspired by Uber's model").
        # Uses the same Groq client at temp=0 — no new dependencies.
        # Only runs if text is long enough to be meaningful and no hard errors already.
        _existing_flags = {v.get('flag') for v in violations}
        if (GROQ_AVAILABLE and len(niche_text) > 40
                and 'inappropriate_content' not in _existing_flags
                and 'too_short' not in _existing_flags):
            try:
                _sem_rsp = _groq_create(
                    model="llama-3.1-8b-instant",
                    temperature=0.0,
                    max_tokens=60,
                    messages=[{
                        "role": "user",
                        "content": (
                            "Classify this student business niche description. "
                            "Reply ONLY with a valid JSON object, no explanation, no markdown.\n\n"
                            f'Text: "{niche_text[:400]}"\n\n'
                            '{"is_copycat": false, "is_price_only": false, "is_mass_market": false}'
                        )
                    }]
                )
                _sem_raw = _sem_rsp.choices[0].message.content.strip()
                # strip markdown code fences if model wraps in ```json ... ```
                if _sem_raw.startswith('```'):
                    _sem_raw = _sem_raw.split('```')[-2].strip().lstrip('json').strip()
                _sem = json.loads(_sem_raw)

                if _sem.get('is_copycat') and 'pure_copycat' not in _existing_flags:
                    violations.append({
                        'id': 'NR013S', 'flag': 'pure_copycat', 'severity': 'warning',
                        'message': (
                            "Your idea appears to closely follow an existing platform or business model "
                            "without a clear reason why customers would choose you over the original.\n\n"
                            "What to do: Name one specific thing the existing version gets wrong for your "
                            "customer — that gap is your differentiation."
                        ),
                        'socratic_hint': (
                            "The student's niche appears to copy an existing business model. "
                            "Ask: what does the established version fail to deliver for this specific customer? "
                            "That failure is the only reason to start a competing business."
                        ),
                    })

                if _sem.get('is_price_only') and 'price_only_niche' not in _existing_flags:
                    violations.append({
                        'id': 'NR014S', 'flag': 'price_only_niche', 'severity': 'warning',
                        'message': (
                            "Being cheaper seems to be the main reason a customer would choose you. "
                            "Price alone is not a sustainable advantage — a bigger competitor can always "
                            "undercut you.\n\n"
                            "What to do: What is one non-price reason your specific customer would choose "
                            "you — speed, personalisation, local knowledge, a skill?"
                        ),
                        'socratic_hint': (
                            "The student's value proposition relies primarily on price. "
                            "Ask: if a larger business charged the same price tomorrow, why would "
                            "customers still choose you?"
                        ),
                    })

                if _sem.get('is_mass_market') and 'vague_audience' not in _existing_flags:
                    violations.append({
                        'id': 'NR002S', 'flag': 'vague_audience', 'severity': 'error',
                        'message': (
                            "Your description sounds like it targets everyone, which means it targets no one. "
                            "A real niche must exclude someone.\n\n"
                            "What to do: Who would NOT be your customer? Naming who you're not for makes "
                            "your actual customer much clearer."
                        ),
                        'socratic_hint': (
                            "The student is targeting a mass-market audience without specific criteria. "
                            "Ask: who would not want this product or service? Exclusion defines a niche."
                        ),
                    })
            except Exception:
                pass  # semantic check is best-effort — never block validation on LLM failure
        # ── end semantic check ──

        has_errors       = any(v['severity'] == 'error' for v in violations)
        symbolic_score   = _auditor.symbolic_from_violations(violations, len(rules))

        if not has_errors:
            # PASS — run both LLMs, compute triple truth, save to session
            pure_text   = call_niche_pure_llm(niche_text, stage1_answers)
            hybrid_text = call_niche_socratic_llm(niche_text, violations, stage1_answers)
            triple      = _auditor.triple_truth(symbolic_score, pure_text, hybrid_text, 'niche')
            _save_audit_with_delta('idea', triple, violations)
            save_research_log('idea', symbolic_score, violations,
                              pure_text, triple['pure_truth'],
                              hybrid_text, triple['hybrid_truth'],
                              triple['sycophancy_gap'])
            radar = _update_chapter_radar('Target', triple['hybrid_truth'])
            return jsonify({
                'passed':         True,
                'violations':     violations,
                'total_rules':    len(rules),
                'symbolic_score': symbolic_score,
                'pure_llm':       pure_text,
                'hybrid_llm':     _strip_mentor_score_line(hybrid_text),
                'triple':         triple,
                'radar_scores':   radar,
            })

        # FAIL — deterministic mentor feedback + pure LLM for sycophancy gap display
        socratic_q = call_niche_socratic_llm(niche_text, violations, stage1_answers)
        pure_text_fail  = call_niche_pure_llm(niche_text, stage1_answers)
        _clean_fail, _llm_fail = _parse_llm_score(pure_text_fail)
        pure_truth_fail = _llm_fail if _llm_fail is not None else _ai_optimist_score(symbolic_score)
        pure_text_fail  = _clean_fail
        # Mentor = min(rules score, pure LLM score) — AI Optimist must always be most generous
        hybrid_truth_fail = min(symbolic_score, pure_truth_fail)
        _inappropriate  = any(v.get('flag') == 'inappropriate_content' for v in violations)
        _impossible     = any(v.get('flag') in ('unrealistic_concept', 'non_human_customer') for v in violations)
        _disrespectful  = any(v.get('flag') == 'disrespectful_language' for v in violations)
        if _inappropriate:
            hybrid_truth_fail = 0
        elif _impossible:
            hybrid_truth_fail = min(hybrid_truth_fail, 20)
        elif _disrespectful:
            hybrid_truth_fail = 20
        fail_triple = {
            'symbolic_score': symbolic_score,
            'pure_truth':     pure_truth_fail,
            'hybrid_truth':   hybrid_truth_fail,
            'sycophancy_gap': pure_truth_fail - hybrid_truth_fail,
        }
        return jsonify({
            'passed':            False,
            'violations':        violations,
            'total_rules':       len(rules),
            'symbolic_score':    symbolic_score,
            'pure_llm':          pure_text_fail,
            'hybrid_llm':        _strip_mentor_score_line(socratic_q),
            'socratic_question': socratic_q,
            'triple':            fail_triple,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── API: Analyse free-text business idea ───────────────────────
@app.route('/api/analyse_idea', methods=['POST'])
def api_analyse_idea():
    """Analyse a raw idea string, return missing elements + battery score + one-sentence tip."""
    _rl = _check_llm_rate_limit()
    if _rl:
        return jsonify({'error': f'Too many requests — please wait {_rl}s before re-running analysis.', 'retry_after': _rl}), 429
    try:
        idea_text = request.json.get('idea_text', '').strip()
        stage1    = request.json.get('stage1_answers', {})
        if not idea_text:
            return jsonify({'error': 'No idea provided'}), 400

        niche_rules_data = _load_json('niche_rules.json')
        rules       = niche_rules_data.get('rules', [])
        violations  = validate_niche(idea_text, rules)
        violations += validate_cross_chapter_conflicts(idea_text, stage1)

        # Archetype tagging first — solution alignment uses archetype as context (C9-13)
        archetype = _detect_archetype(idea_text, stage1)
        # O10-06: invalidate downstream chapter data if archetype changed significantly
        old_archetype = session.get('idea_archetype', '')
        if old_archetype and old_archetype != archetype and old_archetype not in ('other', 'non-viable'):
            _ARCHETYPE_PROXIMITY_GROUPS = {
                frozenset(['physical-product', 'food-beverage', 'resale-retail', 'maker']): 'product',
                frozenset(['service', 'skills-for-hire', 'event-experience', 'expert']): 'service',
                frozenset(['digital-product', 'content-media', 'marketplace']): 'digital',
            }
            old_group = next((g for s, g in _ARCHETYPE_PROXIMITY_GROUPS.items() if old_archetype in s), None)
            new_group = next((g for s, g in _ARCHETYPE_PROXIMITY_GROUPS.items() if archetype in s), None)
            if old_group != new_group:
                # Major archetype group change — clear customer chapter answers since proximity/channel options differ
                for k in ('customer_location', 'how_to_reach', 'audit_scores'):
                    if k == 'audit_scores':
                        audit = session.get('audit_scores', {})
                        audit.pop('customer', None)
                        session['audit_scores'] = audit
                    else:
                        session.pop(k, None)
        session['idea_archetype'] = archetype
        session.setdefault('stage1_answers', {})['_archetype'] = archetype
        session.modified = True

        # Solution alignment — runs after archetype so LLM has business-model context
        violations += _check_solution_alignment(idea_text, archetype)
        symbolic_score = _auditor.symbolic_from_violations(violations, len(rules))

        # non-viable archetype: inject a violation so the rules engine treats it as an error
        # even if the text somehow passed the NR blacklists (jokes, pranks, etc. slip through)
        if archetype == 'non-viable' and not any(
            v.get('flag') in ('inappropriate_content', 'unrealistic_concept') for v in violations
        ):
            violations.append({
                'id': 'NR012',
                'flag': 'non_viable_concept',
                'severity': 'error',
                'message': (
                    "📌 This doesn't look like a real business idea — it reads like a joke, "
                    "a prank, something physically impossible, or something unsafe. "
                    "What real product or service could you offer to a paying customer?"
                ),
            })
            symbolic_score = _auditor.symbolic_from_violations(violations, len(rules))

        # Derive element presence from violations — single source of truth.
        # The rules engine (including LLM fallback) already determined what's missing.
        # Never duplicate that logic here with a separate keyword list.
        violation_flags = {v['flag'] for v in violations}
        # Blocking flags that invalidate the entire niche — all elements are unresolved.
        _all_missing_flags = {'too_short', 'intangible_product', 'inappropriate_content', 'unrealistic_concept', 'non_human_customer'}
        if violation_flags & _all_missing_flags:
            who_found = where_found = prob_found = False
        else:
            who_found   = 'missing_who'      not in violation_flags
            where_found = 'missing_context'  not in violation_flags
            prob_found  = 'missing_need'     not in violation_flags

        elements = [
            {'type':'who',     'emoji':'🧑',  'title':'Who is your customer?',
             'found': who_found,
             'hint': 'Think about a specific type of person — what is their age, situation, or role?'},
            {'type':'problem', 'emoji':'😤',  'title':'What problem do you fix?',
             'found': prob_found,
             'hint': 'Describe the real frustration or pain your customer experiences right now.'},
            {'type':'where',   'emoji':'📍',  'title':'Where will customers find you?',
             'found': where_found,
             'hint': 'Where will you sell — at school, online, at local events? And when do your customers need you most?'},
        ]

        found_main = sum(1 for e in elements if e['found'])
        battery = min(100, found_main * 30 + (symbolic_score // 5))

        # For inappropriate content: never send the text to any LLM.
        _has_inappropriate = any(v.get('flag') == 'inappropriate_content' for v in violations)
        if _has_inappropriate:
            pure_text   = "This business idea is not appropriate for evaluation."
            hybrid_text = (
                "🚫 This isn't a suitable business idea for this platform. "
                "LrnBiz is a school-safe environment — please describe a real, legal, "
                "and school-appropriate business idea."
            )
        else:
            # Run full triple truth LLM analysis (same as validate_niche does at lock-in)
            # Only the pure LLM runs concurrently; hybrid feedback is now deterministic when violations exist.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _pool:
                _f_pure   = _pool.submit(call_niche_pure_llm, idea_text, stage1)
                _f_hybrid = _pool.submit(call_niche_socratic_llm, idea_text, violations, stage1)
                pure_text   = _f_pure.result()
                hybrid_text = _f_hybrid.result()

        # Parse the LLM's self-reported score from pure_text — genuine sycophancy measurement.
        # When violations exist, Mentor = symbolic_score (rules ground truth); pure = real LLM score.
        _clean_pure, _llm_pure_score = _parse_llm_score(pure_text)
        pure_text = _clean_pure  # strip SCORE line from displayed text
        if violations:
            pure_truth   = _llm_pure_score if _llm_pure_score is not None else _ai_optimist_score(symbolic_score)
            # Mentor is capped at symbolic (rules ground truth) AND at pure_truth.
            # The AI Optimist must always be the most generous score — if even the
            # optimist gives 62 for "i am tired", the mentor can't report 75.
            hybrid_truth = min(symbolic_score, pure_truth)
            triple = {
                'module_id':         'niche',
                'symbolic_score':    symbolic_score,
                'pure_truth':        pure_truth,
                'hybrid_truth':      hybrid_truth,
                'sycophancy_gap':    pure_truth - hybrid_truth,
                'pure_score_source': 'llm' if _llm_pure_score is not None else 'fallback',
            }
        else:
            triple = _auditor.triple_truth(symbolic_score, pure_text, hybrid_text, 'niche')

        # Hard-cap Mentor score for inappropriate / impossible / non-human concepts
        _inappropriate  = any(v.get('flag') == 'inappropriate_content' for v in violations)
        _impossible     = any(v.get('flag') in ('unrealistic_concept', 'non_human_customer') for v in violations)
        _intangible     = any(v.get('flag') == 'intangible_product' for v in violations)
        _disrespectful  = any(v.get('flag') == 'disrespectful_language' for v in violations)
        if _inappropriate:
            triple['hybrid_truth']   = 0
            triple['sycophancy_gap'] = triple['pure_truth'] - 0
        elif _impossible or _intangible:
            triple['hybrid_truth']   = min(triple['hybrid_truth'], 20)
            triple['sycophancy_gap'] = triple['pure_truth'] - triple['hybrid_truth']
        elif _disrespectful:
            triple['hybrid_truth']   = 20
            triple['sycophancy_gap'] = triple['pure_truth'] - 20

        _save_audit_with_delta('idea', triple, violations)
        save_research_log('idea', symbolic_score, violations,
                          pure_text, triple['pure_truth'],
                          hybrid_text, triple['hybrid_truth'],
                          triple['sycophancy_gap'])

        # Keep a concise tip for the kid-view mentor bubble (fallback to hybrid text)
        tip = hybrid_text

        # O17-02: Signal archetype change to frontend so it can prompt the student
        # to refresh Chapter 3 and Chapter 4 guidance.
        _prev_archetype = request.json.get('stage1_answers', {}).get('_archetype', '')
        _archetype_changed = bool(_prev_archetype and _prev_archetype != archetype
                                  and _prev_archetype not in ('other', 'non-viable'))

        return jsonify({
            'battery':          battery,
            'elements':         elements,
            'violations':       violations,
            'total_rules':      len(rules),
            'symbolic_score':   symbolic_score,
            'tip':              tip,
            'pure_llm':         pure_text,
            'hybrid_llm':       _strip_mentor_score_line(hybrid_text),
            'triple':           triple,
            'archetype':        archetype,
            'archetype_changed': _archetype_changed,
            'prev_archetype':   _prev_archetype,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── API: Generate rival cards via LLM ─────────────────────────
@app.route('/api/idea_rivals', methods=['POST'])
def api_idea_rivals():
    FALLBACK = [
        {'name':'Local Shop',  'emoji':'🏪', 'strength':'Already trusted by people nearby',           'weakness':'Does not specialise in your exact customer need'},
        {'name':'Big Brand',   'emoji':'🏬', 'strength':'Large budget and wide marketing reach',       'weakness':'Too expensive and impersonal for your specific niche'},
        {'name':'Online Store','emoji':'📦', 'strength':'Available 24/7 from anywhere',               'weakness':'Slow delivery and no personal touch'},
    ]
    try:
        idea_text = request.json.get('idea_text', '').strip()
        if not idea_text or not GROQ_AVAILABLE:
            return jsonify({'rivals': FALLBACK[:2]})
        rsp = _groq_create(
            model='llama-3.1-8b-instant',
            messages=[
                {'role': 'system', 'content': (
                    'You are a business teacher for school students (ages 10-16). '
                    'Given a student\'s business idea, name 2-3 rival TYPES (not specific brand names — '
                    'use categories like "school canteen", "online marketplace", "local corner shop"). '
                    'For each rival: a short name (2-3 words), a relevant emoji, their ONE strength, '
                    'and their ONE weakness your student can exploit. '
                    'Return JSON: {"rivals": [{"name":"...","emoji":"...","strength":"...","weakness":"..."}]}. '
                    'Keep each field under 12 words. Be specific to the idea.'
                )},
                {'role': 'user', 'content': f'Student idea: "{idea_text[:200]}". Generate rival cards.'}
            ],
            max_tokens=350, temperature=0.0,
            response_format={'type': 'json_object'},
        )
        data   = json.loads(rsp.choices[0].message.content)
        rivals = data.get('rivals', [])
        if not rivals or not isinstance(rivals, list):
            rivals = FALLBACK[:2]
        return jsonify({'rivals': rivals})
    except Exception:
        return jsonify({'rivals': FALLBACK[:2]})


# ── API: Parse idea text into structured sentence fields ────────
@app.route('/api/parse_idea_sentence', methods=['POST'])
def api_parse_idea_sentence():
    """Extract sell_what/sell_to/sell_where/sell_price from ideaText + element answers."""
    try:
        idea_text = request.json.get('idea_text', '').strip()
        elem_answers = request.json.get('element_answers', {})
        combined = (idea_text + ' ' + ' '.join(v for v in elem_answers.values() if v))[:400]

        fallback = {
            'sell_what':  elem_answers.get('what', ''),
            'sell_to':    elem_answers.get('who', ''),
            'sell_where': elem_answers.get('where', ''),
            'sell_price': '',
        }

        if not GROQ_AVAILABLE:
            return jsonify(fallback)

        rsp = _groq_create(
            model='llama-3.1-8b-instant',
            messages=[
                {'role': 'system', 'content': (
                    'Extract business idea fields from student text. '
                    'Return JSON with exactly these keys: '
                    '"sell_what" (what is being sold, 3-6 words), '
                    '"sell_to" (who the customer is, 3-6 words), '
                    '"sell_where" (place/context, 2-5 words), '
                    '"sell_price" (leave empty string). '
                    'If a field is unclear, use empty string. Be concise.'
                )},
                {'role': 'user', 'content': f'Student idea: "{combined}"\nExtract the 4 fields:'}
            ],
            max_tokens=150, temperature=0.0,
            response_format={'type': 'json_object'},
        )
        data = json.loads(rsp.choices[0].message.content)
        return jsonify({
            'sell_what':  data.get('sell_what', fallback['sell_what']),
            'sell_to':    data.get('sell_to',   fallback['sell_to']),
            'sell_where': data.get('sell_where',fallback['sell_where']),
            'sell_price': data.get('sell_price',''),
        })
    except Exception:
        return jsonify({'sell_what': '', 'sell_to': '', 'sell_where': '', 'sell_price': ''})


# ── Certificate of Innovation ──────────────────────────────────
@app.route('/certificate')
def certificate():
    # Fallback chain so certificate always has something meaningful to show
    problem = (session.get('problem_solved') or
               session.get('niche_description') or
               'a real everyday problem')
    idea_data = {
        'sell_what':      session.get('sell_what') or session.get('business_type') or 'an innovative product',
        'sell_to':        session.get('sell_to') or 'their community',
        'sell_where':     session.get('sell_where') or session.get('location') or '',
        'problem_solved': problem,
        'student_name':   session.get('student_name') or 'Young Entrepreneur',
        'grade':          session.get('grade_level') or '',
        'modules_done':   len(session.get('audit_scores', {})),
    }
    return render_template('certificate.html', idea_data=idea_data)


# ── Journey Progress Dashboard ────────────────────────────────
@app.route('/progress')
def progress():
    audit_scores = session.get('audit_scores', {})
    story_data = {
        'sell_what':           session.get('sell_what', ''),
        'sell_to':             session.get('sell_to', ''),
        'sell_where':          session.get('sell_where', ''),
        'sell_price':          session.get('sell_price') or session.get('unit_price') or '',
        'customer_age_range':  session.get('customer_age_range', ''),
        'customer_problem':    session.get('customer_problem', ''),
        'differentiation':     session.get('differentiation', ''),
        'startup_cost':        session.get('startup_cost', ''),
        'unit_cost':           session.get('unit_cost', ''),
        'unit_price':          session.get('unit_price', ''),
        'monthly_units':       session.get('monthly_units', ''),
        'interviews_completed':session.get('interviews_completed', ''),
        'insight_applied':     session.get('insight_applied', ''),
        'grade_level':         session.get('grade_level', ''),
        'location':            session.get('location', ''),
        'prior_experience':    session.get('prior_experience', ''),
        'student_name':        session.get('student_name', ''),
    }
    return render_template(
        'progress.html',
        modules=MODULE_REGISTRY,
        audit_scores=audit_scores,
        audit_scores_json=json.dumps(audit_scores),
        story=story_data,
        idea_archetype=session.get('idea_archetype', 'other'),
    )


# ── API: Validate customer + triple truth ─────────────────────
@app.route('/api/validate_customer', methods=['POST'])
def api_validate_customer():
    _rl = _check_llm_rate_limit()
    if _rl:
        return jsonify({'error': f'Too many requests — please wait {_rl}s before re-running analysis.', 'retry_after': _rl}), 429
    try:
        data    = request.json
        answers = data.get('answers', {})
        # Pass the student's actual business idea text to the LLM (not stale structured fields)
        idea_context = (session.get('niche_description') or session.get('problem_solved') or
                        session.get('sell_what') or '')
        if idea_context:
            answers['_business_idea'] = idea_context
        rules_data = _load_json('customer_rules.json')
        rules   = rules_data.get('rules', [])
        violations = validate_simple_rules(answers, rules)
        # Contradiction detection — inject archetype + stage1 context
        answers['archetype']    = session.get('idea_archetype', '')
        answers['business_type']= session.get('business_type', '')
        answers['hours_per_week']= session.get('hours_per_week', '')
        answers['budget']       = session.get('budget', '')
        violations += _detect_contradictions(answers)

        # O10-05: Proximity vs Ch1 location cross-check
        ch1_loc   = session.get('location', '')
        proximity = answers.get('customer_location', '')
        _PROX_CONFLICTS = {
            ('Rural', 'City-wide'):
                "You said you're in a Rural area (Chapter 1) but selected City-wide customer proximity — these don't match. Is your business truly city-wide, or more local?",
            ('Rural', 'National'):
                "Rural location (Chapter 1) with National reach usually requires an online component. Is your business online-enabled?",
        }
        if (ch1_loc, proximity) in _PROX_CONFLICTS:
            violations.append({'id': 'CC007', 'flag': 'proximity_location_mismatch', 'severity': 'warning',
                'message': _PROX_CONFLICTS[(ch1_loc, proximity)],
                'socratic_hint': 'Does the reach you selected for customers actually match where you said you live?'})

        # O10-07 / O13 (CR008): Vague persona detection — no upper length cap
        persona = answers.get('customer_problem', '')
        if len(persona) > 10:
            _VAGUE = {'people', 'everyone', 'anyone', 'all', 'somebody', 'those', 'customers', 'users', 'humans'}
            _SPECIFIC = {'who', 'age', 'student', 'parent', 'owner', 'worker', 'professional', 'women', 'men',
                         'teen', 'kid', 'adult', 'senior', 'driver', 'employee', 'runner', 'athlete',
                         'year', 'grade', 'suburb', 'city', 'near', 'local', 'their', 'need', 'want',
                         'problem', 'struggle', 'frustrat', 'busy', 'can\'t', 'without'}
            words = set(persona.lower().split())
            if words & _VAGUE and not any(s in persona.lower() for s in _SPECIFIC):
                violations.append({'id': 'CR008', 'flag': 'vague_persona', 'severity': 'warning',
                    'message': "Your customer description is quite general. Name a specific type of person — their role, age bracket, location, or situation.",
                    'socratic_hint': 'Instead of "people who need X", try "Grade 10–12 students at suburban schools who struggle with X".'})

        # O13-07 (CR009): Channel-archetype mismatch check
        _archetype_cr = session.get('idea_archetype', 'other')
        _channel = answers.get('how_to_reach', '')
        _DIGITAL_ONLY_CHANNELS = {'App store / platform listing', 'Content marketing / SEO', 'Online communities / forums', 'Social media ads'}
        _PHYSICAL_ONLY_CHANNELS = {'Door-to-door / flyers', 'School / workplace notice board', 'Local events / markets', 'Door-to-door / flyers'}
        if _channel:
            if _archetype_cr == 'digital-product' and _channel in _PHYSICAL_ONLY_CHANNELS:
                violations.append({'id': 'CR009', 'flag': 'channel_archetype_mismatch', 'severity': 'warning',
                    'message': (f"📣 '{_channel}' is a physical channel — digital products are usually discovered and bought online. "
                                f"Consider: app store listings, social media, or online communities to reach your customer."),
                    'socratic_hint': 'Where do people who would buy a digital product actually spend time? Think about where they already discover apps or software.'})
            elif _archetype_cr == 'food-beverage' and _channel in _DIGITAL_ONLY_CHANNELS:
                violations.append({'id': 'CR009', 'flag': 'channel_archetype_mismatch', 'severity': 'warning',
                    'message': (f"📣 '{_channel}' is primarily a digital channel — food and beverage businesses are usually found locally. "
                                f"Are you able to sell your food through this channel, or do customers need to physically come to you?"),
                    'socratic_hint': 'How would a customer actually get their food from you through this channel? Make sure the channel and your delivery model match.'})
            elif _archetype_cr == 'maker' and _channel in {'App store / platform listing', 'Content marketing / SEO'}:
                violations.append({'id': 'CR009', 'flag': 'channel_archetype_mismatch', 'severity': 'warning',
                    'message': (f"📣 '{_channel}' works for scalable digital products, but handmade/maker businesses need to handle physical fulfilment for every order. "
                                f"How will customers receive their item? Consider local markets, social media with DM orders, or a platform like Etsy."),
                    'socratic_hint': 'For a handmade product, how does the item physically reach the customer? Digital discovery channels work, but you still need a delivery or pickup plan.'})
            elif _archetype_cr == 'service' and _channel == 'Content marketing / SEO':
                violations.append({'id': 'CR009', 'flag': 'channel_archetype_mismatch', 'severity': 'warning',
                    'message': (f"📣 Content marketing / SEO works best at scale — a local service business at student level is usually found through word-of-mouth, school networks, or local community pages rather than search engines. "
                                f"Is your target customer likely to Google for your service, or would they hear about you through someone they trust?"),
                    'socratic_hint': 'Who does your target customer ask when they need this type of service? They probably ask a friend, not a search engine. Think about where word-of-mouth naturally happens for your customer.'})
            elif _archetype_cr == 'event-experience' and _channel == 'Social media ads':
                violations.append({'id': 'CR009', 'flag': 'channel_archetype_mismatch', 'severity': 'warning',
                    'message': (f"📣 Paid social media ads for a first event is an expensive way to fill seats. "
                                f"Most student-run events succeed by selling to existing networks first — school notice boards, direct messages, and local community groups. "
                                f"Ads work after you've proven the event can fill seats organically."),
                    'socratic_hint': 'Who already knows about your event and would tell a friend? Start there before spending on ads. What channels do your target attendees already use to find local events?'})

        # CR009-LOCAL: Hyper-local proximity paired with a purely global/paid channel
        # Catches: "Within 1–2 km" + "Social media ads" or "Content marketing / SEO"
        _LOCAL_PROXIMITY_SIGNALS = {'my street', 'block', 'walking distance', '1–2 km', '500m',
                                    'same school', 'same neighbourhood', 'same suburb', 'local pickup',
                                    'in-person consulting', 'local community', 'delivery within 5'}
        _GLOBAL_CHANNELS = {'Social media ads', 'Content marketing / SEO', 'App store / platform listing'}
        _loc = (answers.get('customer_location') or '').lower()
        if _channel in _GLOBAL_CHANNELS and any(sig in _loc for sig in _LOCAL_PROXIMITY_SIGNALS):
            violations.append({'id': 'CR009L', 'flag': 'local_business_global_channel', 'severity': 'warning',
                'message': (f"📣 Your customer is highly local but '{_channel}' is a global or paid channel that may not reliably reach them. "
                            f"For a neighbourhood-scale business, word-of-mouth, local community groups, school notice boards, or door-to-door flyers typically work far better.\n\n"
                            f"What to do: Think about where your exact customer physically is and how they'd discover you there — not how strangers on the internet discover things."),
                'socratic_hint': 'The student has a hyper-local business but is relying on a global/paid channel. Ask: how would a neighbour or classmate specifically find out about your business? Name one local channel they already use that you could reach them through.'})

        # O10-08: Spending power vs sell price cross-chapter check
        _WTP_RANGES = {
            'Under $20':   (0,  20),
            '$20–$50':     (20, 50),
            '$50–$150':    (50, 150),
            'Over $150':   (150, 9999),
        }
        wtp = answers.get('willingness_to_pay', '')
        sell_price = 0.0
        try:
            sell_price = float(session.get('sell_price') or session.get('unit_price') or 0)
        except (ValueError, TypeError):
            pass
        if wtp in _WTP_RANGES and sell_price > 0:
            _, hi = _WTP_RANGES[wtp]
            if sell_price > hi * 1.5:
                violations.append({'id': 'CC008', 'flag': 'price_exceeds_wtp', 'severity': 'error',
                    'message': (f"Your price (${sell_price:.0f}) is much higher than what your customer can spend ('{wtp}'). "
                                f"Either raise the spending power tier or lower your price."),
                    'socratic_hint': 'Would someone in your target group actually pay your price? What\'s the most they\'d realistically spend?'})
            elif sell_price > hi:
                violations.append({'id': 'CC008', 'flag': 'price_exceeds_wtp', 'severity': 'warning',
                    'message': (f"Your price (${sell_price:.0f}) is above the '{wtp}' spending tier you selected. "
                                f"Check that your target customer can actually afford your product."),
                    'socratic_hint': 'Does your target customer normally spend more than this, or is your price stretching their budget?'})
        elif wtp in _WTP_RANGES and sell_price == 0:
            # O11-09: sell_price not yet set — advise rather than silently skip
            violations.append({'id': 'CC008', 'flag': 'price_not_set', 'severity': 'warning',
                'message': (
                    "No sell price set yet — this check compares your price against your customer's budget, "
                    "but you haven't reached Chapter 4 (Money) yet.\n\n"
                    "What to do: Continue to Chapter 4 next — once you set your price there, this check will resolve automatically."
                ),
                'socratic_hint': 'This is a forward-reference check. The student just needs to continue to Chapter 4 (Money) — no action needed here.'})

        has_errors = any(v['severity'] == 'error' for v in violations)
        if not has_errors:
            session['customer_passed'] = True
        # C8-11: customer module emphasises structural completeness (who/where/how)
        symbolic_score = _auditor.symbolic_from_violations(violations, len(rules), error_weight=30, warning_weight=10)

        pure_text   = call_customer_pure_llm(answers)
        hybrid_text = call_customer_hybrid_llm(answers, violations)
        triple = _auditor.triple_truth(symbolic_score, pure_text, hybrid_text, 'customer')
        _save_audit_with_delta('customer', triple, violations)
        _post_disc_customer = 'discovery' in session.get('audit_scores', {})
        save_research_log('customer', symbolic_score, violations,
                          pure_text, triple['pure_truth'],
                          hybrid_text, triple['hybrid_truth'],
                          triple['sycophancy_gap'],
                          post_discovery=_post_disc_customer)
        radar = _update_chapter_radar('Influence', triple['hybrid_truth'])
        socratic_tip = violations[0].get('socratic_hint', '') if violations else ("Great work on this module!" if not has_errors else "")
        return jsonify({'passed': not has_errors, 'violations': violations, 'total_rules': len(rules),
                        'triple': triple, 'pure_llm': pure_text, 'hybrid_llm': _strip_mentor_score_line(hybrid_text),
                        'symbolic_score': symbolic_score, 'socratic_tip': socratic_tip,
                        'radar_scores': radar, 'top_issues': _top_issues_from_violations(violations)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _check_archetype_finance_rules(answers: dict, archetype: str) -> list:
    """Archetype-specific financial validation — calibrated per business type."""
    v = []
    try:
        unit_cost    = float(answers.get('unit_cost', 0) or 0)
        unit_price   = float(answers.get('sell_price') or answers.get('unit_price', 0) or 0)
        startup_cost = float(answers.get('startup_cost', 0) or 0)
    except (ValueError, TypeError):
        return v

    if archetype in ('physical-product', 'maker', 'food-beverage'):
        if unit_cost <= 0:
            label = {'maker': 'raw materials per item', 'food-beverage': 'ingredients + packaging per portion'}.get(archetype, 'materials/production cost per unit')
            v.append({'id': 'AF001', 'flag': 'no_material_cost', 'severity': 'error',
                'message': f"For a {archetype.replace('-',' ')} business you must include {label} in your unit cost — profit will be wrong without it.",
                'socratic_hint': 'What does it cost you to make one unit? Include materials, ingredients, and packaging.'})

    if archetype in ('resale-retail',):
        if unit_cost <= 0:
            v.append({'id': 'AF002', 'flag': 'no_purchase_cost', 'severity': 'error',
                'message': 'A resale/retail business buys stock before selling it. Enter your wholesale purchase price as the unit cost.',
                'socratic_hint': 'How much do you pay to buy each item you plan to resell? Don\'t forget shipping to you.'})
        elif unit_price > 0 and unit_cost > 0 and unit_price < unit_cost * 1.5:
            v.append({'id': 'AF003', 'flag': 'low_retail_markup', 'severity': 'warning',
                'message': f'Your markup is less than 50% (buy ${unit_cost:.2f}, sell ${unit_price:.2f}). Retail needs 2–3× markup to cover shipping, storage, and returns.',
                'socratic_hint': 'Have you added your shipping cost, storage, and your own time into the selling price?'})

    if archetype in ('service', 'skills-for-hire', 'expert'):
        if unit_cost <= 0:
            v.append({'id': 'AF004', 'flag': 'no_time_cost', 'severity': 'warning',
                'message': 'For a service business, include your time + any travel or supply cost per job. Your time has real value — don\'t work for free!',
                'socratic_hint': 'How long does one job take you? What travel costs or supplies do you use per service?'})

    if archetype == 'maker':
        if startup_cost <= 0:
            v.append({'id': 'AF005', 'flag': 'no_maker_startup', 'severity': 'warning',
                'message': 'Maker businesses need tools, equipment, and starter materials before the first sale. Add your estimated startup cost.',
                'socratic_hint': 'What tools or equipment do you need before you can make your first item?'})

    if archetype == 'food-beverage':
        if startup_cost <= 0:
            v.append({'id': 'AF006', 'flag': 'no_food_startup', 'severity': 'warning',
                'message': 'Food/beverage businesses need startup costs for equipment, packaging, and possibly food-handling permits.',
                'socratic_hint': 'Do you need any equipment, containers, or permits before your first sale?'})

    if archetype == 'digital-product':
        if unit_price > 0 and unit_cost > 0 and unit_cost > unit_price * 0.5:
            v.append({'id': 'AF007', 'flag': 'high_digital_cost', 'severity': 'warning',
                'message': f'Digital products have very low per-sale costs (platform/payment fees only). Your unit cost (${unit_cost:.2f}) looks high — are you including one-off development cost here by mistake?',
                'socratic_hint': 'Development costs go in startup cost. Per-sale cost should only be platform commission or payment-processing fees.'})

    if archetype == 'event-experience':
        if unit_cost <= 0:
            v.append({'id': 'AF008', 'flag': 'no_event_cost', 'severity': 'warning',
                'message': 'Event/experience businesses must include venue, materials, and staffing cost per event as unit cost.',
                'socratic_hint': 'What does it cost you to run one event? Include venue hire, materials, and your own time.'})
        # O12-08: Event businesses also need startup costs — venue deposit, equipment, initial marketing
        if startup_cost <= 0:
            v.append({'id': 'AF009', 'flag': 'no_event_startup', 'severity': 'warning',
                'message': 'Event/experience businesses need startup costs before the first event — venue deposit, equipment hire, initial marketing, or permits. Add your estimated startup cost.',
                'socratic_hint': 'What do you need to pay before you can run your first event? Include deposits, any equipment, and promotion costs.'})

    # O13-06 (AF010): Content-media startup cost — creators need equipment even if distribution is free
    if archetype == 'content-media' and unit_price > 0 and startup_cost <= 0:
        v.append({'id': 'AF010', 'flag': 'no_content_startup', 'severity': 'warning',
            'message': 'Content creators need equipment before monetising — microphone, lighting, camera, or editing software. Even basic gear has a cost. Add your estimated startup cost.',
            'socratic_hint': 'What device, microphone, or software do you need before creating your first piece of content? List everything, then add it up.'})

    # O11-08 / O12-06: Round-number estimates — both sell price and unit cost are multiples of 10 and ≥ $10
    def _is_suspiciously_round(val: float) -> bool:
        return val >= 10 and val % 10 == 0 and val % 1 == 0
    if unit_price > 0 and unit_cost > 0 and _is_suspiciously_round(unit_price) and _is_suspiciously_round(unit_cost):
        v.append({'id': 'CC018', 'flag': 'round_number_estimates', 'severity': 'warning',
            'message': (
                f"🟡 Your sell price (${unit_price:.0f}) and unit cost (${unit_cost:.0f}) are both round numbers — "
                f"these look like rough estimates rather than real calculations.\n\n"
                f"What to do: Check your actual supplier prices, platform fees, or material costs to replace estimates with real numbers."
            ),
            'socratic_hint': 'Have you looked up the actual price of materials, or checked what similar businesses charge?'})

    # ── Volume sanity: per-archetype realistic monthly limits for a student ────
    try:
        monthly_units = float(answers.get('monthly_units', 0) or 0)
    except (ValueError, TypeError):
        monthly_units = 0

    # (warn_threshold, error_threshold, unit_label, hours_per_unit_hint)
    _VOL_LIMITS = {
        'content-media':    (12,  30,  'videos/posts',
            'That\'s more than one video per day. Even full-time creators post 4–8/month. '
            'Each video needs filming, editing, and uploading — how many hours does that take you?'),
        'physical-product': (100, 300, 'items',
            'Handmade or physical items take real production time. At that volume you\'d need to make '
            'several items every single day including weekends. Is that realistic alongside school?'),
        'maker':            (20,  60,  'custom items',
            'Custom builds are time-intensive — most student makers produce 5–15 items/month. '
            'How many hours does each item take? Multiply by your monthly target and check if that fits your schedule.'),
        'food-beverage':    (200, 600, 'servings/portions',
            'High volume food production requires commercial-scale equipment, storage, and preparation time. '
            'A student kitchen realistically handles 50–150 servings/month. How long does each batch take?'),
        'service':          (20,  40,  'jobs',
            'Each job takes travel + delivery time. At that volume you\'d need multiple jobs every single day. '
            'How many hours per week do you have available alongside school?'),
        'skills-for-hire':  (15,  30,  'sessions',
            'At that session count you\'d be working almost every day. How long is each session? '
            'Multiply by your target — does that fit around school and other commitments?'),
        'expert':           (12,  25,  'consulting sessions',
            'Consulting and coaching take deep focus — 3+ sessions/week is a heavy schedule for a student. '
            'Quality matters more than volume for expert services.'),
        'resale-retail':    (80,  200, 'items',
            'At that volume you\'d need to source, photograph, list, pack, and ship multiple items daily. '
            'How long does each sale take end-to-end? Is storage a constraint?'),
        'event-experience': (150, 400, 'attendees',
            'Running events at this scale requires a venue, staffing, and significant logistics. '
            'Are you factoring in setup/pack-down time and capacity limits?'),
        'digital-product':  (500, 2000, 'downloads/sales',
            'Selling hundreds of digital products/month requires a large existing audience or paid ads. '
            'New creators typically see 5–30 sales/month. What\'s your plan to reach that many buyers?'),
        'marketplace':      (200, 1000, 'transactions',
            'Marketplace volume this high requires both a large buyer AND seller base already on your platform. '
            'New platforms typically see 10–50 transactions/month in their first year.'),
    }

    if archetype in _VOL_LIMITS and monthly_units > 0:
        warn_t, err_t, unit_lbl, hint = _VOL_LIMITS[archetype]
        if monthly_units > err_t:
            v.append({'id': 'AF-VOL', 'flag': 'unrealistic_volume', 'severity': 'error',
                'message': (
                    f"🔴 {int(monthly_units)} {unit_lbl}/month is extremely unrealistic for a student-run business "
                    f"(typical: {warn_t // 2}–{warn_t}/month to start).\n\n"
                    f"{hint}\n\nLower your monthly target to something you can actually achieve in the first 3 months."
                ),
                'socratic_hint': f'If you had to commit to {int(monthly_units)} {unit_lbl} every month starting next week, what would your daily schedule look like?'})
        elif monthly_units > warn_t:
            v.append({'id': 'AF-VOL-W', 'flag': 'high_volume_check', 'severity': 'warning',
                'message': (
                    f"🟡 {int(monthly_units)} {unit_lbl}/month is ambitious for a student-run business "
                    f"(typical starting range: {warn_t // 3}–{warn_t // 2}/month).\n\n"
                    f"{hint}\n\nThis is possible but requires a solid plan — make sure your time estimate is realistic."
                ),
                'socratic_hint': f'Walk through a typical week: how many {unit_lbl} would you produce each day? Does that fit around your school schedule?'})

    return v


# ── API: Validate money math + triple truth ────────────────────
@app.route('/api/validate_money', methods=['POST'])
def api_validate_money():
    _rl = _check_llm_rate_limit()
    if _rl:
        return jsonify({'error': f'Too many requests — please wait {_rl}s before re-running analysis.', 'retry_after': _rl}), 429
    try:
        data    = request.json
        answers = data.get('answers', {})
        # Resolve archetype — session is authoritative; fall back to frontend-rendered value
        _page_arch_fe = data.get('_page_archetype') or ''
        _effective_arch = session.get('idea_archetype', '') or _page_arch_fe or 'other'
        # Inject into session if session was missing it (e.g. direct navigation to /money)
        if _effective_arch and _effective_arch != 'other' and not session.get('idea_archetype'):
            session['idea_archetype'] = _effective_arch
        # Derive computed fields
        try:
            unit_cost  = float(answers.get('unit_cost', 0) or 0)
            unit_price = float(answers.get('sell_price') or answers.get('unit_price', 0) or 0)
            monthly_units = float(answers.get('monthly_units', 0) or 0)
            startup_cost  = float(answers.get('startup_cost', 0) or 0)
            monthly_revenue = unit_price * monthly_units
            monthly_profit  = (unit_price - unit_cost) * monthly_units
            months_to_be    = (startup_cost / monthly_profit) if monthly_profit > 0 else 999
            _arch_early = _effective_arch
            # For content-media, $0 earnings is valid pre-monetisation — don't fire price_below_cost
            if _arch_early == 'content-media' and unit_price == 0:
                answers['unit_price_gt_unit_cost'] = 'Yes'  # suppress MR001 — handled by CC-MEDIA below
            else:
                answers['unit_price_gt_unit_cost'] = 'Yes' if unit_price > unit_cost else 'No'

            # O13-02: Archetype-aware monthly profit threshold (replaces flat revenue check)
            # content-media uses $10 floor — $8/month (4 videos × $2) is ~$0.67/hr worked,
            # which IS worth flagging so the student thinks about whether it's worthwhile.
            # digital-product: low variable cost, so $10 floor is reasonable.
            _arch = _effective_arch
            _profit_floor = {
                'food-beverage': 25, 'resale-retail': 25,
                'content-media': 10,
                'digital-product': 10,
            }.get(_arch, 20)
            answers['monthly_profit_lt_threshold'] = 'Yes' if 0 < monthly_profit < _profit_floor else 'No'

            # O13-08: Archetype-aware break-even threshold (replaces flat 12-month rule)
            # content-media gets 36 months — channels take 1-2 years to monetise meaningfully
            _be_caps = {
                'food-beverage': 6, 'physical-product': 6, 'maker': 6,
                'service': 12, 'skills-for-hire': 12, 'expert': 12, 'resale-retail': 12,
                'digital-product': 18, 'content-media': 36,
                'event-experience': 9, 'marketplace': 12,
            }
            _be_cap = _be_caps.get(_arch, 12)
            answers['months_to_breakeven_gt_threshold'] = 'Yes' if months_to_be > _be_cap else 'No'

            # MR006: Implied hourly rate below minimum wage (~$5/hr floor)
            # hours_per_week is stored in session from Chapter 1
            _hrs_pw = 0.0
            try:
                _hrs_pw = float(str(session.get('hours_per_week', '0') or '0').replace('+', ''))
            except (ValueError, TypeError):
                pass
            if _hrs_pw > 0 and monthly_profit > 0:
                _hourly_rate = monthly_profit / (_hrs_pw * 4)
                answers['hourly_rate_lt_minimum'] = 'Yes' if _hourly_rate < 5 else 'No'
            else:
                answers['hourly_rate_lt_minimum'] = 'No'
        except (ValueError, TypeError):
            pass

        # C8-06: cross-check startup_cost against Ch1 budget
        # Ch1 fields are stored flat in the session (session['budget'], session['hours_per_week'], etc.)
        # session['stage1_answers'] only holds _archetype — never the form fields.
        stage1 = {
            'budget':        session.get('budget', ''),
            'hours_per_week': session.get('hours_per_week', ''),
            'business_type': session.get('business_type', ''),
        }
        try:
            ch1_budget = float(str(stage1.get('budget', '0') or '0').split('.')[0])
        except (ValueError, TypeError):
            ch1_budget = 0
        if startup_cost > 0 and ch1_budget >= 0 and startup_cost > ch1_budget * 1.5:
            violations_prefix = [{
                'id': 'CC006M', 'flag': 'startup_exceeds_budget', 'severity': 'warning',
                'message': (
                    f"🟡 Your startup cost (${startup_cost:,.0f}) is much higher than the budget "
                    f'you set in <a href="/context" style="color:#A78BFA;text-decoration:underline">Chapter 1</a> (${ch1_budget:,.0f}).\n\n'
                    f'What to do: Update your <a href="/context" style="color:#A78BFA;text-decoration:underline">Chapter 1</a> budget if you now have more funding, or '
                    f"reduce your startup costs to match your available capital."
                ),
                'socratic_hint': "Ask: where is the extra money for startup costs coming from?",
            }]
        else:
            violations_prefix = []

        # O11-06 / O12-07: Low units — annual revenue doesn't cover startup cost repayment
        # Skip for content-media: 4 videos/month is realistic, not "low volume".
        # Content channels build audience before monetising — payback logic doesn't apply early on.
        # For service archetypes with high unit price (≥ $50), threshold is ≤ 3 to avoid false positives.
        _service_archetypes = {'service', 'skills-for-hire', 'expert'}
        _cc017_threshold = 3 if (_effective_arch in _service_archetypes and unit_price >= 50) else 5
        if _effective_arch != 'content-media' and monthly_units > 0 and monthly_units <= _cc017_threshold and startup_cost > 0 and unit_price > 0:
            annual_revenue = monthly_revenue * 12
            if annual_revenue < startup_cost:
                violations_prefix.append({
                    'id': 'CC017', 'flag': 'low_units_no_payback', 'severity': 'warning',
                    'message': (
                        f"🟡 At {int(monthly_units)} unit(s)/month, your annual revenue would be ~${annual_revenue:,.0f} — "
                        f"less than your startup cost of ${startup_cost:,.0f}. "
                        f"You'd never recover your startup investment at this sales volume.\n\n"
                        f"What to do: Increase your monthly sales target, raise your price, or reduce your startup costs."
                    ),
                    'socratic_hint': 'How many sales per month would you need to pay back your startup costs within a year?',
                })

        # CC019: Profit margin vs willingness to pay (Gap 2 cross-chapter fix)
        # CC008 (in customer validation) already checks if price exceeds WTP ceiling.
        # This complementary check asks a different question: even if the price is
        # within WTP, is the unit_cost so high that there is no viable margin left?
        # E.g. WTP ceiling = $50, unit_cost = $44 → max possible profit = $6 (12%).
        # The student technically passes CC008 but their economics are broken.
        _wtp_session = session.get('willingness_to_pay', '')
        _WTP_RANGES_MONEY = {
            'Under $20':  (0,   20),
            '$20–$50':    (20,  50),
            '$50–$150':   (50,  150),
            'Over $150':  (150, 9999),
        }
        if (_wtp_session in _WTP_RANGES_MONEY
                and unit_cost > 0
                and unit_price > unit_cost          # MR001 not already firing
                and _arch_early != 'content-media'):
            _, _wtp_hi = _WTP_RANGES_MONEY[_wtp_session]
            if _wtp_hi < 9999:                      # skip "Over $150" — ceiling is open-ended
                _max_possible_margin = _wtp_hi - unit_cost
                _margin_pct = (_max_possible_margin / _wtp_hi) if _wtp_hi else 1
                if _max_possible_margin <= 0:
                    # Cost already exceeds WTP ceiling — CC008 should have caught price,
                    # but if unit_cost alone is above WTP, flag it directly.
                    violations_prefix.append({
                        'id': 'CC019', 'flag': 'cost_exceeds_wtp', 'severity': 'error',
                        'message': (
                            f"Your cost per unit (${unit_cost:,.0f}) is already higher than the "
                            f"maximum your target customers will pay ('{_wtp_session}'). "
                            f"There is no price you can set that would be both profitable and affordable."
                        ),
                        'socratic_hint': (
                            "If it costs more to make than customers will pay, the business model doesn't work. "
                            "Can you reduce your cost, target a higher-budget customer, or change what you're selling?"
                        ),
                    })
                elif _margin_pct < 0.20:
                    # Cost is so close to WTP ceiling that even charging maximum leaves < 20% margin
                    violations_prefix.append({
                        'id': 'CC019', 'flag': 'thin_margin_vs_wtp', 'severity': 'warning',
                        'message': (
                            f"🟡 Your cost per unit (${unit_cost:,.0f}) leaves very little room for profit "
                            f"within your customer's budget ('{_wtp_session}'). "
                            f"Even charging the maximum they'd pay, your margin would be "
                            f"just ${_max_possible_margin:,.0f} per unit.\n\n"
                            f"What to do: Either reduce your cost per unit or reconsider whether your "
                            f"target customer has a higher spending power than you selected."
                        ),
                        'socratic_hint': (
                            "What would you need to change — your costs, your price, or your target customer — "
                            "to make a meaningful profit on each sale?"
                        ),
                    })

        # O15-07 (MR003 gap): Break-even pricing advisory — $0 profit is valid but worth flagging
        if unit_price > 0 and unit_cost > 0 and monthly_units > 0 and monthly_profit == 0:
            violations_prefix.append({
                'id': 'MR003Z', 'flag': 'zero_profit', 'severity': 'warning',
                'message': (
                    "🟡 Your monthly profit is exactly $0 — you're pricing at cost, which means you're working for free.\n\n"
                    "Is this intentional? Some students price at cost to test demand, but you'll need a plan to earn something eventually. "
                    "Who pays for your time? Even a small margin ($1–$5 per unit) would add up."
                ),
                'socratic_hint': 'If price equals cost, what are you earning for your time and effort? Is this a temporary strategy, or have you forgotten to include your own labour as a cost?',
            })

        # CC-MEDIA: Content-media archetype — monetisation path required
        if _arch_early == 'content-media':
            # Sanity-check unit_cost: production cost per video >$20 is almost never realistic for a student.
            # Likely the student typed their video count (e.g. 400) into the cost field by mistake.
            if unit_cost > 20:
                violations_prefix.append({
                    'id': 'CC-MEDIA-COST', 'flag': 'implausible_content_cost', 'severity': 'error',
                    'message': (
                        f"📺 Your production cost (${unit_cost:,.2f} per video) looks wrong. "
                        f"Student creators typically spend $0–$5 to produce one video "
                        f"(free tools like CapCut, DaVinci Resolve, phone cameras).\n\n"
                        f"Did you accidentally type your video count ({int(monthly_units)}) into the cost field? "
                        f"Clear the 'cost per video' field and enter the real cost to produce ONE video."
                    ),
                    'socratic_hint': f'What does it actually cost you to make one video? Think: equipment you already own, free editing software, any paid assets.',
                })
                # Suppress MR001 (price-below-cost) for content-media with implausible cost —
                # the real issue is the wrong cost entry, not the price strategy.
                answers['unit_price_gt_unit_cost'] = 'Yes'
            if unit_price == 0:
                violations_prefix.append({
                    'id': 'CC-MEDIA1', 'flag': 'no_monetisation_plan', 'severity': 'warning',
                    'message': (
                        "📺 You entered $0 earnings per video — that's fine for a pre-monetisation channel, "
                        "but you need a clear plan for how you'll eventually earn money.\n\n"
                        "YouTube options: 🎯 Ads (need 1,000 subs + 4,000 watch hours) · "
                        "💼 Sponsorships (from ~500 engaged followers) · "
                        "🛍️ Merch or digital products (no follower minimum).\n\n"
                        "Enter your realistic earnings per video once you reach your first milestone."
                    ),
                    'socratic_hint': 'Ask: what is your first monetisation milestone? 1,000 subscribers for ads? A sponsorship at 500 subs? A digital product from day one? How much per video would that earn?',
                })
            elif unit_price < 1:
                violations_prefix.append({
                    'id': 'CC-MEDIA2', 'flag': 'very_low_content_earnings', 'severity': 'warning',
                    'message': (
                        f"📺 ${unit_price:.2f} per video is very low — even a small channel with 500 views/video "
                        f"and a $2 CPM earns $1.00 per video from ads alone.\n\n"
                        f"Consider: ads + one sponsor deal per month could meaningfully increase your per-video earnings. "
                        f"Try entering at least $1–$2 per video as your realistic target."
                    ),
                    'socratic_hint': 'How many views per video do you expect? Multiply by CPM ÷ 1000 to get ad earnings. Is there a sponsor deal you could add on top?',
                })

        rules_data = _load_json('money_rules.json')
        rules   = rules_data.get('rules', [])
        violations = violations_prefix + validate_simple_rules(answers, rules)
        # Replace generic MR003 message for content-media — "raise your price or sell more units"
        # is wrong advice for a YouTube channel. Replace with growth-oriented framing.
        if _arch_early == 'content-media':
            for _v in violations:
                if _v.get('id') == 'MR003':
                    _v['message'] = (
                        f"📺 At ${monthly_profit:.2f}/month (${monthly_profit / max(monthly_units,1):.2f}/video × "
                        f"{int(monthly_units)} videos), this is a realistic starting point — but it's "
                        f"only about ${monthly_profit * 12:.0f}/year.\n\n"
                        f"To make this worthwhile, you need a growth plan: more videos, higher CPM "
                        f"(better niche → better ads), or a second revenue stream (sponsorships, merch, digital products)."
                    )
                    _v['socratic_hint'] = (
                        f'At ${monthly_profit:.2f}/month, divide that by your hours spent creating — '
                        f'what is your effective hourly rate? What would you need to change to reach $50/month within 6 months?'
                    )
        # Note: _detect_contradictions is NOT called here — money chapter only validates
        # financial math. Semantic contradictions (channel vs demographics, differentiation
        # vs problem) belong to the customer chapter where those fields actually exist.
        # Calling it here with sparse financial data causes the 8B model to hallucinate violations.
        answers['archetype']     = _effective_arch
        answers['business_type'] = stage1.get('business_type', '')
        answers['hours_per_week']= stage1.get('hours_per_week', '')
        violations += _check_archetype_finance_rules(answers, _effective_arch)

        # Cross-check monthly volume vs Ch1 hours_per_week
        # Session stores a numeric value from the slider (e.g. '5', '14'), not range strings.
        _hours_raw = str(stage1.get('hours_per_week', '') or '')
        try:
            _weekly_hrs = float(_hours_raw)
        except ValueError:
            # Fallback for legacy sessions that stored range strings
            _legacy_map = {'1–5 hours': 4, '5–10 hours': 7, '10–20 hours': 15, '20+ hours': 20}
            _weekly_hrs = _legacy_map.get(_hours_raw, 0)
        _hours_str = f'{_weekly_hrs:.0f} hrs' if _weekly_hrs > 0 else _hours_raw
        if _weekly_hrs > 0 and monthly_units > 0:
            # Estimate hours per unit for time-sensitive archetypes
            _hrs_per_unit = {
                'content-media':   3.0,   # film + edit + upload
                'maker':           4.0,   # design + build
                'service':         1.5,   # travel + deliver
                'skills-for-hire': 1.5,   # session delivery
                'expert':          1.5,
                'physical-product':0.5,   # assembly
                'food-beverage':   0.25,  # per portion (batch production)
                'event-experience':0.5,   # per attendee (event prep spread)
            }.get(_effective_arch, 0)
            # Skip CC-TIME if AF-VOL already fired — volume is already flagged as unrealistic
            _af_vol_fired = any(v['id'] in ('AF-VOL', 'AF-VOL-W') for v in violations)
            if _hrs_per_unit > 0 and not _af_vol_fired:
                _monthly_hrs_needed = monthly_units * _hrs_per_unit
                _monthly_hrs_available = _weekly_hrs * 4.3
                if _monthly_hrs_needed > _monthly_hrs_available * 1.5:
                    violations.append({
                        'id': 'CC-TIME', 'flag': 'time_capacity_conflict', 'severity': 'warning',
                        'message': (
                            f"⏱️ Time conflict: you said you have {_hours_str}/week in Chapter 1, "
                            f"but {int(monthly_units)} units/month × ~{_hrs_per_unit:.1f} hrs each = "
                            f"~{_monthly_hrs_needed:.0f} hrs/month needed vs "
                            f"~{_monthly_hrs_available:.0f} hrs available.\n\n"
                            f"Lower your monthly target or go back to Chapter 1 and update your available hours."
                        ),
                        'socratic_hint': f'You have {_hours_str}/week. At {_hrs_per_unit:.1f} hrs per unit, {int(monthly_units)} units needs {_monthly_hrs_needed:.0f} hrs/month — but you only have {_monthly_hrs_available:.0f} hrs available. What\'s a realistic monthly target?',
                    })

        has_errors = any(v['severity'] == 'error' for v in violations)
        # C8-11: money module emphasises financial accuracy (errors very costly)
        symbolic_score = _auditor.symbolic_from_violations(violations, len(rules), error_weight=25, warning_weight=12)

        pure_text   = call_money_pure_llm(answers)
        # Sort violations for LLM: volume errors (AF-VOL) first so they aren't drowned out by
        # other errors like MR001. Then errors before warnings.
        _VOL_IDS = {'AF-VOL', 'AF-VOL-W', 'CC-TIME', 'CC-MEDIA-COST'}
        _triggered_for_llm = sorted(
            violations,
            key=lambda v: (0 if v['id'] in _VOL_IDS else 1 if v['severity'] == 'error' else 2)
        ) if violations else []
        hybrid_text = call_hybrid_llm(answers, {'symbolic': symbolic_score}, _triggered_for_llm)
        # 8b model ignores the [10-65] score range instruction when violations exist.
        # Mentor uses harder weights than symbolic to reflect honest financial viability.
        if violations and hybrid_text:
            import re as _re_money
            _n_err  = sum(1 for v in violations if v.get('severity') == 'error')
            _n_warn = sum(1 for v in violations if v.get('severity') == 'warning')
            if _n_err > 0:
                _money_score = max(15, min(55, 100 - _n_err * 30 - _n_warn * 8))
            else:
                _money_score = max(50, min(75, 100 - _n_warn * 10))
            if _re_money.search(r'MENTOR_SCORE:\s*\d', hybrid_text):
                hybrid_text = _re_money.sub(r'MENTOR_SCORE:\s*\d{1,3}', f'MENTOR_SCORE: {_money_score}', hybrid_text)
            else:
                hybrid_text += f'\nMENTOR_SCORE: {_money_score}'
        triple = _auditor.triple_truth(symbolic_score, pure_text, hybrid_text, 'money')
        _save_audit_with_delta('money', triple, violations)
        _post_disc_money = 'discovery' in session.get('audit_scores', {})
        save_research_log('money', symbolic_score, violations,
                          pure_text, triple['pure_truth'],
                          hybrid_text, triple['hybrid_truth'],
                          triple['sycophancy_gap'],
                          post_discovery=_post_disc_money)
        radar = _update_chapter_radar('Gold', triple['hybrid_truth'])
        socratic_tip = violations[0].get('socratic_hint', '') if violations else ("Great work on this module!" if not has_errors else "")
        return jsonify({'passed': not has_errors, 'violations': violations, 'total_rules': len(rules),
                        'triple': triple, 'pure_llm': pure_text, 'hybrid_llm': _strip_mentor_score_line(hybrid_text),
                        'symbolic_score': symbolic_score, 'socratic_tip': socratic_tip,
                        'radar_scores': radar, 'top_issues': _top_issues_from_violations(violations)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── API: Validate discovery + triple truth ─────────────────────
@app.route('/api/validate_discovery', methods=['POST'])
def api_validate_discovery():
    _rl = _check_llm_rate_limit()
    if _rl:
        return jsonify({'error': f'Too many requests — please wait {_rl}s before re-running analysis.', 'retry_after': _rl}), 429
    try:
        data    = request.json
        answers = data.get('answers', {})
        # Coerce numeric fields
        try:
            answers['interviews_completed'] = int(answers.get('interviews_completed', 0) or 0)
        except (ValueError, TypeError):
            answers['interviews_completed'] = 0

        # Save user-provided answers back to session so that O15-02 auto-re-run
        # on the next page visit uses the latest values, not the stale form-POST values.
        for _k in ('interviews_completed', 'interview_sources', 'question_type',
                   'discovery_changed_plan', 'insight_applied'):
            _v = answers.get(_k)
            if _v is not None:
                session[_k] = str(_v)
        session.modified = True  # Flask-Session filesystem backend requires explicit mark

        rules_data = _load_json('discovery_rules.json')
        rules   = rules_data.get('rules', [])
        violations = validate_simple_rules(answers, rules)

        # O10-09: Archetype-calibrated interview floor check
        # O13: 'other' archetype gets floor=5 (unknown type = more validation needed)
        archetype   = session.get('idea_archetype', 'other')
        arch_params = ARCHETYPE_PARAMS.get(archetype, ARCHETYPE_PARAMS['other'])
        interview_floor = 5 if archetype == 'other' else arch_params.get('interview_floor', 3)
        n_interviews    = answers.get('interviews_completed', 0)
        if 0 < n_interviews < interview_floor:
            # O11-07: Marketplace archetype needs extra explanation — two-sided interview requirement
            if archetype == 'marketplace':
                dr006_message = (
                    f"For a marketplace business, aim for at least {interview_floor} interviews — "
                    f"but these must cover BOTH sides: potential buyers AND potential sellers (you have {n_interviews}). "
                    f"A marketplace fails if either side has no interest. One-sided validation misses half the risk."
                )
                dr006_hint = (
                    "Have you interviewed both the people who would buy through your marketplace AND "
                    "the people who would sell or list on it? Both groups are your customers."
                )
            else:
                dr006_message = (
                    f"For a {archetype.replace('-', ' ')} business, aim for at least "
                    f"{interview_floor} interviews to validate your idea (you have {n_interviews}). "
                    f"This business type has higher customer complexity."
                )
                dr006_hint = (
                    f"Who are the specific people you still need to interview for a "
                    f"{archetype.replace('-', ' ')} business?"
                )
            violations.append({
                'id': 'DR006', 'flag': 'below_archetype_interview_floor', 'severity': 'warning',
                'message': dr006_message,
                'socratic_hint': dr006_hint,
            })

        # Contradiction detection — full plan context available at Ch5
        # Ch1 fields are stored flat in session, not in stage1_answers dict
        answers['archetype']     = session.get('idea_archetype', '')
        answers['business_type'] = session.get('business_type', '')
        answers['hours_per_week']= session.get('hours_per_week', '')
        answers['budget']        = session.get('budget', '')
        answers['sell_price']    = session.get('unit_price', '')
        answers['unit_cost']     = session.get('unit_cost', '')
        answers['monthly_units'] = session.get('monthly_units', '')

        # DR008: Cross-reference interview sources against who the student is targeting.
        # If their target customer (sell_to) is primarily adults, businesses, or professionals
        # but they only interviewed friends and family, the bias risk is much higher —
        # friends/family can't proxy for a B2B or adult professional customer.
        _interview_src = str(answers.get('interview_sources', '') or '')
        _sell_to       = str(session.get('sell_to', '') or '').lower()
        _b2b_keywords  = ['business', 'company', 'shop', 'store', 'restaurant', 'professional',
                          'employer', 'manager', 'adult', 'parent', 'teacher', 'client']
        _is_b2b_target = any(kw in _sell_to for kw in _b2b_keywords)
        if _interview_src == 'Friends and family only' and _is_b2b_target:
            violations.append({
                'id': 'DR008',
                'flag': 'audience_interview_mismatch',
                'severity': 'warning',
                'message': (
                    f"⚠️ Your target customer is '{session.get('sell_to', 'adults/professionals')}' "
                    "but your interviews were with friends and family. Friends and family can't give you "
                    "honest feedback about a product or service they're not the real customer for. "
                    "You need at least one conversation with someone who actually matches your customer profile."
                ),
                'socratic_hint': (
                    "The student's target customer is adults or professionals but they only interviewed "
                    "friends/family. Ask: can a friend your age give you the same information as a real paying customer? "
                    "Who is one specific adult or professional you could reach this week — even via email or social media?"
                ),
            })
        # _detect_contradictions is NOT called here — discovery answers don't include
        # how_to_reach / differentiation / problem_confirmed, so the 8B model hallucinates.
        # Contradiction detection belongs in customer chapter where those fields exist.

        # O13-01 / O13-10: Insight quality check — don't fire DR005 if student wrote substantive reflections
        # Also: soften DR005 if insight text exists but insight_applied flag is "No"
        _insight_text = str(answers.get('discovery_insight_text', '') or '').strip()
        _insight_chars = len(_insight_text)
        _new_violations = []
        for v_item in violations:
            if v_item.get('id') == 'DR005':
                if _insight_chars >= 60:
                    # Student wrote substantive insight text — change to a nudge rather than a warning
                    _new_violations.append({
                        'id': 'DR005', 'flag': 'insight_not_applied_but_written', 'severity': 'warning',
                        'message': (
                            '📝 You captured great reflections in your notes — but make sure you\'ve also gone back '
                            'and updated your earlier chapters (<a href="/customer" style="color:#A78BFA;text-decoration:underline">Chapter 3 Customer</a>, '
                            '<a href="/money" style="color:#A78BFA;text-decoration:underline">Chapter 4 Money</a>) with what you learned. '
                            'That\'s what turns discovery into improvement.'
                        ),
                        'socratic_hint': 'Look at your Chapter 3 persona and Chapter 4 price — do they still match what you heard in your interviews? Update them if needed.',
                    })
                else:
                    _new_violations.append(v_item)
            else:
                _new_violations.append(v_item)
        violations = _new_violations

        has_errors = any(v['severity'] == 'error' for v in violations)
        # C8-11: discovery module rewards attempt over perfection (lighter penalty for warnings)
        symbolic_score = _auditor.symbolic_from_violations(violations, len(rules), error_weight=20, warning_weight=8)

        pure_text   = call_discovery_pure_llm(answers)
        hybrid_text = call_hybrid_llm(answers, {'symbolic': symbolic_score},
                                      [r for r in rules if any(v['id'] == r['id'] for v in violations)])
        # 8b model ignores the [10-65] score range instruction when violations exist.
        # Mentor uses harder weights than symbolic (symbolic is lenient to avoid discouraging students;
        # Mentor should reflect honest research quality, not encouragement).
        if violations and hybrid_text:
            import re as _re_disc
            _n_err  = sum(1 for v in violations if v.get('severity') == 'error')
            _n_warn = sum(1 for v in violations if v.get('severity') == 'warning')
            if _n_err > 0:
                _disc_score = max(20, min(55, 100 - _n_err * 30 - _n_warn * 8))
            else:
                _disc_score = max(50, min(75, 100 - _n_warn * 10))
            if _re_disc.search(r'MENTOR_SCORE:\s*\d', hybrid_text):
                hybrid_text = _re_disc.sub(r'MENTOR_SCORE:\s*\d{1,3}', f'MENTOR_SCORE: {_disc_score}', hybrid_text)
            else:
                hybrid_text += f'\nMENTOR_SCORE: {_disc_score}'
        triple = _auditor.triple_truth(symbolic_score, pure_text, hybrid_text, 'discovery')
        _save_audit_with_delta('discovery', triple, violations)
        save_research_log('discovery', symbolic_score, violations,
                          pure_text, triple['pure_truth'],
                          hybrid_text, triple['hybrid_truth'],
                          triple['sycophancy_gap'])
        radar = _update_chapter_radar('Knowledge', triple['hybrid_truth'])
        socratic_tip = violations[0].get('socratic_hint', '') if violations else ("Great work on this module!" if not has_errors else "")
        return jsonify({'passed': not has_errors, 'violations': violations, 'total_rules': len(rules),
                        'triple': triple, 'pure_llm': pure_text, 'hybrid_llm': _strip_mentor_score_line(hybrid_text),
                        'symbolic_score': symbolic_score, 'socratic_tip': socratic_tip,
                        'radar_scores': radar, 'top_issues': _top_issues_from_violations(violations)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── API: Save rival edge to session ────────────────────────────
@app.route('/api/save_rival_edge', methods=['POST'])
def api_save_rival_edge():
    try:
        data       = request.json or {}
        rival_name = data.get('rival_name', 'Unknown')
        edge       = data.get('edge', '')
        if not edge:
            return jsonify({'ok': False, 'error': 'No edge provided'}), 400
        edges = session.get('rival_edges', [])
        edges.append({'rival': rival_name, 'edge': edge})
        session['rival_edges'] = edges
        session.modified = True
        return jsonify({'ok': True, 'total': len(edges)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── API: Return test personas ──────────────────────────────────
@app.route('/api/personas')
def api_personas():
    try:
        return jsonify(_load_json('test_profiles.json'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════
#  EVALUATION MODULE — 20-persona ITS benchmark
# ══════════════════════════════════════════════════════════════
@app.route('/eval')
def eval_dashboard():
    return render_template('eval.html')


@app.route('/api/eval/personas')
def api_eval_personas():
    try:
        return jsonify(_load_json('eval_personas.json'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eval/run_symbolic', methods=['GET', 'POST'])
def api_eval_run_symbolic():
    """Run all 20 personas through symbolic rules only — fast, no LLM calls."""
    try:
        personas    = _load_json('eval_personas.json')
        biz_rules   = _load_json('business_rules.json')
        niche_rules = _load_json('niche_rules.json').get('rules', [])
        cust_rules  = _load_json('customer_rules.json').get('rules', [])
        money_rules = _load_json('money_rules.json').get('rules', [])
        disc_rules  = _load_json('discovery_rules.json').get('rules', [])

        results = []
        for p in personas:
            pid = p['id']
            row = {'id': pid, 'name': p['name'], 'quality': p['quality'], 'bio': p.get('bio',''), 'modules': {}}

            # Context
            ctx_ans = p.get('context', {})
            ctx_scores, ctx_triggered = compute_health_scores(ctx_ans, biz_rules)
            ctx_sym = int(sum(ctx_scores.values()) / len(ctx_scores))
            row['modules']['context'] = {
                'symbolic': ctx_sym,
                'violations': len(ctx_triggered),
                'errors': sum(1 for r in ctx_triggered if r.get('severity') == 'error'),
            }

            # Idea (niche)
            idea_ans  = p.get('idea', {})
            niche_txt = idea_ans.get('niche_text', '')
            viols     = validate_niche(niche_txt, niche_rules)
            row['modules']['idea'] = {
                'symbolic': _auditor.symbolic_from_violations(viols, len(niche_rules)),
                'violations': len(viols),
                'errors': sum(1 for v in viols if v['severity'] == 'error'),
            }

            # Customer
            cust_ans = p.get('customer', {})
            viols    = validate_simple_rules(cust_ans, cust_rules)
            row['modules']['customer'] = {
                'symbolic': _auditor.symbolic_from_violations(viols, len(cust_rules)),
                'violations': len(viols),
                'errors': sum(1 for v in viols if v['severity'] == 'error'),
            }

            # Money — compute derived fields
            money_ans = dict(p.get('money', {}))
            try:
                uc = float(money_ans.get('unit_cost', 0) or 0)
                up = float(money_ans.get('unit_price', 0) or 0)
                money_ans['unit_price_gt_unit_cost']   = 'Yes' if up > uc else 'No'
            except (ValueError, TypeError):
                pass
            viols = validate_simple_rules(money_ans, money_rules)
            row['modules']['money'] = {
                'symbolic': _auditor.symbolic_from_violations(viols, len(money_rules)),
                'violations': len(viols),
                'errors': sum(1 for v in viols if v['severity'] == 'error'),
            }

            # Discovery
            disc_ans = dict(p.get('discovery', {}))
            try:
                disc_ans['interviews_completed'] = int(disc_ans.get('interviews_completed', 0) or 0)
            except (ValueError, TypeError):
                disc_ans['interviews_completed'] = 0
            viols = validate_simple_rules(disc_ans, disc_rules)
            row['modules']['discovery'] = {
                'symbolic': _auditor.symbolic_from_violations(viols, len(disc_rules)),
                'violations': len(viols),
                'errors': sum(1 for v in viols if v['severity'] == 'error'),
            }

            results.append(row)

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eval/run_full/<int:persona_id>', methods=['POST'])
def api_eval_run_full(persona_id):
    """Run full triple-truth (symbolic + pure LLM + hybrid) for one persona."""
    try:
        personas = _load_json('eval_personas.json')
        p = next((x for x in personas if x['id'] == persona_id), None)
        if not p:
            return jsonify({'error': f'Persona {persona_id} not found'}), 404

        biz_rules   = _load_json('business_rules.json')
        niche_rules = _load_json('niche_rules.json').get('rules', [])
        cust_rules  = _load_json('customer_rules.json').get('rules', [])
        money_rules = _load_json('money_rules.json').get('rules', [])
        disc_rules  = _load_json('discovery_rules.json').get('rules', [])

        triples = {}

        # Context
        ctx_ans = p.get('context', {})
        ctx_scores, ctx_triggered = compute_health_scores(ctx_ans, biz_rules)
        ctx_sym = int(sum(ctx_scores.values()) / len(ctx_scores))
        pure_t  = call_context_pure_llm(ctx_ans, ctx_scores)
        hyb_t   = call_hybrid_llm(ctx_ans, ctx_scores, ctx_triggered)
        if not ctx_triggered and hyb_t:
            import re as _re_ctx_eval1
            _ctx_floor1 = max(70, ctx_sym - 5)
            _ctx_m1 = _re_ctx_eval1.search(r'MENTOR_SCORE:\s*(\d{1,3})', hyb_t)
            if _ctx_m1 and int(_ctx_m1.group(1)) < _ctx_floor1:
                hyb_t = _re_ctx_eval1.sub(r'MENTOR_SCORE:\s*\d{1,3}', f'MENTOR_SCORE: {_ctx_floor1}', hyb_t)
        triples['context'] = _auditor.triple_truth(ctx_sym, pure_t, hyb_t, 'context')

        # Idea
        idea_ans  = p.get('idea', {})
        niche_txt = idea_ans.get('niche_text', '')
        viols     = validate_niche(niche_txt, niche_rules)
        sym       = _auditor.symbolic_from_violations(viols, len(niche_rules))
        pure_t    = call_niche_pure_llm(niche_txt, p.get('context', {}))
        hyb_t     = call_niche_socratic_llm(niche_txt, viols, p.get('context', {}))
        triples['idea'] = _auditor.triple_truth(sym, pure_t, hyb_t, 'idea')

        # Customer
        cust_ans = p.get('customer', {})
        viols    = validate_simple_rules(cust_ans, cust_rules)
        sym      = _auditor.symbolic_from_violations(viols, len(cust_rules))
        pure_t   = call_customer_pure_llm(cust_ans)
        hyb_t    = call_customer_hybrid_llm(cust_ans, viols)
        triples['customer'] = _auditor.triple_truth(sym, pure_t, hyb_t, 'customer')

        # Money
        money_ans = dict(p.get('money', {}))
        try:
            uc = float(money_ans.get('unit_cost', 0) or 0)
            up = float(money_ans.get('unit_price', 0) or 0)
            monthly_units = float(money_ans.get('monthly_units', 0) or 0)
            startup_cost  = float(money_ans.get('startup_cost', 0) or 0)
            monthly_profit = (up - uc) * monthly_units
            months_to_be   = (startup_cost / monthly_profit) if monthly_profit > 0 else 999
            _arch = p.get('context', {}).get('business_type', '')
            # unit_price_gt_unit_cost (MR001)
            money_ans['unit_price_gt_unit_cost'] = 'Yes' if up > uc else 'No'
            # monthly_profit_lt_threshold (MR003) — archetype-aware floor
            _profit_floor = {'food-beverage': 25, 'resale-retail': 25, 'content-media': 10,
                             'digital-product': 10}.get(_arch, 20)
            money_ans['monthly_profit_lt_threshold'] = 'Yes' if 0 < monthly_profit < _profit_floor else 'No'
            # months_to_breakeven_gt_threshold (MR002) — archetype-aware cap
            _be_caps = {'food-beverage': 6, 'physical-product': 6, 'maker': 6,
                        'service': 12, 'skills-for-hire': 12, 'expert': 12, 'resale-retail': 12,
                        'digital-product': 18, 'content-media': 36, 'event-experience': 9, 'marketplace': 12}
            _be_cap = _be_caps.get(_arch, 12)
            money_ans['months_to_breakeven_gt_threshold'] = 'Yes' if months_to_be > _be_cap else 'No'
            # hourly_rate_lt_minimum (MR006)
            _hrs_pw = float(str(p.get('context', {}).get('hours_per_week', '0') or '0'))
            if _hrs_pw > 0 and monthly_profit > 0:
                money_ans['hourly_rate_lt_minimum'] = 'Yes' if (monthly_profit / (_hrs_pw * 4)) < 5 else 'No'
            else:
                money_ans['hourly_rate_lt_minimum'] = 'No'
        except (ValueError, TypeError):
            pass
        viols  = validate_simple_rules(money_ans, money_rules)
        sym    = _auditor.symbolic_from_violations(viols, len(money_rules))
        pure_t = call_money_pure_llm(money_ans)
        hyb_t  = call_hybrid_llm(money_ans, {'symbolic': sym},
                                  [r for r in money_rules if any(v['id'] == r['id'] for v in viols)])
        # MENTOR_SCORE override — mirrors live money route
        if viols and hyb_t:
            import re as _re_money_eval
            _n_err  = sum(1 for v in viols if v.get('severity') == 'error')
            _n_warn = sum(1 for v in viols if v.get('severity') == 'warning')
            _money_score = max(15, min(55, 100 - _n_err * 30 - _n_warn * 8)) if _n_err > 0 \
                           else max(50, min(75, 100 - _n_warn * 10))
            if _re_money_eval.search(r'MENTOR_SCORE:\s*\d', hyb_t):
                hyb_t = _re_money_eval.sub(r'MENTOR_SCORE:\s*\d{1,3}', f'MENTOR_SCORE: {_money_score}', hyb_t)
            else:
                hyb_t += f'\nMENTOR_SCORE: {_money_score}'
        triples['money'] = _auditor.triple_truth(sym, pure_t, hyb_t, 'money')

        # Discovery
        disc_ans = dict(p.get('discovery', {}))
        try:
            disc_ans['interviews_completed'] = int(disc_ans.get('interviews_completed', 0) or 0)
        except (ValueError, TypeError):
            disc_ans['interviews_completed'] = 0
        viols  = validate_simple_rules(disc_ans, disc_rules)
        sym    = _auditor.symbolic_from_violations(viols, len(disc_rules))
        pure_t = call_discovery_pure_llm(disc_ans)
        hyb_t  = call_hybrid_llm(disc_ans, {'symbolic': sym},
                                  [r for r in disc_rules if any(v['id'] == r['id'] for v in viols)])
        # MENTOR_SCORE override — mirrors live discovery route
        if viols and hyb_t:
            import re as _re_disc_eval
            _n_err  = sum(1 for v in viols if v.get('severity') == 'error')
            _n_warn = sum(1 for v in viols if v.get('severity') == 'warning')
            _disc_score = max(20, min(55, 100 - _n_err * 30 - _n_warn * 8)) if _n_err > 0 \
                          else max(50, min(75, 100 - _n_warn * 10))
            if _re_disc_eval.search(r'MENTOR_SCORE:\s*\d', hyb_t):
                hyb_t = _re_disc_eval.sub(r'MENTOR_SCORE:\s*\d{1,3}', f'MENTOR_SCORE: {_disc_score}', hyb_t)
            else:
                hyb_t += f'\nMENTOR_SCORE: {_disc_score}'
        triples['discovery'] = _auditor.triple_truth(sym, pure_t, hyb_t, 'discovery')

        return jsonify({'persona_id': persona_id, 'name': p['name'], 'triples': triples})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eval/run_all_full', methods=['POST'])
def api_eval_run_all_full():
    """Run full triple-truth for ALL personas — used for CSV export."""
    try:
        personas    = _load_json('eval_personas.json')
        biz_rules   = _load_json('business_rules.json')
        niche_rules = _load_json('niche_rules.json').get('rules', [])
        cust_rules  = _load_json('customer_rules.json').get('rules', [])
        money_rules = _load_json('money_rules.json').get('rules', [])
        disc_rules  = _load_json('discovery_rules.json').get('rules', [])

        results = []
        for p in personas:
            try:
                triples = {}

                # Context
                ctx_ans = p.get('context', {})
                ctx_scores, ctx_triggered = compute_health_scores(ctx_ans, biz_rules)
                ctx_sym = int(sum(ctx_scores.values()) / len(ctx_scores)) if ctx_scores else 0
                pure_t  = call_context_pure_llm(ctx_ans, ctx_scores)
                hyb_t   = call_hybrid_llm(ctx_ans, ctx_scores, ctx_triggered)
                if not ctx_triggered and hyb_t:
                    import re as _re_ctx_all
                    _ctx_floor_all = max(70, ctx_sym - 5)
                    _ctx_m_all = _re_ctx_all.search(r'MENTOR_SCORE:\s*(\d{1,3})', hyb_t)
                    if _ctx_m_all and int(_ctx_m_all.group(1)) < _ctx_floor_all:
                        hyb_t = _re_ctx_all.sub(r'MENTOR_SCORE:\s*\d{1,3}',
                                                 f'MENTOR_SCORE: {_ctx_floor_all}', hyb_t)
                triples['context'] = _auditor.triple_truth(ctx_sym, pure_t, hyb_t, 'context')

                # Idea
                idea_ans  = p.get('idea', {})
                niche_txt = idea_ans.get('niche_text', '')
                viols     = validate_niche(niche_txt, niche_rules)
                sym       = _auditor.symbolic_from_violations(viols, len(niche_rules))
                pure_t    = call_niche_pure_llm(niche_txt, p.get('context', {}))
                hyb_t     = call_niche_socratic_llm(niche_txt, viols, p.get('context', {}))
                triples['idea'] = _auditor.triple_truth(sym, pure_t, hyb_t, 'idea')

                # Customer
                cust_ans = p.get('customer', {})
                viols    = validate_simple_rules(cust_ans, cust_rules)
                sym      = _auditor.symbolic_from_violations(viols, len(cust_rules))
                pure_t   = call_customer_pure_llm(cust_ans)
                hyb_t    = call_customer_hybrid_llm(cust_ans, viols)
                triples['customer'] = _auditor.triple_truth(sym, pure_t, hyb_t, 'customer')

                # Money
                money_ans = dict(p.get('money', {}))
                try:
                    uc = float(money_ans.get('unit_cost', 0) or 0)
                    up = float(money_ans.get('unit_price', 0) or 0)
                    monthly_units = float(money_ans.get('monthly_units', 0) or 0)
                    startup_cost  = float(money_ans.get('startup_cost', 0) or 0)
                    monthly_profit = (up - uc) * monthly_units
                    months_to_be   = (startup_cost / monthly_profit) if monthly_profit > 0 else 999
                    _arch = p.get('context', {}).get('business_type', '')
                    money_ans['unit_price_gt_unit_cost'] = 'Yes' if up > uc else 'No'
                    _profit_floor = {'food-beverage': 25, 'resale-retail': 25, 'content-media': 10,
                                     'digital-product': 10}.get(_arch, 20)
                    money_ans['monthly_profit_lt_threshold'] = 'Yes' if 0 < monthly_profit < _profit_floor else 'No'
                    _be_caps = {'food-beverage': 6, 'physical-product': 6, 'maker': 6,
                                'service': 12, 'skills-for-hire': 12, 'expert': 12, 'resale-retail': 12,
                                'digital-product': 18, 'content-media': 36, 'event-experience': 9, 'marketplace': 12}
                    _be_cap = _be_caps.get(_arch, 12)
                    money_ans['months_to_breakeven_gt_threshold'] = 'Yes' if months_to_be > _be_cap else 'No'
                    _hrs_pw = float(str(p.get('context', {}).get('hours_per_week', '0') or '0'))
                    if _hrs_pw > 0 and monthly_profit > 0:
                        money_ans['hourly_rate_lt_minimum'] = 'Yes' if (monthly_profit / (_hrs_pw * 4)) < 5 else 'No'
                    else:
                        money_ans['hourly_rate_lt_minimum'] = 'No'
                except (ValueError, TypeError):
                    pass
                viols  = validate_simple_rules(money_ans, money_rules)
                sym    = _auditor.symbolic_from_violations(viols, len(money_rules))
                pure_t = call_money_pure_llm(money_ans)
                hyb_t  = call_hybrid_llm(money_ans, {'symbolic': sym},
                                         [r for r in money_rules if any(v['id'] == r['id'] for v in viols)])
                # MENTOR_SCORE override — mirrors live money route
                if viols and hyb_t:
                    import re as _re_money_all
                    _n_err  = sum(1 for v in viols if v.get('severity') == 'error')
                    _n_warn = sum(1 for v in viols if v.get('severity') == 'warning')
                    _money_score = max(15, min(55, 100 - _n_err * 30 - _n_warn * 8)) if _n_err > 0 \
                                   else max(50, min(75, 100 - _n_warn * 10))
                    if _re_money_all.search(r'MENTOR_SCORE:\s*\d', hyb_t):
                        hyb_t = _re_money_all.sub(r'MENTOR_SCORE:\s*\d{1,3}', f'MENTOR_SCORE: {_money_score}', hyb_t)
                    else:
                        hyb_t += f'\nMENTOR_SCORE: {_money_score}'
                triples['money'] = _auditor.triple_truth(sym, pure_t, hyb_t, 'money')

                # Discovery
                disc_ans = dict(p.get('discovery', {}))
                try:
                    disc_ans['interviews_completed'] = int(disc_ans.get('interviews_completed', 0) or 0)
                except (ValueError, TypeError):
                    disc_ans['interviews_completed'] = 0
                viols  = validate_simple_rules(disc_ans, disc_rules)
                sym    = _auditor.symbolic_from_violations(viols, len(disc_rules))
                pure_t = call_discovery_pure_llm(disc_ans)
                hyb_t  = call_hybrid_llm(disc_ans, {'symbolic': sym},
                                         [r for r in disc_rules if any(v['id'] == r['id'] for v in viols)])
                # MENTOR_SCORE override — mirrors live discovery route
                if viols and hyb_t:
                    import re as _re_disc_all
                    _n_err  = sum(1 for v in viols if v.get('severity') == 'error')
                    _n_warn = sum(1 for v in viols if v.get('severity') == 'warning')
                    _disc_score = max(20, min(55, 100 - _n_err * 30 - _n_warn * 8)) if _n_err > 0 \
                                  else max(50, min(75, 100 - _n_warn * 10))
                    if _re_disc_all.search(r'MENTOR_SCORE:\s*\d', hyb_t):
                        hyb_t = _re_disc_all.sub(r'MENTOR_SCORE:\s*\d{1,3}', f'MENTOR_SCORE: {_disc_score}', hyb_t)
                    else:
                        hyb_t += f'\nMENTOR_SCORE: {_disc_score}'
                triples['discovery'] = _auditor.triple_truth(sym, pure_t, hyb_t, 'discovery')

                results.append({'id': p['id'], 'name': p['name'], 'triples': triples})
            except Exception as pe:
                # Don't fail the whole batch — record error for this persona
                results.append({'id': p['id'], 'name': p['name'], 'error': str(pe), 'triples': {}})

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/share', methods=['POST'])
def api_share():
    """Generate a one-time teacher share token capturing current session progress."""
    snapshot = {
        'name':             session.get('name', 'Student'),
        'grade_level':      session.get('grade_level', ''),
        'business_type':    session.get('business_type', ''),
        'sell_what':        session.get('sell_what', ''),
        'sell_to':          session.get('sell_to', ''),
        'sell_where':       session.get('sell_where', ''),
        'audit_scores':     dict(session.get('audit_scores', {})),
        'radar_scores':     dict(session.get('radar_scores', {})),
        'story':            session.get('story', ''),
        'created_at':       datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # C8-30: Full Chapter 1 profile for teacher context
        'ch1_profile': {
            'budget':          session.get('budget', ''),
            'hours_per_week':  session.get('hours_per_week', ''),
            'grade_level':     session.get('grade_level', ''),
            'business_type':   session.get('business_type', ''),
            'delivery_model':  session.get('delivery_model', ''),
            'prior_experience':session.get('prior_experience', ''),
        },
    }
    token = secrets.token_urlsafe(16)
    with _teacher_shares_lock:
        # Keep at most 200 tokens in memory and on disk
        if len(_teacher_shares) >= 200:
            oldest = sorted(_teacher_shares, key=lambda t: _teacher_shares[t]['created_at'])[:50]
            for t in oldest:
                del _teacher_shares[t]
        _teacher_shares[token] = snapshot
        _save_share_tokens(_teacher_shares)  # C6-40: persist to disk
    share_url = url_for('teacher_view', token=token, _external=True)
    return jsonify({'token': token, 'url': share_url})


@app.route('/about')
def about():
    """Architecture explainer — how Triple Truth scoring works."""
    return render_template('about.html')


# ── Demo class — always available at code DEMO00 ──
_DEMO_CLASS = {
    'code': 'DEMO00',
    'class_name': 'Demo Class (Year 9 Business)',
    'teacher_name': 'Ms Johnson',
    'created_at': '2026-01-15T09:00:00',
    'students': [
        {
            'name': 'Alice Chen',
            'joined_at': '2026-01-16T10:30:00',
            'share_url': '/share/sample',
            'scores': {
                'context':   {'hybrid_truth': 74, 'pure_truth': 85, 'symbolic_score': 70, 'sycophancy_gap': 11},
                'idea':      {'hybrid_truth': 68, 'pure_truth': 80, 'symbolic_score': 65, 'sycophancy_gap': 12},
                'customer':  {'hybrid_truth': 72, 'pure_truth': 82, 'symbolic_score': 68, 'sycophancy_gap': 10},
                'money':     {'hybrid_truth': 78, 'pure_truth': 88, 'symbolic_score': 74, 'sycophancy_gap': 10},
                'discovery': {'hybrid_truth': 81, 'pure_truth': 89, 'symbolic_score': 76, 'sycophancy_gap':  8},
            }
        },
        {
            'name': 'Ben Okafor',
            'joined_at': '2026-01-16T11:15:00',
            'share_url': None,
            'scores': {
                'context': {'hybrid_truth': 55, 'pure_truth': 72, 'symbolic_score': 50, 'sycophancy_gap': 17},
                'idea':    {'hybrid_truth': 48, 'pure_truth': 70, 'symbolic_score': 42, 'sycophancy_gap': 22},
            }
        },
        {
            'name': 'Priya Sharma',
            'joined_at': '2026-01-17T09:45:00',
            'share_url': None,
            'scores': {
                'context':  {'hybrid_truth': 88, 'pure_truth': 91, 'symbolic_score': 85, 'sycophancy_gap': 3},
                'idea':     {'hybrid_truth': 82, 'pure_truth': 87, 'symbolic_score': 79, 'sycophancy_gap': 5},
                'customer': {'hybrid_truth': 85, 'pure_truth': 90, 'symbolic_score': 82, 'sycophancy_gap': 5},
                'money':    {'hybrid_truth': 79, 'pure_truth': 84, 'symbolic_score': 75, 'sycophancy_gap': 5},
            }
        },
        {
            'name': 'Marcus Webb',
            'joined_at': '2026-01-17T13:20:00',
            'share_url': None,
            'scores': {}
        },
    ]
}


@app.route('/teacher')
def teacher_view():
    """Read-only teacher summary page for a student's progress snapshot."""
    token = request.args.get('token', '')
    if not token:
        return redirect(url_for('teacher_dashboard'))
    with _teacher_shares_lock:
        snapshot = _teacher_shares.get(token)
    if not snapshot:
        return render_template('teacher.html', error='Invalid or expired share link.', snapshot=None)
    return render_template('teacher.html', error=None, snapshot=snapshot)


@app.route('/teacher/dashboard')
def teacher_dashboard():
    """Teacher landing page — create a class, view a class roster, or look up a student plan.
    Visiting this page sets teacher_mode in session, which reveals the Eval + Teacher nav links."""
    session['teacher_mode'] = True
    session.modified = True
    return render_template('teacher_dashboard.html')


@app.route('/teacher/export.csv')
def teacher_export_csv():
    """Export all student share tokens as a CSV file for marking.
    Columns: name, grade, business idea, chapters done, avg mentor score, avg sycophancy gap.
    Only real student entries are included (not the demo/sample token)."""
    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        'Name', 'Grade', 'Business Idea', 'Target Customer',
        'Chapters Done',
        'Context Mentor', 'Idea Mentor', 'Customer Mentor', 'Money Mentor', 'Discovery Mentor',
        'Avg Mentor Score', 'Avg Sycophancy Gap',
        'Share Link', 'Submitted At',
    ])
    _chapter_ids = ['context', 'idea', 'customer', 'money', 'discovery']
    with _teacher_shares_lock:
        tokens_snapshot = dict(_teacher_shares)
    for token, snap in tokens_snapshot.items():
        if token == 'sample':
            continue
        scores = snap.get('audit_scores') or snap.get('scores') or {}
        chapter_scores = [scores.get(c, {}).get('hybrid_truth') for c in _chapter_ids]
        done_scores = [s for s in chapter_scores if s is not None]
        avg_mentor = round(sum(done_scores) / len(done_scores)) if done_scores else ''
        gap_vals = [scores.get(c, {}).get('sycophancy_gap') for c in _chapter_ids]
        done_gaps = [g for g in gap_vals if g is not None]
        avg_gap   = round(sum(done_gaps) / len(done_gaps)) if done_gaps else ''
        share_url = url_for('teacher_view', token=token, _external=True) if not token.startswith('code_') else ''
        writer.writerow([
            snap.get('name', ''),
            snap.get('grade_level', ''),
            snap.get('sell_what', ''),
            snap.get('sell_to', ''),
            len(done_scores),
            *(s if s is not None else '' for s in chapter_scores),
            avg_mentor,
            avg_gap,
            share_url,
            snap.get('created_at', snap.get('joined_at', '')),
        ])
    csv_bytes = buf.getvalue().encode('utf-8-sig')  # utf-8-sig adds BOM so Excel opens it correctly
    return Response(
        csv_bytes,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename="lrnbiz_class_export.csv"'},
    )


@app.route('/share/sample')
def sample_plan():
    """C8-17: Static sample completed plan shown to new students on the homepage."""
    sample_snapshot = {
        'name': 'Alex (Sample)',
        'grade_level': 'Year 9',
        'business_type': 'food-beverage',
        'sell_what': 'Healthy snack boxes for secondary school students',
        'sell_to': 'Year 7–10 students at local schools',
        'sell_where': 'School canteen pop-up, once a week',
        'unit_price': 4.50,
        'unit_cost': 1.80,
        'startup_cost': 120,
        'niche_statement': 'I help Year 7–10 students at my school who want a healthy snack option by selling pre-packed fruit-and-nut boxes at the canteen every Friday.',
        'audit_scores': {
            'context':   {'symbolic_score': 72, 'pure_truth': 85, 'hybrid_truth': 82, 'sycophancy_gap': 3},
            'idea':      {'symbolic_score': 68, 'pure_truth': 82, 'hybrid_truth': 78, 'sycophancy_gap': 4},
            'customer':  {'symbolic_score': 74, 'pure_truth': 88, 'hybrid_truth': 84, 'sycophancy_gap': 4},
            'money':     {'symbolic_score': 71, 'pure_truth': 80, 'hybrid_truth': 80, 'sycophancy_gap': -1},
            'discovery': {'symbolic_score': 65, 'pure_truth': 78, 'hybrid_truth': 75, 'sycophancy_gap': 3},
        },
        'ch1_profile': {
            'budget': 120,
            'hours_per_week': 6,
            'grade_level': 'Year 9',
            'business_type': 'food-beverage',
            'delivery_model': 'In-person',
            'prior_experience': 'None',
        },
        'is_sample': True,
        'created_at': '2026-01-15T10:30:00',
        'radar_scores': {'Knowledge': 78, 'Customer': 84, 'Money': 71, 'Discovery': 75},
        'token': 'sample',
        'story': 'Alex noticed that healthy food options at the school canteen were limited. After interviewing 8 classmates, Alex discovered most students wanted something quick, filling and not too expensive. The snack box idea was born from that research.',
    }
    return render_template('teacher.html', error=None, snapshot=sample_snapshot)


@app.route('/api/save_code', methods=['POST'])
def api_save_code():
    """C7-48: Generate a 6-char restore code that saves key session data server-side."""
    payload = {
        'name':          session.get('name', ''),
        'grade_level':   session.get('grade_level', ''),
        'sell_what':     session.get('sell_what', ''),
        'sell_to':       session.get('sell_to', ''),
        'sell_where':    session.get('sell_where', ''),
        'business_type': session.get('business_type', ''),
        'unit_price':    session.get('unit_price', ''),
        'unit_cost':     session.get('unit_cost', ''),
        'startup_cost':  session.get('startup_cost', ''),
        'audit_scores':  dict(session.get('audit_scores', {})),
        'created_at':    datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    # 6-char alphanumeric code
    code = ''.join(secrets.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(6))
    with _teacher_shares_lock:
        _teacher_shares['code_' + code] = payload
        _save_share_tokens(_teacher_shares)
    return jsonify({'code': code})


@app.route('/api/restore_code', methods=['POST'])
def api_restore_code():
    """C7-48: Restore session data from a 6-char save code."""
    data = request.json or {}
    code = (data.get('code', '') or '').strip().upper()
    with _teacher_shares_lock:
        payload = _teacher_shares.get('code_' + code)
    if not payload:
        return jsonify({'error': 'Code not found or expired'}), 404
    # Restore non-empty fields into session
    for k, v in payload.items():
        if k not in ('created_at',) and v:
            session[k] = v
    # Normalise any legacy business_type value (e.g. 'Product' → 'physical-product')
    if 'business_type' in session:
        session['business_type'] = _normalise_business_type(session['business_type'])
    # Also normalise inside stage1_answers if present
    sa = session.get('stage1_answers', {})
    if sa.get('business_type'):
        sa['business_type'] = _normalise_business_type(sa['business_type'])
        session['stage1_answers'] = sa
    return jsonify({'ok': True, 'name': payload.get('name', ''), 'chapters': len(payload.get('audit_scores', {}))})


@app.route('/api/hint', methods=['POST'])
def api_hint():
    """C6-27: Single Socratic hint for a student who is stuck mid-chapter."""
    if not GROQ_AVAILABLE:
        return jsonify({'hint': '💡 Try to think about your customer more specifically — who exactly has this problem, and why would they pay to solve it?'})
    data    = request.json or {}
    page    = data.get('page', 'context')
    answers = data.get('answers', {})
    filled  = ', '.join(f"{k}: {v}" for k, v in list(answers.items())[:6] if v)
    page_ctx = {
        'context':   'Chapter 1 Business DNA (grade, budget, location, experience)',
        'idea':      'Chapter 2 Business Idea (WHO/PROBLEM/WHERE/SOLUTION)',
        'customer':  'Chapter 3 Target Customer (age, problem, differentiation)',
        'money':     'Chapter 4 Money Math (prices, costs, break-even)',
        'discovery': 'Chapter 5 Customer Discovery (interviews, what they learned)',
    }.get(page, 'business planning')
    prompt = (
        f"A student is stuck on {page_ctx}. Their current answers: {filled or '(none filled yet)'}. "
        "Ask ONE short, friendly Socratic question (no more than 2 sentences) to help them think more deeply. "
        "Don't give the answer. Be encouraging and specific to their situation."
    )
    try:
        rsp = _groq_create(
            model='llama-3.1-8b-instant',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0, max_tokens=80,
        )
        return jsonify({'hint': rsp.choices[0].message.content.strip()})
    except Exception:
        return jsonify({'hint': '💡 Think about your customer more specifically — who exactly is experiencing this problem, and how much would they pay to solve it?'})


@app.route('/api/regenerate_story', methods=['POST'])
def api_regenerate_story():
    """C6-34: Generate a fresh AI business narrative from current session data."""
    if not GROQ_AVAILABLE:
        return jsonify({'error': _NO_KEY_MSG})
    name       = session.get('student_name', 'Young Entrepreneur')
    sell_what  = session.get('sell_what', '')
    sell_to    = session.get('sell_to', '')
    sell_where = session.get('sell_where', '')
    audit      = session.get('audit_scores', {})
    avg_score  = round(sum((audit[m].get('hybrid_truth', 0) for m in audit), 0) / max(1, len(audit)))
    chapters_done = len(audit)
    prompt = (
        f"Write a short, encouraging 3-4 sentence entrepreneurship story for a student named {name}. "
        f"Their business idea: I sell {sell_what} to {sell_to}"
        f"{(' at ' + sell_where) if sell_where else ''}. "
        f"They completed {chapters_done}/5 chapters with an average mentor score of {avg_score}/100. "
        "Make it personal, celebratory, and specific to their idea. Keep it under 100 words."
    )
    try:
        rsp = _groq_create(
            model='llama-3.1-8b-instant',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.8, max_tokens=150,
        )
        story = rsp.choices[0].message.content.strip()
        session['story'] = story
        session.modified = True
        return jsonify({'story': story})
    except Exception as e:
        return jsonify({'error': f'Could not generate story: {str(e)}'})


# ══════════════════════════════════════════════════════════════
#  C8-31 — CLASS CODE ROSTER
#  Teachers create a class code; students enter it on their
#  homepage. Teacher sees all students in one dashboard.
# ══════════════════════════════════════════════════════════════
_CLASS_CODES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_codes.json')
_class_codes_lock = threading.Lock()

def _load_class_codes() -> dict:
    try:
        if os.path.exists(_CLASS_CODES_PATH):
            with open(_CLASS_CODES_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_class_codes(codes: dict):
    try:
        with open(_CLASS_CODES_PATH, 'w', encoding='utf-8') as f:
            json.dump(codes, f, indent=2)
    except Exception:
        pass

_class_codes = _load_class_codes()


@app.route('/api/class/create', methods=['POST'])
def api_class_create():
    """Teacher creates a class — returns a 6-char class code."""
    data = request.json or {}
    teacher_name = (data.get('teacher_name') or 'Teacher').strip()[:60]
    class_name   = (data.get('class_name') or 'My Class').strip()[:80]
    code = ''.join(secrets.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(6))
    entry = {
        'code': code,
        'teacher_name': teacher_name,
        'class_name': class_name,
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'students': [],
    }
    with _class_codes_lock:
        _class_codes[code] = entry
        _save_class_codes(_class_codes)
    return jsonify({'code': code, 'class_name': class_name})


@app.route('/api/class/join', methods=['POST'])
def api_class_join():
    """Student joins a class with the 6-char code — snapshots their current progress."""
    data = request.json or {}
    code = (data.get('code') or '').strip().upper()
    with _class_codes_lock:
        cls = _class_codes.get(code)
    if not cls:
        return jsonify({'error': 'Class code not found'}), 404
    share_token = data.get('share_token', '')
    student_entry = {
        'name':        session.get('name', 'Student'),
        'joined_at':   datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'grade_level': session.get('grade_level', ''),
        'budget':      session.get('budget', ''),
        'scores': {
            mid: {'hybrid_truth': sc.get('hybrid_truth', 0), 'analysed_at': sc.get('analysed_at', '')}
            for mid, sc in session.get('audit_scores', {}).items()
        },
        'share_token': share_token,
        'share_url':   url_for('teacher_view', token=share_token, _external=True) if share_token else '',
    }
    with _class_codes_lock:
        cls = _class_codes.get(code, {})
        students = cls.get('students', [])
        # Update existing entry if same student name re-joins
        students = [s for s in students if s.get('name') != student_entry['name']]
        students.append(student_entry)
        cls['students'] = students
        _class_codes[code] = cls
        _save_class_codes(_class_codes)
    session['class_code'] = code
    session.modified = True
    return jsonify({'ok': True, 'class_name': cls.get('class_name', '')})


@app.route('/class/<code>')
def class_roster(code):
    """Teacher roster view — all students in a class."""
    code = code.upper()
    if code == 'DEMO00':
        return render_template('class_roster.html', error=None, cls=_DEMO_CLASS)
    with _class_codes_lock:
        cls = _class_codes.get(code)
    if not cls:
        return render_template('class_roster.html', error='Class code not found.', cls=None)
    return render_template('class_roster.html', error=None, cls=cls)


# ══════════════════════════════════════════════════════════════
#  C8-32 — RESEARCH LOG FILTERED EXPORT
# ══════════════════════════════════════════════════════════════
@app.route('/admin/research')
@_require_admin
def admin_research():
    """Research log viewer with optional filters: ?chapter=idea&from=2026-01-01&flag=NR001"""
    log_path = os.path.join(app.root_path, 'researchlogs.json')
    logs = []
    try:
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
    except Exception:
        pass

    chapter_filter = request.args.get('chapter', '').strip()
    from_filter    = request.args.get('from', '').strip()
    flag_filter    = request.args.get('flag', '').strip().upper()
    fmt            = request.args.get('fmt', 'json')

    if chapter_filter:
        logs = [e for e in logs if e.get('module_id') == chapter_filter]
    if from_filter:
        logs = [e for e in logs if e.get('timestamp', '') >= from_filter]
    if flag_filter:
        def _has_flag(e):
            return (flag_filter in e.get('cc_flags', []) or
                    flag_filter in e.get('nr_flags', []) or
                    any(v.get('id') == flag_filter for v in e.get('violations', [])))
        logs = [e for e in logs if _has_flag(e)]

    if fmt == 'csv':
        import csv, io
        buf = io.StringIO()
        fieldnames = ['timestamp','module_id','symbolic_score','pure_truth','hybrid_truth',
                      'sycophancy_gap','cc_flags','nr_flags']
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for e in logs:
            row = dict(e)
            row['cc_flags'] = ','.join(e.get('cc_flags', []))
            row['nr_flags'] = ','.join(e.get('nr_flags', []))
            writer.writerow(row)
        return Response(buf.getvalue(), mimetype='text/csv',
                        headers={'Content-Disposition': 'attachment;filename=research_export.csv'})

    return jsonify({'count': len(logs), 'entries': logs})


@app.route('/api/save_reflection', methods=['POST'])
def api_save_reflection():
    """C8-18: Save end-of-chapter reflection text to session + research log."""
    data    = request.json or {}
    chapter = data.get('chapter', 'unknown')[:20]
    text    = (data.get('text') or '').strip()[:200]
    if not text:
        return jsonify({'ok': False, 'error': 'No text'}), 400
    reflections = session.setdefault('reflections', {})
    reflections[chapter] = {
        'text': text,
        'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    session.modified = True
    # Log to research data
    log_path = os.path.join(app.root_path, 'researchlogs.json')
    try:
        with _research_log_lock:
            logs = []
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            logs.append({
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'type': 'reflection',
                'chapter': chapter,
                'text': text,
            })
            if len(logs) > 500:
                logs = logs[-500:]
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2)
    except Exception:
        pass
    return jsonify({'ok': True})


@app.route('/api/set_name', methods=['POST'])
def api_set_name():
    """S21-09: Store the student's name in session (called from welcome modal)."""
    data = request.json or {}
    name = (data.get('name') or '').strip()[:60]
    if name:
        session['name'] = name
        session.modified = True
    return jsonify({'ok': True})


@app.route('/health')
def health_check():
    """C6-43: Health check endpoint for monitoring."""
    return jsonify({'status': 'ok', 'version': '1.0', 'ai_available': GROQ_AVAILABLE})


if __name__ == '__main__':
    app.run(debug=False, use_reloader=True)
