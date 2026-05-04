# LrnBiz — AI-Powered Intelligent Tutoring System for Young Entrepreneurs

**LrnBiz** is a 5-chapter web-based Intelligent Tutoring System (ITS) that guides school students (grades 6–12) through building and validating a real business plan. It combines a deterministic rules engine with two simultaneous AI models to detect AI sycophancy — showing students where automated feedback is over-optimistic versus what an honest mentor actually thinks.

This project was built as an academic submission exploring how AI tutoring tools can be made *honest* and *pedagogically sound*, rather than just encouraging.

---

## Live Demo

**[lrnbiz.onrender.com](https://lrnbiz.onrender.com)**

> First load may take ~30 seconds if the server is sleeping (free tier on Render).

---

## What It Does

Students work through five sequential chapters. Each chapter unlocks only when the previous one is complete:

| Chapter | Topic | What students do |
|---------|-------|-----------------|
| 1 | **Context** | Set their budget, location, grade, business type, hours per week |
| 2 | **Business Idea** | Describe their idea; validated by niche rules + AI scoring |
| 3 | **Target Customer** | Define their customer persona; checks alignment with idea |
| 4 | **Money Math** | Model pricing, costs, break-even, and profit — validated by financial rules |
| 5 | **Customer Discovery** | Log interview evidence; checks quality of customer research |

After all five chapters, students receive a **certificate** with their final Business DNA radar chart showing readiness across six axes: Passion, Energy, Gold, Influence, Knowledge, Target.

---

## The Triple Truth Scoring System

The core academic contribution of LrnBiz is its **Triple Truth** approach to scoring student plans. Each chapter produces three simultaneous scores:

```
┌────────────────────┬──────────────────────────────────────────────────────┐
│ Score Type         │ How It Works                                         │
├────────────────────┼──────────────────────────────────────────────────────┤
│ Rules Check        │ Deterministic rules engine — always consistent,      │
│ (Symbolic Score)   │ no randomness. Fires JSON-defined rules against the  │
│                    │ student's answers. Each rule has a severity (error /  │
│                    │ warning) and a score impact.                         │
├────────────────────┼──────────────────────────────────────────────────────┤
│ AI Optimist        │ llama-3.1-8b-instant with a "blindly positive"       │
│ (Pure LLM Score)   │ system prompt. Ignores all problems, only praises.   │
│                    │ Self-reports its own score (SCORE: N) so sycophancy  │
│                    │ is measured genuinely, not estimated.                │
├────────────────────┼──────────────────────────────────────────────────────┤
│ Mentor Score       │ llama-3.1-8b-instant given the rule violations    │
│ (Hybrid Score)     │ as context. Asks ONE Socratic question about the     │
│                    │ most critical issue. Self-reports MENTOR_SCORE: N.   │
└────────────────────┴──────────────────────────────────────────────────────┘

Sycophancy Gap = AI Optimist Score − Mentor Score

A positive gap means the AI was over-enthusiastic relative to the rules-grounded mentor.
Students can see this gap — it teaches them that AI praise is not the same as good advice.
```

---

## Architecture

```
lrnbiz/
├── app.py                  Flask application — all routes, rule engine, LLM calls (~4000 lines)
├── templates/
│   ├── base.html           Shared layout: nav, radar chart, modals, nudge system
│   ├── context.html        Chapter 1 — personal context form
│   ├── idea.html           Chapter 2 — business idea + niche validation
│   ├── customer.html       Chapter 3 — customer persona builder
│   ├── money.html          Chapter 4 — money math + break-even calculator
│   ├── discovery.html      Chapter 5 — customer discovery evidence
│   ├── final.html          Summary dashboard with all five scores
│   ├── certificate.html    Printable completion certificate
│   ├── teacher.html        Teacher login / class creation
│   └── teacher_dashboard.html  Class roster, student plan viewer
├── static/
│   ├── main.js             Shared JS: radar chart, badge system, sound FX
│   ├── dashboard.css       Scores/progress page styles
│   └── style.css           Core application styles
├── *_rules.json            Business validation rules (5 files, one per chapter)
├── eval_personas.json      20 synthetic student personas for ITS benchmarking
├── researchlogs.json       Anonymised scoring audit log (auto-generated)
├── share_tokens.json       Teacher share tokens (auto-generated)
├── flask_sessions/         Server-side session storage (auto-generated)
├── requirements.txt        Python dependencies
└── .env                    Environment variables (not committed to git)
```

### Rule Files

Each chapter has its own JSON rule file:

| File | Chapter | Rules |
|------|---------|-------|
| `business_rules.json` | Context (Ch.1) | Budget/location/grade constraints, card-lock logic |
| `niche_rules.json` | Idea (Ch.2) | Niche validation, idea-type rules |
| `customer_rules.json` | Customer (Ch.3) | Persona completeness, alignment checks |
| `money_rules.json` | Money Math (Ch.4) | Price > cost, break-even thresholds, profit floor |
| `discovery_rules.json` | Discovery (Ch.5) | Interview count, question quality, insight evidence |

---

## How to Run Locally

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Create a `.env` file in the project root
```
LRNBIZ_SECRET_KEY=any-long-random-string-at-least-32-chars
GROQ_API_KEY=your_key_from_console.groq.com
LRNBIZ_ADMIN_PASSWORD=choose-a-password
```

- Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.
- Generate a secret key: `python -c "import secrets; print(secrets.token_hex(32))"`
- The app runs without a Groq key — the rules engine still works, AI feedback is disabled.

### 3. Start the server
```bash
python -m flask run
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 4. Kill stale processes (Windows — if the port is already in use)
```bash
netstat -ano | findstr :5000
taskkill /F /PID <pid>
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| Triple Truth Scoring | Rules engine + AI Optimist + Hybrid Mentor — three simultaneous scores per chapter |
| Sycophancy Detection | Measures how much the AI over-praised relative to the mentor's honest assessment |
| Business DNA Radar | Live 6-axis spider chart (Passion, Energy, Gold, Influence, Knowledge, Target) |
| Chapter Guard | Linear progression — chapter 3 cannot be accessed until chapters 1 and 2 are done |
| Audit Trail | After any chapter, students can go back, improve their answers, and re-run analysis |
| Score History | Each chapter tracks up to 10 analysis attempts — students can see their improvement |
| Pulse Reminder | The "Run Analysis" button pulses when the student changes inputs without re-running |
| Run Analysis Nudge | If the student tries to proceed without running analysis, a modal asks them to run it first |
| Teacher Dashboard | Create class codes, view student plans, class roster with all scores |
| ITS Evaluation Lab | `/eval` — run triple-truth scoring on 20 synthetic student personas for benchmarking |
| Research Log | `/admin/research` — anonymised log of all scoring sessions for academic analysis |
| Gamification | XP points, badges, streak tracking, improvement celebrations |
| Offline Mode | App works without a Groq API key — rules engine always runs, AI feedback is disabled |
| Bfcache Safe | All pages handle browser back-forward cache restore correctly (pageshow event) |

---

## Teacher and Admin Access

| URL | Access | Purpose |
|-----|--------|---------|
| `/teacher` | Open | Create class codes for students |
| `/teacher/dashboard` | Class code required | View student plans and scores |
| `/eval` | Open | Run ITS benchmarking on 20 synthetic personas |
| `/admin/research` | HTTP Basic Auth (`admin` / `LRNBIZ_ADMIN_PASSWORD`) | View anonymised research log |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, Flask 3.1, Flask-Session 0.8 |
| AI / LLM | Groq API — `llama-3.1-8b-instant` (AI Optimist), `llama-3.1-8b-instant` (Mentor) |
| Frontend | Vanilla JavaScript, Chart.js 4.4.3 for radar charts |
| Fonts | Nunito, Fredoka One (Google Fonts) |
| Session Storage | Filesystem (flask_sessions/) — avoids the 4KB cookie limit |
| Rule Files | JSON flat files, loaded once at startup via `functools.lru_cache` |
| Deployment | Render.com (free tier), Gunicorn WSGI server |

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `LRNBIZ_SECRET_KEY` | **Yes** | Flask session signing — must be a long random string |
| `GROQ_API_KEY` | No | Enables AI feedback. Without it, only the rules engine runs |
| `LRNBIZ_ADMIN_PASSWORD` | No | Protects the `/admin/research` endpoint |

---

## Known Limitations

| Limitation | Impact | Notes |
|------------|--------|-------|
| No persistent database | Student progress is lost if the server restarts or session expires | Flask-Session writes to disk but sessions are not portable |
| No user authentication | Each browser session is independent — no login or account | Students cannot resume on a different device |
| No CSRF protection | Forms are not protected against cross-site request forgery | Acceptable for classroom demo; not for open public deployment |
| Single-server only | Filesystem sessions cannot be shared across multiple instances | Would need Redis or a database to scale |
| LLM score variability | Even at temperature=0, different Groq model versions may produce slightly different scores | Score drift between sessions is possible |
| Rules are threshold-based | Pass/fail is binary at fixed thresholds (e.g. break-even > 6 months = error) | Real businesses sit on a spectrum — a threshold cannot capture every edge case |
| No input sanitisation beyond basic trimming | Long or malformed text inputs pass through to the LLM | Not a security issue for classroom use but relevant for public deployment |

---

## Design Decisions

### Why Flask instead of a full framework?
The application is a single-teacher, classroom-scale tool. A monolithic Flask file keeps the entire backend readable in one place, which is appropriate for an academic project where the code must be understandable by evaluators.

### Why filesystem sessions instead of cookies?
A student completing all five chapters accumulates ~8–12 KB of session data (LLM text, violation lists, score history). Browser cookies are capped at 4 KB. Filesystem sessions avoid silent data loss.

### Why two different LLM models?
The AI Optimist uses the smaller, faster `llama-3.1-8b-instant` to simulate a naive AI reviewer. The Hybrid Mentor uses the larger `llama-3.1-8b-instant` to produce more nuanced Socratic questions. The gap between them is the pedagogical point.

### Why temperature=0 for both models?
Consistency. A student re-running analysis on the same answers should get the same score. Temperature=0 minimises random variation, making the sycophancy measurement repeatable.

### Why JSON rule files instead of a database?
Rules are edited by a teacher/author, not by students. Flat JSON files are human-readable, version-controllable, and editable without a database migration. They are loaded once at server start via `lru_cache`.

---

## File-by-File Code Notes

See inline comments in `app.py` for detailed explanations of every major section. Key areas:

- **Lines 114–163**: Rule engine (`_eval_cond`, `compute_health_scores`) — deterministic, mirrors client-side JS
- **Lines 187–229**: Sentiment scoring and LLM score parsing helpers
- **Lines 243–317**: `ModuleAuditor.triple_truth()` — combines all three scores into one result dict
- **Lines 364–388**: Per-session LLM rate limiter (max 10 calls/minute)
- **Lines 535–682**: Two LLM call functions with their system prompts
- **Lines 776–788**: Chapter guard — enforces linear chapter progression
- **Lines 791 onwards**: Flask routes, one per chapter
