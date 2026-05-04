# LrnBiz — Testing Guide

## Links

| Resource | URL |
|----------|-----|
| Live Tool | https://lrnbiz.onrender.com |
| GitHub Repository | https://github.com/YashKapoor1102/lrnbiz |

> **Note:** The live tool is hosted on Render.com free tier. If the server is sleeping, the first page load may take up to 30 seconds. Wait for it to fully load before testing.

---

## Quick Start — Complete a Full Student Session (5–10 minutes)

This walks through all five chapters and the certificate.

1. Go to **https://lrnbiz.onrender.com**
2. Enter any name (e.g. "Test Student") and press **Start**
3. Complete **Chapter 1 — Context**: select a grade, budget, location, business type, and hours per week. Click **Analyse**.
4. Complete **Chapter 2 — Business Idea**: describe a business idea (e.g. "Selling handmade candles online"). Click **Analyse** to see the three scores (Rules Score, AI Optimist, Hybrid Mentor) and the Sycophancy Gap.
5. Complete **Chapter 3 — Target Customer**: fill in the customer persona fields. Click **Analyse**.
6. Complete **Chapter 4 — Money Math**: enter a unit price, unit cost, startup cost, and expected monthly sales. Try setting the unit price **lower than** the unit cost to see rule MR001 fire. Click **Analyse**.
7. Complete **Chapter 5 — Customer Discovery**: log at least one customer interview. Click **Analyse**.
8. Click **View Final Results** to see the summary dashboard and Business DNA radar chart.
9. Click **Get Certificate** to view the completion certificate.

---

## Testing the Sycophancy Gap

To observe the largest measurable gap, use the following inputs in Chapter 4 (Money Math):

- Unit price: **£3**
- Unit cost: **£5**
- Startup cost: **£200**
- Monthly sales: **50**

Expected result:  The AI Optimist praises the plan; the Mentor flags that the price is below the cost.

To observe a near-zero gap, complete all chapters with coherent, complete inputs (no rule violations). Both models will converge, and the gap will be close to zero.

---

## Automated Evaluation Lab — 25 Synthetic Personas

The `/eval` route runs the full Triple Truth pipeline on 25 pre-built synthetic student personas and displays all scores and gaps in a table.

1. Go to **https://lrnbiz.onrender.com/eval**
2. Click **Run All Analyses** to run all 25 personas, or click **Run** next to any individual persona.
3. Results show Rules Score, AI Optimist score, Hybrid Mentor score, and Sycophancy Gap for each persona.
4. A CSV export button downloads all results for offline analysis.




---

## Running Locally (Optional)

If you prefer to run the tool locally:

```bash
git clone https://github.com/YashKapoor1102/lrnbiz.git
cd lrnbiz
pip install -r requirements.txt
```

Create a `.env` file:
```
LRNBIZ_SECRET_KEY=any-long-random-string-at-least-32-chars
GROQ_API_KEY=your_key_from_console.groq.com
LRNBIZ_ADMIN_PASSWORD=choose-a-password
```

Start the server:
```bash
python -m flask run
```

Open **http://127.0.0.1:5000** in your browser.

> A free Groq API key (no credit card required) is available at https://console.groq.com. Without a key, the rules engine still runs but AI feedback is disabled.

---

## Key Features Summary

| Feature | Where to test |
|---------|--------------|
| Three simultaneous scores per chapter | Any chapter — click Analyse |
| Sycophancy Gap display | Chapter 4 with price below cost |
| Rule violations with Socratic questions | Any chapter with weak inputs |
| Chapter progression guard | Try navigating to `/money` before completing Chapter 1 — redirects to `/context` |
| Business DNA radar chart | Final results page |
| Automated 25-persona evaluation | `/eval` |
| Teacher dashboard and CSV export | `/teacher` and `/teacher/dashboard` |
