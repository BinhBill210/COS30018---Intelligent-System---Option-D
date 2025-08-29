## TASK ALLOCATION ##
 - T1, T2  - anh Binh
 - T3, T4 - Dinh Nam
 - T5 - Duc Tri
 - T6 - Minh Hoang
 









1. [Dataset & Conventions](#dataset--conventions)
    
2. [Tool Specs (6 Total)](#tool-specs-6-total)
    
    - T1 HybridRetrieve
        
    - T2 BusinessPulse
        
    - T3 AspectABSA ⭐
        
    - T4 DriftSegment
        
    - T5 ImpactRanker ⭐
        
    - T6 ActionPlanner ⭐
        
3. [Task → Tool Choreography](#task--tool-choreography)
    
4. [LLM Handoff (Bundle Contract)](#llm-handoff-bundle-contract)
    
5. [Orchestration Graph](#orchestration-graph)
    
6. [Prompt Templates](#prompt-templates)
    
7. [Edge Cases, Guardrails, and Caching](#edge-cases-guardrails-and-caching)
    

---

## Dataset & Conventions

**Canonical schema (adapter to `review_cleaned.csv`):**

- `review_id` _(str)_
    
- `user_id` _(str)_
    
- `business_id` _(str)_
    
- `stars` _(int|float, 1..5)_
    
- `useful`, `funny`, `cool` _(int)_
    
- `text` _(str)_
    
- `date` _(ISO `YYYY-MM-DD`)_
    

**Derived / shared rules used by all tools**

- **Helpfulness weight**: `helpfulness = log1p(useful + funny + cool)`
    
- **Rating bands**: detractors = 1–2★, passives = 3★, promoters = 4–5★
    
- **Evidence objects**: every tool returns `evidence: [{review_id, quote, stars, date, helpfulness}]`
    
- **Meta block**: every tool returns `meta: {generated_at, elapsed_ms, params}`
    
- **Retrieval**: Hybrid semantic (FAISS MiniLM) + lexical (BM25).
    
- **Aspect taxonomy**: optional `aspects.yaml` stabilizes names/aliases.
    

---

## Tool Specs (6 Total)

### T1) HybridRetrieve — _search + curated evidence_

**Purpose**  
High-recall lexical+semantic retrieval with diverse, helpfulness-weighted quotes to ground any claim.

**Input**

`{   "business_id": "str",   "query": "str",   "top_k": 50,   "filters": {     "date_from": "YYYY-MM-DD",     "date_to": "YYYY-MM-DD",     "stars": [1, 5]   } }`

**Output**

`{   "hits": [     {"review_id":"r1","score":0.82,"text":"...","stars":2,"date":"2021-06-03","helpfulness":5.2}   ],   "evidence": [     {"review_id":"r1","quote":"Waited 40 minutes...","stars":2,"date":"2021-06-03","helpfulness":5.2}   ],   "meta": {"generated_at":"ISO","elapsed_ms":12,"params":{"top_k":50}} }`

**Notes**

- Deduplicate with MMR; prioritize higher helpfulness.
    

---

### T2) BusinessPulse — _overview & sanity checks_

**Purpose**  
Quick health snapshot for a business (counts, star mix, top terms, sentiment, mismatch rate).

**Input**

`{"business_id":"str"}`

**Output**

`{   "summary": {     "n_reviews": 1941,     "stars_dist": {"1":0.12,"2":0.08,"3":0.10,"4":0.23,"5":0.47},     "date_range": ["2016-02-01","2021-12-30"]   },   "text_sentiment": {"pos":0.60,"neu":0.22,"neg":0.18},   "top_positive_terms": ["friendly staff","tasty","clean"],   "top_negative_terms": ["wait time","overpriced","refund"],   "consistency_check": {"star_vs_text_mismatch_pct": 0.07},   "evidence": [     {"review_id":"r123","quote":"Staff were super friendly","stars":5,"date":"2020-09-11","helpfulness":2.7}   ],   "meta": {"generated_at":"ISO","elapsed_ms":38,"params":{}} }`

**Notes**

- Term mining via noun/adj phrase extraction; mismatch via classifier vs stars.
    

---

### T3) AspectABSA ⭐ — _aspect mining + aspect-level sentiment_

**Purpose**  
Create a stable list of **aspects** and compute **per-aspect sentiment** and **support**.

**Input**

`{   "business_id":"str",   "seed_aspects":["service_speed","staff","price","quality","cleanliness","ambience","booking","refund","delivery","parking"],   "top_k_aspects": 30 }`

**Output**

`{   "aspects": [     {       "name":"service_speed",       "aliases":["wait time","long line","slow service"],       "sentiment":{"neg":0.62,"neu":0.23,"pos":0.15},       "support":{"mentions":384,"weighted_mentions":512.7},       "top_neg_phrases":["waited 40 minutes","line barely moved"]     },     {       "name":"price",       "aliases":["value","expensive"],       "sentiment":{"neg":0.41,"neu":0.29,"pos":0.30},       "support":{"mentions":290}     }   ],   "evidence": [{"review_id":"r77","quote":"Waited 40 minutes...","stars":2,"date":"2021-06-03","helpfulness":9.0}],   "meta": {"generated_at":"ISO","elapsed_ms":210,"params":{"top_k_aspects":30}} }`

**Notes**

- Use embeddings + keyphrase extraction; ABSA via classifier or rule-guided LLM; weight by helpfulness.
    

---

### T4) DriftSegment — _trends & segments for an aspect_

**Purpose**  
Show **when** problems happen and **who** feels them (trend + segment pockets).

**Input**

`{   "business_id":"str",   "aspect":"service_speed",   "window":"M",   "segments":["rating_band","helpfulness_quartile"] }`

**Output**

`{   "trend": [     {"period":"2021-05","neg_frac":0.41},     {"period":"2021-06","neg_frac":0.68,"anomaly":{"z":3.1,"note":"spike"}}   ],   "segments": [     {"segment":{"rating_band":"1-2","helpfulness_q":"Q4"},"neg_frac":0.79,"support":186},     {"segment":{"rating_band":"3","helpfulness_q":"Q4"},"neg_frac":0.36,"support":90}   ],   "evidence": [{"review_id":"r910","quote":"Hold music for half an hour","stars":1,"date":"2021-06-10","helpfulness":6.1}],   "meta": {"generated_at":"ISO","elapsed_ms":95,"params":{"window":"M"}} }`

**Notes**

- Time series built from dated reviews; anomalies via z-score; segments configurable.
    

---

### T5) ImpactRanker ⭐ — _prioritize areas to improve_

**Purpose**  
Rank aspects by **severity × frequency × recency × segment lift − effort**.

**Input**

`{   "business_id":"str",   "candidates":["service_speed","refund","price","cleanliness"],   "weights":{"severity":0.35,"frequency":0.30,"recency":0.20,"segmentity":0.10,"effort":-0.05},   "half_life_days":120,   "effort_overrides":{"refund":0.6,"cleanliness":0.3} }`

**Output**

`{   "ranked": [     {       "target":"service_speed",       "score":0.81,       "components":{"severity":0.85,"frequency":0.75,"recency":0.68,"segmentity":0.70,"effort":0.40},       "why":"High negative share, frequent, recent spike, concentrated in 1–2★ high-helpfulness reviews."     },     {       "target":"refund",       "score":0.64,       "components":{"severity":0.61,"frequency":0.42,"recency":0.50,"segmentity":0.38,"effort":0.60}     }   ],   "evidence": [{"review_id":"r221","quote":"No refund after defective item","stars":1,"date":"2021-05-14","helpfulness":6.0}],   "meta": {"generated_at":"ISO","elapsed_ms":52,"params":{"half_life_days":120}} }`

**Notes**

- `severity` = strong-negative aspect share; `frequency` = aspect mention share; `recency` = time-decayed neg; `segmentity` = lift vs overall; `effort` = heuristic 0..1 (lower is easier).
    

---

### T6) ActionPlanner ⭐ — _evidence-linked actions + KPIs_

**Purpose**  
Produce **concrete recommendations**, with **effort/impact**, **KPIs**, and **citations**.

**Input**

`{   "business_id":"str",   "target":"service_speed",   "constraints":{"budget_tier":"low|medium|high","policy_changes_ok":true} }`

**Output**

`{   "recommendations":[     {       "title":"Introduce queue triage & callback",       "rationale":"Per ImpactRanker, service_speed is top pain; callbacks reduce perceived wait.",       "estimated_effort":"medium",       "expected_impact":"high",       "kpis":["neg_share(service_speed)","1-2★ ratio","avg_star"],       "implementation_notes":["enable callback after 120s hold","post expected wait at entrance"],       "evidence_refs":["r77","r221"]     },     {       "title":"Display real-time wait estimates",       "estimated_effort":"low",       "expected_impact":"medium",       "kpis":["neg_share(service_speed)"]     }   ],   "evidence":[{"review_id":"r77","quote":"Waited 40 minutes...","stars":2,"date":"2021-06-03","helpfulness":9.0}],   "meta":{"generated_at":"ISO","elapsed_ms":44,"params":{"budget_tier":"low"}} }`

**Notes**

- Guarantees KPI names for UI tracking; pulls citations from prior tool evidence.
    

---

## Task → Tool Choreography

Below are the **user tasks** and the **exact tool sequence** (with what each contributes). All outputs are LLM-ready and can be concatenated into a final JSON bundle.

### 1) “Quick health check for Business X”

**Flow:** `T2 BusinessPulse` → _(optional)_ `T3 AspectABSA (top 5)`  
**LLM gets:** overview + 1–2 supporting quotes; optional top aspects with sentiment.

---

### 2) “Top areas to improve for Business X”

**Flow:** `T3 AspectABSA` → `T4 DriftSegment` (for top-neg aspects) → `T5 ImpactRanker` → _(optional quotes)_ `T1 HybridRetrieve`  
**LLM gets:** aspects+sentiment/support → trend & segments → ranked targets with component breakdown → extra quotes.

---

### 3) “Why is <aspect> #1?”

**Flow:** `T5 ImpactRanker` (component scores) → `T4 DriftSegment` (anomaly & segments) → `T1 HybridRetrieve`  
**LLM gets:** weighted justification + time/segment context + curated evidence.

---

### 4) “What actions should we take (low budget / policy OK)?”

**Flow:** `T5 ImpactRanker` (tune effort penalty) → `T6 ActionPlanner`  
**LLM gets:** prioritized targets → concrete, KPI-linked actions with evidence refs.

---

### 5) “What changed recently? Any spikes?”

**Flow:** `T3 AspectABSA` (pick problematic aspects) → `T4 DriftSegment`  
**LLM gets:** time series, anomalies, and quotes for a “Recent Changes” section.

---

### 6) “Which segments are hurting most?”

**Flow:** `T3 AspectABSA` (choose 3–5 aspects) → `T4 DriftSegment` (`segments=["rating_band","helpfulness_quartile"]`)  
**LLM gets:** segment rows `{segment, neg_frac, support}` + quotes; LLM suggests targeted fixes.

---

### 7) “Show evidence for <claim>”

**Flow:** `T1 HybridRetrieve`  
**LLM gets:** diverse, high-helpfulness quotes with IDs/dates.

---

### 8) “Compare two businesses (or two periods)”

**Flow:** For each side: `T2 BusinessPulse` + `T3 AspectABSA` → Joint: `T5 ImpactRanker` → _(optional)_ `T1 HybridRetrieve` per side  
**LLM gets:** side-by-side bundles + unified priority ranking + quotes.

---

## LLM Handoff (Bundle Contract)

For multi-tool answers, the orchestrator hands a **single JSON** to the LLM:

`{   "business_id": "b_42",   "question": "Top areas to improve & what to do",   "pulse": { /* T2 output */ },   "aspects": { /* T3 output */ },   "drift": { /* T4 outputs keyed by aspect */ },   "priority": { /* T5 output */ },   "actions": { /* T6 output for top targets */ } }`

**Guarantees for the LLM**

- Every section contains `evidence` with `review_id/quote/stars/date/helpfulness`.
    
- KPI names in `T6` are canonical for dashboards.
    
- Component scores in `T5` explain “why” a target ranks high.
    

---

## Orchestration Graph

        `(Dataset + FAISS/BM25)                    │            ┌───────┴────────┐            │  T2 BusinessPulse  (overview)            └───────┬────────┘                    │            ┌───────┴────────┐            │  T3 AspectABSA   (aspects + ABSA)            └───────┬────────┘                    │ top negative aspects            ┌───────┴────────┐            │  T4 DriftSegment (trends + segments)            └───────┬────────┘                    │ trend/segment signals            ┌───────┴────────┐            │  T5 ImpactRanker (priority + why)            └───────┬────────┘                    │ top targets            ┌───────┴────────┐            │  T6 ActionPlanner (actions + KPIs + evidence)            └───────┬────────┘                    │            ┌───────┴────────┐            │  LLM Composer   (final report; can pull extra quotes via T1 HybridRetrieve)            └─────────────────┘`

---

## Prompt Templates

**Final Report (areas to improve + actions)**

> You are given a JSON bundle with fields: `pulse`, `aspects`, `drift`, `priority`, `actions`.
> 
> 1. List top 3 areas to improve with 1–2 sentences each.
>     
> 2. For each, include **one short quote** from `evidence`.
>     
> 3. Propose **1–2 actions** from `actions.recommendations` per area, including **KPIs**.
>     
> 4. Only make claims supported by quotes or metrics in the JSON.
>     
> 5. Keep the answer under 300 words.
>     

**Justification (“Why is X #1?”)**

> Use `priority.ranked` component scores and `drift.trend/segments`.  
> Explain in 3 bullets: severity/frequency, recency/anomaly, segment impact.  
> Include 2 quotes from `evidence`.

---

## Edge Cases, Guardrails, and Caching

- **Low review count:** Tools return `support.mentions`; if `< 50`, LLM must note “limited evidence”.
    
- **Missing dates:** `T4 DriftSegment` returns empty `trend`; other tools still work.
    
- **Noisy/mismatched reviews:** `T2.consistency_check.star_vs_text_mismatch_pct` prompts down-weighting in ABSA.
    
- **Evidence requirement:** If any section lacks `evidence`, the LLM must either call `T1 HybridRetrieve` or state “insufficient direct evidence”.
    
- **Caching:**
    
    - Pre-embed all `text` once (FAISS).
        
    - Cache `T3 AspectABSA` per `business_id`.
        
    - Reuse evidence objects across tools for consistent citations.
        
- **Config files:**
    
    - `aspects.yaml` (names/aliases/effort defaults).
        
    - `impact_weights.json` (weights & half-life)



 