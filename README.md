# Clinical Trial Matching & Research Assistant

A full-stack application that deterministically matches patients against clinical
trial eligibility criteria and answers natural-language clinical questions with an
LLM whose every claim is grounded in, and validated against, real evidence pulled
from a patient's FHIR record — never invented, never hallucinated.

Built on synthetic [Synthea](https://github.com/synthetichealth/synthea)-generated
patient data. **For research/demo purposes only — not a real clinical system and
not connected to any real patient data or real clinical trials.**

---

## Key features

- **Deterministic FHIR ingestion** — Synthea-generated FHIR JSON bundles are parsed
  and stored in MongoDB, then normalized into flat, application-friendly patient
  profiles.
- **Deterministic eligibility matching** — every patient is evaluated against trial
  criteria with a strict three-state result per criterion: `PASS` / `FAIL` /
  `UNKNOWN`. Missing data is never silently forced into a PASS or a FAIL — an
  overall result is `UNKNOWN` whenever any criterion is `UNKNOWN`, never
  auto-resolved.
- **Evidence traceability** — every clinical fact used anywhere in the system (in a
  match result or a grounded answer) is traceable back to a specific FHIR resource
  (`resource_type` + `resource_id`), never presented as a bare, unsourced claim.
- **Hybrid evidence retrieval** — a structured, code/registry-based retriever and a
  dependency-free semantic (bag-of-words) retriever are combined and deterministically
  ranked. No embeddings, no vector database, no external ML service.
- **Grounded Research Assistant** — ask natural-language questions about a patient
  (`"What medications is the patient taking?"`, `"What is the patient's blood
  pressure?"`) and get an answer that is validated post-generation: any citation the
  LLM makes that doesn't correspond to real, supplied evidence causes the whole
  answer to be rejected in favor of an explicit `insufficient_evidence` result —
  never a partially-trusted, partially-fabricated answer.
- **Pre-generation safety gate** — unsupported or no-evidence questions are
  rejected *before* the LLM is ever called, and missing evidence is never converted
  into a negative clinical claim (e.g. "no evidence found" is not "patient does not
  have this condition").
- **Full audit trail** — every question asked is recorded (patient, query,
  retrieval status, answer status, evidence references, timestamp) for later review,
  without ever storing raw LLM output, prompts, or secrets.
- **Strict patient isolation** — enforced structurally at the data-model level (not
  just by convention) at every layer, from the MongoDB query up through the LLM
  prompt and the final response.
- **Modern dashboard UI** — a React/TypeScript frontend for browsing patients,
  inspecting evidence, running trial matches, using the Research Assistant, and
  reviewing audit history.

---

## Architecture

```
Synthea FHIR JSON
       |
FHIR ingestion  --------------------------->  MongoDB (fhir_resources)
       |                                            |
Patient normalization                       Evidence extraction
       |                                            |
Patient profiles                      Structured + Semantic retrievers
                                                     |
                                            Hybrid retriever (ranked)
                                                     |
                                              GroundedContext
                                                     |
                                        Pre-generation safety gate
                                                     |
                                           Grounded prompt builder
                                                     |
                                          LLMProvider (Anthropic)
                                                     |
                                            Answer validation
                                                     |
                                             GroundedAnswer
                                                     |
                                              Audit record  ----> MongoDB (audit_records)
```

- **Backend**: FastAPI + PyMongo + Pydantic, Python 3.11+
- **Database**: MongoDB
- **Frontend**: React 19 + TypeScript + Vite + React Router
- **LLM provider**: Anthropic (via a small, swappable `LLMProvider` interface — no
  vendor SDK, plain HTTP)

---

## Prerequisites

- Python 3.11+ (developed and tested with 3.14) and a virtual environment tool
- Node.js 18+ and npm
- A running MongoDB instance (local `mongod` or a connection string to any
  MongoDB-compatible server)
- (Optional, only needed for the Research Assistant's actual LLM calls) an
  [Anthropic API key](https://console.anthropic.com/)

---

## Setup

### 1. Backend

```bash
# from the project root
python -m venv venv
source venv/Scripts/activate      # Windows (Git Bash) / venv\Scripts\activate on cmd
# source venv/bin/activate        # macOS/Linux

pip install -r requirements.txt

cp .env.example .env
```

Edit `.env` and set at least `MONGODB_URI` / `MONGODB_DATABASE` if your MongoDB
isn't running on the default `mongodb://localhost:27017`. Leave `LLM_API_KEY` blank
to run everything except real Research Assistant answers (see
[Configuration](#configuration) below).

### 2. Load the demo data

Place Synthea-generated patient FHIR bundles under `data/synthea/fhir/` (a handful
of sample patients are enough), then run the ingestion pipeline:

```bash
python -m app.services.fhir_ingestion          # loads FHIR bundles into MongoDB
python -m app.services.patient_normalization   # builds normalized patient profiles
python -m app.services.trial_ingestion         # loads the trial fixture (data/trials/dev_trials.json)
```

Each is idempotent — safe to re-run.

### 3. Start the backend

```bash
python -m uvicorn app.main:app --reload
```

Verify it's up:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Interactive API docs are auto-served at `http://localhost:8000/docs`.

### 4. Frontend

```bash
cd frontend
npm install
cp .env.example .env    # defaults to VITE_API_BASE_URL=http://localhost:8000
npm run dev
```

Open `http://localhost:5173`.

> **Note (Windows only):** if your project folder path contains an `&` character,
> npm's auto-generated script shims can break (`npm run build`/`npm run dev`
> failing with `'X' is not recognized as an internal or external command`). This
> repo's `frontend/package.json` scripts already work around it by invoking
> `node ./node_modules/<pkg>/bin/...` directly — no extra steps needed, but worth
> knowing if you ever see that error after editing the scripts yourself.

---

## Configuration

All backend configuration lives in `.env` (see `.env.example` for the full list
with defaults):

| Variable | Purpose |
|---|---|
| `MONGODB_URI` / `MONGODB_DATABASE` | MongoDB connection |
| `SYNTHEA_FHIR_DIR` | Directory the FHIR ingestion script reads from |
| `TRIALS_DATA_PATH` | Path to the trial fixture JSON |
| `LLM_API_KEY` | Your Anthropic API key — **required only for the Research Assistant's actual generation step**. Everything else (patients, evidence, trial matching, audit) works without it. |
| `LLM_MODEL` / `LLM_TIMEOUT_SECONDS` | LLM request tuning, sensible defaults provided |
| `CORS_ALLOWED_ORIGINS` | Browser origins allowed to call the API (defaults to Vite's dev ports) |

`.env` is gitignored — never commit real credentials. If `LLM_API_KEY` is unset,
Research Assistant questions that would otherwise reach the LLM fail fast with a
clear, secret-free `"LLM_API_KEY is not configured"` error instead of hanging or
fabricating an answer; questions that don't need the LLM at all (unsupported
queries, or genuinely no evidence found) still work normally.

The frontend's `frontend/.env` holds only `VITE_API_BASE_URL` — it never sees or
stores any backend credential.

---

## Running tests

```bash
pytest -q
```

The full suite runs entirely offline against an in-memory MongoDB mock
(`mongomock`) and fake LLM providers — no real database or LLM connection is ever
required to run it.

---

## Project structure

```
app/
├── api/routes/       FastAPI routers (patients, trials, fhir)
├── models/            Pydantic models — request/response contracts and domain types
├── repositories/       MongoDB query layer (one module per collection)
├── services/           Business logic: ingestion, normalization, eligibility
│                       matching, evidence retrieval, grounding, LLM provider,
│                       answer validation, audit
├── db/                MongoDB connection + index setup
└── config.py           Settings (environment-driven)

tests/                 Full pytest suite (mongomock + fake providers, no live deps)

frontend/
├── src/api/            Typed API client — every backend call goes through here
├── src/components/     Layout, shared UI primitives, charts
├── src/pages/           Dashboard, Patients, Patient Detail, Trial Matching,
│                        Research Assistant, Audit History
├── src/hooks/           useScopedQuery — the patient-isolation-safe data hook
└── src/types/           TypeScript types mirroring the backend's Pydantic models

data/
├── synthea/fhir/        Synthea-generated patient FHIR bundles (sample data)
└── trials/               Synthetic clinical trial fixture data
```

---

## API overview

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `GET /api/patients` | Paginated patient list |
| `GET /api/patients/{id}` | One patient |
| `GET /api/patients/{id}/profile` | Normalized demographics/clinical summary |
| `GET /api/patients/{id}/resources` | Paginated raw FHIR resources for a patient |
| `GET /api/patients/{id}/evidence` | Paginated, structured (non-raw-FHIR) evidence |
| `GET /api/patients/{id}/matches` | Trial eligibility results across candidate trials |
| `GET /api/patients/{id}/matches/{trial_id}` | Eligibility result for one trial |
| `POST /api/patients/{id}/ask` | Ask the grounded Research Assistant a question |
| `GET /api/patients/{id}/audit` | Paginated audit history for that patient |
| `GET /api/trials` | Paginated trial catalog |
| `GET /api/fhir/{resource_type}/{resource_id}` | One raw FHIR resource |

Full interactive documentation (request/response schemas, try-it-out) is available
at `/docs` while the backend is running.

---

## Important disclaimers

- All patient and clinical trial data is **synthetic**, generated by Synthea or
  hand-written as development fixtures. Nothing in this system reflects a real
  person or a real clinical trial.
- This is a research/demo project, not a certified medical device or clinical
  decision-support tool. Do not use it to make real clinical or research
  decisions.
- Match scores and eligibility results are transparent counts of criteria
  satisfied against the data on record — never a probability of trial success or
  a substitute for clinical judgment.
