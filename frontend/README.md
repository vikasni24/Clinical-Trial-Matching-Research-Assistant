# Clinical Trial Matching & Research Assistant — Frontend

A React + TypeScript + Vite frontend for the existing FastAPI backend. This
frontend is a pure consumer of the backend's REST API — it contains no
clinical logic, no eligibility rules, and no LLM calls of its own. Every
three-state semantic the backend enforces (PASS/FAIL/UNKNOWN eligibility;
answered/insufficient_evidence/unsupported research-assistant answers) is
preserved and displayed as-is, never reinterpreted.

## Setup

```bash
npm install
cp .env.example .env   # adjust VITE_API_BASE_URL if the backend runs elsewhere
npm run dev
```

The backend must be running separately (see the project root README/
`app/main.py`) — by default at `http://localhost:8000`.

## Scripts

- `npm run dev` — start the Vite dev server (default: http://localhost:5173)
- `npm run build` — type-check (`tsc -b`) and produce a production build in `dist/`
- `npm run preview` — preview the production build locally
- `npm run lint` — run oxlint

## Structure

```
src/
├── api/          one module per backend resource area; all HTTP calls go
│                 through api/client.ts's request() — never scattered fetch()s
├── components/   layout/, common/ (badges, loading/error/empty states,
│                 pagination), evidence/, matching/
├── pages/        one component per route
├── types/        TypeScript types mirroring the backend's Pydantic models
│                 exactly (app/models/*.py) — verified against source, not guessed
├── hooks/        useScopedQuery — the patient-isolation-safe data-fetching hook
└── utils/        formatting + FHIR field extraction helpers
```

## Patient isolation

Every patient-scoped page fetches data via `useScopedQuery`, keyed on the
current `patientId` (and any other relevant parameters, e.g. page number).
Whenever that key changes, the hook immediately clears its data and enters
a loading state *before* issuing the new request, and discards any
still-in-flight response from a now-stale key. This makes it structurally
impossible for one patient's data to render — even momentarily — while a
different patient's page is loading.

## A note on `npm run` scripts on this machine

This project's path contains an `&` character (`.../Clinical Trial Matching
& Research Assistant/frontend`). npm's auto-generated `node_modules/.bin/*.cmd`
shims on Windows don't reliably quote that character when re-invoking Node,
which breaks `npx tsc`, `npx vite`, and — before this fix — `npm run build`/
`npm run dev` themselves. The scripts in `package.json` call
`node ./node_modules/<pkg>/bin/<entry>.js` directly instead of through those
shims, which sidesteps the bug entirely. No behavior changes as a result —
`npm run dev` / `npm run build` / `npm run preview` work exactly as normal.

## Known limitations

- The backend has no endpoint for patient search or global (cross-patient)
  recent activity, so the Dashboard does not display a "recent audit
  activity" feed — that would require either an unscoped query (violating
  patient isolation) or an endpoint that doesn't exist. A "quick patient
  lookup by ID" is provided instead.
- `GET /api/patients` has no search/filter parameter server-side, so the
  patient list is browse/paginate only.
