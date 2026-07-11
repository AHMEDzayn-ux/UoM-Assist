---
title: Nexus RAG Backend
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Nexus Telecom RAG Backend

FastAPI backend for the multi-tenant RAG chatbot platform (Nexus Telecom demo). Serves the
`/api/*` routes consumed by the frontend (deployed separately, e.g. on Vercel) — set that
frontend's `VITE_API_URL` to this Space's URL.

No persistent disk here: the SQLite DB, FAISS vector stores, and uploaded documents reset on
every Space rebuild/restart. See the repo root README for the full project and the
post-deploy bootstrap steps (create the `nexus` client, upload its knowledge base, seed demo
telecom data) that need re-running after a rebuild.
