# PennyStockLLM → Enterprise: AI Engineer Roadmap

**Strategy:** Don't "study" the skills from the job list — *build* them into this project.
Every phase upgrades PennyStockLLM toward a production system AND covers specific
skills from your Hirist saved-jobs analysis. You finish each phase with a demo-able
artifact for your resume/interviews.

**Cadence assumption:** ~10 focused hrs/week → full roadmap ≈ 4–5 months.
Phases are ordered so the app always works end-to-end; you can pause after any phase.

---

## ✅ Phase 0 — What you ALREADY have (don't undersell this)

Skills from the PDF you can already claim, with proof in this repo:

| PDF skill | Your proof |
|---|---|
| Python, LLMs, Generative AI | entire project |
| RAG | Chroma + BM25s **hybrid/ensemble retriever** over a 700-page book |
| LangChain + **LangGraph** | multi-agent supervisor pipeline in `Stocks_PG_Code.ipynb` |
| **Agentic AI** | 3-agent system (MarketData → Analyst → Prometheus judge) + tool routing |
| Prompt Engineering | constrained agent prompts, judge rubric prompt |
| Vector Databases | ChromaDB (`penny_stock_vector_db/`) |
| Fine-tuning / LLaMA | **QLoRA fine-tune of Llama-3-8B** (MLX), balanced synthetic dataset, iterative debugging (fixed ticker-collapse failure), fuse→GGUF→Ollama deploy |
| AI Integration | custom `penny-llama` model serving the notebook via Ollama |

**Day 0 tasks (this week):**
- [ ] Step 10: flip `llm_data_agent` to `ChatOllama(model="penny-llama", ...)` and run the full pipeline once.
- [ ] Create a GitHub repo, push everything (add `.gitignore` for models/DBs; **rotate the HF token hardcoded in cell 2 first**).
- [ ] Write a README with an architecture diagram (agents, tools, RAG, fine-tuned model). This alone puts you ahead of most applicants.

---

## Phase 1 — Engineering hardening (2–3 wks)
**PDF skills: Python (senior-level), Data Structures & Algorithms, Git**

The jump from "notebook" to "enterprise" starts here: notebooks demo, packages ship.

Build:
1. Restructure into a package:
   ```
   pennystock/
     agents/ (data_agent.py, analyst.py, judge.py, supervisor.py)
     retrieval/ (ensemble.py, bm25.py, chroma.py)
     tools/ (market_data.py, researcher.py)
     config.py  (pydantic-settings; NO hardcoded secrets)
   tests/
   ```
2. `pytest` suite: unit-test the retriever merge logic, ticker formatting (`SUZLON`→`SUZLON.NS`), agent routing rules. Target ~70% coverage on core logic.
3. Type hints + `ruff` + `mypy`; pre-commit hooks.
4. Keep one thin notebook that imports from the package (for demos).

Also start (ongoing through all phases): 2–3 LeetCode/DSA problems per week in Python — interviews test this separately.

**Proof artifact:** a clean, tested, importable Python package on GitHub.

---

## Phase 2 — RAG excellence + Evaluation (2–3 wks)
**PDF skills: RAG (advanced), NLP, Vector DBs (Qdrant), Prompt Engineering, Data Science**

Enterprise RAG = measured RAG. You already judge answers with Prometheus; formalize it.

Build:
1. **Golden eval set:** ~50 Q/A pairs from the penny-stock book (you write them once; they become your regression suite).
2. **RAG eval harness** with RAGAS (or custom): faithfulness, answer relevance, context precision/recall. Baseline your current ensemble retriever.
3. **Improve retrieval, measure the delta:** add a cross-encoder **reranker** (e.g. bge-reranker) on top of the hybrid retriever; tune chunk size/overlap; compare eval scores before/after. That before/after table is interview gold.
4. **Migrate Chroma → Qdrant** running in Docker (named in JDs: Pinecone/Weaviate/Qdrant). Keep the retriever interface identical — that's clean architecture in practice.
5. **Agent eval:** turn `finetune/test_adapter.py` into a scored suite — 100 held-out queries → % correct tool name / ticker / exchange. Quote this number ("97% tool-call accuracy after QLoRA vs X% base").

**Proof artifact:** an `evals/` folder + a results table in the README.

---

## Phase 3 — API & Data layer (3 wks)
**PDF skills: REST APIs, FastAPI, SQL, PostgreSQL, Data Modeling, Data Analytics, Data Pipelines**

Turn the pipeline into a product with a real backend.

Build:
1. **FastAPI service** wrapping the LangGraph app:
   - `POST /analyze` (async, **SSE streaming** of agent progress), `GET /health`, `GET /reports/{id}`
   - API-key auth middleware + rate limiting; pydantic request/response models.
2. **PostgreSQL** (SQL is in half the JDs — this is where you actually learn it):
   - Schema you design (= data modeling): `users`, `analysis_requests`, `agent_runs`,
     `tool_calls`, `market_snapshots` (time-series of fetched quotes), `eval_results`.
   - SQLAlchemy + Alembic migrations.
3. **Analytics in SQL** (= data analytics): most-queried tickers, agent latency percentiles,
   tool-failure rates, eval-score trends — as SQL views + a small dashboard notebook.
4. **Scheduled data pipeline:** nightly job (cron/APScheduler) that pulls EOD data for
   tracked tickers into `market_snapshots` — a real, small ETL pipeline.
5. Optional stretch: **pgvector** for embeddings inside Postgres (SQL + vectors in one).

**Proof artifact:** `curl` demo of a streaming analysis + an ERD diagram in the README.

---

## Phase 4 — MLOps & Observability (3–4 wks)
**PDF skills: MLOps, Docker, CI/CD, experiment tracking, monitoring**

This phase is the single biggest differentiator for "AI Engineer" vs "knows LangChain".

Build:
1. **Dockerize everything:** `docker-compose.yml` with 4 services — FastAPI app, Ollama
   (serving penny-llama), Postgres, Qdrant. One command brings up the whole stack.
2. **Experiment tracking:** re-run your QLoRA fine-tune logging to **MLflow** (params: rank,
   lr, iters; metrics: train/val loss; artifact: adapter). Register model versions
   (v1 = the "BANK" failure, v2 = the fix — great interview story, now with charts).
3. **LLM observability:** add **Langfuse** (self-hosted, free) tracing to every agent run —
   spans per agent, per tool call, token counts, latencies.
4. **CI/CD (GitHub Actions):**
   - on PR: ruff + mypy + pytest
   - nightly/on-demand: the Phase-2 eval suite; **fail the pipeline if tool-call accuracy
     or RAG faithfulness drops** below threshold → that's a model-quality gate, the heart
     of LLMOps.
5. **Monitoring basics:** Prometheus metrics endpoint in FastAPI + a Grafana dashboard
   (request rate, latency, error rate, tokens/request).

**Proof artifact:** `docker compose up` → full stack + traces + dashboards; a green Actions badge.

---

## Phase 5 — Cloud (AWS) + Kubernetes (3–4 wks)
**PDF skills: AWS, SageMaker, Kubernetes, cloud deployment**

Budget-conscious path (your model also runs CPU-only at Q4_K_M):

1. **S3 + ECR:** store model artifacts (GGUF, adapters, datasets) in S3 with versioned
   paths; push Docker images to ECR.
2. **Deploy the stack to one EC2 instance** (or ECS) with compose — public demo URL,
   HTTPS via a small nginx/caddy container. (Use RDS free tier for Postgres if you want
   managed SQL on the resume.)
3. **SageMaker touchpoint** (JDs name it): deploy the *fused HF model* (`penny-llama-fused`)
   as a SageMaker endpoint once, benchmark it, write it up, then tear it down (cost!).
   Alternative if budget-blocked: SageMaker local mode / a detailed written design.
4. **Kubernetes — learn locally, deploy minimally:** minikube/k3s on your Mac →
   manifests (Deployment, Service, ConfigMap, Secret, HPA) for the API + Qdrant;
   optionally EKS for a weekend, then destroy. Helm-chart the app.

**Proof artifact:** architecture diagram (S3/ECR/EC2/RDS), live demo link, k8s manifests in repo.

---

## Phase 6 — Deep-learning credibility + Portfolio polish (2–3 wks)
**PDF skills: PyTorch, Deep Learning, NLP, Statistical/Predictive Modeling, Time Series**

Fill the one real gap: JDs want PyTorch fundamentals, and MLX abstracts them away.

1. **PyTorch project inside PennyStockLLM:** fine-tune **FinBERT** (or DistilBERT) on
   financial sentiment (Financial PhraseBank dataset) in raw PyTorch — your own training
   loop: DataLoader, optimizer, scheduler, eval. Ship it as a new `sentiment` tool the
   Analyst agent can call. Now your "PyTorch" resume line has a story.
2. Optional stretch: a small **time-series** baseline (Prophet/ARIMA vs a simple LSTM) on
   `market_snapshots` data → covers predictive modeling honestly (label it a baseline,
   not a trading system).
3. **Portfolio packaging:**
   - README: architecture diagram, eval tables, MLflow screenshots, 2-min demo video/GIF.
   - 2–3 blog posts (the QLoRA "BANK bug" debugging story is genuinely excellent content).
   - Resume bullets with numbers (see below).
   - LinkedIn/Hirist profile refresh mapping to these keywords.

**Resume bullet templates (fill your real numbers):**
- "Fine-tuned Llama-3-8B (QLoRA on Apple Silicon/MLX) for tool-calling; raised tool-call accuracy from X%→Y% on a held-out suite; deployed via GGUF/Ollama."
- "Built multi-agent LangGraph pipeline (data-fetch, RAG analyst over 700-pg corpus with hybrid BM25+dense retrieval + reranking, LLM-as-judge evaluator); RAG faithfulness improved X→Y after reranker."
- "Productionized as FastAPI + Postgres + Qdrant + Ollama stack (Docker Compose, K8s manifests); CI/CD quality gates on model evals (GitHub Actions, MLflow, Langfuse); deployed on AWS."

---

## Skill-coverage map (PDF → phase)

| PDF tier | Skill | Covered in |
|---|---|---|
| Must-have | Python | 0,1 | 
| Must-have | AI/ML fundamentals | 2,6 |
| Must-have | LLMs | 0 (done) |
| Must-have | PyTorch | **6** |
| Very high | RAG | 0 + **2** |
| Very high | GenAI / Prompt Eng | 0 (done) |
| Very high | LangChain | 0 (done) |
| Very high | MLOps | **4** |
| Very high | Deep Learning / NLP | **6** (+2) |
| Very high | SQL | **3** |
| Frequent | Agentic AI / LangGraph | 0 (done) |
| Frequent | Vector DBs | 2 (Qdrant) |
| Frequent | AWS / SageMaker | **5** |
| Frequent | Data Modeling / Analytics | 3 |
| Support | Docker / Kubernetes | 4 / 5 |
| Support | REST APIs / FastAPI | 3 |
| Support | Data Pipelines | 3 |
| Skip | Java/Spring, Product Analyst, Node.js | outliers — different career tracks; ignore |

## Ground rules
- **One phase at a time; the app must always run.** Enterprise = working software.
- Every phase ends with something you can *show* (demo, table, dashboard, link).
- Numbers beat adjectives in interviews: log every before/after metric.
- Start applying from **end of Phase 3** — you don't need the whole list; interviews
  themselves teach you what to deepen (many hires happen at "Phase 3.5").
