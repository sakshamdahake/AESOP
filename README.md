# Aesop: Autonomous Evidence Synthesis & Observation Platform

**Aesop** is a distributed, multi-agent system designed to automate the creation of "Living Systematic Reviews" for medical professionals. It utilizes a **Self-Improving Agentic RAG** architecture to scrape, grade, and synthesize medical literature in real-time.

---

## 🏗️ System Architecture

The system follows a **Monorepo** structure utilizing a Distributed Micro-Agent architecture.

| Component | Tech Stack | Responsibility |
| :--- | :--- | :--- |
| **Orchestrator** | **FastAPI + LangGraph** | Manages agent state, routing, and API endpoints. |
| **Dependency Mgr** | **uv (Astral)** | Ultra-fast Python package management. |
| **Vector DB** | **PostgreSQL + pgvector** | Stores embeddings of medical abstracts. |
| **Graph DB** | **Neo4j** | Maps citation networks to detect retracted/conflicting science. |
| **Queue/Cache** | **Redis** | Manages distributed agent tasks and caching. |
| **Infrastructure** | **Docker Compose** | Orchestrates the entire stack locally. |

---

## 🚀 Getting Started

### Prerequisites
* **Docker Desktop** (Running and updated)
* **Git**

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/aesop.git](https://github.com/yourusername/aesop.git)
cd aesop

```

### 2. Environment Configuration

Create a `.env` file in the root directory. You can copy the example below:

**File:** `.env`

```ini
# --- AI Provider ---
OPENAI_API_KEY=sk-your-openai-key-here

# --- Database Defaults (Matches docker-compose.yml) ---
DATABASE_URL=postgresql://aesop:aesop_pass@postgres:5432/aesop_db
REDIS_URL=redis://redis:6379/0
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=aesop_graph_pass

```

### 3. Launch the Stack

We use `docker-compose` to spin up the Backend, Database, Redis, and Neo4j simultaneously.

```bash
docker-compose up --build

```

* **First Run:** This may take a few minutes to download images and compile dependencies.
* **Success:** You will see `Uvicorn running on http://0.0.0.0:8000`.

---

## 🔍 Verification

Once the stack is running, verify the services:

1. **API Docs (Swagger):** [http://127.0.0.1:8000/docs](https://www.google.com/search?q=http://127.0.0.1:8000/docs)
* *Test:* Click "Try it out" on the `/health` endpoint.


2. **Neo4j Browser:** [http://localhost:7474](https://www.google.com/search?q=http://localhost:7474)
* *Login:* `neo4j` / `aesop_graph_pass`



---

## 📂 Project Structure

```text
aesop-monorepo/
├── docker-compose.yml       # Orchestrates all services
├── .env                     # Secrets (Not committed to Git)
├── backend/                 # Python/FastAPI Source Code
│   ├── Dockerfile           # Universal Dockerfile (Windows/Mac/Linux compatible)
│   ├── pyproject.toml       # Dependencies (Managed by uv)
│   ├── uv.lock              # Lockfile
│   └── app/
│       ├── main.py          # API Entrypoint
│       ├── core/            # Config & Logging
│       └── agents/          # LangGraph Agent Logic
└── frontend/                # (Placeholder for React App)

```

---

## 🛠️ Development Workflow

### Adding New Python Packages

We use `uv` for package management. Do not use `pip` manually.

1. **Enter the container** (easiest way to ensure compatibility):
```bash
docker exec -it aesop_backend bash

```


2. **Add the package**:
```bash
uv add numpy pandas

```


3. **Exit and Rebuild**:
The `pyproject.toml` and `uv.lock` will update on your host machine (via volume mount).
```bash
exit
docker-compose up --build

```



### Hot Reloading

The `backend` service is configured with a volume mount.

* **Edit** `backend/app/main.py` locally in VS Code.
* **Save** the file.
* **Watch** the terminal logs. The server will restart automatically.

---

## 🚑 Troubleshooting

### 1. "Executor not found" or "no such file or directory: /app/.venv..."

**Cause:** The virtual environment paths are misaligned between the build stage and runtime stage.
**Fix:** Ensure your `Dockerfile` copies the venv to `/opt/venv` and sets `ENV UV_PROJECT_ENVIRONMENT="/opt/venv"`. (The current Dockerfile handles this).
**Command:**

```bash
docker-compose down --volumes
docker-compose up --build --force-recreate

```

### 2. "Empty Response" or "Connection Refused" on localhost

**Cause:** Docker networking issues or IPv6 conflicts.
**Fix:**

* Use `http://127.0.0.1:8000` instead of `localhost`.
* Ensure the app is listening on `0.0.0.0`, NOT `127.0.0.1` inside the container (Check `main.py`).

### 3. "Port is already allocated"

**Cause:** You have another Postgres or Redis running locally.
**Fix:** Stop local services or kill the specific port.

```bash
# MacOS/Linux
lsof -i :5432
kill -9 <PID>

```

### 4. Windows Volume Mount Issues

**Cause:** Docker on Windows file sharing permissions.
**Fix:**

* Ensure Docker Desktop has permission to access the Drive where the project is located.
* **WSL2:** It is highly recommended to clone this repo inside your WSL2 filesystem (e.g., `\\wsl$\Ubuntu\home\user\aesop`), not on the Windows `C:` drive.

---

## 📜 License

Private / Proprietary.

```

```