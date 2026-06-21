# DS-Agent: AI-Powered Data Science Agent

DS-Agent is a full-stack autonomous data science platform that transforms a raw CSV into dataset profiling, data cleaning, feature engineering, model training, chart generation, HTML reporting, and AI-generated narrative insights with minimal human intervention. It combines a Next.js frontend, FastAPI backend, classical ML pipelines, and local LLM inference through Ollama and Qwen2.5.

## Executive Overview 

DS-Agent delivers end-to-end data science automation in a single workflow:
- 📤 **Upload** your dataset via web interface
- 🤖 **AI Agent** analyzes and improves data quality automatically
- 🎯 **Train & Compare** multiple ML models
- 📊 **Generate** visualizations and interactive HTML reports
- 📝 **Produce** AI-powered narrative insights and recommendations
  
#### ` Scroll down for a demo video walkthough🔥 `

## What It Does

DS-Agent automates the entire data science workflow with intelligent decision-making:

- ✅ **Dataset Profiling** — Analyzes columns, missing values, uniqueness, and numeric distributions
- 🧹 **Data Cleaning** — Handles missing values, outliers, and normalization before modeling
- ⚡ **Feature Engineering** — Performs controlled feature creation with safeguards against feature explosion
- 🔄 **Autonomous Improvement Loop** — Iteratively evaluates and refines model performance with early stopping
- 🏆 **Model Selection** — Compares multiple ML models and automatically selects the best performer
- 📈 **Visualizations** — Generates feature importance plots, distribution charts, and correlation matrices
- 📄 **HTML Reports** — Creates downloadable, shareable executive reports with all findings
- 🎓 **Narrative Generation** — Produces natural-language executive summaries and actionable recommendations using Qwen2.5 via Ollama

## Architecture

### System Design
The platform is built on a modern, scalable architecture:

<img width="4680" height="1293" alt="image" src="https://github.com/user-attachments/assets/10333e6d-94dc-46c4-97d7-423b9d9163a0" />

```


┌─────────────────────────────────────────────────────────────────┐
│                        Next.js Frontend                          │
│                 (React, TypeScript, TailwindCSS)                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                       FastAPI Backend                            │
│                   (Python, Uvicorn, Async)                      │
├─────────────────────────────────────────────────────────────────┤
│  • Job Queue & Background Processing                            │
│  • Step-level Progress Tracking                                 │
│  • Upload Sanitization & Validation                             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
│  ML Pipeline   │   │  Reporting      │   │  LLM Inference  │
│  • Profiling   │   │  • Matplotlib   │   │  • Ollama       │
│  • Cleaning    │   │  • Seaborn      │   │  • Qwen2.5:3B   │
│  • FE & Train  │   │  • HTML Export  │   │  • Async Queue  │
└────────────────┘   └─────────────────┘   └─────────────────┘
```

### ML Pipeline Flow

<img width="8904" height="1608" alt="image" src="https://github.com/user-attachments/assets/5dcfd13d-449b-4ce0-8843-830339206b33" />

```
Raw CSV → Profile → Clean → Feature Engineer → Train Models → 
Evaluate → Compare → Select Best → Generate Charts → Export Report → 
Generate Narrative (Async)
```

### Key Design Principles
- **Non-Blocking UI**: Background jobs keep the frontend responsive during long analyses
- **Progress Visibility**: Step-level tracking shows real-time pipeline execution status
- **Smart Feature Engineering**: Cardinality-aware constraints prevent runaway feature growth
- **Decoupled Narratives**: LLM inference runs independently from core ML results
- **Security First**: Upload sanitization reduces path traversal and injection risks

## Why This Project Stands Out

DS-Agent is not just an ML demo—it's a production-style autonomous workflow designed with real engineering tradeoffs and best practices.

### Engineering Highlights
- 🏗️ **Background-Job Architecture** — Keeps UI responsive during long analyses via async task queues
- 👀 **Step-Level Progress Tracking** — Users get real-time visibility into exactly where the pipeline is
- 🛡️ **Cardinality Safeguards** — Feature engineering includes intelligent cardinality checks to prevent explosive feature growth
- ⏳ **Async Narrative Generation** — LLM tasks run in background; results are retrieved asynchronously without blocking
- 🔒 **Upload Sanitization** — Path traversal and injection attack mitigations on all file inputs
- 📦 **Modular Design** — Each pipeline stage is independently testable and upgradeable

### Performance Wins
Through iterative optimization, DS-Agent achieved:
- **Feature reduction**: From 10,394 to 29 features after cardinality safeguards
- **Training speedup**: From 363 seconds to 1.58 seconds per iteration
- **Better UX**: Blocking narrative generation converted to async background task
- **Improved reliability**: Progress tracker synchronized with actual backend execution

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Frontend** | Next.js, React, TypeScript, TailwindCSS |
| **Backend** | FastAPI, Python, Uvicorn |
| **ML & Data** | Pandas, NumPy, Scikit-Learn |
| **Visualization** | Matplotlib, Seaborn |
| **LLM** | Ollama, Qwen2.5:3B (local inference) |
| **DevOps** | Git, GitHub, Docker-ready design |

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker (recommended)
- Ollama (for local LLM inference)

### Installation

#### Clone the repository
```bash
git clone https://github.com/Def-Not-Faiz/ds-agent.git
cd ds-agent
```

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

#### Frontend Setup
```bash
cd frontend
npm install
```

#### LLM Setup (Ollama)
```bash
# Install Ollama from https://ollama.ai
# Pull Qwen2.5:3B model
ollama pull qwen2.5:3b
ollama serve  # Keep running in background
```

### Running the Application

#### Development Mode
```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev

# Terminal 3: Ollama (if not already running)
ollama serve
```

#### Production Mode (Docker)
```bash
docker-compose up
```

Access the application at `http://localhost:3000`

## Usage

1. **Upload Dataset** — Click "Upload CSV" and select your data file
2. **Configure** — Review auto-detected column types and settings
3. **Analyze** — Let the agent profile, clean, and engineer features
4. **Train** — Watch models train and improve iteratively
5. **Review** — Examine visualizations, feature importance, and metrics
6. **Export** — Download HTML report and narrative summary
7. **Share** — Distribute insights to stakeholders

## Demo & Results

### Example Workflow
A typical 10,000-row dataset with 100+ initial features:
- ✅ Profiling: ~2 seconds
- ✅ Cleaning: ~5 seconds
- ✅ Feature Engineering: ~3 seconds (reduced to 29 key features)
- ✅ Model Training: ~2 seconds (vs 363 seconds without optimization)
- ✅ Report Generation: ~1 second
- ✅ Narrative Generation: ~10 seconds (async, doesn't block UI)

**Total time to insights: ~23 seconds**


## Demo Video : ( Use the sound )

https://github.com/user-attachments/assets/5f7add83-edbe-4388-b712-bc9756f21a5b

(Sorry for the low Quality)

Screenshots can be accessed through the link at the bottom of this file.

## Project Structure

```
ds-agent/
├── frontend/                 # Next.js React application
│   ├── components/          # React components
│   ├── pages/              # Next.js pages and API routes
│   └── styles/             # TailwindCSS styles
├── backend/                 # FastAPI Python backend
│   ├── api/                # API endpoints
│   ├── ml/                 # ML pipeline modules
│   ├── models/             # Data models and schemas
│   ├── utils/              # Utility functions
│   └── main.py            # FastAPI app entry point
├── docker-compose.yml       # Docker orchestration
├── README.md               # This file

```

## Roadmap & Future Work

### Near Term
- [ ] GPU acceleration for faster local LLM inference
- [ ] Optional cloud LLM providers (OpenAI, Anthropic, Gemini)
- [ ] Advanced AutoML with hyperparameter search and ensembles
- [ ] Time-series forecasting support

### Medium Term
- [ ] Multi-agent decomposition (specialized agents for cleaning, FE, model selection)
- [ ] Direct database connectivity (MySQL, PostgreSQL, MongoDB)
- [ ] Deep learning support with PyTorch/TensorFlow
- [ ] Explainability improvements (SHAP, LIME integration)

### Long Term
- [ ] Federated learning for privacy-preserving analysis
- [ ] Real-time streaming data support
- [ ] Advanced ensemble techniques and stacking
- [ ] Natural language query interface for data exploration

## Executive Graphic 

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/8fa4deb1-9aaa-4c71-a02e-53264a184f2a" />

## Screenshots : https://drive.google.com/drive/folders/1M4LJdh4EinAc9ZcRGybX2qsPXYCeD10M?usp=sharing
