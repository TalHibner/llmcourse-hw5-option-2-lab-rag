# Technical Design Document
## RAG Context Window Research System

**Version:** 1.0
**Date:** December 10, 2025
**Project:** LLM Course HW5 - RAG Research Lab

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Research System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Data Gen   │  │  Experiment  │  │    Analysis &    │  │
│  │    Module    │→ │   Execution  │→ │  Visualization   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│         ↓                  ↓                    ↓            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Synthetic   │  │   Results    │  │   Graphs &       │  │
│  │   Dataset    │  │   Storage    │  │   Statistics     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                   Core Infrastructure                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐  │
│  │  Ollama  │  │LangChain │  │  Chroma   │  │  Config   │  │
│  │   API    │  │   RAG    │  │ Vector DB │  │  Manager  │  │
│  └──────────┘  └──────────┘  └───────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Component Interaction Diagram

```
┌──────────────┐
│   User/CLI   │
└──────┬───────┘
       │
       ├─► [1] Generate Data ──────► data_generator.py
       │                                    │
       │                                    ▼
       │                            Synthetic Facts JSON
       │                                    │
       ├─► [2] Run Experiment 1 ──────────►│
       │         (Lost in Middle)           │
       │                │                   │
       │                └──► Ollama API ◄───┘
       │                         │
       │                         ▼
       │                   Results CSV
       │                         │
       ├─► [3] Run Experiment 2 ─┼─► Add Noise
       │      (Noise Impact)      │         │
       │                          │         ▼
       │                          │   Ollama + Embeddings
       │                          │         │
       │                          ▼         ▼
       │                      Results CSV
       │                          │
       ├─► [4] Run Experiment 3 ─┼─► Build Chroma DB
       │        (RAG Solution)    │         │
       │                          │         ▼
       │                          │   RAG Pipeline
       │                          │    (LangChain)
       │                          │         │
       │                          ▼         ▼
       │                      Results CSV
       │                          │
       └─► [5] Analyze Results ───┴─► analysis_notebook.ipynb
                                        │
                                        ├─► Graphs (PNG/PDF)
                                        └─► Statistics (LaTeX)
```

---

## 2. Technology Stack

### 2.1 Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.10+ | Primary implementation language |
| **Package Manager** | UV | Latest | Fast, reliable dependency management |
| **LLM Inference** | Ollama | Latest | Local LLM serving (llama2/mistral/phi) |
| **RAG Framework** | LangChain | 0.1.0+ | RAG pipeline components |
| **Vector Store** | ChromaDB | 0.4.0+ | Embedding storage and retrieval |
| **Embeddings** | Ollama Embeddings | - | nomic-embed-text or similar |
| **Data Analysis** | Pandas | 2.0+ | Data manipulation and analysis |
| **Numerical** | NumPy | 1.24+ | Statistical computations |
| **Visualization** | Matplotlib | 3.7+ | Graph generation |
| **Visualization** | Seaborn | 0.12+ | Statistical visualizations |
| **Notebooks** | Jupyter | Latest | Analysis and experimentation |
| **Testing** | pytest | 7.0+ | Unit and integration testing |
| **Coverage** | pytest-cov | 4.0+ | Code coverage reporting |

### 2.2 Model Selection Strategy

**Primary LLM Options (via Ollama):**
1. **llama2:7b** - Balanced performance and speed
2. **mistral:7b** - Strong reasoning capabilities
3. **phi-2** - Lightweight, fast inference

**Embedding Model:**
- **nomic-embed-text** - High-quality text embeddings, 768 dimensions

**Selection Criteria:**
- Model must run locally on standard hardware
- Context window ≥ 4096 tokens
- Inference time < 10s per query on CPU

---

## 3. Data Generation Strategy

### 3.1 Synthetic Fact Document Schema

**Fact Document Structure:**
```json
{
  "id": "fact_001",
  "category": "geography",
  "fact": "Paris is the capital of France",
  "question": "What is the capital of France?",
  "answer": "Paris",
  "metadata": {
    "complexity": "simple",
    "entity_type": "city",
    "created_at": "2025-12-10"
  }
}
```

**Fact Categories (for diversity):**
- Geography: Capitals, landmarks, regions
- Science: Chemical symbols, physical constants, discoveries
- History: Dates, events, historical figures
- Mathematics: Formulas, theorems, definitions
- Literature: Authors, works, characters

### 3.2 Fact Generation Algorithm

```python
def generate_fact_corpus(n_facts: int = 25) -> List[Fact]:
    """
    Generate n_facts diverse, non-overlapping synthetic facts

    Requirements:
    - Each fact must be verifiable (single correct answer)
    - Questions must be unambiguous
    - Facts should not semantically overlap
    - Distribute across categories
    """
    templates = {
        'geography': "{city} is the capital of {country}",
        'science': "The chemical symbol for {element} is {symbol}",
        'history': "{event} occurred in the year {year}",
        'math': "The value of {constant} is {value}",
        'literature': "{book} was written by {author}"
    }
    # Implementation details in src/data_generator.py
```

### 3.3 Noise Document Generation

**Noise Document Characteristics:**
- Similar length to fact documents (50-100 tokens)
- Grammatically correct but irrelevant
- No factual overlap with core facts
- Varied topics to ensure semantic diversity

**Noise Types:**
1. **Random sentences** from Wikipedia-style text
2. **Synthetic descriptions** of fictional entities
3. **General knowledge** statements unrelated to test questions

**Noise Ratio Control:**
```
Total Documents = Core Facts + Noise Documents
Noise Ratio = Noise Documents / Total Documents

Example for 70% noise with 10 facts:
  Noise Documents = 10 × 0.70 / 0.30 ≈ 23
  Total = 10 + 23 = 33 documents
```

---

## 4. Experiment Methodologies

### 4.1 Experiment 1: Lost in the Middle

**Implementation Design:**

```python
class ContextWindowExperiment:
    """
    Tests LLM accuracy based on fact position in context

    Input Data:
    - facts: List[Fact] - 25-30 fact documents
    - questions: List[Question] - corresponding questions

    Setup Data:
    - model_name: str - Ollama model identifier
    - temperature: float = 0.0 - for determinism
    - n_runs: int = 5 - repetitions per condition

    Output Data:
    - results: DataFrame with columns:
      [run_id, fact_id, position, position_category,
       question, predicted_answer, correct_answer,
       is_correct, context_length]
    """

    def build_context(self, facts: List[Fact], target_idx: int) -> str:
        """Concatenate all facts with target at specific position"""
        pass

    def classify_position(self, idx: int, total: int) -> str:
        """Classify position as 'beginning', 'middle', or 'end'"""
        # Beginning: 0-33%ile
        # Middle: 34-66%ile
        # End: 67-100%ile
        pass

    def run_experiment(self) -> pd.DataFrame:
        """Execute full experiment with all position permutations"""
        pass
```

**Position Categorization:**
- **Beginning:** Documents 1-8 (top 33%)
- **Middle:** Documents 9-17 (middle 33%)
- **End:** Documents 18-25 (bottom 33%)

**Prompt Template:**
```
Context: {all_facts_concatenated}

Question: {question}

Answer:
```

**Expected Execution:**
- 25 facts × 3 positions × 5 runs = 375 queries
- Estimated time: ~30 minutes on CPU

---

### 4.2 Experiment 2: Noise and Irrelevance

**Implementation Design:**

```python
class NoiseImpactExperiment:
    """
    Measures accuracy degradation with increasing noise

    Input Data:
    - core_facts: List[Fact] - 10 core facts
    - noise_documents: List[str] - pool of 100+ noise docs
    - noise_levels: List[float] - [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

    Setup Data:
    - embedding_model: str - Ollama embedding model
    - use_embeddings: bool - whether to embed before LLM call

    Output Data:
    - results: DataFrame with columns:
      [run_id, noise_ratio, fact_id, question,
       predicted_answer, is_correct, is_hallucination,
       context_length, response_confidence]
    """

    def sample_noise_docs(self, n_noise: int) -> List[str]:
        """Randomly sample n_noise documents from pool"""
        pass

    def detect_hallucination(self, answer: str, fact: Fact) -> bool:
        """Check if answer is confident but wrong"""
        # Hallucination = confident response != correct answer
        pass

    def run_experiment(self) -> pd.DataFrame:
        """Execute across all noise levels"""
        pass
```

**Hallucination Detection:**
```python
def is_hallucination(response: str, correct_answer: str) -> bool:
    """
    Hallucination criteria:
    1. Response is not "I don't know" or uncertain
    2. Response != correct answer
    3. Response appears confident (no hedging language)
    """
    uncertainty_markers = ["I don't know", "uncertain",
                          "not sure", "cannot determine"]
    is_uncertain = any(marker in response.lower()
                      for marker in uncertainty_markers)
    is_wrong = correct_answer.lower() not in response.lower()

    return (not is_uncertain) and is_wrong
```

---

### 4.3 Experiment 3: RAG Solution

**RAG Pipeline Architecture:**

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Embed Query     │  ← Ollama Embeddings
└──────┬──────────┘
       │
       ▼
┌──────────────────────┐
│ Vector Similarity    │  ← Chroma DB
│ Search (top-k)       │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ [Optional] Reranking │  ← Cross-encoder or LLM
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Retrieve Documents   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Build Context        │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Generate Answer      │  ← Ollama LLM
└──────┬───────────────┘
       │
       ▼
┌─────────────┐
│   Answer    │
└─────────────┘
```

**Implementation Design:**

```python
class RAGExperiment:
    """
    RAG-based retrieval and generation

    Input Data:
    - documents: List[Document] - all facts + noise
    - questions: List[Question]
    - noise_levels: List[float]

    Setup Data:
    - vector_db: ChromaDB instance
    - embedding_model: str
    - llm_model: str
    - top_k: int = 3 - number of documents to retrieve
    - reranking_enabled: bool = False

    Output Data:
    - results: DataFrame with columns:
      [run_id, noise_ratio, question,
       retrieved_doc_ids, retrieval_precision,
       predicted_answer, is_correct,
       retrieval_time_ms, generation_time_ms]
    """

    def build_vector_db(self, documents: List[Document]):
        """Initialize Chroma DB with embedded documents"""
        pass

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        """Vector similarity search"""
        pass

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Optional: rerank retrieved documents"""
        # BONUS implementation
        pass

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """LLM generation with retrieved context"""
        pass

    def calculate_retrieval_precision(
        self, retrieved: List[str], relevant: List[str]
    ) -> float:
        """Precision = |retrieved ∩ relevant| / |retrieved|"""
        pass
```

**ChromaDB Configuration:**
```python
import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./data/chromadb"
))

collection = client.create_collection(
    name="fact_documents",
    metadata={"hnsw:space": "cosine"}  # Cosine similarity
)
```

**LangChain Integration:**
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Vector store
vectorstore = Chroma(
    collection_name="facts",
    embedding_function=embeddings,
    persist_directory="./data/chromadb"
)

# LLM
llm = Ollama(model="llama2", temperature=0.0)

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)
```

**Top-k Parameter Sweep:**
- Test k ∈ {1, 3, 5, 10}
- Analyze precision/recall trade-off
- Report optimal k value

---

## 5. Statistical Analysis Approach

### 5.1 Descriptive Statistics

For each experiment, compute:

**Central Tendency:**
- Mean accuracy: $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$
- Median accuracy
- Mode (most common outcome)

**Dispersion:**
- Standard deviation: $\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2}$
- Variance: $\sigma^2$
- Interquartile range (IQR)

**Confidence Intervals (95%):**
$$CI_{95\%} = \bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{\sigma}{\sqrt{n}}$$

Where $t_{\alpha/2, n-1}$ is the critical t-value for 95% confidence

### 5.2 Inferential Statistics

**Hypothesis Testing:**

For Experiment 1 (Position Effect):
- **Test:** One-way ANOVA
- **Null Hypothesis (H₀):** $\mu_{beginning} = \mu_{middle} = \mu_{end}$
- **Alternative (H₁):** At least one mean differs
- **Significance level:** $\alpha = 0.05$

Post-hoc testing (if ANOVA significant):
- Tukey's HSD for pairwise comparisons

**Effect Size:**
- Cohen's d: $d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$
- Interpretation: small (0.2), medium (0.5), large (0.8)

**Correlation Analysis:**

For Experiment 2 (Noise Impact):
- Pearson correlation: $r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$
- Test for significance of correlation
- Report $R^2$ (coefficient of determination)

### 5.3 Statistical Software

```python
import scipy.stats as stats
from scipy.stats import ttest_ind, f_oneway, pearsonr
import statsmodels.api as sm

# Example: One-way ANOVA for position effect
f_stat, p_value = stats.f_oneway(
    beginning_accuracies,
    middle_accuracies,
    end_accuracies
)

# Effect size (Cohen's d)
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std
```

---

## 6. Visualization Plan

### 6.1 Graph Specifications

**Common Settings:**
- DPI: 300 (publication quality)
- Figure size: (10, 6) inches
- Font: DejaVu Sans, size 12
- Color palette: ColorBrewer qualitative/sequential
- Grid: Light gray, alpha=0.3
- File formats: PNG (for notebooks), PDF (vector graphics)

**Style Configuration:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# Color palette
colors = sns.color_palette("Set2", 8)

# Figure template
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
```

### 6.2 Experiment 1 Visualizations

**Graph 1: Position vs Accuracy (Bar Chart)**
- X-axis: Position category (Beginning, Middle, End)
- Y-axis: Accuracy (%)
- Error bars: 95% confidence intervals
- Annotations: Sample size, mean values
- Title: "LLM Accuracy by Fact Position in Context"

**Graph 2: Position Distribution (Box Plot)**
- Shows distribution of accuracies per position
- Outlier detection visible
- Median line emphasized

**LaTeX Table: Statistical Summary**
```latex
\begin{table}[h]
\centering
\caption{Accuracy by Position (n=125 per category)}
\begin{tabular}{lccc}
\toprule
Position & Mean ± SD & 95\% CI & Effect Size (vs Middle) \\
\midrule
Beginning & 0.92 ± 0.08 & [0.90, 0.94] & d = 1.2 (large) \\
Middle    & 0.58 ± 0.12 & [0.55, 0.61] & - \\
End       & 0.88 ± 0.09 & [0.86, 0.90] & d = 1.1 (large) \\
\bottomrule
\end{tabular}
\end{table}
```

### 6.3 Experiment 2 Visualizations

**Graph 3: Noise Ratio vs Accuracy (Line Plot)**
- X-axis: Noise percentage (0-90%)
- Y-axis: Accuracy (%)
- Line: Mean accuracy with shaded 95% CI
- Second Y-axis: Hallucination rate overlay
- Title: "Performance Degradation with Increasing Noise"

**Graph 4: Accuracy Degradation Curve Fit**
- Scatter plot of raw data points
- Fitted regression line (linear or exponential)
- Equation and R² displayed
- Formula: $Accuracy = \beta_0 + \beta_1 \times NoiseRatio$ or $Accuracy = e^{\beta_0 + \beta_1 \times NoiseRatio}$

### 6.4 Experiment 3 Visualizations

**Graph 5: RAG vs Classic Comparison (Grouped Bar Chart)**
- X-axis: Noise percentage
- Y-axis: Accuracy (%)
- Groups: RAG vs Classic
- Error bars: 95% CI
- Title: "RAG vs Classic Context: Accuracy Comparison"

**Graph 6: Retrieval Precision Heatmap**
- X-axis: Noise percentage
- Y-axis: Top-k value
- Cell color: Retrieval precision
- Annotations: Precision values
- Title: "RAG Retrieval Precision by Noise Level and Top-K"

**Graph 7: Multi-Metric Comparison (Radar Chart)**
- Metrics: Accuracy, Precision, Speed, Noise Tolerance
- Lines: RAG vs Classic
- Normalized to [0, 1] scale

---

## 7. Module Design

### 7.1 Project Structure

```
llmcourse-hw5-option-2-lab-rag/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Configuration management
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── fact_generator.py     # Synthetic fact creation
│   │   └── noise_generator.py    # Noise document creation
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── base_experiment.py    # Abstract base class
│   │   ├── experiment1_context.py
│   │   ├── experiment2_noise.py
│   │   └── experiment3_rag.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py         # Embedding utilities
│   │   ├── vector_store.py       # Chroma DB wrapper
│   │   ├── retriever.py          # Retrieval logic
│   │   └── reranker.py           # (Bonus) Reranking
│   ├── llm/
│   │   ├── __init__.py
│   │   └── ollama_client.py      # Ollama API wrapper
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── statistics.py         # Statistical functions
│   │   └── visualization.py      # Plotting functions
│   └── utils/
│       ├── __init__.py
│       ├── logging.py            # Logging configuration
│       └── helpers.py            # Utility functions
├── experiments/
│   ├── experiment1_notebook.ipynb
│   ├── experiment2_notebook.ipynb
│   ├── experiment3_notebook.ipynb
│   └── comprehensive_analysis.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_data_generation.py
│   ├── test_experiments.py
│   ├── test_rag.py
│   └── test_llm.py
├── data/
│   ├── facts/
│   │   └── synthetic_facts.json
│   ├── noise/
│   │   └── noise_documents.json
│   └── chromadb/                 # Vector DB persistence
├── results/
│   ├── experiment1/
│   │   ├── raw_data.csv
│   │   └── graphs/
│   ├── experiment2/
│   │   ├── raw_data.csv
│   │   └── graphs/
│   └── experiment3/
│       ├── raw_data.csv
│       └── graphs/
├── docs/
│   └── architecture_diagrams/
├── config/
│   ├── config.yaml
│   └── example.env
├── PRD.md
├── DESIGN.md
├── TASKS.md
├── README.md
├── pyproject.toml
├── .gitignore
└── .env (not tracked)
```

### 7.2 Core Module Interfaces

**Data Generation Module:**
```python
# src/data_generation/fact_generator.py

from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Fact:
    """Building block for fact representation"""
    id: str
    category: str
    fact_text: str
    question: str
    answer: str
    metadata: Dict

    def validate(self) -> bool:
        """Input validation"""
        pass

class FactGenerator:
    """
    Generates synthetic fact documents

    Input Data:
    - n_facts: int (20-30)
    - categories: List[str]

    Setup Data:
    - random_seed: int
    - diversity_threshold: float

    Output Data:
    - facts: List[Fact]
    """

    def generate_facts(self, n: int) -> List[Fact]:
        pass

    def ensure_diversity(self, facts: List[Fact]) -> bool:
        """Validate no semantic overlap"""
        pass
```

**LLM Client Module:**
```python
# src/llm/ollama_client.py

from typing import Optional, List
import requests

class OllamaClient:
    """
    Wrapper for Ollama API

    Setup Data:
    - base_url: str = "http://localhost:11434"
    - model_name: str
    - temperature: float = 0.0
    - timeout: int = 30

    Input Data:
    - prompt: str
    - context: Optional[str]

    Output Data:
    - response: str
    - metadata: Dict (tokens, time, etc.)
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.temperature = kwargs.get('temperature', 0.0)
        # Input validation
        self._validate_config()

    def generate(self, prompt: str, context: str = "") -> str:
        """Generate text response"""
        # Input validation
        self._validate_input(prompt)

        # API call
        response = self._call_api(prompt, context)

        # Output validation
        return response

    def embed(self, text: str) -> List[float]:
        """Generate embeddings"""
        pass
```

**RAG Pipeline Module:**
```python
# src/rag/retriever.py

from typing import List
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

class RAGRetriever:
    """
    RAG retrieval component

    Setup Data:
    - vector_store: Chroma
    - top_k: int = 3
    - reranking: bool = False

    Input Data:
    - query: str

    Output Data:
    - retrieved_docs: List[Document]
    - relevance_scores: List[float]
    """

    def retrieve(self, query: str) -> List[Document]:
        """Vector similarity search"""
        # Input validation
        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Retrieval
        docs = self.vector_store.similarity_search(
            query, k=self.top_k
        )

        # Optional reranking
        if self.reranking:
            docs = self.reranker.rerank(query, docs)

        return docs
```

**Statistics Module:**
```python
# src/analysis/statistics.py

import numpy as np
from scipy import stats
from typing import Tuple

class StatisticalAnalyzer:
    """
    Statistical analysis utilities

    Implements:
    - Descriptive statistics
    - Hypothesis testing
    - Effect size calculations
    - Confidence intervals
    """

    @staticmethod
    def confidence_interval(
        data: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval"""
        mean = np.mean(data)
        se = stats.sem(data)
        margin = se * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return (mean - margin, mean + margin)

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        pass

    @staticmethod
    def anova_test(groups: List[np.ndarray]) -> Tuple[float, float]:
        """One-way ANOVA"""
        return stats.f_oneway(*groups)
```

---

## 8. Configuration Management

### 8.1 Configuration Schema

**config/config.yaml:**
```yaml
# LLM Configuration
llm:
  provider: ollama
  model_name: llama2
  base_url: http://localhost:11434
  temperature: 0.0
  max_tokens: 512
  timeout: 30

# Embedding Configuration
embeddings:
  model_name: nomic-embed-text
  dimension: 768
  batch_size: 32

# Vector Store Configuration
vector_store:
  type: chroma
  persist_directory: ./data/chromadb
  collection_name: fact_documents
  distance_metric: cosine

# Experiment Configuration
experiments:
  random_seed: 42
  n_runs: 5

  experiment1:
    n_facts: 25
    position_categories: [beginning, middle, end]

  experiment2:
    n_core_facts: 10
    noise_levels: [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    n_noise_pool: 100

  experiment3:
    top_k_values: [1, 3, 5, 10]
    reranking_enabled: false

# Analysis Configuration
analysis:
  confidence_level: 0.95
  significance_alpha: 0.05

# Visualization Configuration
visualization:
  dpi: 300
  figure_size: [10, 6]
  style: whitegrid
  palette: Set2
  save_formats: [png, pdf]

# Paths
paths:
  data_dir: ./data
  results_dir: ./results
  figures_dir: ./results/figures
```

### 8.2 Environment Variables (.env)

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/experiment.log

# API Keys (if needed for future extensions)
# OPENAI_API_KEY=sk-...  # Not used, for reference only
```

---

## 9. Error Handling and Validation

### 9.1 Input Validation Strategy

Every building block implements comprehensive input validation:

```python
def validate_fact(fact: Fact) -> None:
    """Validate fact structure"""
    if not fact.id:
        raise ValueError("Fact ID cannot be empty")
    if not fact.question.strip():
        raise ValueError("Question cannot be empty")
    if not fact.answer.strip():
        raise ValueError("Answer cannot be empty")
    if len(fact.fact_text) < 10:
        raise ValueError("Fact text too short")

def validate_noise_ratio(ratio: float) -> None:
    """Validate noise ratio parameter"""
    if not 0 <= ratio <= 1:
        raise ValueError(f"Noise ratio must be in [0, 1], got {ratio}")
```

### 9.2 Exception Hierarchy

```python
# src/utils/exceptions.py

class RAGExperimentError(Exception):
    """Base exception for all experiment errors"""
    pass

class DataGenerationError(RAGExperimentError):
    """Errors in data generation"""
    pass

class OllamaAPIError(RAGExperimentError):
    """Errors from Ollama API"""
    pass

class VectorStoreError(RAGExperimentError):
    """Errors in vector store operations"""
    pass

class StatisticalAnalysisError(RAGExperimentError):
    """Errors in statistical computations"""
    pass
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

**Coverage Target:** ≥70% overall, ≥90% for core modules

**Test Structure:**
```python
# tests/test_data_generation.py

import pytest
from src.data_generation import FactGenerator, Fact

class TestFactGenerator:
    """Test suite for FactGenerator"""

    def test_generate_facts_count(self):
        """Test correct number of facts generated"""
        generator = FactGenerator(random_seed=42)
        facts = generator.generate_facts(n=25)
        assert len(facts) == 25

    def test_fact_diversity(self):
        """Test facts are semantically diverse"""
        generator = FactGenerator(random_seed=42)
        facts = generator.generate_facts(n=25)
        assert generator.ensure_diversity(facts)

    def test_fact_validation(self):
        """Test fact validation catches errors"""
        invalid_fact = Fact(id="", category="geo", ...)
        with pytest.raises(ValueError):
            invalid_fact.validate()
```

### 10.2 Integration Tests

```python
# tests/test_experiments.py

def test_experiment1_end_to_end():
    """Test Experiment 1 runs successfully"""
    exp = ContextWindowExperiment(model_name="llama2")
    results = exp.run_experiment()

    # Verify output structure
    assert 'position_category' in results.columns
    assert 'is_correct' in results.columns

    # Verify expected number of rows
    assert len(results) > 0

    # Verify results are valid
    assert results['is_correct'].dtype == bool
```

---

## 11. Performance Considerations

### 11.1 Execution Time Estimates

| Component | Estimated Time | Notes |
|-----------|---------------|-------|
| Data Generation | < 1 minute | Synthetic generation is fast |
| Experiment 1 | 20-40 minutes | ~375 LLM calls at 3-5s each |
| Experiment 2 | 15-30 minutes | ~300 LLM calls |
| Experiment 3 (RAG) | 30-60 minutes | Includes vector DB build + queries |
| Analysis | < 5 minutes | Statistical computation is fast |
| Visualization | < 2 minutes | Graph generation |
| **Total** | **~2 hours** | On CPU, single-threaded |

### 11.2 Optimization Strategies

1. **Batch Processing:** Group similar queries
2. **Caching:** Cache LLM responses with same prompt
3. **Parallelization:** Use multiprocessing for independent runs
4. **Model Selection:** Use smaller models (phi-2) for faster inference if needed

---

## 12. Reproducibility Measures

### 12.1 Random Seed Management

```python
import random
import numpy as np

def set_random_seeds(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    # If using torch: torch.manual_seed(seed)
```

### 12.2 Environment Documentation

Create `environment.txt` with exact versions:
```bash
uv pip freeze > requirements.txt
```

### 12.3 Experiment Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiment.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log all experiment parameters
logger.info(f"Starting Experiment 1 with seed={seed}, model={model}")
```

---

## 13. Deliverable Formats

### 13.1 Data Formats

**Synthetic Facts:** JSON
```json
[
  {
    "id": "fact_001",
    "category": "geography",
    "fact": "Paris is the capital of France",
    "question": "What is the capital of France?",
    "answer": "Paris"
  }
]
```

**Experimental Results:** CSV
```csv
run_id,experiment,noise_ratio,fact_id,question,predicted_answer,correct_answer,is_correct,timestamp
1,exp2,0.0,fact_001,"What is the capital of France?","Paris","Paris",True,2025-12-10 10:30:15
```

### 13.2 Graph Formats

- **Primary:** PNG (300 DPI) for notebooks
- **Secondary:** PDF (vector) for publications
- **Naming:** `experiment1_position_accuracy.png`

---

## 14. Security Considerations

1. **No Hardcoded Secrets:** All credentials in `.env` (gitignored)
2. **Input Sanitization:** Validate all external inputs
3. **API Rate Limiting:** Respect Ollama local limits
4. **Data Privacy:** Only synthetic data, no real PII

---

## 15. Future Extensions

Potential enhancements beyond current scope:
1. **Multi-Model Comparison:** Test across llama2, mistral, phi simultaneously
2. **Advanced Reranking:** Implement multiple reranking strategies
3. **Query Expansion:** Automatic query reformulation
4. **Hybrid Search:** Combine dense and sparse retrieval
5. **Interactive Dashboard:** Streamlit/Gradio UI for experiments

---

**Document Status:** Complete v1.0
**Last Updated:** December 10, 2025
**Next Steps:** Create TASKS.md for implementation breakdown
