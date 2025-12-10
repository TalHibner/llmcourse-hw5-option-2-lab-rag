# Product Requirements Document (PRD)
## RAG Impact on Context Window Research

**Version:** 1.0
**Date:** December 10, 2025
**Author:** Graduate Research Project
**Course:** LLM Course - Homework 5, Option 2

---

## 1. Executive Summary

This research project investigates the fundamental challenges of large context windows in Large Language Models (LLMs) and demonstrates how Retrieval-Augmented Generation (RAG) addresses these limitations. Through systematic experimentation, we will quantify the "lost in the middle" phenomenon, measure performance degradation under noisy contexts, and validate RAG as a solution.

---

## 2. Research Question

**Primary Question:**
*How does Retrieval-Augmented Generation (RAG) mitigate the context window limitations of LLMs, particularly the "lost in the middle" problem and noise-induced performance degradation?*

**Sub-Questions:**
1. How does fact retrieval accuracy vary based on position within a long context (beginning/middle/end)?
2. To what extent does irrelevant information (noise) degrade LLM performance in question answering?
3. Can RAG with vector-based retrieval maintain high accuracy even with noisy, large-scale document collections?

---

## 3. Objectives

### 3.1 Primary Objectives
1. **Quantify Context Window Problem**: Empirically measure how LLM accuracy degrades when relevant information is positioned at different locations within long contexts
2. **Measure Noise Impact**: Determine the relationship between noise ratio and accuracy degradation
3. **Validate RAG Solution**: Demonstrate that RAG maintains consistent high performance regardless of context length or noise levels

### 3.2 Secondary Objectives
1. Implement reranking mechanisms to further improve RAG performance (bonus)
2. Create publication-quality visualizations of experimental results
3. Develop reusable, modular codebase following software engineering best practices
4. Generate comprehensive statistical analysis with confidence intervals

---

## 4. Hypotheses

### Hypothesis 1: Lost in the Middle (Experiment 1)
**H0 (Null):** Position of information within context has no significant effect on retrieval accuracy
**H1 (Alternative):** Information positioned in the middle of long contexts will show significantly lower retrieval accuracy compared to information at the beginning or end

**Expected Result:** U-shaped accuracy curve with highest accuracy at beginning/end and lowest in the middle

### Hypothesis 2: Noise-Induced Degradation (Experiment 2)
**H0 (Null):** Addition of irrelevant documents does not affect LLM accuracy
**H1 (Alternative):** Accuracy decreases monotonically as noise ratio increases

**Expected Result:** Linear or exponential decay in accuracy as noise percentage increases from 0% to 90%

### Hypothesis 3: RAG Resilience (Experiment 3)
**H0 (Null):** RAG performance is comparable to baseline classic approach
**H1 (Alternative):** RAG maintains accuracy >90% regardless of noise levels, significantly outperforming classical long-context approaches

**Expected Result:** RAG achieves 95%+ retrieval precision even with 80%+ noise ratio

---

## 5. Success Metrics

### 5.1 Quantitative Metrics

**Primary Metrics:**
- **Accuracy:** Percentage of correctly answered questions
- **Retrieval Precision:** P = TP / (TP + FP) where TP = correct retrievals, FP = incorrect retrievals
- **Position-Based Accuracy:** Accuracy stratified by document position (beginning/middle/end terciles)
- **Noise Tolerance:** Maximum noise percentage maintaining >80% accuracy

**Secondary Metrics:**
- **Hallucination Rate:** Percentage of confident but incorrect responses
- **Response Latency:** Average time to generate answers (ms)
- **Retrieval Recall:** Percentage of relevant documents successfully retrieved by RAG

### 5.2 Statistical Significance
- Minimum 5 runs per experimental condition
- 95% confidence intervals for all measurements
- Cohen's d effect size calculation for key comparisons
- Statistical significance threshold: p < 0.05

### 5.3 Minimum Performance Thresholds

**Experiment 1 (Context Window):**
- Middle accuracy must be ≥20% lower than beginning/end (demonstrating problem)
- Minimum 30 questions tested across 3 position categories

**Experiment 2 (Noise):**
- Clear monotonic decrease in accuracy with noise increase
- 0% noise accuracy: ≥90%
- 80% noise accuracy: ≤50% (showing significant degradation)

**Experiment 3 (RAG):**
- RAG accuracy with 80% noise: ≥90%
- RAG vs Classic improvement: ≥40 percentage points at high noise
- Retrieval precision: ≥95%

---

## 6. Experimental Design

### 6.1 Experiment 1: Context Window Problem - "Lost in the Middle"

**Objective:** Demonstrate that LLMs struggle to retrieve information from the middle of long contexts

**Method:**
1. Generate 20-30 synthetic fact documents (e.g., "Paris is the capital of France")
2. Create questions targeting specific facts (e.g., "What is the capital of France?")
3. Concatenate all documents into a single long context
4. Systematically vary target fact position: beginning (docs 1-10), middle (docs 11-20), end (docs 21-30)
5. Query LLM with context + question using Ollama API
6. Record accuracy by position category

**Independent Variable:** Position of target fact (beginning/middle/end)
**Dependent Variable:** Answer accuracy
**Control Variables:** Same facts, same questions, same LLM model, same prompt template

**Data Collection:**
- 30 unique facts × 3 positions = 90 queries
- 5 repetitions = 450 total measurements

**Expected Output:**
- Position vs Accuracy graph (bar chart or line plot)
- Statistical summary table
- Example successes and failures per position

---

### 6.2 Experiment 2: Noise and Irrelevance - "The Failure"

**Objective:** Quantify how irrelevant information degrades LLM performance

**Method:**
1. Start with 10 core fact documents
2. Generate 90 "filler" documents (random sentences, unrelated facts)
3. Vary noise ratio: 0%, 20%, 40%, 60%, 80%, 90%
4. Embed documents using Ollama's embedding model (e.g., nomic-embed-text)
5. Present full document set to LLM as context
6. Measure accuracy and hallucination rate

**Independent Variable:** Noise percentage
**Dependent Variable:** Accuracy, hallucination rate
**Control Variables:** Same 10 core facts, same questions, same LLM

**Data Collection:**
- 10 questions × 6 noise levels × 5 repetitions = 300 measurements

**Expected Output:**
- Noise Ratio vs Accuracy graph (line plot with error bars)
- Hallucination rate overlay
- Performance degradation curve fit (linear/exponential)

---

### 6.3 Experiment 3: RAG Solution

**Objective:** Show that RAG maintains high accuracy even with noise and large document collections

**Method:**
1. Build vector database using Chroma DB
2. Generate embeddings for all documents using Ollama (nomic-embed-text or similar)
3. Implement RAG pipeline:
   - Query → Embed query → Retrieve top-k documents → Generate answer with retrieved context
4. Compare RAG vs Classic (full context) across noise levels
5. **Bonus:** Implement reranking using cross-encoder or LLM-based reranking

**Independent Variable:** Noise percentage, retrieval method (RAG vs Classic)
**Dependent Variable:** Accuracy, retrieval precision, latency
**Control Variables:** Same document set, same questions, same LLM for generation

**Data Collection:**
- 10 questions × 6 noise levels × 2 methods × 5 repetitions = 600 measurements
- Top-k parameter sweep: k ∈ {1, 3, 5, 10}

**Expected Output:**
- RAG vs Classic comparison chart (grouped bar chart)
- Retrieval precision metrics
- Latency comparison
- Noise resilience curves

---

## 7. Scope and Constraints

### 7.1 In Scope
- Local LLM inference using Ollama
- Synthetic dataset generation
- Three core experiments
- Statistical analysis and visualization
- Complete Python implementation with tests
- Comprehensive documentation

### 7.2 Out of Scope
- Real-world dataset evaluation
- Multi-modal RAG (text-only)
- Production deployment
- Commercial API usage (OpenAI, Anthropic)
- Fine-tuning or model training

### 7.3 Technical Constraints
- Must use Ollama for local inference
- Must use LangChain for RAG components
- Must use Chroma as vector store (as recommended by lecturer)
- Minimum 20, maximum 30 synthetic documents
- Graduate-level statistical rigor required

---

## 8. Deliverables

### 8.1 Code Deliverables
1. **Source Code** (`src/` directory):
   - Data generation module
   - Experiment 1 implementation
   - Experiment 2 implementation
   - Experiment 3 implementation (RAG pipeline)
   - Utilities and configuration

2. **Tests** (`tests/` directory):
   - Unit tests with ≥70% coverage
   - Integration tests for RAG pipeline
   - Data validation tests

3. **Experiments** (`experiments/` directory):
   - Jupyter notebooks for each experiment
   - Analysis notebook with LaTeX equations
   - Reproducible execution scripts

### 8.2 Data Deliverables
1. **Synthetic Dataset** (`data/` directory):
   - 20-30 fact documents (JSON/CSV)
   - Filler/noise documents
   - Question-answer pairs
   - Data generation scripts

2. **Results** (`results/` directory):
   - Raw experimental data (CSV)
   - Statistical summaries
   - All generated graphs (high-resolution PNG/PDF)
   - Results tables

### 8.3 Documentation Deliverables
1. **Project Documentation**:
   - PRD.md (this document)
   - DESIGN.md (technical architecture)
   - TASKS.md (implementation breakdown)
   - README.md (setup and usage guide)

2. **Analysis Documentation**:
   - Comprehensive analysis notebook
   - LaTeX-formatted mathematical explanations
   - Interpretation of results
   - Conclusions and insights

### 8.4 Configuration Deliverables
1. `pyproject.toml` or `requirements.txt`
2. `.gitignore`
3. Example configuration files
4. Environment setup instructions

---

## 9. Stakeholders

**Primary Stakeholder:** Course Instructor (Dr. Yoram Segal)
**Secondary Stakeholders:** Graduate students, LLM researchers
**Target Audience:** Academic community studying RAG systems

---

## 10. Timeline and Milestones

| Milestone | Deliverable | Status |
|-----------|-------------|--------|
| M1 | Documentation complete (PRD, DESIGN, TASKS, README) | In Progress |
| M2 | Project setup and data generation | Pending |
| M3 | Experiment 1 complete with analysis | Pending |
| M4 | Experiment 2 complete with analysis | Pending |
| M5 | Experiment 3 complete with analysis | Pending |
| M6 | Comprehensive analysis notebook | Pending |
| M7 | All visualizations generated | Pending |
| M8 | Tests written and passing | Pending |
| M9 | Final documentation and review | Pending |
| M10 | Project submission | Pending |

---

## 11. Acceptance Criteria

### 11.1 Technical Acceptance
- [ ] All code follows PEP 8 style guidelines
- [ ] Unit test coverage ≥70%
- [ ] All experiments run successfully on fresh environment
- [ ] No hardcoded API keys or secrets
- [ ] Modular, reusable code architecture
- [ ] Proper error handling and input validation

### 11.2 Scientific Acceptance
- [ ] Minimum 5 experimental runs per condition
- [ ] Statistical significance demonstrated (p < 0.05)
- [ ] Confidence intervals included in all measurements
- [ ] Clear hypothesis validation or rejection
- [ ] Publication-quality figures (300+ DPI)
- [ ] Reproducible results with random seeds

### 11.3 Documentation Acceptance
- [ ] Complete README with installation instructions
- [ ] All functions have docstrings
- [ ] Architecture diagrams included
- [ ] Code comments explain "why" not "what"
- [ ] Analysis notebook includes LaTeX equations
- [ ] Results interpretation clearly explained

---

## 12. Risks and Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Ollama installation issues | High | Medium | Provide detailed setup guide, Docker alternative |
| LLM non-determinism | Medium | High | Use temperature=0, multiple runs, random seeds |
| Insufficient performance difference | High | Low | Carefully design noise levels, ensure clear baselines |
| Time constraints | Medium | Medium | Prioritize core experiments, reranking as bonus |
| Compute limitations | Medium | Medium | Use smaller models, limit experiment scale if needed |

---

## 13. Open Questions

1. Which specific Ollama model should be used? (e.g., llama2, mistral, phi)
2. What embedding dimension provides best trade-off for Chroma DB?
3. Should we implement multiple reranking strategies or just one?
4. What constitutes "good" synthetic filler documents that don't overlap with facts?
5. Should we test multiple LLM model sizes to show generalizability?

---

## 14. References

1. Liu, N. F., et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts"
2. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
3. Izacard, G., & Grave, E. (2021). "Leveraging Passage Retrieval with Generative Models"
4. LangChain Documentation: https://python.langchain.com/
5. Ollama Documentation: https://ollama.ai/
6. ChromaDB Documentation: https://docs.trychroma.com/

---

**Document Status:** Draft v1.0
**Last Updated:** December 10, 2025
**Next Review:** Upon completion of DESIGN.md
