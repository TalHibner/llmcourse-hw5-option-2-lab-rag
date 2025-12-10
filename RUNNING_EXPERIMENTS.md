# Running Experiments Guide

This guide explains how to execute the three experiments and generate results for the RAG Context Window research project.

## Prerequisites

### 1. Install Ollama

Ollama is required to run the local LLM for experiments.

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**
Download from https://ollama.com/download

### 2. Start Ollama Service

**Linux/macOS:**
```bash
ollama serve
```

Leave this running in a separate terminal.

### 3. Pull Required Models

```bash
# Language model for generation (7B parameters, ~4GB)
ollama pull llama2

# Embedding model for vector search (~274MB)
ollama pull nomic-embed-text
```

Verify models are available:
```bash
ollama list
```

You should see:
```
NAME                    ID              SIZE    MODIFIED
llama2:latest          78e26419b446    3.8 GB  ...
nomic-embed-text:latest 0a109f422b47    274 MB  ...
```

### 4. Verify Ollama is Running

```bash
curl http://localhost:11434/api/tags
```

Should return JSON with model list.

## Running Experiments

### Quick Start - Run All Experiments

From the project root:

```bash
# Activate virtual environment (if using one)
source .venv/bin/activate  # or: uv venv && source .venv/bin/activate

# Run all three experiments in sequence
python3 scripts/run_experiment1.py
python3 scripts/run_experiment2.py
python3 scripts/run_experiment3.py
```

### Experiment 1: Context Window Limitations

**Purpose**: Demonstrate "Lost in the Middle" phenomenon

**Command:**
```bash
python3 scripts/run_experiment1.py
```

**What it does:**
- Tests 25 facts at different positions (beginning, middle, end)
- Measures accuracy at each position
- Expected: U-shaped accuracy curve (low in middle)

**Output:**
- `results/experiment1_results.json`: Detailed results per fact
- `results/experiment1/summary_statistics.json`: Accuracy by position
- Console: Progress bar and accuracy summary

**Expected Runtime**: ~5-10 minutes (depends on system)

### Experiment 2: Noise Impact

**Purpose**: Measure degradation with increasing noise

**Command:**
```bash
python3 scripts/run_experiment2.py
```

**What it does:**
- Tests 10 facts with noise ratios [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
- Measures accuracy and hallucination rate
- Expected: Linear degradation (accuracy drops as noise increases)

**Output:**
- `results/experiment2_results.json`: Detailed results per fact/noise level
- `results/experiment2/summary_statistics.json`: Accuracy by noise ratio
- Console: Progress bar and accuracy trends

**Expected Runtime**: ~10-15 minutes

### Experiment 3: RAG Solution

**Purpose**: Demonstrate RAG maintains high accuracy with noise

**Command:**
```bash
python3 scripts/run_experiment3.py
```

**What it does:**
- Creates vector database (ChromaDB) with facts
- Tests RAG retrieval with high noise (90%)
- Compares top-k values [3, 5]
- Expected: >90% accuracy even with high noise

**Output:**
- `results/experiment3_results.json`: Detailed results per fact
- `results/experiment3/summary_statistics.json`: Accuracy and retrieval metrics
- `data/chromadb/`: Vector database files
- Console: Progress bar and RAG performance

**Expected Runtime**: ~15-20 minutes (includes embedding generation)

## Analyzing Results

After running experiments, analyze results with the Jupyter notebook:

```bash
# Install Jupyter if needed
pip install jupyter

# Start Jupyter
jupyter notebook notebooks/comprehensive_analysis.ipynb
```

The notebook will:
1. Load results from all experiments
2. Perform statistical analysis (ANOVA, Cohen's d)
3. Generate publication-quality visualizations
4. Export summary statistics to CSV

## Expected Results

### Experiment 1: Context Window
- **Hypothesis**: U-shaped accuracy curve
- **Beginning position**: 80-90% accuracy
- **Middle position**: 40-60% accuracy (Lost in the Middle)
- **End position**: 70-85% accuracy

### Experiment 2: Noise Impact
- **Hypothesis**: Linear degradation
- **0% noise**: 90-95% accuracy
- **50% noise**: 60-70% accuracy
- **90% noise**: 30-50% accuracy

### Experiment 3: RAG Solution
- **Hypothesis**: RAG maintains >90% accuracy
- **90% noise with RAG**: 90-95% accuracy
- **Baseline at 90% noise**: 30-50% accuracy
- **Improvement**: >40 percentage points

## Troubleshooting

### Ollama Connection Error

**Error**: `Failed to connect to Ollama at http://localhost:11434`

**Solution**:
1. Check Ollama is running: `ps aux | grep ollama`
2. Start Ollama: `ollama serve`
3. Verify connectivity: `curl http://localhost:11434/api/tags`

### Model Not Found

**Error**: `model 'llama2' not found`

**Solution**:
```bash
ollama pull llama2
ollama pull nomic-embed-text
```

### Slow Performance

**Issue**: Experiments taking very long

**Solutions**:
1. **Use faster model** (smaller but less accurate):
   ```yaml
   # In config/config.yaml
   llm:
     model_name: "llama2:7b-q4_0"  # 4-bit quantized
   ```

2. **Reduce test set**:
   ```yaml
   # In config/config.yaml
   experiments:
     experiment1:
       facts_to_test: 10  # Instead of 25
   ```

3. **Use GPU**: Ollama automatically uses GPU if available (NVIDIA CUDA or Apple Metal)

### Out of Memory

**Error**: Ollama crashes or system freezes

**Solution**:
1. **Use smaller model**:
   ```bash
   ollama pull phi  # 2.7B parameters, ~1.6GB
   ```

   Update `config/config.yaml`:
   ```yaml
   llm:
     model_name: "phi"
   ```

2. **Increase swap space** (Linux):
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### ChromaDB Errors

**Error**: `ChromaDB collection already exists`

**Solution**:
```bash
# Remove existing vector database
rm -rf data/chromadb/*
python3 scripts/run_experiment3.py
```

## Advanced Configuration

### Custom Parameters

Edit `config/config.yaml` to customize:

```yaml
llm:
  temperature: 0.0  # Increase for more creative answers
  max_tokens: 100   # Increase for longer responses
  timeout: 60       # Increase if getting timeouts

experiments:
  experiment1:
    positions: [0.0, 0.25, 0.5, 0.75, 1.0]  # Test more positions

  experiment2:
    noise_ratios: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Finer granularity

  experiment3:
    top_k_values: [1, 3, 5, 7, 10]  # Test more retrieval sizes
```

### Running on Remote Server

If running on a server without display:

```bash
# Use virtual display for matplotlib
export MPLBACKEND=Agg

# Run experiments
python3 scripts/run_experiment1.py
python3 scripts/run_experiment2.py
python3 scripts/run_experiment3.py
```

### Batch Processing

Run all experiments in background:

```bash
nohup bash -c "
  python3 scripts/run_experiment1.py && \
  python3 scripts/run_experiment2.py && \
  python3 scripts/run_experiment3.py
" > experiment_run.log 2>&1 &

# Monitor progress
tail -f experiment_run.log
```

## Validation Checklist

After running experiments, verify:

- [ ] All three results JSON files exist:
  - `results/experiment1_results.json`
  - `results/experiment2_results.json`
  - `results/experiment3_results.json`

- [ ] Summary statistics generated:
  - `results/experiment1/summary_statistics.json`
  - `results/experiment2/summary_statistics.json`
  - `results/experiment3/summary_statistics.json`

- [ ] ChromaDB populated:
  - `data/chromadb/` contains database files

- [ ] Results match expected patterns:
  - Experiment 1: U-shaped curve
  - Experiment 2: Linear degradation
  - Experiment 3: >90% accuracy with RAG

- [ ] No errors in logs

## Next Steps

Once experiments complete successfully:

1. **Analyze results**: Open Jupyter notebook for statistical analysis
2. **Generate visualizations**: Notebook creates publication-quality plots
3. **Commit results**: `git add results/ data/chromadb/ && git commit -m "Add experiment results"`
4. **Review findings**: Compare against hypotheses in PRD.md
5. **Document insights**: Update README.md with key findings

## Support

If you encounter issues not covered here:

1. Check Ollama logs: `journalctl -u ollama -f` (Linux)
2. Review experiment logs in `logs/` directory
3. Test Ollama manually:
   ```bash
   curl http://localhost:11434/api/generate -d '{
     "model": "llama2",
     "prompt": "Hello, world!"
   }'
   ```

## Performance Tips

- **GPU acceleration**: Ollama automatically uses NVIDIA CUDA or Apple Metal
- **Model caching**: First run downloads models, subsequent runs are faster
- **Parallel experiments**: Cannot run in parallel (share Ollama instance)
- **Incremental runs**: Can rerun individual experiments without affecting others

## Time Estimates

| Step | Time | Notes |
|------|------|-------|
| Ollama installation | 5 min | One-time setup |
| Model downloads | 10-15 min | One-time, ~4GB total |
| Experiment 1 | 5-10 min | 25 facts × 3 positions |
| Experiment 2 | 10-15 min | 10 facts × 6 noise levels |
| Experiment 3 | 15-20 min | Includes embedding generation |
| Analysis notebook | 2-3 min | If results exist |
| **Total first run** | **45-60 min** | Including setup |
| **Subsequent runs** | **30-45 min** | Models cached |

---

**Ready to start?** Run the Quick Start commands at the top of this guide!
