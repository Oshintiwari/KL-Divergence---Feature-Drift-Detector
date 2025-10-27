# KL Divergence Feature Drift Detector

===
RL Task — Data Drift Detection

## Objective
This task trains an LLM to implement a realistic ML engineering skill: **detecting feature drift** between two datasets using Kullback–Leibler (KL) divergence.  
It mimics how ML engineers monitor model inputs over time to identify data distribution shifts.

## Task Description
The model receives a prompt (`drift_prompt.md`) describing how to:
- Discretize numeric features into 20 equal-width bins.
- Compute KL divergence between two distributions.
- Apply numerical stability using epsilon.
- Report all features with drift above a threshold.

The grading script (`main_drift.py`) checks whether the model:
- Implements correct bin alignment and normalization.
- Handles NaNs safely.
- Produces KL values close to the reference implementation.

## Output Example

Drifted: age (KL=0.1562)
Drifted: income (KL=0.3366)


## Key Engineering Concepts
- Feature-level distribution monitoring for data drift.
- Histogram-based approximation for continuous variables.
- Numerical stability via epsilon and safe normalization.
- Integration of LLM tool-use (Claude + Python executor).

## Why This Task
This task teaches an LLM to reason like an ML engineer monitoring production pipelines — a core, real-world responsibility in maintaining reliable machine learning systems.


Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.
