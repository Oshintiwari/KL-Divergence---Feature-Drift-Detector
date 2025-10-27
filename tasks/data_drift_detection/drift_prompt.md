You are an ML engineer monitoring production data for drift.
Implement a Python function that detects feature drift between two datasets using KL divergence.

You are given two Pandas DataFrames: old_df and new_df, each numeric-only.

Implement and print the output of:

def detect_drift(old_df, new_df, threshold=0.1):
    """
    Steps:
    1) For each numeric column:
       - Discretize both datasets into the same 20 equal-width bins
       - Compute KL divergence between the two histograms
       - Use epsilon=1e-10 for stability; ignore NaNs
    2) Return a dict {column_name: kl_value} (floats)
    3) Print every feature with KL > threshold as:
       Drifted: <column> (KL=<value>)
    """

---

Output Format Instructions
Write your entire solution as valid Python code implementing the above function and printing its output.

Your code must:
- Import pandas and numpy.
- Create two sample DataFrames (old_df and new_df) for testing.
- Call detect_drift(old_df, new_df) at the end.
- Print results directly in the required format.

Very Important:
Wrap your entire solution in triple backticks like this:

```python
# your complete Python solution here
```

Do not include any explanations, reasoning, or text outside the code block.
The grader will extract and execute the Python code block only.
