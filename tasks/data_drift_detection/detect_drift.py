import pandas as pd
import numpy as np

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
    epsilon = 1e-10
    kl_values = {}
    
    for column in old_df.columns:
        # Remove NaNs
        old_data = old_df[column].dropna()
        new_data = new_df[column].dropna()
        
        # Determine bin edges based on combined data range
        combined_min = min(old_data.min(), new_data.min())
        combined_max = max(old_data.max(), new_data.max())
        bins = np.linspace(combined_min, combined_max, 21)
        
        # Create histograms
        old_hist, _ = np.histogram(old_data, bins=bins)
        new_hist, _ = np.histogram(new_data, bins=bins)
        
        # Normalize to get probabilities
        old_prob = old_hist / old_hist.sum()
        new_prob = new_hist / new_hist.sum()
        
        # Add epsilon for stability
        old_prob = old_prob + epsilon
        new_prob = new_prob + epsilon
        
        # Renormalize after adding epsilon
        old_prob = old_prob / old_prob.sum()
        new_prob = new_prob / new_prob.sum()
        
        # Compute KL divergence
        kl_div = np.sum(new_prob * np.log(new_prob / old_prob))
        kl_values[column] = float(kl_div)
        
        # Print if drifted
        if kl_div > threshold:
            print(f"Drifted: {column} (KL={kl_div})")
    
    return kl_values

# Create sample DataFrames for testing
np.random.seed(42)
old_df = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.uniform(0, 10, 1000),
    'feature3': np.random.exponential(2, 1000)
})

new_df = pd.DataFrame({
    'feature1': np.random.normal(0.5, 1.2, 1000),  # Shifted mean and variance
    'feature2': np.random.uniform(0, 10, 1000),    # Similar distribution
    'feature3': np.random.exponential(3, 1000)     # Different scale
})

result = detect_drift(old_df, new_df)