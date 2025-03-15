The simple evaluation metric:

```python
def compute_metrics(eval_preds):
    """Compute metrics for evaluation."""
    preds, labels = eval_preds
    
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Simple exact match metric
    exact_matches = sum(1 for p, l in zip(pred_str, label_str) if p.strip() == l.strip())
    exact_match_percentage = exact_matches / len(pred_str) * 100
    
    return {
        "exact_match": exact_match_percentage,
    }
```

This metric uses exact string matching to determine if the model's prediction exactly matches the reference answer.

## Issues with Exact Match for ConvFinQA

Looking at the ConvFinQA dataset and how it's processed:

1. **Numerical Precision Issues**: The dataset involves financial calculations with numerical answers (e.g., "14.1%" or "10.1%"). Exact match would fail if the model produces "14.1" vs "14.1%" or "0.141" vs "14.1%", even though these are semantically equivalent.

2. **Structured Output Format**: The dataset is processed to include structured thinking with `<|begin_of_thought|>` and `<|begin_of_solution|>` tags. Small formatting differences in these sections could cause exact match to fail.

3. **Multi-turn Nature**: The dataset consists of multi-turn conversations where each turn builds on previous calculations. Exact match doesn't account for partial correctness in reasoning steps.

4. **Rounding Differences**: In the examples, we see answers like "14.1%" which is a rounded version of the exact calculation (0.14136). The model might produce a different but valid rounding (e.g., "14.14%").

## Better Metrics for ConvFinQA

Here are more appropriate metrics for this dataset:

1. **Solution-Only Exact Match**: Extract just the final numerical answer from the `<|begin_of_solution|>` section and compare only that part, ignoring formatting differences.

2. **Numerical Tolerance Match**: For numerical answers, use a tolerance-based comparison (e.g., consider answers within ±0.1% to be correct).

3. **Step-by-Step Evaluation**: Evaluate both the reasoning steps and the final answer separately, giving partial credit for correct reasoning even if the final answer is slightly off.

4. **Normalized Numerical Comparison**: Convert all percentage/decimal representations to a standard form before comparison.

## Recommendation

I recommend implementing a custom metric that:

1. Extracts the numerical solution from between the `<|begin_of_solution|>` and `<|end_of_solution|>` tags
2. Normalizes the numerical format (removing % signs, standardizing decimal representation)
3. Uses a tolerance-based comparison for numerical values
4. Optionally evaluates the reasoning steps separately

Here's a proposed implementation:

```python
def compute_metrics(eval_preds):
    """Compute metrics for financial QA evaluation."""
    preds, labels = eval_preds
    
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Extract solutions and normalize
    def extract_solution(text):
        solution_pattern = r'<\|begin_of_solution\|>\s*(.*?)\s*<\|end_of_solution\|>'
        match = re.search(solution_pattern, text, re.DOTALL)
        if match:
            solution = match.group(1).strip()
            # Normalize numerical values
            solution = solution.replace('%', '')
            try:
                # Convert to float for numerical comparison
                return float(solution)
            except:
                # If not a number, return as is
                return solution
        return text.strip()
    
    pred_solutions = [extract_solution(p) for p in pred_str]
    label_solutions = [extract_solution(l) for l in label_str]
    
    # Exact matches (for non-numerical answers)
    exact_matches = 0
    # Numerical matches with tolerance
    numerical_matches = 0
    tolerance = 0.001  # 0.1% tolerance
    
    for pred, label in zip(pred_solutions, label_solutions):
        if isinstance(pred, float) and isinstance(label, float):
            # Numerical comparison with tolerance
            if abs(pred - label) <= tolerance * max(1, abs(label)):
                numerical_matches += 1
        elif pred == label:
            exact_matches += 1
    
    total_matches = exact_matches + numerical_matches
    match_percentage = total_matches / len(pred_str) * 100
    
    return {
        "numerical_match": match_percentage,
    }
```

This approach would be much more appropriate for evaluating model performance on the ConvFinQA dataset, as it accounts for the numerical nature of the answers and allows for minor formatting differences while still ensuring the core calculations are correct.
