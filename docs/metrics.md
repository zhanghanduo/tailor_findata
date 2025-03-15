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

3. **Program-Step-Match**: Just check the program outputs in the specifice `<begin_of_program>` tag and conduct matching.


I design two types of loss `program_match` and `answer_match`, and build several metrics on top of them:
```python
return {
        "program_match": program_match_percentage,
        "answer_match": answer_match_percentage,
        "format_valid": format_valid_percentage,  # Add this new metric
        "program_matches_count": program_matches,
        "answer_matches_count": answer_matches,
        "format_valid_count": format_valid_count,  # Add this new count
        "total_samples": len(pred_str),
    }
```

The metric code:
```python
    pred_programs = []
    label_programs = []
    pred_answers = []
    label_answers = []
    
    for p in pred_str:
        pred_programs.append(extract_program_tokens(p))
        pred_answers.append(extract_answer(p))
    
    for l in label_str:
        label_programs.append(extract_program_tokens(l))
        label_answers.append(extract_answer(l))
    # Program token match
    program_matches = 0
    for pred_prog, label_prog in zip(pred_programs, label_programs):
        if pred_prog == label_prog:
            program_matches += 1
    
    # Answer matches (numerical with tolerance or exact)
    answer_matches = 0
    tolerance = 0.01  # 1% tolerance - increased from 0.1%

    # Calculate metrics
    program_match_percentage = program_matches / max(1, len(pred_str)) * 100
    answer_match_percentage = answer_matches / max(1, len(pred_str)) * 100
```
