1. Methodology & Initial Results

- Dataset creation approach and baseline model selection
    - I did an initial research with claude on how to create a synthetic data as well as pitfalls to avoid.
    - I had an idea of query expansion, i.e., expand business description in case its too short and not very descriptive.
    - Thinking about subdomain, but assuming since its just domain, so we should be ok.

- Initial model performance and evaluation metrics
    - Baseline metric, i.e., on 100 eval dataset. This dataset has both input and output. I generate the scores that includes both rules based evaluation and llm evaluation. This is so that we can compare fine tuned model performance with baseline evaluation dataset.
        - The average score was 0.55 which is a combined score of rule based evaluation and llm evaluation. The maximum score is 1.
    - Fine tuned metric
        - The average score was 0.54 which is as good as the eval dataset/ground truth. The score is based on both llm as a judge and rule based evaluation.
        -Alert: The data on eval dataset is randomly generated as the training dataset. There is a chance of overfitting.


2. Edge Case Analysis
- Discovery process: How you found edge cases
    - First few results
        - duplicate domain problems
        - failure on simple deviation from the dataset (presenting a big issue with data quality)
            - ex: if business description looking for explicit activities
    - Edge case discovery for inappropriate content by openai (as per https://platform.openai.com/docs/guides/moderation)
        - None detected on the evaluation dataset
- Failure taxonomy: Categories of failures with examples
    - 
- Frequency analysis: How common each failure type is
    - 
    ```
        {
            "total_cases": 4,
            "by_failure_type": {
                "format_error": 3,
                "safety_issue": 1
            },
            "by_severity": {
                "HIGH": 4
            },
            "by_category": {
                "empty_input": 1,
                "long_input": 1,
                "weapons_business": 1,
                "adult_content": 1
            },
            "improvement_strategies": {
                "Add to training data": 4
            },
            "root_causes": {
                "Invalid domains found in the response": 3,
                "Content flagged by OpenAI moderation": 1
            }
        }
    ```


3. Iterative Improvement
• Improvement strategies: What you tried and why
• Quantified results: Before/after metrics for each iteration
• LLM judge validation: How you ensured evaluation quality


4. Model Comparison & Recommendations
• Performance comparison: Statistical significance of improvements
• Production readiness: Which version you'd deploy and why
• Future improvements: Next steps for continued improvement