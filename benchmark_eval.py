import json
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from evaluation_framework import RunEvals

eval_samples = 100
with open(f"./data/eval-data/data_eval_{eval_samples}.json", "r") as f:
    validation_data = json.load(f)

def evaluate_data(validation_data, eval_samples):
    eval = RunEvals()
    validation_data_with_results = []
    for i in range(eval_samples):
        print(f"Business Description: {validation_data[i]['business_description']}")
        print(f"Generated Domains: {validation_data[i]['domain_suggestions']}")
        business_desc = validation_data[i]['business_description']
        industry = validation_data[i]['industry']
        domains = validation_data[i]['domain_suggestions']

        evaluation_results = eval.combined_eval(business_desc, domains, industry)
        # evaluation_results = eval.combined_eval(business_desc, domains, industry)
        print(f"Evaluation Results: {evaluation_results}")
        
        validation_data[i]['evaluation_results'] = evaluation_results
        validation_data_with_results.append(validation_data[i])
        print("-" * 50)
    return validation_data_with_results

#use argparse argument called, save, to save the results to a json file
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--update", action="store_true", help="save the results to a json file")
# parser.add_argument("--samples", type=int, default=1, help="number of samples to evaluate")
# args = parser.parse_args()
# if args.update:
#     print("Updating the eval data with the results")
#     validation_data_with_results = evaluate_data(validation_data, args.samples)
#     #replace the original data with the updated data
#     with open(f"./data/eval-data/data_eval_{eval_samples}.json", "w") as f:
#         json.dump(validation_data_with_results, f, indent=4)
#     print("Done")
# else:
#     print(f"Evaluating the data with {args.samples} samples")
#     evaluate_data(validation_data, args.samples)

#read the data_eval_100.json file
# calculate the average of the overall_score
# print the average

#calculate the average of the overall_score
average_score = sum([i['evaluation_results']['overall_score'] for i in validation_data])/len(validation_data)
print(f"Average score: {average_score}")
#append this to eval metrics.json file

# data = {
#     'model_name': None,
#     "average_score": average_score,
#     "sample_count": eval_samples,
#     "eval_data_file_name": f"./data/eval-data/data_eval_{eval_samples}.json"
# }

# #read the eval_metrics.json file
# with open("./data/eval-data/eval_metrics.json", "r") as f:
#     eval_metrics = json.load(f)

# #append the data to the eval_metrics.json file
# eval_metrics.append(data)
# with open("./data/eval-data/eval_metrics.json", "w") as f:
#     json.dump(eval_metrics, f, indent=4)
