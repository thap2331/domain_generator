
import json
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from evaluation_framework import RunEvals, QuickEvaluator
from utils import Utils

total_samples = 5000
model_name = f"flan-t5-domain-generator-final-{total_samples}"

eval = QuickEvaluator()
utils = Utils(model_name=model_name)

eval_samples = 100
eval_data_file_name = f"./data/eval-data/data_eval_{eval_samples}.json"
with open(eval_data_file_name, "r") as f:
    validation_data = json.load(f)


def evaluate_data(business_desc, domains):
    evaluation_results = eval.fine_tuned_calculate_overall_score(business_desc, domains)
    
    return evaluation_results


def save_final_metric(average_score):
    data = {
        'model_name': model_name,
        "average_score": average_score,
        "sample_count": eval_samples,
        "eval_data_file_name": eval_data_file_name
    }

    #read the eval_metrics.json file
    with open("./data/eval-data/eval_metrics.json", "r") as f:
        eval_metrics = json.load(f)

    #append the data to the eval_metrics.json file
    eval_metrics.append(data)
    with open("./data/eval-data/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)


def save_ground_truth_and_generated_domains(data, file_name):
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)

def try_manual_input_text(input_text: str=None):
    domains = utils.query_model([input_text])[0]
    print(f"Domains: {domains}")

def predict_domains_for_all_input_texts():
    all_input_texts = [i['business_description'] for i in validation_data]
    print(f'Predicting domains for {len(all_input_texts)} input texts')
    all_domains = utils.query_model(all_input_texts)

    all_data = []
    scores = []
    print(f"Evaluating {len(all_domains)} domains")
    for i in range(len(all_domains)):
        if i%10 == 0:
            print(f"Evaluating {i} of {len(all_domains)} business descriptions")
        business_desc = validation_data[i]['business_description']
        domains = all_domains[i]
        industry = validation_data[i]['industry']
        all_data.append({
            'business_description': business_desc,
            'ground_truth_domains': validation_data[i]['domain_suggestions'],
            'industry': industry,
            'predicted_domains': domains,
            'evaluation_results': eval.fine_tuned_calculate_overall_score(business_desc, domains)
        })
        scores.append(eval.fine_tuned_calculate_overall_score(business_desc, domains)['overall_score'])
    average_score = sum(scores)/len(scores)
    save_final_metric(average_score)

    #save the data to a json file
    file_name = f"./data/eval-data/predicted_domains_and_ground_truth_{eval_samples}_{model_name}.json"
    save_ground_truth_and_generated_domains(all_data, file_name)

# print(f"Model name: {model_name}")
# print('business description: Scam people online')
# try_manual_input_text('Scam people online')
# print('business description: Best way to kill weeds in your garden')
# try_manual_input_text('Best way to kill weeds in your garden')
# print('business description: organic coffee shop in downtown area')
# try_manual_input_text('organic coffee shop in downtown area')
# print('business description: adult content website with explicit nude content')
# try_manual_input_text('adult content website with explicit nude content')

# predict_domains_for_all_input_texts()
