
import json
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from evaluation_framework import RunEvals, QuickEvaluator

total_samples = 5000
model_name = f"flan-t5-domain-generator-final-{total_samples}"

eval = QuickEvaluator()

eval_samples = 100
eval_data_file_name = f"./data/eval-data/data_eval_{eval_samples}.json"
with open(eval_data_file_name, "r") as f:
    validation_data = json.load(f)

def suggest_domains(input_texts):
    tokenizer = T5Tokenizer.from_pretrained(f"./models/{model_name}")
    fine_tuned_model = T5ForConditionalGeneration.from_pretrained(f"./models/{model_name}")
    #Convert input texts to input IDs and attention masks
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",  # Return PyTorch tensors
        padding=True,
        truncation=True,
        max_length=128
    )

    #use fine tuned model to generate domains for the validation dataset
    generated_domains = fine_tuned_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_return_sequences=1
        )
    decoded_domains = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_domains]
    decoded_domains = [i.split(',') for i in decoded_domains]
    decoded_domains = [[j.strip() for j in i] for i in decoded_domains]
    decoded_domains = [list(set(i)) for i in decoded_domains]

    return decoded_domains

def evaluate_data(business_desc, domains):
    evaluation_results = eval.fine_tuned_calculate_overall_score(business_desc, domains)
    
    return evaluation_results

# scores = []

# input_texts = [i['business_description'] for i in validation_data]
# domains_list = suggest_domains(input_texts)

# for i in range(len(domains_list)):
#     # print(f"Business Description: {input_texts[i]}")
#     if i%100 == 0:
#         print(f"Processing {i} of {len(input_texts)}")
#     industry = validation_data[i]['industry']
#     domains = domains_list[i]
#     if len(domains) == 1 and len(domains[0]) == 0:
#         print(f"No domains generated for {input_texts[i]} and industry {industry}")
#         print('-' * 50)
    # print(f"Industry: {industry}, i: {i}")

    # print(f"Generated Domains: {domains[i]}, {len(domains[i]), len(domains[i][0])}")
    # evaluation_results = evaluate_data(input_texts[i], domains[i])
    # print(f"Evaluation Results: {evaluation_results}")
    # overall_score = evaluation_results['overall_score']
    # print("")
    # scores.append(overall_score)
    # print('-' * 50)

# average_score = sum(scores)/len(scores)
# print(f"Average score: {average_score}")

# def get_suggestions_and_evaluate(input_texts):
#     domains_list = suggest_domains(input_texts)

#     for i in range(len(domains_list)):
#         # print(f"Business Description: {input_texts[i]}")
#         if i%100 == 0:
#             print(f"Processing {i} of {len(input_texts)}")
#         industry = validation_data[i]['industry']
#         domains = domains_list[i]
        

def save_final_metric(data, average_score):
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
    domains = suggest_domains([input_text])
    print(f"Domains: {domains}")

def predict_domains_for_all_input_texts():
    all_input_texts = [i['business_description'] for i in validation_data]
    print(f'Predicting domains for {len(all_input_texts)} input texts')
    all_domains = suggest_domains(all_input_texts)

    all_data = []
    for i in range(len(all_domains)):
        if i%10 == 0:
            print(f"Processing {i} of {len(all_domains)}")
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

    #save the data to a json file
    file_name = f"./data/eval-data/predicted_domains_and_ground_truth_{eval_samples}.json"
    save_ground_truth_and_generated_domains(all_data, file_name)

try_manual_input_text('Scam people online')