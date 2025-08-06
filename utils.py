from transformers import T5Tokenizer, T5ForConditionalGeneration
import json

class Utils:
    def __init__(self, model_name: str='flan-t5-domain-generator-final-5000'):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(f"./models/{self.model_name}")
        self.fine_tuned_model = T5ForConditionalGeneration.from_pretrained(f"./models/{self.model_name}")
        
    def query_model(self, input_texts):
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        generated_domains = self.fine_tuned_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_return_sequences=1
            )
        decoded_domains = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_domains]
        decoded_domains = [i.split(',') for i in decoded_domains]
        decoded_domains = [[j.strip() for j in i] for i in decoded_domains]
        decoded_domains = [list(set(i)) for i in decoded_domains]

        return decoded_domains

    def load_jsonl(self, file_path):
        with open(file_path, "r") as f:
            return [json.loads(line) for line in f]

    def clean_decoded_domains(self, decoded_domains):
        decoded_domains = [i.split(',') for i in decoded_domains]
        decoded_domains = [[j.strip() for j in i] for i in decoded_domains]
        decoded_domains = [list(set(i)) for i in decoded_domains]

        return decoded_domains

    def save_final_metric(self, average_score, model_name, eval_samples, eval_data_file_name):
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
