from transformers import T5Tokenizer, T5ForConditionalGeneration
import json

class Utils:
    def __init__(self):
        samples = 5000
        self.model_name = f'flan-t5-domain-generator-final-{samples}'
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