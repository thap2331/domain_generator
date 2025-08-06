import json
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
import os
from utils import Utils
from evaluation_framework import QuickEvaluator

eval = QuickEvaluator()
utils = Utils()

# Initialize MLflow
mlflow.set_experiment("Domain Generation Experiment")

# Load model and tokenizer
model_name = f"flan-t5-domain-generator-final-5000"
model = T5ForConditionalGeneration.from_pretrained(f"./models/{model_name}")
tokenizer = T5Tokenizer.from_pretrained(f"./models/{model_name}")

# Load your training data
training_data_path = "data/edge-cases/data-addendum/training-data.jsonl"
training_data = Utils().load_jsonl(file_path=training_data_path)

# Format data for T5 (instruction format)
def format_data(data):
    inputs = []
    targets = []
    
    for i in data:
        # Create instruction-style input
        input_text = f"Generate domain names for this business: {i['input']}"
        inputs.append(input_text)
        targets.append(i['output'])
    
    return {"input_text": inputs, "target_text": targets}

# Prepare datasets
formatted_data = format_data(training_data)
train_data, eval_data = train_test_split(
    list(zip(formatted_data["input_text"], formatted_data["target_text"])), 
    test_size=0.1, 
    random_state=42
)

train_inputs, train_targets = zip(*train_data)
eval_inputs, eval_targets = zip(*eval_data)

train_dataset = Dataset.from_dict({
    "input_text": train_inputs,
    "target_text": train_targets
})
print("train_dataset", train_dataset[0:10])
eval_dataset = Dataset.from_dict({
    "input_text": eval_inputs,
    "target_text": eval_targets
})
print("\n\neval_dataset", eval_dataset[0:10])

# Tokenization function
def preprocess_function(examples):
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=128,
        truncation=True,
        padding=True
    )
    
    # Tokenize targets
    labels = tokenizer(
        examples["target_text"],
        max_length=128,
        truncation=True,
        padding=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Validation preprocessing function
def preprocess_validation_function(examples):
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=128,
        truncation=True,
        padding=True
    )
    
    # Tokenize targets
    labels = tokenizer(
        examples["target_text"],
        max_length=128,
        truncation=True,
        padding=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_version", mlflow.active_run().info.run_id)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("train_size", len(train_dataset))
    mlflow.log_param("eval_size", len(eval_dataset))
    
    # Training code
    # Training arguments optimized for creativity
    training_args = TrainingArguments(
        output_dir="./flan-t5-domain-generator-iterative",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-4,  # Higher learning rate for T5
        warmup_steps=10,
        logging_steps=5,
        eval_steps=15,
        save_steps=15,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        dataloader_pin_memory=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the model
    print("Saving model...")
    #create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)
    old_train_samples = 5000
    new_training_samples = len(training_data)
    total_samples = old_train_samples + new_training_samples
    trainer.save_model(f"./models/flan-t5-domain-generator-final-{total_samples}")
    tokenizer.save_pretrained(f"./models/flan-t5-domain-generator-final-{total_samples}")
    print("Training completed and model saved!")


    input_example = {
        "input_text": ["Generate domain names for this business: premium coffee roasting business"]
    }
    # Log model
    mlflow.pytorch.log_model(model, f"model-{total_samples}", input_example=input_example)

    print("Predicting domains and running evaluation")
    validation_data = []
    eval_samples = 100
    eval_data_file_name = "./data/eval-data/data_eval_100.json"
    with open(eval_data_file_name, "r") as f:
        validation_data = json.load(f)
    # Extract inputs and targets from validation data
    validation_inputs = [item["business_description"] for item in validation_data]
    validation_targets = [item["domain_suggestions"] for item in validation_data]

    # Create validation dataset
    validation_dataset = Dataset.from_dict({
        "input_text": validation_inputs,
        "target_text": validation_targets
    })
    input_texts = list(validation_dataset["input_text"])
    # Convert input texts to input IDs and attention masks
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",  # Return PyTorch tensors
        padding=True,
        truncation=True,
        max_length=128
    )

    # use fine tuned model to predict domains for the validation dataset
    fine_tuned_model_name = f"flan-t5-domain-generator-final-{total_samples}"
    fine_tuned_model = T5ForConditionalGeneration.from_pretrained(f"./models/{fine_tuned_model_name}")
    generated_domains = fine_tuned_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_return_sequences=1
        )
    decoded_domains = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_domains]
    #save the data to a file
    file_name = f"./data/eval-data/predicted_domains_and_ground_truth_{total_samples}_{fine_tuned_model_name}.json"
    with open(file_name, "w") as f:
        json.dump(decoded_domains, f)

    #evaluate the generated domains
    all_domains = utils.clean_decoded_domains(decoded_domains)
    all_data, scores = [], []
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
    utils.save_final_metric(average_score, fine_tuned_model_name, eval_samples, eval_data_file_name)
    #log the average score to mlflow
    mlflow.log_metric(f"average_score_on_eval_data_{eval_samples}", average_score)

    #save the data to a json file
    with open(file_name, "w") as f:
        json.dump(all_data, f, indent=4)

