import json
import torch
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Initialize MLflow
mlflow.set_experiment("Domain Generation Experiment")

# Load model and tokenizer
model_name = "google/flan-t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load your training data
def load_data(file_path):
    """Load JSONL data"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Example data format (replace with your actual data loading)
# training_data = [
#     {"input": "established affordable home goods option", "output": "homeapp.org, boutiquedepot.dev, superboutique.ai, shopoutlet.ly, storegallery.dev"},
#     {"input": "premium coffee roasting business", "output": "roastcraft.com, premiumbean.co, craftroast.io, beanmaster.net, roasthaus.com"},
#     # Add more examples...
# ]

# Generate dataset
training_data = []
total_samples = 5000
with open(f"training_data_{total_samples}.jsonl", "r") as f:
    for line in f:
        training_data.append(json.loads(line))

# Format data for T5 (instruction format)
def format_data(examples):
    inputs = []
    targets = []
    
    for example in examples:
        # Create instruction-style input
        input_text = f"Generate domain names for this business: {example['input']}"
        inputs.append(input_text)
        targets.append(example['output'])
    
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
        output_dir="./flan-t5-domain-generator",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-4,  # Higher learning rate for T5
        warmup_steps=100,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb logging
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
    trainer.save_model(f"./models/flan-t5-domain-generator-final-{total_samples}")
    tokenizer.save_pretrained(f"./models/flan-t5-domain-generator-final-{total_samples}")
    print("Training completed and model saved!")


    input_example = {
        "input_text": ["Generate domain names for this business: premium coffee roasting business"]
    }
    # Log model
    mlflow.pytorch.log_model(model, f"model-{total_samples}", input_example=input_example)

    #Use data_eval.jsonl to evaluate the model and create the evaluation metrics, 
    #accuracy, precision, recall, f1-score, etc.
    #and log the metrics to mlflow
    validation_data = []
    with open("data_eval.jsonl", "r") as f:
        for line in f:
            validation_data.append(json.loads(line))
    # Extract inputs and targets from validation data
    validation_inputs = [item["input"] for item in validation_data]
    validation_targets = [item["output"] for item in validation_data]

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


    # use fine tuned model to generate domains for the validation dataset
    fine_tuned_model = T5ForConditionalGeneration.from_pretrained(f"./models/flan-t5-domain-generator-final-{total_samples}")
    generated_domains = fine_tuned_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_return_sequences=1
        )
    decoded_domains = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_domains]

    
    print("\n\ngenerated_domains", decoded_domains)

    # evaluate the generated domains
    # calculate the accuracy, precision, recall, f1-score, etc.
    # log the metrics to mlflow
    accuracy = accuracy_score(validation_targets, decoded_domains)
    precision = precision_score(validation_targets, decoded_domains, average='macro')
    recall = recall_score(validation_targets, decoded_domains, average='macro')
    f1 = f1_score(validation_targets, decoded_domains, average='macro')
    print("\n\naccuracy", accuracy)
    print("\n\nprecision", precision)
    print("\n\nrecall", recall)
    print("\n\nf1", f1)

    mlflow.log_metric("eval_accuracy", accuracy)
    mlflow.log_metric("eval_precision", precision)
    mlflow.log_metric("eval_recall", recall)
    mlflow.log_metric("eval_f1", f1)
