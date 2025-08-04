import os
import openai
import json
import boto3
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
from typing import List, Dict
import re
import numpy as np
from pydantic import BaseModel


# Global variables for caching
model = None
tokenizer = None
s3_client = boto3.client('s3')

BUCKET_NAME = 'mlpcacourts'
MODEL_PREFIX = 'models/flan-t5-domain-generator-final-5000/'
LOCAL_MODEL_PATH = '/tmp/model'

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def download_model_from_s3():
    """Download all model files from S3 to /tmp"""
    print("Downloading model from S3...")
    start_time = time.time()
    
    # Create local directory
    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
    
    # List all files in the model directory
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=MODEL_PREFIX
    )
    
    if 'Contents' not in response:
        raise Exception(f"No model files found at s3://{BUCKET_NAME}/{MODEL_PREFIX}")
    
    # Download each file
    for obj in response['Contents']:
        s3_key = obj['Key']
        # Get filename relative to model prefix
        filename = s3_key.replace(MODEL_PREFIX, '')
        
        if filename:  # Skip if it's just the prefix/directory
            local_file_path = os.path.join(LOCAL_MODEL_PATH, filename)
            
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            print(f"Downloading {filename}...")
            s3_client.download_file(BUCKET_NAME, s3_key, local_file_path)
    
    download_time = time.time() - start_time
    print(f"Model downloaded in {download_time:.2f} seconds")

def load_model():
    """Load model from /tmp, downloading from S3 if needed"""
    global model, tokenizer
    
    if model is None:
        # Check if model exists locally, if not download it
        if not os.path.exists(LOCAL_MODEL_PATH) or not os.listdir(LOCAL_MODEL_PATH):
            download_model_from_s3()
        
        print("Loading model into memory...")
        start_time = time.time()
        
        # Load tokenizer and model
        tokenizer = T5Tokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        model = T5ForConditionalGeneration.from_pretrained(
            LOCAL_MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto"
        )
        
        model.eval()
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
    
    return model, tokenizer

def lambda_handler(event, context):
    try:
        #Load model (downloads from S3 on first invocation, cached afterward)
        #model, tokenizer = load_model()
        
        #Get input parameters
        print(f"Event: {event}")
        input_text = json.loads(event['body']).get('business_description', '')
        print(f"Input text: {input_text}")
 
        if not input_text:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'input_text is required'})
            }
        
        print(f"Generating text for input: {input_text}...")
        is_flagged, flagged_categories = openai_moderation(input_text)
        if is_flagged:
            return {
                "suggestions": [],
                "status": "blocked",
                "message": "Request contains inappropriate content"
                }
        
        generated_domains = query_model([input_text])[0]
        print(f"Text generation completed in seconds")
        print(f"Generated domains: {generated_domains}")

        if len(generated_domains) == 1 and len(generated_domains[0]) == 0:
            return {
                "suggestions": [],
                "status": "blocked",
                "message": "Request contains inappropriate content"
                }

        domain_with_confidence_score = calculate_domains_confidence_score(input_text, generated_domains)

        print(f"Confidence score per domain: {domain_with_confidence_score}")
        
        return {
                "suggestions": domain_with_confidence_score,
                "status": "success",
                }

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'model_loaded': model is not None
            })
        }

def query_model(input_texts):
    fine_tuned_model, tokenizer = load_model()
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

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

def openai_moderation(business_desc: str):
    '''
    This checks the domains for inappropriate content.
    '''
    response = openai_client.moderations.create(
        model="omni-moderation-latest",
        input=business_desc
    )
    is_flagged = response.results[0].flagged
    flagged_categories = response.results[0].categories

    return is_flagged, flagged_categories


def calculate_domains_confidence_score(business_desc: str, domains: List[str]) -> List[Dict]:
    '''
    This calculates the confidence score for the domains.
    '''
    rule_based_evaluation = _rule_based_evaluation(business_desc, domains)
    openai_evaluation = _openai_evaluation(business_desc, domains)
    domain_confidence_scores = [{'domain': domain_score['domain'], 'confidence': ((domain_score['score']+openai_evaluation['domain_scores'][i]['score'])/2)/10} for i, domain_score in enumerate(rule_based_evaluation['domain_scores'])]
    
    return domain_confidence_scores


def _rule_based_evaluation(business_desc: str, domains: List[str]) -> Dict:
    #Simple rule-based scoring
    scores = []
    for domain in domains:
        score = 0
        
        #Check relevance (keywords from business description)
        business_words = set(business_desc.lower().split())
        domain_words = set(re.findall(r'[a-zA-Z]+', domain.lower()))
        overlap = len(business_words.intersection(domain_words))
        score += overlap * 2
        
        #Check length (prefer 6-15 characters before extension)
        domain_name = domain.split('.')[0]
        if 6 <= len(domain_name) <= 15:
            score += 3
        
        #Check for common extensions
        if domain.endswith(('.com', '.org', '.net')):
            score += 2
        
        #Avoid numbers and hyphens
        if not re.search(r'[0-9-]', domain_name):
            score += 1
            
        scores.append(min(score, 10))
    
    return {
        'overall_score': np.mean(scores) if scores else 0,
        'domain_scores': [{'domain': d, 'score': s} for d, s in zip(domains, scores)],
        'evaluation_method': 'rule_based'
    }


class DomainScore(BaseModel):
    domain: str
    score: int

class DomainEvaluation(BaseModel):
    overall_score: float
    domain_scores: List[DomainScore]

    
def _openai_evaluation(business_desc: str, domains: List[str]) -> Dict:
    domains_str = "\n".join([f"{i+1}. {domain}" for i, domain in enumerate(domains)])
    prompt = f"""
    Evaluate these domain names for the business: "{business_desc}"
    Domains:
    {domains_str}
    Rate each domain 1-10 on: relevance, memorability, brandability.
    Return JSON format:
    {{"overall_score": float, "domain_scores": [{{"domain": "name", "score": int}}]}}
    """

    #force the json format in response
    
    try:
        response = openai_client.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format=DomainEvaluation
        )
        result = json.loads(response.choices[0].message.content)
        result['evaluation_method'] = 'openai_prompt'
        return result
    except Exception as e:
        print(f"OpenAI evaluation failed: {e}")
        return
