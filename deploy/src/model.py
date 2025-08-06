import os
import openai
import json
import re
import requests
from typing import List, Dict
from pydantic import BaseModel
import numpy as np

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

        payload = {
                        "inputs": [
                            input_text
                        ],
                        "parameters": {
                            "max_new_tokens": 256,
                            "temperature": 0.1,
                            "return_full_text": False
                        }
                    }
        
        generated_domains_str = query_model(payload)
        generated_domains = generated_domains_str.split(',')
        generated_domains = list(set([i.strip() for i in generated_domains]))
        print(f"Text generation completed in seconds")
        print(f"Generated domains: {generated_domains}")

        if len(generated_domains) == 1 and len(generated_domains[0]) == 0:
            return {
                "suggestions": [],
                "status": "blocked",
                "message": "Request contains inappropriate content"
                }

        #remove empty domains
        generated_domains = [domain for domain in generated_domains if domain]
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
            })
        }

def query_model(payload):
    headers = {
        "Accept" : "application/json",
        "Content-Type": "application/json" 
    }

    response = requests.post(
        "https://fi31ip7vhnqcjfg7.us-east-1.aws.endpoints.huggingface.cloud",
        headers=headers,
        json=payload
    )

    return response.json()[0][0]['generated_text']

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
