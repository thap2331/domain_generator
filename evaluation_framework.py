import re
import json
import numpy as np
from typing import List, Dict
import openai
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI



load_dotenv()
openai_client = OpenAI()


class DomainScore(BaseModel):
    domain: str
    score: int

class DomainEvaluation(BaseModel):
    overall_score: float
    domain_scores: List[DomainScore]

class QuickEvaluator:
    '''
    1. Rule based evaluation: This evaluates the domains based on a set of rules such as:
        - Relevance (keywords from business description)
        - Length (prefer 6-15 characters before extension)
        - Common extensions (com, org, net)
        - Avoid numbers and hyphens
    2. OpenAI evaluation: This evaluates the domains based on a prompt to the OpenAI API.
        - Relevance (keywords from business description)
    3. Similarity check: This checks the similarity between the business description and the domains
        using embeddings.
    4. Openai moderation api: This checks the domains for inappropriate content.
    Create a score for each domain based on the above methods.
    '''

    
    def _rule_based_evaluation(self, business_desc: str, domains: List[str], industry: str=None) -> Dict:
        #Simple rule-based scoring
        scores = []
        if industry == "adult_entertainment" and len(domains) == 0:
            return {
                'overall_score': 10,
                'domain_scores': [],
                'evaluation_method': 'rule_based'
            }
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

    def rule_based_evaluation_and_openai_moderation(self, business_desc: str, domains: List[str]) -> Dict:
        #check for openai moderation
        is_flagged, flagged_categories = self._openai_moderation(business_desc)
        # print(f"is_flagged: {is_flagged}, flagged_categories: {flagged_categories}, domains: {domains}, business_desc: {business_desc}")
        if len(domains) == 0 and len(domains[0]) == 0 and is_flagged:
            return {
                'overall_score': 10,
                'domain_scores': [],
                'evaluation_method': 'fine_tuned_rule_based_and_openai_moderation',
                'flagged_categories': flagged_categories,
                'is_flagged': is_flagged
            }
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
            'evaluation_method': 'fine_tuned_rule_based_and_openai_moderation'
        }

    
    def _openai_evaluation(self, business_desc: str, domains: List[str]) -> Dict:
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


    def similarity_check(self, business_desc: str, domains: List[str]) -> Dict:
        '''
        This is almost the same as the openai evaluation, but keeping it here for future reference
        '''

        #get all embeddings in a single request to make it cheaper
        embeddings = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[business_desc, * domains]
        )
        #get the embedding of the business description
        business_desc_embedding = embeddings.data[0].embedding
        
        #get the embedding of the domains
        domains_embeddings = [embedding.embedding for embedding in embeddings.data[1:]]
        

        #get the cosine similarity score between business description and domains for each domain
        similarity_scores = []
        for i in range(len(domains)):
            similarity_scores.append(np.dot(business_desc_embedding, domains_embeddings[i]) / (np.linalg.norm(business_desc_embedding) * np.linalg.norm(domains_embeddings[i])))
        
        #get a dictionary of domain and similarity score
        similarity_scores_dict = {domains[i]: similarity_scores[i] for i in range(len(domains))}
        return similarity_scores_dict


    def _openai_moderation(self, business_desc: str) -> Dict:
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


    def calculate_confidence_score(self, business_desc: str, domains: List[str], industry: str=None) -> float:
        '''
        This calculates the confidence score for the domains.
        '''
        rule_based_evaluation = self._rule_based_evaluation(business_desc, domains, industry)
        openai_evaluation = self._openai_evaluation(business_desc, domains)
        domain_confidence_scores = [{'domain': domain_score['domain'], 'confidence': ((domain_score['score']+openai_evaluation['domain_scores'][i]['score'])/2)/10} for i, domain_score in enumerate(rule_based_evaluation['domain_scores'])]
        
        return domain_confidence_scores

    def calculate_overall_score(self, business_desc: str, domains: List[str], industry: str=None) -> float:
        '''
        This calculates the overall score for the domains.
        '''
        rule_based_evaluation = self._rule_based_evaluation(business_desc, domains, industry)
        openai_evaluation = self._openai_evaluation(business_desc, domains)
        #get in decimal form, i.e., divvy by 10 
        score = ((rule_based_evaluation['overall_score'] + openai_evaluation['overall_score']) / 2)/10

        return score

    def fine_tuned_calculate_overall_score(self, business_desc: str, domains: List[str]) -> float:
        '''
        This calculates the overall score for the domains.
        '''
        scores = {}
        openai_evaluation = self._openai_evaluation(business_desc, domains)
        rule_and_openai_moderation_score = self.rule_based_evaluation_and_openai_moderation(business_desc, domains)
        score = float(((openai_evaluation['overall_score'] + rule_and_openai_moderation_score['overall_score']) / 2)/10)
        scores['overall_score'] = score
        if rule_and_openai_moderation_score.get('is_flagged', None):
            scores['flagged_categories'] = rule_and_openai_moderation_score.get('flagged_categories')
            scores['is_flagged'] = rule_and_openai_moderation_score.get('is_flagged')

        return scores


def run_comprehensive_test():
    """Run a comprehensive test of the domain generation system"""
    
    print("=== COMPREHENSIVE DOMAIN GENERATOR TEST ===\n")
    
    # Initialize components
    evaluator = QuickEvaluator()  # Will use rule-based evaluation
    #edge_detector = EdgeCaseDetector()
    #load test cases from data/eval-data/data_eval_100.json
    with open('data/eval-data/data_eval_100.json', 'r') as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test cases")

    
    results = []
    
    # for i, data_dict in enumerate(test_cases[42:46]):
    #     business_desc = data_dict["business_description"]
    #     #empty list if no domains are generated
    #     output_domains = data_dict["domain_suggestions"]
    #     industry = data_dict["industry"]
    #     # if industry == "adult_entertainment":
    #     #     print('Here', i, business_desc, output_domains)
    #     print("-" * 50)
    #     print(f"Test Case {i+1}: {business_desc}; and output domains: {output_domains}")

    #     # Evaluate domains
    #     rule_based_evaluation = evaluator._rule_based_evaluation(business_desc, output_domains, industry)
    #     openai_evaluation = evaluator._openai_evaluation(business_desc, output_domains)
    #     similarity_scores = evaluator.similarity_check(business_desc, output_domains)
    #     print(f"\nRule based evaluation: {rule_based_evaluation}")
    #     print(f"\nOpenAI evaluation: {openai_evaluation}")
    #     print(f"\nSimilarity scores: {similarity_scores}")
        
        
        # Check for safety issues first
        # edge_issues = edge_detector.detect_issues(business_desc, [])
        
        # if edge_issues['inappropriate_request']:
        #     print("❌ BLOCKED: Inappropriate content detected")
        #     result = {
        #         'business_description': business_desc,
        #         'status': 'blocked',
        #         'domains': [],
        #         'evaluation': {'overall_score': 0},
        #         'edge_issues': edge_issues
        #     }
        # else:
        #     # Generate domains (using fallback method since model isn't trained yet)
        #     domains = generate_fallback_domains(business_desc)
        #     print(f"Generated domains: {domains}")
            
        #     # Evaluate domains
        #     evaluation = evaluator.evaluate_domains(business_desc, domains)
        #     print(f"Overall score: {evaluation['overall_score']:.2f}")
            
        #     # Check for edge cases
        #     edge_issues = edge_detector.detect_issues(business_desc, domains)
            
        #     if edge_issues['invalid_domains']:
        #         print(f"⚠️  Invalid domains detected: {edge_issues['invalid_domains']}")
        #     if edge_issues['poor_quality_domains']:
        #         print(f"⚠️  Poor quality domains: {edge_issues['poor_quality_domains']}")
            
        #     result = {
        #         'business_description': business_desc,
        #         'status': 'success',
        #         'domains': domains,
        #         'evaluation': evaluation,
        #         'edge_issues': edge_issues
        #     }
        
        # results.append(result)
        # print("\n")
    
    # return results
# run_comprehensive_test()

class RunEvals:
    def __init__(self):
        self.evaluator = QuickEvaluator()  # Will use rule-based evaluation

    def single_eval(self, business_desc: str, domains: List[str], industry: str=None, get_overall_score: bool=False) -> Dict:
        '''
        This evaluates a single business description and domains.
        '''
        domain_confidence_scores = self.evaluator.calculate_confidence_score(business_desc, domains, industry)
        evaluation_results = {
            'domain_confidence_scores': domain_confidence_scores,
            'overall_score': None
        }
        overall_score = self.evaluator.calculate_overall_score(business_desc, domains, industry)
        if get_overall_score:
            evaluation_results['overall_score'] = overall_score
        
        return evaluation_results


    def combined_eval(self, business_desc: str, domains: List[str], industry: str=None) -> Dict:
        '''
        This evaluates a single business description and domains.
        '''
        combined_score = float(self.evaluator.calculate_overall_score(business_desc, domains, industry))
        r = {'overall_score': combined_score}

        return r

    def model_eval(self, business_desc: str, domains: List[str]) -> Dict:
        '''
        This evaluates a single business description and domains.
        '''
        combined_score = float(self.evaluator.rule_based_evaluation_and_openai_moderation(business_desc, domains))

        r = {'overall_score': combined_score}

        return r