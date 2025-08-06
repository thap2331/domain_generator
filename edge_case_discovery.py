import csv
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from openai import OpenAI
import time
import re
from datetime import datetime
import json
from dotenv import load_dotenv
from utils import Utils
import os

load_dotenv()
openai_client = OpenAI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FailureType(Enum):
    FACTUAL_ERROR = "factual_error"
    REASONING_ERROR = "reasoning_error"
    SAFETY_ISSUE = "safety_issue"
    HALLUCINATION = "hallucination"
    FORMAT_ERROR = "format_error"
    REFUSAL_ERROR = "refusal_error"
    CONSISTENCY_ERROR = "consistency_error"
    INVALID_DOMAIN = "invalid_domain"
    POOR_QUALITY_DOMAIN = "poor_quality_domain"
    NO_DOMAIN_GENERATION = "no_domain_generation"

class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class EdgeCase:
    id: str
    input_text: str
    predicted_domains: str
    failure_type: FailureType
    severity: Severity
    category: str
    root_cause: str
    improvement_strategy: str
    timestamp: str
    metadata: Dict[str, Any]

class EdgeCaseDiscovery:
    def __init__(self, model_name: str):
        self.edge_cases: List[EdgeCase] = []
        self.test_categories = [
            "boundary_tests",
            "adversarial_tests", 
            "ambiguous_tests",
            "format_tests",
            "safety_tests"
        ]
        self.inappropriate_terms = [
            "porn",
            "sex",
            "nude",
            "naked",
            "xxx",
        ]
        self.utils = Utils(model_name=model_name)
        self.model_name = model_name

    def generate_test_cases(self, category: str, num_cases: int = 10) -> List[Dict[str, str]]:
        with open(f"data/discovery.json", 'r') as f:
            data = json.load(f)
        return data[category][:num_cases]
    
    def manual_inappropriate_checks(self, predicted_domains: List[str]) -> List[str]:
        inappropriate_domains = []
        for domain in predicted_domains:
            if any(term.lower() in domain.lower() for term in self.inappropriate_terms):
                inappropriate_domains.append(domain)
        
        return inappropriate_domains

    def openai_moderation(self, business_desc: str) -> Dict:
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

    def _check_invalid_domains(self, domains: List[str]) -> List[str]:
        invalid = []
        for domain in domains:
            # Check for invalid characters or patterns
            if not re.match(r'^[a-zA-Z0-9-]+\.[a-zA-Z]{2,}', domain):
                invalid.append(domain)
            elif len(domain.split('.')[0]) > 30:
                invalid.append(domain)
            elif domain.startswith('-') or domain.endswith('-'):
                invalid.append(domain)
        return invalid
    
    def _check_poor_quality(self, domains: List[str]) -> List[str]:
        poor_quality = []
        for domain in domains:
            domain_name = domain.split('.')[0]
            # Too many numbers or special chars
            if len(re.findall(r'[0-9]', domain_name)) > 2:
                poor_quality.append(domain)
            # Too short or too long
            elif len(domain_name) < 3 or len(domain_name) > 20:
                poor_quality.append(domain)
        return poor_quality
    
    
    def evaluate_response(self, input_text: str, predicted_domains: list, category: str) -> EdgeCase:
        """Evaluate if the response represents an edge case failure"""
        
        # Simple evaluation logic - enhance this based on your needs
        is_failure = False
        failure_type = FailureType.FACTUAL_ERROR
        severity = Severity.LOW
        root_cause = "Unknown"
        improvement_strategy = "Add to training data"
        invalid_domains = self._check_invalid_domains(predicted_domains)
        poor_quality_domains = self._check_poor_quality(predicted_domains)
        check_openai_moderation = self.openai_moderation(input_text)
        inappropriate_domains = self.manual_inappropriate_checks(predicted_domains)
        
        # Enhanced failure detection for domain name suggestions
        if not isinstance(predicted_domains, list):
            is_failure = True
            failure_type = FailureType.FORMAT_ERROR
            severity = Severity.HIGH
            root_cause = "Domain list is not a list error"
        elif len(predicted_domains) == 0 and len(predicted_domains[0]) == 0:
            is_failure = True
            failure_type = FailureType.FORMAT_ERROR
            severity = Severity.HIGH
            root_cause = "Could not generate any domains. This could be because of adult content. Check the input."
        elif invalid_domains:
            is_failure = True
            failure_type = FailureType.INVALID_DOMAIN
            severity = Severity.HIGH
            root_cause = "Invalid domains found in the response"
        elif poor_quality_domains:
            is_failure = True
            failure_type = FailureType.POOR_QUALITY_DOMAIN
            severity = Severity.HIGH
            root_cause = "Poor quality domains found in the response"
        elif check_openai_moderation[0]:
            is_failure = True
            failure_type = FailureType.SAFETY_ISSUE
            severity = Severity.HIGH
            root_cause = "Content flagged by OpenAI moderation"
        elif inappropriate_domains:
            is_failure = True
            failure_type = FailureType.SAFETY_ISSUE
            severity = Severity.HIGH
            root_cause = "Inappropriate domains checked by manual check"
        
        if is_failure:
            edge_case = EdgeCase(
                id=f"edge_{len(self.edge_cases)}_{int(time.time())}",
                input_text=input_text,
                predicted_domains=predicted_domains,
                failure_type=failure_type,
                severity=severity,
                category=category,
                root_cause=root_cause,
                improvement_strategy=improvement_strategy,
                timestamp=datetime.now().isoformat(),
                metadata={"evaluation_version": "1.0"}
            )
            # print(f"\n in evaluate_response, Edge case: {edge_case}")
            return edge_case
        
        return None
    
    def run_discovery(self, categories: List[str] = None, cases_per_category: int = 10) -> List[EdgeCase]:
        """Run systematic edge case discovery"""
        if categories is None:
            categories = self.test_categories
        
        discovered_cases = []
        
        for category in categories:
            logger.info(f"Testing category: {category}")
            test_cases = self.generate_test_cases(category, cases_per_category)
            print(f"\n in run_discovery, Test cases: {test_cases}")
            
            for test_case in test_cases:
                logger.info(f"Testing: {test_case['input'][:50]}...")
                
                # Query the model
                predicted_domains = self.utils.query_model([test_case['input']])[0]
                print(f"\n in run_discovery, Actual response: {predicted_domains}")
                
                # Evaluate for edge cases
                edge_case = self.evaluate_response(
                    test_case['input'],
                    predicted_domains,
                    test_case['category']
                )
                
                if edge_case:
                    discovered_cases.append(edge_case)
                    logger.warning(f"Edge case discovered: {edge_case.failure_type.value}")
                
                # Rate limiting
                time.sleep(0.5)
        
        self.edge_cases.extend(discovered_cases)
        return discovered_cases
    
    def analyze_failures(self) -> Dict[str, Any]:
        """Analyze discovered edge cases and generate insights"""
        if not self.edge_cases:
            return {"message": "No edge cases found"}
        
        analysis = {
            "total_cases": len(self.edge_cases),
            "by_failure_type": {},
            "by_severity": {},
            "by_category": {},
            "improvement_strategies": {},
            "root_causes": {}
        }
        
        for case in self.edge_cases:
            # Count by failure type
            ft = case.failure_type.value
            analysis["by_failure_type"][ft] = analysis["by_failure_type"].get(ft, 0) + 1
            
            # Count by severity
            sev = case.severity.name
            analysis["by_severity"][sev] = analysis["by_severity"].get(sev, 0) + 1
            
            # Count by category
            cat = case.category
            analysis["by_category"][cat] = analysis["by_category"].get(cat, 0) + 1
            
            # Count improvement strategies
            strat = case.improvement_strategy
            analysis["improvement_strategies"][strat] = analysis["improvement_strategies"].get(strat, 0) + 1
            
            # Count root causes
            cause = case.root_cause
            analysis["root_causes"][cause] = analysis["root_causes"].get(cause, 0) + 1
        
        return analysis
    
    def export_results(self, filename: str = None):
        """Export edge cases and analysis to files"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"edge_cases_{timestamp}"
        
        # Export edge cases as JSON
        cases_data = [asdict(case) for case in self.edge_cases]
        for case_data in cases_data:
            case_data['failure_type'] = case_data['failure_type'].value
            case_data['severity'] = case_data['severity'].value
        
        os.makedirs(f"./data/edge-cases/{self.model_name}", exist_ok=True)
        with open(f"./data/edge-cases/{self.model_name}/{filename}_cases.json", 'w') as f:
            json.dump(cases_data, f, indent=2)
        
        # Export analysis
        analysis = self.analyze_failures()
        if not os.path.exists("./data/edge-cases"):
            os.makedirs("./data/edge-cases")
        with open(f"./data/edge-cases/{self.model_name}/{filename}_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Results exported to {filename}_*.json")
    
    def generate_training_data(self) -> List[Dict[str, str]]:
        """Generate training data from edge cases for fine-tuning"""
        training_data = []
        
        for case in self.edge_cases:
            if case.severity in [Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]:
                # Create training example with corrected response
                training_example = {
                    "input": case.input_text,
                    "metadata": {
                        "source": "edge_case_discovery",
                        "failure_type": case.failure_type.value,
                        "original_output": case.predicted_domains
                    }
                }
                training_data.append(training_example)
        
        return training_data

def main():
    """Example usage for domain name suggestion testing"""
    samples = 5000
    model_name = f'flan-t5-domain-generator-final-{samples}'
    discovery = EdgeCaseDiscovery(model_name=model_name)
    
    # Run discovery on domain name specific categories
    logger.info("Starting domain name suggestion edge case discovery...")
    discovery.run_discovery(
        categories=["boundary_tests", "adversarial_tests", "ambiguous_tests", "format_tests", "safety_tests"],
        cases_per_category=2
    )
    
    # Analyze results
    analysis = discovery.analyze_failures()
    print("\nDomain Name Suggestion Analysis Results:")
    print(json.dumps(analysis, indent=2))
    
    # Export results
    discovery.export_results("domain_name_edge_cases")
    
    # Generate training data for fine-tuning
    training_data = discovery.generate_training_data()
    print(f"\nGenerated {len(training_data)} training examples from domain name edge cases")
    
    # Save training data
    os.makedirs(f"./data/edge-cases/data-addendum/{model_name}", exist_ok=True)
    #save as jsonl
    with open(f"./data/edge-cases/data-addendum/{model_name}/domain_name_training_data.jsonl", 'w') as f:
        for i in training_data:
            f.write(json.dumps(i) + "\n")
    
    print("\nExample training data:")
    for i, example in enumerate(training_data[:3]):
        print(f"\nExample {i+1}:")
        print(f"Input: {example['input']}")
        print(f"Failure Type: {example['metadata']['failure_type']}")

if __name__ == "__main__":
    main()