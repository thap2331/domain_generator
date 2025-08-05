import random
import json
import re
from typing import List, Dict, Tuple


class BusinessDomainGenerator:
    def __init__(self, industries_file='industries.json'):
        # Industry-specific data
        self.industries = self.load_json(industries_file)
        
        # Business description templates - shorter and more varied
        self.description_templates = [
            "{service} for {target}",
            "{keyword} specializing in {service}",
            "Professional {service} company",
            "{service} in {location}",
            "Premium {service} provider",
            "Local {keyword} offering {service}",
            "{industry_type} {keyword}",
            "Modern {service} solutions",
            "{target}-focused {service}",
            "Boutique {keyword} serving {target}",
            "Online {service} platform",
            "Custom {service} for {target}",
            "{service} with {benefit}",
            "Affordable {service} option",
            "High-end {keyword} experience"
        ]
        
        # Location descriptors for more realistic descriptions
        self.locations = [
            'downtown area', 'city center', 'suburb', 'business district', 'shopping mall',
            'strip mall', 'main street', 'waterfront', 'historic district', 'residential area',
            'industrial zone', 'commercial plaza', 'uptown', 'midtown', 'airport area'
        ]
        
        # Industry types for simple descriptions
        self.industry_types = [
            'family-owned', 'boutique', 'luxury', 'discount', 'premium', 'organic', 'sustainable',
            'eco-friendly', 'artisan', 'handcrafted', 'custom', 'personalized', 'traditional',
            'modern', 'innovative', 'cutting-edge', 'established', 'award-winning', 'certified'
        ]
        
        # Company name generators
        self.company_prefixes = ['Smart', 'Quick', 'Pro', 'Elite', 'Prime', 'Apex', 'Digital', 'Modern', 'Advanced', 'Optimal']
        self.company_suffixes = ['Solutions', 'Systems', 'Technologies', 'Innovations', 'Labs', 'Works', 'Hub', 'Group', 'Partners', 'Dynamics']
        
        
        self.tlds = ['.com', '.io', '.co', '.tech', '.app', '.dev', '.ai', '.ly', '.me', '.org']

    def load_json(self, filename: str) -> Dict:
        """Load JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)

    def clean_for_domain(self, text: str) -> str:
        """Clean text for domain name use"""
        # Remove spaces, special characters, make lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', text.lower())
        return cleaned

    def generate_business_description(self, industry: str) -> Tuple[str, str]:
        """Generate a business description for a specific industry"""
        if industry not in self.industries:
            industry = random.choice(list(self.industries.keys()))
        
        industry_data = self.industries[industry]
        
        #Select random elements
        service = random.choice(industry_data['services'])
        target = random.choice(industry_data['targets'])
        benefit = random.choice(industry_data['benefits'])
        keyword = random.choice(industry_data['keywords'])
        location = random.choice(self.locations)
        industry_type = random.choice(self.industry_types)
        
        # Generate company name
        company_name = f"{random.choice(self.company_prefixes)}{random.choice(self.company_suffixes)}"
        
        # Select and fill template
        template = random.choice(self.description_templates)
        
        # Add different template styles
        description = template.format(
            service=service,
            target=target,
            benefit=benefit,
            keyword=keyword,
            location=location,
            industry_type=industry_type,
            company_name=company_name
        )
        
        # Sometimes add simple qualifiers for more realistic variety
        qualifiers = ['', 'new ', 'established ', 'local ', 'independent ', 'family-owned ', 'premium ']
        if random.random() < 0.3:  # 30% chance to add qualifier
            qualifier = random.choice(qualifiers)
            if not description.startswith(('New ', 'Established ', 'Local ', 'Independent ', 'Family-owned ', 'Premium ')):
                description = qualifier + description.lower()
        
        return description, industry

    def generate_domain_suggestions(self, description: str, industry: str, count: int = 5) -> List[str]:
        """Generate domain name suggestions based on business description"""
        industry_data = self.industries.get(industry, self.industries['saas'])

        if industry == 'adult_entertainment':
            return None
        else:
            domains = []
            
            # Extract keywords from description
            description_words = re.findall(r'\b[a-zA-Z]{3,}\b', description.lower())
            key_words = [word for word in description_words if len(word) > 3 and word not in ['help', 'them', 'with', 'through', 'using', 'provide', 'offer', 'deliver']]
            
            # Generate domains using different strategies
            strategies_used = set()
            
            max_iterations = 1000  #Define a maximum number of iterations to avoid infinite loop
            iteration_count = 0  # Initialize iteration counter
            while len(domains) < count and len(strategies_used) < 10:
                if iteration_count >= max_iterations:
                    print("Max iterations reached, exiting loop.")
                    break
                iteration_count += 1  # Increment iteration counter
                strategy = random.choice(['prefix_keyword', 'keyword_suffix', 'compound', 'brandable', 'descriptive'])
                
                if strategy in strategies_used and len(domains) < count // 2:
                    continue
                strategies_used.add(strategy)
                
                if strategy == 'prefix_keyword':
                    prefix = random.choice(industry_data['domain_prefixes'])
                    keyword = random.choice(industry_data['keywords'][:5])
                    domain = f"{prefix}{self.clean_for_domain(keyword)}"
                    
                elif strategy == 'keyword_suffix':
                    keyword = random.choice(key_words[:3] if key_words else industry_data['keywords'][:3])
                    suffix = random.choice(industry_data['domain_suffixes'])
                    domain = f"{self.clean_for_domain(keyword)}{suffix}"
                    
                elif strategy == 'compound':
                    word1 = random.choice(industry_data['keywords'])
                    word2 = random.choice(industry_data['domain_suffixes'])
                    domain = f"{self.clean_for_domain(word1)}{self.clean_for_domain(word2)}"
                    
                elif strategy == 'brandable':
                    # Create made-up but pronounceable words
                    prefixes = ['app', 'sync', 'flow', 'dash', 'zoom', 'spark', 'nova', 'axis', 'flux', 'zeta']
                    suffixes = ['ify', 'ly', 'io', 'co', 'ai', 'hub', 'lab', 'tek', 'pro', 'max']
                    domain = f"{random.choice(prefixes)}{random.choice(suffixes)}"
                    
                elif strategy == 'descriptive':
                    if key_words:
                        word = random.choice(key_words[:3])
                        domain = f"{self.clean_for_domain(word)}{random.choice(['pro', 'hub', 'app', 'io', 'co'])}"
                    else:
                        continue
                
                # Add TLD
                tld = random.choice(self.tlds)
                full_domain = f"{domain}{tld}"
                
                # Avoid duplicates and ensure reasonable length
                if full_domain not in domains and 5 <= len(domain) <= 15:
                    domains.append(full_domain)
            
            # Fill remaining slots if needed
            while len(domains) < count:
                fallback_domain = f"{random.choice(industry_data['domain_prefixes'])}{random.choice(industry_data['domain_suffixes'])}{random.choice(self.tlds)}"
                if fallback_domain not in domains:
                    domains.append(fallback_domain)
            
            return domains[:count]

    def generate_dataset(self, total_samples: int = 1000, domains_per_description: int = 5) -> List[Dict]:
        """Generate a complete dataset"""
        dataset = []
        industries = list(self.industries.keys())
        
        # Ensure balanced distribution across industries
        samples_per_industry = total_samples // len(industries)
        extra_samples = total_samples % len(industries)
        
        for i, industry in enumerate(industries):
            if i % 10 == 0:
                print(f"Generating {industry}...on {i} of {len(industries)}")
            industry_samples = samples_per_industry + (1 if i < extra_samples else 0)
            
            for _ in range(industry_samples):
                print(f"Generating {industry}...on {i} of {len(industries)}")
                description, detected_industry = self.generate_business_description(industry)
                print(f"Generating domains for {description} for {industry}")
                domain_suggestions = self.generate_domain_suggestions(
                    description, detected_industry, domains_per_description
                )
                print(f"Generated domains for {description} for {industry}")
                
                dataset.append({
                    "business_description": description,
                    "domain_suggestions": domain_suggestions,
                    "industry": detected_industry,
                    "training_text": f"Business: {description}\nDomains: {', '.join(domain_suggestions)}"
                })
            print(f"Generated {industry_samples} samples for {industry}. Total samples: {len(dataset)}")
        
        # Shuffle the dataset
        print(f"Shuffling dataset of size {len(dataset)}")
        random.shuffle(dataset)
        return dataset

    def save_dataset(self, dataset: List[Dict], filename: str = "business_domain_dataset.json"):
        """Save dataset to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filename}")

    def export_for_training(self, dataset: List[Dict], filename: str = "training_data.jsonl"):
        """Export in JSONL format for fine-tuning"""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in dataset:
                # Format for instruction fine-tuning
                training_item = {
                    "input": item["business_description"],
                    "output": ", ".join(item["domain_suggestions"]),
                    "training_text": item["training_text"]
                }
                f.write(json.dumps(training_item) + "\n")
        print(f"Training data exported to {filename}")

# Example usage
if __name__ == "__main__":
    generator = BusinessDomainGenerator(industries_file='industries.json')
    
    #Generate a small sample
    # print("=== Sample Generations ===")
    # for i in range(100):
    #     description, industry = generator.generate_business_description(random.choice(list(generator.industries.keys())))
    #     domains = generator.generate_domain_suggestions(description, industry)
        # if industry == 'adult_entertainment':
        #     print(f"\nSample {i+1}:")
        #     print(f"Industry: {industry}")
        #     print(f"Description: {description}")
        #     print(f"Domains: {domains}")
    
    # Generate full dataset
    print("\n=== Generating Full Dataset ===")
    total_samples = 50
    dataset = generator.generate_dataset(total_samples=total_samples, domains_per_description=5)
    
    # # Save in different formats
    print(f"Saving dataset to business_domain_dataset_{total_samples}.json")
    generator.save_dataset(dataset, f"business_domain_dataset_{total_samples}.json")
    print(f"Saving training data to training_data_{total_samples}.jsonl")
    generator.export_for_training(dataset, f"training_data_{total_samples}.jsonl")
    
    print(f"\nGenerated {len(dataset)} samples across {len(generator.industries)} industries")
    
    # Show industry distribution
    industry_counts = {}
    for item in dataset:
        industry = item['industry']
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
    
    print("\nIndustry distribution:")
    for industry, count in industry_counts.items():
        print(f"  {industry}: {count} samples")

