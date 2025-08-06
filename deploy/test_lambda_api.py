#test_lambda_api.py
#make a post request to the lambda url
import requests

lambda_url = "https://fwomgr25ypqdf52e2655xhwvt40yprkg.lambda-url.us-west-2.on.aws/"

business_description = "organic coffee shop in downtown area"
response = requests.post(
    lambda_url, json={"business_description": business_description}
    )
print(f'\nbusiness_description: {business_description} and domains: {response.json()}')

business_description = "adult content website with explicit nude content"
response = requests.post(
    lambda_url, json={"business_description": business_description}
    )
print(f'\nbusiness_description: {business_description} and domains: {response.json()}')

business_description = "furniture like ikea"
response = requests.post(
    lambda_url, json={"business_description": business_description}
    )
print(f'\nbusiness_description: {business_description} and domains: {response.json()}')

business_description = "modern mlops in ai space"
response = requests.post(
    lambda_url, json={"business_description": business_description}
    )
print(f'\nbusiness_description: {business_description} and domains: {response.json()}')
