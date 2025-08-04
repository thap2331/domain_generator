#test_lambda_api.py
#make a post request to the lambda url
import requests

lambda_url = "https://p4sscoi5qngsrpt6aqsh2znvwu0zbttl.lambda-url.us-west-2.on.aws/"

response = requests.post(
    lambda_url, json={"business_name": "organic coffee shop in downtown area"}
    )

print(response.json())