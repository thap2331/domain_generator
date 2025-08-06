# Solcae

## Overview


## Prerequisites

- AWS SAM CLI, `aws-sam-cli`
- Python 3.10
- AWS account with access to Secrets Manager
- OPENAI api key

## Setup

1. **Save Secrets**
- **Production**
  - Store your OpenAI API key in AWS Secrets Manager under the name `OPENAI_API_KEY`. Keep it under `openai` in `us-west-2` region.

- **Development**
  - Keep OPENAI_API_KEY in `.env` file
  - For testing locally using `sam local invoke`, keep the secrets in .env.json 
    ```
    {"Parameters": 
      {"OPENAI_API_KEY":"abcd"}
    }
    ```

2. **Install Dependencies**: 
   ```bash
   pip install -r src/requirements.txt
   ```

## Build and Test Locally

  - **Build the Project**:
    ```bash
    sam build
    ```

  - **Invoke Locally**:
    ```bash
    sam build && sam local invoke --event event.json --env-vars .env.json
    ```

## Deployment

  - **Deploy the Application**:
    ```bash
    sam deploy --guided
    ```

  - **Sync code**
    ```bash
    sam sync --watch --stack-name domaingenerator
    ```