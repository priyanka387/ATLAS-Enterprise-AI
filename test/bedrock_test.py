import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL")

# Amazon Nova Lite
BEDROCK_CHAT_MODEL = "amazon.nova-lite-v1:0"

# Validate required environment variables
required_vars = {
    "AWS_REGION": AWS_REGION,
    "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
    "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
    "BEDROCK_EMBED_MODEL": BEDROCK_EMBED_MODEL,
}

missing = [name for name, value in required_vars.items() if not value]

if missing:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing)}"
    )

print(f"AWS Region: {AWS_REGION}")
print(f"Embedding model: {BEDROCK_EMBED_MODEL}")
print(f"Chat model: {BEDROCK_CHAT_MODEL}")


client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


# ============================================================
# Test 1: Amazon Titan Embeddings
# ============================================================

embedding_response = client.invoke_model(
    modelId=BEDROCK_EMBED_MODEL,
    body=json.dumps({
        "inputText": "Hello Amazon Bedrock",
        "dimensions": 1024,
        "normalize": True,
    }),
    contentType="application/json",
    accept="application/json",
)

embedding_data = json.loads(
    embedding_response["body"].read()
)

print(
    "Embedding dimension:",
    len(embedding_data["embedding"])
)


# ============================================================
# Test 2: Amazon Nova Lite
# ============================================================

chat_response = client.converse(
    modelId=BEDROCK_CHAT_MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "text": "Explain RAG in one sentence."
                }
            ],
        }
    ],
    inferenceConfig={
        "maxTokens": 200,
    },
)

print(
    "Nova Response:",
    chat_response["output"]["message"]["content"][0]["text"]
)
