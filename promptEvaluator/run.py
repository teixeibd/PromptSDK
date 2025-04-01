# importing important utitilites and libraries
import os
from promptflow.tracing import start_trace
import dotenv
from dotenv import load_dotenv
from dotenv import load_dotenv
from promptflow.core import Prompty, AzureOpenAIModelConfiguration
from promptflow.client import load_flow

#starting the trace
load_dotenv()
#creating the AzureOpenAIModelConfiguration object
model_config = AzureOpenAIModelConfiguration(
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_KEY"),
    azure_deployment=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    api_version="2024-05-01-preview"
)

#loading the custom prompt evaluator
courtesy_eval = load_flow(source="evaluator.prompty", model = {"configuration": model_config})
#evaluating the courtesy score
courtesy_score = courtesy_eval(question="hello how are you?", answer="I am fine. You need to ask me that again")

#displaying the courtesy score
print(courtesy_score)

