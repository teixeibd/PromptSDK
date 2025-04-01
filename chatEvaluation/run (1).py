import os
from promptflow.tracing import start_trace
import dotenv
from dotenv import load_dotenv
from promptflow.core import Prompty, AzureOpenAIModelConfiguration
from promptflow.evals.evaluators import ChatEvaluator

#starting the trace
start_trace()

#loading the .env file
load_dotenv()
#creating the AzureOpenAIModelConfiguration object
model_config = AzureOpenAIModelConfiguration(
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_KEY"),
    azure_deployment=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    api_version="2024-05-01-preview"
)

#creating the Prompty object
prompty = Prompty.load("chat.prompty", model={'configuration': model_config})

#prompting the user with the chat history and chat input
response = prompty(
    chat_history=[
        {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
        {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."}
    ],
    chat_input="Do other Azure AI services support this too?")

#displaying the response
print(response)


#creating the conversation list
conversation = chat_history=[
        {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
        {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
        {"role": "user", "content": "Do other Azure AI services support this too?"},
        {"role": "assistant", "content": response}
    ]

#creating the ChatEvaluator object
chat_eval = ChatEvaluator(model_config=model_config)
score = chat_eval(conversation=conversation)

#displaying the score
print(score)