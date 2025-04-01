import pandas as pd
import os 
from dotenv import load_dotenv
from promptflow.tracing import start_trace
from promptflow.core import Prompty, AzureOpenAIModelConfiguration
from promptflow.evals.evaluators import RelevanceEvaluator, GroundednessEvaluator, ViolenceEvaluator

#opening the csv file in reading mode using pandas dataframe
df=pd.read_csv('evaluation_ai_studio.csv')
num_columns = df.shape[1] #counting the number of columns in the csv file
num_rows = df.shape[0] #counting the number of rows in the csv file

#intializing the count variable to 0
count=0

#creating the AzureOpenAIModelConfiguration object
load_dotenv()
model_config = AzureOpenAIModelConfiguration(
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_KEY"),
    azure_deployment=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    api_version="2024-05-01-preview"
)

#creating the RelevanceEvaluator and GroundednessEvaluator objects from the promptflow.evals.evaluators module in the promptflow-evals SDK for python
relevance_eval = RelevanceEvaluator(model_config)
groundedness_eval = GroundednessEvaluator(model_config)

#creating the lists to store the relevance and groundedness scores
relevance_score_list = []
groundedness_score_list = []

#looping through the rows of the csv file
while(count<num_rows):
    #evaluating the relevance and groundedness scores for each question-answer pair in the csv file
    relevance_score= relevance_eval(
        answer=df.iloc[count][2],
        context=df.iloc[count][1],
        question=df.iloc[count][0]
    )
    
    groundedness_score = groundedness_eval(
        answer = df.iloc[count][2],
        context = df.iloc[count][1],
    )   
    
    
    #printing the question, answer, relevance score and groundedness score for each question-answer pair
    print("question : " + df.iloc[count][0])
    print("answer : " + df.iloc[count][2])
    
    #converting the relevance and groundedness scores to string and printing them from the dict values
    gpt_relevance_score=str(relevance_score['gpt_relevance'])
    gpt_groundedness_score = str(groundedness_score['gpt_groundedness'])
    
    #printing the relevance and groundedness scores
    print("relevance score : " + gpt_relevance_score)
    print("groundedness score : " + gpt_groundedness_score)
    
    #appending the relevance and groundedness scores to the relevance_score_list and groundedness_score_list
    relevance_score_list.append(gpt_relevance_score)
    groundedness_score_list.append(gpt_groundedness_score) 
   
   # incrementing the count variable
    count+=1
    
#adding the relevance and groundedness scores to the csv file    
df['GPT Relevance Score'] = relevance_score_list
df['GPT Groundedness Score'] = groundedness_score_list

#saving the csv file with the relevance and groundedness scores
df.to_csv('evaluation_ai_studio.csv', index=False)

