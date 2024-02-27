from db import RAG
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import gradio as gr
from rich import print, print_json
from ConversationAgentModel import ConversationalAgent
load_dotenv()



#initialize document store
docstore = RAG('final.csv')
docstore.create_collection()

api_key = os.getenv("OPEN_AI_API")

bot = ConversationalAgent(api_key)

def chat_session(query,chat_history):
    
    #add retrieved docs from db to system prompt here
    docs = docstore.retrieve_data(query=query,number_of_documents=2)
    print(docs['documents'])
    bot.system_prompt = bot.system_prompt.replace('{documents}',str(docs))
    results = bot.generate_response(query,chat_history)
    return results

with gr.Blocks() as demo:

    chatbot = gr.Chatbot(label='Contlos', height=600)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    
    msg.submit(chat_session, [msg, chatbot], [msg, chatbot])
    #print(olivia.conversation_history)

demo.launch()

