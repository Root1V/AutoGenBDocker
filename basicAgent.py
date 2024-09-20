# agent basic

import os
import autogen
from autogen import AssistantAgent, UserProxyAgent
import autogen.coding

llmConfig = {
    "model": "gpt-4o-mini",
    "api_key": os.environ["OPENAI_API_KEY"]
}

# Assistant Agent
assitant = AssistantAgent("Alexa", llm_config=llmConfig)

# Code Agent
user_code = UserProxyAgent(
    "Jarvis", 
    code_execution_config={"executor": autogen.coding.LocalCommandLineCodeExecutor(work_dir="coding")}
)

# Start
user_code.initiate_chat(
    assitant,
    message="Dime una broma acerca de la Mecanica Cuantica"
)