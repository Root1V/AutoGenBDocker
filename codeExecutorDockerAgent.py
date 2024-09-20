# codeExcutorDocker.py
import tempfile
import os
from autogen import ConversableAgent
from autogen.coding import DockerCommandLineCodeExecutor

# Creamos una carpeta temporal
temp_dir = tempfile.TemporaryDirectory()

# Creamos un executor para docker
executor = DockerCommandLineCodeExecutor(
    image="python:3.12-slim",
    timeout=10,
    work_dir=temp_dir.name
)

# Creamos un agente 
code_executor_agent = ConversableAgent(
    "Skynet",
    llm_config=False,
    code_execution_config={ "executor": executor},
    human_input_mode="ALWAYS"
)


message_w_code_block=""" This is a message with code block.
The code block is below:
```python
x = 15
y = 6
z = x ** y
print(f'Se ejecuto en docker correctamente.. {z}')
```
This is the end of the message.
"""

# Ejecutamos el agente
reply = code_executor_agent.generate_reply(
    messages=[{"role": "user", "content": message_w_code_block}]
)

print(reply)
print(temp_dir.name)
print(os.listdir(temp_dir.name))
temp_dir.cleanup()



