"""This file shows an example on how to create a simple chatbot without memory using langchain and ollama."""
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import Ollama

model_name = "dolphin-mixtral"  # name of your ollama model
stop_tokens = []  # tokens which should force end a message
num_gpu = 10  # number of layers to offload to gpu
num_ctx = 16000  # number of tokens as context
temperature = 1  # temperature parameter 0 <= temperature <= 1

# Initialize Ollama
ollama = Ollama(
    model=model_name,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    stop=stop_tokens,
    temperature=temperature,
    num_ctx=num_ctx,
    num_gpu=num_gpu,
)

human = "Adrian"

prompt_template = f"""{{history}}\n{human}: {{input}}"""
PROMPT = PromptTemplate(input_variables=["input", "history"], template=prompt_template)

history = ConversationBufferMemory(human_prefix=human, ai_prefix="professor")
history.chat_memory.add_ai_message("""Hello! What is your wish today?""")

conversation = ConversationChain(
    prompt=PROMPT,
    llm=ollama,
    verbose=True,
    memory=history,
)


if __name__ == "__main__":
    while True:
        prompt = input("\nYour prompt: ")
        conversation.predict(input=prompt)
