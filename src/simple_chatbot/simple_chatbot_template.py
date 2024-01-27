"""This file shows an example on how to create a simple chatbot without memory using langchain and ollama."""
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
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

prompt_template = """Tell me a single joke about {theme}. Also translate it to {language}."""
PROMPT = PromptTemplate(input_variables=["theme", "language"], template=prompt_template)


if __name__ == "__main__":
    while True:
        theme = input("\nYour theme: ")
        language = input("\nYour language: ")
        prompt = PROMPT.format(theme=theme, language=language)
        ollama.invoke(input=prompt)
