from langchain_ollama import OllamaLLM
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import textwrap

# Initialize the Ollama LLM with qwen-2.5 model
llm = OllamaLLM(model="qwen2.5:latest")

def call_llm_with_full_text(user_query):
    try:
        # Create the message chain
        messages = [
            SystemMessage(content="You are an expert Natural Language Processing exercise expert."),
            AIMessage(content="1. You can explain read the input and answer in detail"),
            HumanMessage(content=user_query)
        ]
        
        # Invoke the LLM
        response = llm.invoke(messages)
        return response.strip()
    except Exception as e:
        return str(e)

def print_formatted_response(response):
    # Define the width for wrapping the text
    wrapper = textwrap.TextWrapper(width=80)  # Set to 80 columns wide
    wrapped_text = wrapper.fill(text=response)

    # Print the formatted response with a header and footer
    print("\nResponse:")
    print("---------------")
    print(wrapped_text)
    print("---------------\n")

# Example usage
if __name__ == "__main__":
    query = "define a rag store"
    llm_response = call_llm_with_full_text(query)
    print_formatted_response(llm_response)