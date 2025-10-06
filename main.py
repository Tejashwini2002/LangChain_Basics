
'''
# SAMPLE PROGRAM 1: 

from dotenv import load_dotenv
import os
load_dotenv()

def main():
    print("Hello from langchain-course!")
    print("Your OPENAI_API_KEY is:", os.getenv("OPENAI_API_KEY"))
    print("Your OPENAI_API_KEY is:",os.environ.get("OPENAI_API_KEY"))
    
if __name__ == "__main__":
    main()
'''
##
'''
    
# SAMPLE PROGRAM 2:
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

def create_travel_advisor():
    # Initialize the OpenAI model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # You can also use "gpt-4" if you have access
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create a template for travel advice
    template = """You are a helpful travel advisor. 
    Previous conversation: {chat_history}
    Human: {human_input}
    AI: Let me help you with your travel query!
    """

    # Create a prompt from the template
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template
    )

    # Set up memory to remember conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        human_prefix="Human",
        ai_prefix="AI"
    )

    # Create the conversation chain
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    return conversation

def main():
    print("Welcome to your AI Travel Advisor!")
    print("Ask me anything about travel planning, destinations, or tips.")
    print("Type 'quit' to exit.")
    
    # Create the travel advisor
    advisor = create_travel_advisor()
    
    # Start conversation loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        # Get response from the advisor
        response = advisor.predict(human_input=user_input)
        print("\nAI:", response.strip())

if __name__ == "__main__":
    main()
   
 '''
#




# SAMPLE PROGRAM 3: (Using Gemini)
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

def create_travel_advisor():
    # Initialize the language model with Gemini
    llm = ChatGoogleGenerativeAI(
        # model="gemini-pro",
        # model="gemini-1.5-flash",
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),  # Changed to use GEMINI_API_KEY
        temperature=0.7
    )

    # Create a template for travel advice
    template = """You are a helpful travel advisor. 
    Previous conversation: {chat_history}
    Human: {human_input}
    AI: Let me help you with your travel query!
    """

    # Create a prompt from the template
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template
    )

    # Set up memory to remember conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        human_prefix="Human",
        ai_prefix="AI"
    )

    # Create the conversation chain
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    return conversation

def main():
    print("Welcome to your AI Travel Advisor!")
    print("Ask me anything about travel planning, destinations, or tips.")
    print("Type 'quit' to exit.")
    
    # Create the travel advisor
    advisor = create_travel_advisor()
    
    # Start conversation loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        # Get response from the advisor
        response = advisor.predict(human_input=user_input)
        print("\nAI:", response.strip())

if __name__ == "__main__":
    main()

   

# from dotenv import load_dotenv
# import os
# import google.generativeai as genai

# # Load environment variables from .env
# load_dotenv()

# # Get API key from .env
# api_key = os.getenv("GEMINI_API_KEY")

# if not api_key:
#     raise ValueError("GEMINI_API_KEY not found in .env file")

# # Configure the Gemini API
# genai.configure(api_key=api_key)

# # Create model and chat
# model = genai.GenerativeModel("gemini-1.5-flash")
# chat = model.start_chat(history=[])

# print("Welcome to your AI Travel Advisor!")
# print("Ask me anything about travel planning, destinations, or tips.")
# print("Type 'quit' to exit.")

# while True:
#     user_input = input("\nYou: ")
#     if user_input.lower() == "quit":
#         break

#     try:
#         response = chat.send_message(user_input)

#         # Depending on SDK version, use the correct text access method
#         if hasattr(response, "text"):
#             print("\nAssistant:", response.text)
#         else:
#             print("\nAssistant:", response.candidates[0].content.parts[0].text)

#     except Exception as e:
#         print(f"\nError: {type(e).__name__} - {e}")


# import google.generativeai as genai
# from dotenv import load_dotenv
# import os

# load_dotenv()
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# print("Available models:\n")
# for m in genai.list_models():
#     print(m.name)

##############
'''
Step 1 — Install the Gemini SDK

In your activated environment ((langchain-course) prompt is visible ✅), run:

pip install google-generativeai


💡 Note: Use this inside your virtual environment, not globally.

You can verify installation by checking:

pip show google-generativeai


You should see version info like:

Name: google-generativeai
Version: 0.8.3

✅ Step 2 — Re-run your model listing script
'''
##
'''
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Available Gemini models:")
for m in genai.list_models():
    print(m.name)
'''
