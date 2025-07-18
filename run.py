# -------- chat_loop.py --------
from app.agents import app
from langchain_core.messages import HumanMessage
import sys

print("Travel Chatbot — ask anything (type 'exit' to quit)\n")

# Initialize conversation history
conversation_messages = []

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Bye!")
        sys.exit()


    conversation_messages.append(HumanMessage(content=query))
    
    result = app.invoke({"messages": conversation_messages})
    

    if 'messages' in result:
        # The app returns the full state, get the last AI message
        last_message = result['messages'][-1]
        if hasattr(last_message, 'content'):
            response = last_message.content
        else:
            response = "No response content found."
    else:
        response = result.get('response', 'No response returned by agent.')
    
    print(f"Bot: {response}")
    
    
    if 'messages' in result:
        conversation_messages = result['messages']