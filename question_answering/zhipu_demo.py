from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

chat = ChatZhipuAI(
    model="glm-4-flash",
    temperature=0.5, # the temperature of the model
    api_key="your_api_key"
)

messages = [
    AIMessage(content="Hi."),  # AI generated message
    SystemMessage(content="Your role is a poet."),  # the role of the model
    HumanMessage(content="Write a short poem about AI in four lines."),  # the message from the user
]

# get the answer from the model
response = chat.invoke(messages)
print(response.content)