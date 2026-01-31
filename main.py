import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# TOOLS

def multiply(a, b):
    '''
    Multplies two numbers
    '''
    return a * b

def divide(a, b):
    '''
    Divides two numbers
    '''
    return a / b


# TOOL DEFINITIONS

tools = [
    {
        "type" : "function",
        "function" : {
            "name" : "multiply",
            "description" : "Call this function to multiply two numbers.",
            "parameters":{
                "type" : "object",
                "properties":{
                    "a" : {
                        "type" : "integer",
                        "description" : "The first number"
                    },
                    "b" : {
                        "type" : "integer",
                        "description" : "The first number"
                    }
                },
                "required" : ["a", "b"]
            }
        }
    },

    {
        "type":"function",
        "function" : {
            "name" : "divide",
            "description" : "Call this function to divide two numbers.",
            "parameters":{
                "type" : "object",
                "properties":{
                    "a" : {
                        "type" : "integer",
                        "description" : "The first number"
                    },
                    "b" : {
                        "type" : "integer",
                        "description" : "The first number"
                    }
                },
                "required" : ["a", "b"]
            }
        }
    }
]

# ASSESSING TOOL CALL

message = [
    {"role" : "user" , "content" : "What is 123456 multiplied by 654321?"}
]

response = client.chat.completions.create(
    model= "gpt-5",
    tools=tools,
    messages=message
)


available_functions = {
    "multiply" : multiply,
    "divide" : divide 
}
tool_calls = response.choices[0].message.tool_calls
if tool_calls:

    tool_call_id = tool_calls[0].id
    function_name = tool_calls[0].function.name
    function_args = json.loads(tool_calls[0].function.arguments)

    function_to_call = available_functions[function_name]

    result = function_to_call(**function_args)

    print(f"Agent decided to call function : {function_name}")
    print(f"Arguments : {function_args}")
    print(f"Calculated result : {result}")
    message.append(response.choices[0].message)

    message.append({
        "role" : "tool",
        "tool_call_id" : tool_call_id,
        "content" : str(result)
    })

    final_response = client.chat.completions.create(
        model="gpt-5",
        messages = message,
        tools = tools
    )

    print("--" * 10 + "FINAL AGENT RESPONSE" + "--" * 10)
    print(final_response.choices[0].message.content)
else:
    print("Agent answered directly without needing tools :", response.choices[0].message)

