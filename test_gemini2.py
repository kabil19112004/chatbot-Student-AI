import llm

resp = llm.generate_chat_response([
    {"role": "user", "content": "Hello!"}
])
print(resp)
