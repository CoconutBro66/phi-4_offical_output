import transformers

pipeline = transformers.pipeline(
    "text-generation",
    model="/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/phi-4",
    model_kwargs={"torch_dtype": "auto"},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a medieval knight and must provide explanations to modern people."},
    {"role": "user", "content": "How should I explain the Internet?"},
]

outputs = pipeline(messages, max_new_tokens=128)
print(outputs[0]["generated_text"][-1])