import torch
from llama_index.core import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# LLaMA 4 Scout Model ID
MODEL_ID = "meta-llama/Meta-Llama-4-Scout"

# Load HuggingFace LLM via LlamaIndex
llm = HuggingFaceLLM(
    model_name=MODEL_ID,
    tokenizer_name=MODEL_ID,
    device_map="auto",
    dtype=torch.float16,
)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt_template = PromptTemplate("{prompt}")
    formatted_prompt = prompt_template.format(prompt=request.prompt)
    response = llm.complete(formatted_prompt)
    return {"response": response.text}
