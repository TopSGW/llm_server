import torch
from llama_index.core import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# LLaMA 4 Scout Model ID
MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E"

# Load HuggingFace LLM via LlamaIndex
llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-4-Scout-17B-16E",
    tokenizer_name="meta-llama/Llama-4-Scout-17B-16E",
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16},  # Pass dtype within model_kwargs
    # Other parameters as needed
)


class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt_template = PromptTemplate("{prompt}")
    formatted_prompt = prompt_template.format(prompt=request.prompt)
    response = llm.complete(formatted_prompt)
    return {"response": response.text}
