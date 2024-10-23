import gradio as gr
from threading import Thread
from typing import Iterator

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

###
import os
from dotenv import load_dotenv
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
###

model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=huggingface_token)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    token=huggingface_token
).to(device)

def respond(
    message: str,
    history: list[tuple[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> Iterator[str]:

    conversation = [{"role": "system", "content": system_message}]
    
    for user_msg, assistant_msg in history:
        if user_msg:
            conversation.append({"role": "user", "content": user_msg})
        if assistant_msg:
            conversation.append({"role": "assistant", "content": assistant_msg})
    
    conversation.append({"role": "user", "content": message})
    
    input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
    
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_tokens,
        early_stopping=True,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
    )
    
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    response = ""
    for token in streamer:
        response += token
        yield response

chat = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
)

def greet(message):
    conversation = [{"role": "system", "content": "you are a helpful assistant"}]
    history = []
    
    for user_msg, assistant_msg in history:
        if user_msg:
            conversation.append({"role": "user", "content": user_msg})
        if assistant_msg:
            conversation.append({"role": "assistant", "content": assistant_msg})
    
    conversation.append({"role": "user", "content": message})
    
    input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=5,
        early_stopping=True,
        do_sample=True,
        top_p=.90,
        temperature=.9,
        streamer=streamer,
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    response = ""
    for token in streamer:
        response += token
        yield response

textbox = gr.Interface(fn=greet, inputs="textbox", outputs="textbox", flagging_mode='never')


if __name__ == "__main__":
    #chat.launch()
    textbox.launch()
