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

SYSTEM_MESSAGE = "You are a helpful assistant."

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
    system_message: gr.State,
    max_tokens: int,
    temperature: float,
    top_p: float,
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

def handle_system_message_change(new_role: str):
    if new_role.lower() == "translator":
        return "You are a translator. Translate all user inputs from English to Spanish."
    else:
        return "You are a helpful assistant."

with gr.Blocks() as user_interface:

    system_message = gr.State("You are a helpful assistant.")
    
    with gr.Column():

        new_role_input = gr.Dropdown(
            label="Select new system role",
            choices=["Translator", "Assistant"],
            value="Assistant",
            interactive=True
        )
        new_role_input.change(fn=handle_system_message_change, inputs=[new_role_input], outputs=[system_message])

        with gr.Accordion(label="Additional Settings", open=False):
            max_new_tokens_slider = gr.Slider(minimum=1, maximum=2048, value=20, step=1, interactive=True, label="Max new tokens")
            temperature_slider = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, interactive=True, label="Temperature")
            top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, interactive=True, label="Top-p (nucleus sampling)")

        chat = gr.ChatInterface(
            fn=respond,
            additional_inputs=[
                system_message,
                max_new_tokens_slider,
                temperature_slider,
                top_p_slider],
        )

if __name__ == "__main__":
    user_interface.launch()
