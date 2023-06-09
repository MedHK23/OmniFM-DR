import argparse
import os
import random
from PIL import Image

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from demo_init import init_task, ask_answer


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    # parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    # parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    # seed = config.run_cfg.seed + get_rank()
    seed = 1

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
init_task()
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def _upload_img(image, conv, img_list):
    if isinstance(image, str):  # is a image path
        image = Image.open(image).convert('RGB')

    img_list.append(image)
    # conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
    msg = "Received."
    # self.conv.append_message(self.conv.roles[1], msg)
    return msg

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    img_list = []
    llm_message = _upload_img(gr_img, chat_state, img_list)
    if isinstance(gr_img, str):
        image = Image.open(image).convert('RGB')
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    # chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list):
    insturction = chatbot[-1][0]
    print(f"*** Instruction: {insturction}")
    llm_message = ask_answer(img_list[0], insturction)
    # llm_message = ('/data/niziyu/tmp/sunhao/OFA/temp_sunhao/5B1A21A377C692_boxes.jpg', None)
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list

title = """<h1 align="center">Demo of OmniFM-DR</h1>"""
description = """<h3>This is the demo of OmniFM-DR. Upload your images and start testing!</h3>"""
# article = """<p><a href='https://'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p><a href='https://'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>
# """

#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    # gr.Markdown(article)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='DR_Captain')
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first ^-^', interactive=False)
    
    upload_button.click(upload_img, [image, text_input, chat_state], [image, text_input, upload_button, chat_state, img_list])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list], [chatbot, chat_state, img_list]
    )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, upload_button, chat_state, img_list], queue=False)
    gr.Markdown("## Instruction Examples")
    examples = gr.Examples(examples=["what can we get from this chest medical image?", "Where is cardiomegaly?", "what disease does this image have?"],
                           inputs=[text_input])
    gr.Markdown("## Image Examples")
    examples = gr.Examples(examples=["./results/tmp/00a7c9c6-39774f9b-deb1ac06-fbcf0a96-01a94d7e.jpg"],
                           inputs=[image])

demo.launch(share=True, enable_queue=True, server_name='0.0.0.0')