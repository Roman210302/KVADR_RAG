import requests
import gradio as gr
import argparse

parser = argparse.ArgumentParser(description="Input port of API and port of demo")
parser.add_argument('--api_port', type=int, help="Port of model API", required=True)
parser.add_argument('--demo_port', type=int, help="Port of demo", required=True)
args = parser.parse_args()


def chat_with_model(user_input, chat_history=[]):
    api_url = f"http://localhost:{args.api_port}/v1/completions"
    response = requests.post(api_url, json={"question": user_input})
    
    if response.status_code == 200:
        bot_response = response.json().get("response", "Model didn't respond.")
    else:
        bot_response = "Error connecting to model."

    chat_history.append((user_input, bot_response))
    return chat_history, chat_history, ""


with gr.Blocks() as demo:
    gr.Markdown("<h1><center>KVADR RAG Chat</center></h1>")

    chatbox = gr.Chatbot()
    user_input = gr.Textbox(show_label=False, placeholder="Type your message and press Enter")
    send_button = gr.Button("Send")

    chat_history_state = gr.State([])

    send_button.click(fn=chat_with_model, inputs=[user_input, chat_history_state], 
                      outputs=[chatbox, chat_history_state, user_input])
    user_input.submit(fn=chat_with_model, inputs=[user_input, chat_history_state], 
                      outputs=[chatbox, chat_history_state, user_input])
    
demo.launch(server_port=args.demo_port)




