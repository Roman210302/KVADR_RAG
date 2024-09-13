import requests
import gradio as gr
import argparse

# Argument parser for API and demo ports
parser = argparse.ArgumentParser(description="Input port of API and port of demo")
parser.add_argument('--api_port', type=int, help="Port of model API", required=True)
parser.add_argument('--demo_port', type=int, help="Port of demo", required=True)
args = parser.parse_args()

# Function to interact with the model
def chat_with_model(user_input, chat_history=[]):
    api_url = f"http://localhost:{args.api_port}/v1/completions"
    response = requests.post(api_url, json={"question": user_input})
    
    if response.status_code == 200:
        bot_response = response.json().get("response", "Model didn't respond.")
    else:
        bot_response = "Error connecting to model."

    chat_history.append((user_input, bot_response))
    return chat_history, chat_history, ""

# Create Gradio Blocks with custom CSS for styling
with gr.Blocks(css="""
    .header {
        text-align: center;
        color: #0044cc; /* Dark blue for header */
    }
    .subheader {
        text-align: center;
        color: #007f5c; /* Dark green for subheader */
    }
    .gr-chatbot {
        background-color: #e6f7ff; /* Light blue background for chat */
        border: 1px solid #0056b3; /* Blue border */
        border-radius: 10px;
    }
    .gr-textbox {
        border: 2px solid #007f5c; /* Green border for textbox */
        border-radius: 5px;
    }
    .gr-button {
        background-color: #007f5c; /* Green button */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px;
    }
    .gr-button:hover {
        background-color: #005f4b; /* Darker green on hover */
    }
    .image-container {
        display: flex;
        flex-direction: column; /* Stack children vertically */
        align-items: center; /* Center children horizontally */
        justify-content: center; /* Center children vertically */
    }
    .image-container img {
        height: 80px; /* Adjust image height */
        margin-bottom: 10px; /* Space between image and header */
    }
    .footer-icon {
        display: flex;
        justify-content: center; /* Center horizontally */
        margin-top: 20px; /* Space above the icon */
        margin-bottom: 20px; /* Space below the icon */
    }
    .footer-icon img {
        height: 40px; /* Adjust icon height */
    }
""") as demo:
    
    # Create a container for the image and header
    with gr.Column(elem_id="image-container"):
        gr.Image(value="gerb.png", elem_id="header-image")  # Replace with your image path
        gr.Markdown("<h1 class='header'>KVADR RAG Chat</h1>")
    
    gr.Markdown("<h2 class='subheader'>Данный чат-бот умеет отвечать на вопросы по Нормативным правовым актам Ханты-Мансийского округа.</h2>")

    chatbox = gr.Chatbot()
    user_input = gr.Textbox(show_label=False, placeholder="Задайте здесь свой вопрос по НПА")
    send_button = gr.Button("Отправить")

    chat_history_state = gr.State([])

    send_button.click(fn=chat_with_model, inputs=[user_input, chat_history_state], 
                      outputs=[chatbox, chat_history_state, user_input])
    user_input.submit(fn=chat_with_model, inputs=[user_input, chat_history_state], 
                      outputs=[chatbox, chat_history_state, user_input])

    # Add an icon at the bottom of the interface directly without wrapping in a Column
    gr.Image(value="kvadr.png", elem_id="footer-icon-image")  # Replace with your icon image path

demo.launch(server_port=args.demo_port)