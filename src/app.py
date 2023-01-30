import datetime
import os
import gradio as gr

from bot import get_answer, get_chain


def set_openai_api_key(api_key):
    """Set the OpenAI API key and return the chain and search index."""

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        chain, search_index = get_chain()
        os.environ["OPENAI_API_KEY"] = ""
        return chain, search_index


def chat(inp, chain, index):
    """Chat with the bot."""

    if chain is None:
        return "Please paste your OpenAI key to use"
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("inp: " + inp)
    answer = get_answer(inp, chain, index)
    print(answer)
    return answer


block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Devcontainer search</center></h3>")

        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="What's the answer to life, the universe, and everything?",
            lines=1,
        )
        submit = gr.Button(value="Submit", variant="secondary").style(
            full_width=False)

    output = gr.Markdown()

    gr.Examples(
        examples=[
            "What are lifecycle hooks?",
            "Can I specify the order in which features are installed?",
            "What is the difference between a devcontainer and a devcontainer.json?",
        ],
        inputs=message,
    )

    gr.HTML(
        """
    Semantic search for the devcontainer documentation at https://containers.dev"""
    )

    chain_state = gr.State()
    index_state = gr.State()

    submit.click(chat, inputs=[message,
                 chain_state, index_state], outputs=[output])
    message.submit(chat, inputs=[message,
                   chain_state, index_state], outputs=[output])

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox],
        outputs=[chain_state, index_state],
    )

block.launch(debug=True)
