"""Bot that answers questions about devcontainers docs."""

from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from index import get_search_index


def print_answer(question):
    """Print the answer to a question."""

    print(
        chain(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )


EXAMPLE_PROMPT = PromptTemplate(
    template=">Example:\nContent:\n---------\n{page_content}\n----------\nSource: {source}",
    input_variables=["page_content", "source"],
)
template = """You are an AI assistant for devcontainers. The documentation is located at https://containers.dev.
You are given the following extracted parts of a long document and a question. Provide a conversational answer with a hyperlink to the documentation.
You should only use hyperlinks that are explicitly listed as a source in the context. Do NOT make up a hyperlink that is not listed.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about devcontainers, politely inform them that you are tuned to only answer questions about devcontainers.
Question: {question}
=========
{summaries}
=========
Answer in Markdown:"""
PROMPT = PromptTemplate(template=template, input_variables=[
                        "question", "summaries"])

search_index = get_search_index()
chain = load_qa_with_sources_chain(
    OpenAI(temperature=0),
    chain_type="stuff",
    prompt=PROMPT,
    document_prompt=EXAMPLE_PROMPT)

print_answer(
    "Is there a way to specify the order in which features are installed?")
