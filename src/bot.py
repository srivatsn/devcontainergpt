"""Bot that answers questions about devcontainers docs."""

from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from index import get_search_index


def get_chain():
    """Get the chain."""

    example_prompt = PromptTemplate(
        template=">Example:\nContent:\n---------\n{page_content}\n----------\nSource: {source}",
        input_variables=["page_content", "source"],
    )

    template = """You are an AI assistant for the Codespaces team's live site operations.
    You are given the following extracted parts of a long document and a question. Each of the document is a trouble shooting guide (abbrebiated as TSG).
    Provide a conversational answer with a hyperlink to the documentation.
    You should only use hyperlinks that are explicitly listed as a source in the context. Do NOT make up a hyperlink that is not listed.
    Always return a hyperlink to the TSG.
    If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
    If the question is not about Codespaces or live site operations, politely inform them that you are tuned to only answer questions about Codespaces.
    Question: {question}
    =========
    {summaries}
    =========
    Answer in Markdown:"""

    prompt = PromptTemplate(template=template, input_variables=[
                            "question", "summaries"])

    search_index = get_search_index()
    chain = load_qa_with_sources_chain(
        OpenAI(temperature=0),
        chain_type="stuff",
        prompt=prompt,
        document_prompt=example_prompt)

    return chain, search_index


def get_answer(question, chain, search_index):
    """Get the answer to a question."""
    return chain(
        {
            "input_documents": search_index.similarity_search(question, k=4),
            "question": question,
        },
        return_only_outputs=True,
    )["output_text"]


def print_answer(question):
    """Print the answer to a question."""
    print(get_answer(question, *get_chain()))