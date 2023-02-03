"""Create a search index for the devcontainers docs."""

import os
import pathlib
import pickle
import subprocess
import tempfile
import urllib.request
from time import sleep
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter


def get_github_docs(repo_owner, repo_name, pathPrefix=None):
    """Get all markdown files from a github repo."""

    with tempfile.TemporaryDirectory() as d:
        subprocess.check_call(
            f"git clone --depth 1 https://github.com/{repo_owner}/{repo_name}.git .",
            cwd=d,
            shell=True,
        )
        git_sha = (
            subprocess.check_output("git rev-parse HEAD", shell=True, cwd=d)
            .decode("utf-8")
            .strip()
        )
        repo_path = pathlib.Path(d)
        markdown_files = list(repo_path.glob(pathPrefix + "**/*.md")) + list(
            repo_path.glob(pathPrefix + "**/*.mdx")
        )
        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                print(markdown_file)
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}"
                yield Document(page_content=f.read(), metadata={"source": github_url})


def create_source_chunks(sources):
    """Create chunks of text from the sources."""

    source_chunks = []
    splitter = CharacterTextSplitter(
        separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in sources:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(
                Document(page_content=chunk, metadata=source.metadata))
    return source_chunks


def create_search_index():
    """Create a search index by getting all the docs from the devcontainers repo."""

    sources = get_github_docs("github", "ops", "docs/playbooks/codespaces/")

    source_chunks = create_source_chunks(sources)

    first_chunk = source_chunks.pop()
    search_index = FAISS.from_documents([first_chunk], OpenAIEmbeddings())

    i = 0
    print(f"Total chunks: {len(source_chunks)}")
    while i < len(source_chunks):
        print(f"Getting embeddings for chunk {i} to {i + 20}")
        documents = source_chunks[i:i+20]
        search_index.add_texts(
            [d.page_content for d in documents],
            [d.metadata for d in documents])
        i += 20
        # wait a min for rate limiting
        sleep(60)

    with open("search_index.pickle", "wb") as f:
        pickle.dump(search_index, f)

    return search_index


def get_search_index():
    """Get the search index from a pickle file or create it if it doesn't exist."""

    if (os.path.exists("search_index.pickle")):
        with open("search_index.pickle", "rb") as f:
            index = pickle.load(f)
    else:
        try:
            index = pickle.load(urllib.request.urlopen(
                "https://github.com/srivatsn/devcontainergpt/releases/download/index-0.1/search_index.pickle"))
        except Exception:
            return None
    return index

if __name__ == "__main__":
    create_search_index()
