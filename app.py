import gradio as gr
from database import init_db, fetch_arxiv, fetch_semantic_scholar, save_papers, query_papers_from_db, get_query_last_fetched, update_query_timestamp, get_all_papers_from_db
from retrieval import hybrid_search
from summarizer import summarize
import time

TOP_K = 5
STALE_SECONDS = 7 * 24 * 3600  # 1 week
init_db()

def search_interface(query):
    if not query:
        return "Please enter a search query."

    last_fetched = get_query_last_fetched(query)
    if last_fetched is None or time.time() - last_fetched > STALE_SECONDS:
        arxiv = fetch_arxiv(query)
        semantic = fetch_semantic_scholar(query)
        all_new = arxiv + semantic
        save_papers(all_new)
        update_query_timestamp(query)

    local_results = query_papers_from_db(query)
    if len(local_results) < TOP_K:
        local_results = get_all_papers_from_db()

    ranked = hybrid_search(query, local_results)
    top = ranked[:TOP_K]
    summaries = summarize([paper[3] for paper in top])
    output = "\n\n".join(
        [f"**{paper[1]}**\n*{paper[2]}*\n\n{summary}" for paper, summary in zip(top, summaries)]
    )
    return output

demo = gr.Interface(fn=search_interface,
                    inputs="text",
                    outputs="markdown",
                    title="ResearchMate AI",
                    description="Search and summarize academic papers using hybrid semantic + keyword search.")

demo.launch()
