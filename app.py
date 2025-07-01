import gradio as gr
import time
from database import (
    init_db,
    fetch_arxiv,
    fetch_semantic_scholar,
    save_papers,
    query_papers_from_db,
    get_query_last_fetched,
    update_query_timestamp,
    get_all_papers_from_db
)
from retrieval import hybrid_search
from summarizer import summarize

TOP_K = 5
STALE_SECONDS = 7 * 24 * 3600  # 1 week

init_db()

def search_interface(query):
    if not query:
        return "❗ Please enter a search query."

    try:
        print(f"\n🔍 Received query: '{query}'")
        last_fetched = get_query_last_fetched(query)
        if last_fetched is None or time.time() - last_fetched > STALE_SECONDS:
            print("⏬ Fetching new data...")
            arxiv = fetch_arxiv(query)
            try:
                semantic = fetch_semantic_scholar(query)
            except Exception as e:
                print(f"⚠️ Semantic Scholar fetch failed: {e}")
                semantic = []
            all_new = arxiv + semantic
            save_papers(all_new)
            update_query_timestamp(query)
        else:
            print("✅ Using cached results.")

        local_results = query_papers_from_db(query)
        print(f"📄 Local results found: {len(local_results)}")

        if len(local_results) < TOP_K:
            print("⚠️ Not enough results, falling back to full DB.")
            local_results = get_all_papers_from_db()

        if not local_results:
            return "❌ No results found in database."

        ranked = hybrid_search(query, local_results)
        top = ranked[:TOP_K]

        abstracts = [paper[3] for paper in top]
        print("🧠 Summarizing results...")
        summaries = summarize(abstracts)

        output = "\n\n".join(
            [f"**{paper[1]}**\n*{paper[2]}*\n\n{summary}" for paper, summary in zip(top, summaries)]
        )
        return output

    except Exception as e:
        print(f"❌ Uncaught error: {e}")
        return "Something went wrong. Please try again later."

demo = gr.Interface(
    fn=search_interface,
    inputs="text",
    outputs="markdown",
    title="📚 ResearchMate AI",
    description="Search and summarize academic papers using hybrid semantic + keyword search."
)

demo.launch()
