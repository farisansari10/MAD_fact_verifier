import wikipedia
from ddgs import DDGS


def search_web(query: str, max_results: int = 3) -> str:
    try:
        results = []

        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"SOURCE: {r['title']}\n{r['body']}"
                )

        if not results:
            return "No web results found."

        return "\n\n---\n\n".join(results)

    except Exception as exc:
        return f"Web search failed: {exc}"


def search_wikipedia(query: str, sentences: int = 5) -> str:
    try:
        titles = wikipedia.search(query, results=3)

        if not titles:
            return "No Wikipedia articles found."

        summaries = []

        for title in titles[:2]:
            try:
                summary = wikipedia.summary(
                    title,
                    sentences=sentences,
                    auto_suggest=False
                )
                summaries.append(f"ARTICLE: {title}\n{summary}")

            except wikipedia.exceptions.DisambiguationError:
                continue

            except wikipedia.exceptions.PageError:
                continue

            except Exception:
                continue

        if not summaries:
            return "Could not retrieve Wikipedia content."

        return "\n\n---\n\n".join(summaries)

    except Exception as exc:
        return f"Wikipedia search failed: {exc}"