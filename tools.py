import wikipedia                   # library to search and fetch Wikipedia articles
from ddgs import DDGS  # library to search the web using DuckDuckGo — free, no API key needed


def search_web(query: str, max_results: int = 3) -> str:
    # this function searches the web and returns results as plain text
    # Agent 1 (GPT-4o-mini) calls this before forming its opinion on a claim
    # query       = what to search for e.g. "Pakistan floods 2024"
    # max_results = how many results to fetch — 3 is enough, more = slower and costlier

    try:                                                        # try to run — if anything fails jump to except
        results = []                                            # empty list to collect search results

        with DDGS() as ddgs:                                    # open a DuckDuckGo search session
            for r in ddgs.text(query, max_results=max_results): # loop through each result
                results.append(                                 # add each result to our list
                    f"SOURCE: {r['title']}\n{r['body']}"        # format: title on top, snippet below
                )

        if not results:                                         # if list is still empty after searching
            return "No web results found."                      # return this so agent knows nothing was found

        return "\n\n---\n\n".join(results)                      # join all results into one string with dividers

    except Exception as exc:                                    # if ANYTHING goes wrong — network, rate limit etc.
        return f"Web search failed: {exc}"                      # return error message instead of crashing


def search_wikipedia(query: str, sentences: int = 5) -> str:
    # this function searches Wikipedia and returns article summaries as plain text
    # Agent 2 (Mistral 7B) calls this to get background context on a news claim
    # query     = what to search for e.g. "Russia Ukraine war"
    # sentences = how many sentences to pull from each article — 5 is enough

    try:                                                        # try to run — if anything fails jump to except
        titles = wikipedia.search(query, results=3)             # search Wikipedia — returns list of article titles only

        if not titles:                                          # if no titles found
            return "No Wikipedia articles found."               # return this message

        summaries = []                                          # empty list to collect article summaries

        for title in titles[:2]:                                # loop through top 2 titles only — enough context
            try:                                                # inner try — each article could fail individually
                summary = wikipedia.summary(
                    title,                                      # fetch summary for this specific title
                    sentences=sentences,                        # only get first 5 sentences — keeps it short
                    auto_suggest=False                          # don't silently redirect to a different article
                )
                summaries.append(f"ARTICLE: {title}\n{summary}") # add formatted summary to list

            except wikipedia.exceptions.DisambiguationError:   # "Mercury" = planet? element? car? — skip it
                continue                                        # move to next title

            except wikipedia.exceptions.PageError:             # title found in search but page won't load
                continue                                        # move to next title

            except Exception:                                   # any other error on this specific article
                continue                                        # move to next title

        if not summaries:                                       # if we got nothing after looping all titles
            return "Could not retrieve Wikipedia content."      # return this message

        return "\n\n---\n\n".join(summaries)                    # join summaries into one string with dividers

    except Exception as exc:                                    # if the whole function crashes
        return f"Wikipedia search failed: {exc}"                # return error message instead of crashing