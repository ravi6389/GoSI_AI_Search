import os
import trafilatura
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from collections import deque
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import numpy as np
import tiktoken
import hashlib

from tavily import TavilyClient
from firecrawl import Firecrawl
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
import os
from sentence_transformers import SentenceTransformer

import json
import re
from collections import deque
from typing import TypedDict

import streamlit as st
import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go

from tavily import TavilyClient
from firecrawl import Firecrawl
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI


MAX_SUBLINKS_PER_PAGE = 7

# -------------------------------------------------------
# 🔧 LOAD ENV
# -------------------------------------------------------
load_dotenv()
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
FIRECRAWL_API_KEY = st.secrets["FIRECRAWL_API_KEY"]


# -------------------------------------------------------
# IMPORT TOOLS
# -------------------------------------------------------


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# -------------------------------------------------------
# AZURE GPT-4o LLM
# -------------------------------------------------------

# llm = AzureChatOpenAI(
#                                 api_key = st.secrets["AZURE_API_KEY"],
#                                 azure_endpoint =st.secrets["AZURE_ENDPOINT"],
#                                 model = st.secrets["LLM_MODEL"],
#                                 api_version= st.secrets["AZURE_API_VERSION"],
#                                 temperature = 0.
#                                 )
llm = ChatGroq(temperature=0.8, groq_api_key=GROQ_API_KEY,
            #    model_name="llama-3.1-8b-instant", streaming=True)
            model_name="llama-3.3-70b-versatile", streaming=True)

# ============================================================
# CHUNK 1 — Imports, Keys, Helper Utilities, Search + Crawl
# ============================================================




# ============================================================
# 🔑 API KEYS (your existing variable names)
# ============================================================


tavily = TavilyClient(api_key=TAVILY_API_KEY)
firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)

# =====================================================================
# DISC BADGE RENDERER
# =====================================================================


# =========================================================
# DISC BADGE (Humantic Style)
# =========================================================
def linkedin_priority(url: str, intent: str):
    u = url.lower()

    if intent == "person_profile":
        if "/in/" in u:
            return 100
        if "/company/" in u:
            return 5
        if "/posts/" in u or "/feed/" in u or "/pulse/" in u:
            return 2
        return 1

    if intent == "company_profile":
        if "/company/" in u:
            return 100
        if "/in/" in u:
            return 5
        if "/posts/" in u or "/feed/" in u or "/pulse/" in u:
            return 2
        return 1

    return 1



def chunk_text(text, max_tokens=500, overlap=50, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap

    return chunks



def embed_texts(texts):
    """
    Free, local embeddings using sentence-transformers
    """
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))




def classify_linkedin_url(url: str):
    """
    Accurately classify LinkedIn URLs as either:
    - person_profile
    - company_profile
    - unknown

    Uses:
    1. URL structural rules (most accurate)
    2. Final fallback → LLM intent reasoning
    """

    u = url.lower()

    # ---------RULE 1: Pure structural classification----------
    # Person profiles ALWAYS have '/in/' pattern globally
    if re.search(r"linkedin\.com\/in\/[^\/]+\/?$", u):
        return "person_profile"

    # Company profiles ALWAYS have /company/<name>
    if "linkedin.com/company/" in u:
        return "company_profile"

    # ---------RULE 2: Check if LinkedIn FOLLOWER/EMPLOYEE metadata exists---------
    # Tavily sometimes returns metadata indicating company
    meta = url  # not ideal until passed raw metadata
    # You will apply this rule in planner node instead.

    # ---------RULE 3: Fallback → let LLM decide---------
    intent_prompt = """
    Decide if this URL belongs to:
    - a person's LinkedIn profile
    - a company's LinkedIn profile
    - or unknown.

    Return strict JSON:
    {"intent": "person_profile" | "company_profile" | "unknown"}
    """

    resp = llm.invoke([
        {"role": "system", "content": intent_prompt},
        {"role": "user", "content": url}
    ])

    try:
        return json.loads(resp.content.strip())["intent"]
    except:
        return "unknown"
def llm_general_summary(question, extracted_pages):
    """
    LLM decides whether output should be:
    - tabular
    - narrative text
    """

    system_prompt = """
You are a research summarization engine.

Based on the QUESTION and AVAILABLE DATA, decide the BEST output format.

You MUST return STRICT JSON in ONE of the following forms:

OPTION 1 — TABLE:
{
  "format": "table",
  "columns": ["col1", "col2", "..."],
  "rows": [
    ["value1", "value2", "..."],
    ...
  ]
}

OPTION 2 — TEXT:
{
  "format": "text",
  "content": "clear, factual research summary that answers user's question and can be easily understood by layman also"
}

Rules:
- Use TABLE only if data has multiple comparable entities or attributes
- Use TEXT for narrative or explanatory answers
- Do NOT invent data
- Use ONLY the provided extracted content
"""

    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps({
            "question": question,
            "data": extracted_pages
        })}
    ])

    return safe_json(llm, [{"role": "user", "content": resp.content}])


def detect_intent_llm(question, extracted_pages, search_urls=None):
    """
    MUCH more accurate intent detection:
    - Look at question
    - Look at URLs
    - Look at raw_content signals
    """

    # Build context for LLM
    context = {
        "question": question,
        "urls": [],
        "samples": []
    }

    # Add URLs (from search step)
    if search_urls:
        for u in search_urls:
            context["urls"].append(u["url"])
            raw = u.get("raw", {})
            snippet = raw.get("raw_content") or raw.get("content") or ""
            context["samples"].append({
                "url": u["url"],
                "snippet": snippet[:1500]
            })

    # Add extracted_pages (from crawl step)
    if extracted_pages:
        for p in extracted_pages:
            context["urls"].append(p["url"])
            md = p.get("content") or ""
            context["samples"].append({
                "url": p["url"],
                "snippet": md[:500]
            })

    system_prompt = """
You are an intent classifier for LinkedIn research.

You MUST choose exactly one of:
- "person_profile"
- "company_profile"
- "general_research"

Decision rules:
1. If any LinkedIn URL contains "/in/" → strongly person_profile.
2. If any LinkedIn URL contains "/company/" → strongly company_profile.
3. If raw_content mentions job titles, years, experience, followers → person_profile.
4. If raw_content mentions "employees", "industry", "headquarters", etc. → company_profile.
5. If the question is explicitly asking for a LinkedIn profile of a person → person_profile.
6. If the question is asking for a company profile → company_profile.
7. Otherwise → general_research.

Return STRICT JSON:
{"intent": "..."}    
"""

    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(context)}
    ])

    try:
        data = json.loads(resp.content.strip())
        # st.write("content of response is..", resp.content.strip())
        return data.get("intent", "general_research")
    
    except:
        return "general_research"



def render_disc_badge(disc_scores):
    if not disc_scores or not isinstance(disc_scores, dict):
        return ""

    dominant = max(disc_scores, key=disc_scores.get)
    colors = {"D": "#E74C3C", "I": "#F1C40F", "S": "#2ECC71", "C": "#3498DB"}
    color = colors.get(dominant, "#555")

    return f"""
    <div style="
        display:inline-block;
        padding:8px 20px;
        border-radius:30px;
        background:{color};
        color:white;
        font-weight:700;
        font-size:16px;">
        DISC Type: {dominant}
    </div>
    """

# =========================================================
# BIG FIVE RADAR
# =========================================================
def plot_big_five_radar(big_five):
    if not big_five:
        return None

    traits = list(big_five.keys())
    values = list(big_five.values())

    fig = go.Figure(data=go.Scatterpolar(
        theta=traits + [traits[0]],
        r=values + [values[0]],
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False
    )
    return fig

# =========================================================
# DISC WHEEL
# =========================================================
def plot_disc_wheel(disc):
    if not disc:
        return None

    fig = go.Figure(data=[go.Pie(
        labels=list(disc.keys()),
        values=list(disc.values()),
        hole=0.55
    )])
    return fig

# =========================================================
# IMAGE EXTRACTOR
# =========================================================
def extract_profile_image(raw, md):
    for key in ["primary_image", "image_url", "thumbnail", "favicon", "image"]:
        if key in raw and isinstance(raw[key], str) and raw[key].startswith("http"):
            return raw[key]

    m = re.search(r'property="og:image"\s*content="([^"]+)"', md or "")
    if m:
        return m.group(1)

    cdn = re.findall(r'https://media\.licdn\.com[^\s"\']+', md or "")
    if cdn:
        return cdn[0]

    g = re.findall(r'https://encrypted-tbn0\.gstatic\.com[^"\']+', md or "")
    if g:
        return g[0]

    return ""


# ============================================================
# Safe JSON Helper
# ============================================================
def safe_json(llm, messages, max_retries=4):
    """
    Forces LLM to output VALID JSON.
    Retries until success.
    """
    for _ in range(max_retries):
        resp = llm.invoke(messages)
        raw = resp.content.strip()

        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(raw)
        except:
            # Retry prompt
            messages.append({
                "role": "system",
                "content": "Return ONLY valid JSON. No explanations."
            })

    return {}

# ============================================================
# LinkedIn URL Detection
# ============================================================
def is_linkedin(url: str):
    return "linkedin.com" in url.lower()

# ============================================================
# Tavily Search Wrapper
# ============================================================
def search_tool(query):
    # st.write('Query is..', query['query'])
    return tavily.search(
        query=query['query'],
        max_results=4,
        include_raw_content=True
    ).get("results", [])

# ============================================================
# Firecrawl Wrapper for Non-LinkedIn URLs
# ============================================================
MAX_SUBLINKS_PER_PAGE = 6

def crawl_tool(url):
    try:
        out = firecrawl.scrape(url=url, formats=["markdown", "links", "html"])
        return {
            "markdown": out.markdown or "",
            "html": out.html or "",
            "links": out.links or [],
        }
    except:
        return {"markdown": "", "html": "", "links": []}

# ============================================================
# LLM Decider: Should we crawl this page?
# ============================================================
# def llm_crawl_decider(snippet, question):
#     decision = safe_json(llm, [
#         {
#             "role": "system",
#             "content": 'Return ONLY {"crawl": true} or {"crawl": false}.'
#         },
#         {"role": "user", "content": f"QUESTION:\n{question}"},
#         {"role": "user", "content": f"PAGE SNIPPET:\n{snippet}"}
#     ])
#     return decision.get("crawl", False)

# ============================================================
# Image Extractor (LinkedIn profile picture)
# ============================================================
# def extract_profile_image(raw, md):
#     """
#     Priority:
#     1) Tavily metadata (primary_image, thumbnail, etc.)
#     2) OG:image tags
#     3) LinkedIn CDN (media.licdn.com)
#     4) Google CDN fallback
#     """
#     # Tavily fields
#     for key in ["primary_image", "image_url", "thumbnail", "favicon", "image"]:
#         if key in raw and isinstance(raw[key], str) and raw[key].startswith("http"):
#             return raw[key]

#     # OG tags
#     if md:
#         m = re.search(r'property="og:image"\s*content="([^"]+)"', md)
#         if m:
#             return m.group(1)

#     # LinkedIn CDN
#     cdn = re.findall(r'https://media\.licdn\.com[^\s"\']+', md or "")
#     if cdn:
#         return cdn[0]

#     # Google fallback
#     gcache = re.findall(r'https://encrypted-tbn0\.gstatic\.com[^"\']+', md or "")
#     if gcache:
#         return gcache[0]

#     return ""

# ============================================================
# CHUNK 2 — Planner → Search → Crawl → Extract
# ============================================================

MAX_PAGES = 7
MAX_DEPTH = 2
# PROMISING_KEYWORDS = ["case", "note", "study", "customer", "client", "research"]
def llm_promising_keywords(question, max_keywords=6):
    """
    Extract promising crawl keywords from the user's question.
    Returns a list of lowercase keywords.
    """

    prompt = """
You extract keywords that help discover relevant sub-pages.

User question:
{question}

Return STRICT JSON only:
{{"keywords": ["keyword1", "keyword2", "..."]}}

Rules:
- Max {max_keywords} keywords
- Use nouns or noun phrases
- NO explanations
""".format(question=question, max_keywords=max_keywords)

    resp = llm.invoke([
        {"role": "system", "content": "Return ONLY valid JSON."},
        {"role": "user", "content": prompt}
    ])

    raw = resp.content.strip().replace("```json", "").replace("```", "")

    try:
        data = json.loads(raw)
        return [k.lower() for k in data.get("keywords", []) if isinstance(k, str)]
    except Exception:
        return []

# ------------------------------------------------------------
# PLANNER NODE
# ------------------------------------------------------------


def decompose_query(question):
    """
    Breaks a complex question into search-ready sub-queries
    """

    system_prompt = """
You are a research query decomposer.

Given a user question, identify:
1. Core intent
2. Key entities
3. Independent sub-questions needed to answer fully

Return STRICT JSON:
{
  "intent": "...",
  "entities": {
    "organizations": [],
    "products": [],
    "domains": []
  },
  "sub_queries": [
    "...",
    "..."
  ]
}

Rules:
- Sub-queries must be answerable independently
- Do NOT invent facts
- Prefer factual research steps
"""

    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ])

    return safe_json(llm, [{"role": "user", "content": resp.content}])

def llm_relevance_score(question, tavily_result, debug=False):
    text = tavily_result.get("raw_content") or tavily_result.get("content") or ""
    url = tavily_result.get("url", "")

    snippet = text[:1000]

    # FIXED: escape all literal braces → {{ }}
    prompt = """
You are a strict relevance scoring engine.

Rate how relevant this page is to the user's question.

Question:
{question}

URL:
{url}

Content Snippet:
{snippet}

Return STRICT JSON ONLY:
{{"score": <number between 0 and 1>}}
""".format(question=question, url=url, snippet=snippet)

    resp = llm.invoke([
        {"role": "system", "content": "Return ONLY JSON. No commentary."},
        {"role": "user", "content": prompt}
    ])

    raw = resp.content.strip().replace("```json", "").replace("```", "")

    try:
        data = json.loads(raw)
        score = float(data.get("score", 0))
    except Exception:
        score = 0.0

    if debug:
        st.write("🔍 LLM Relevance Debug →", url)
        st.write("Question is..", question)
        st.json({"score": score, "snippet": snippet})

    return score




def filter_relevant_results(results, threshold=0.30, debug=False):

    kept = []

    for r in results:
        search_query = r.get("search_query", "")
        raw = r.get("raw", {})

        text = raw.get("raw_content") or raw.get("content") or ""
        if not text.strip() or not search_query:
            continue
        
        # st.write('New serach query is..', search_query)
        score = llm_relevance_score(search_query, raw, debug=debug)

        if debug:
            st.write(f"Score for {r['url']} (query='{search_query}'): {score}")

        if score >= threshold:
            kept.append(r)

    return kept



def planner_node(state):
    question = state["question"].strip()

    decomposition = decompose_query(question)

    raw_subs = decomposition.get("sub_queries", [])

    new_questions = []
    # st.write('Raw_subs is..', raw_subs)
    for q in raw_subs:
       
        new_questions.append(q["query"])
     

    if not new_questions:
        new_questions = [question]

    state["new_questions"] = new_questions
    state["refined_question"] = " ".join(new_questions)

    st.write("🧠 New research questions:", new_questions)

    intent = detect_intent_llm(question, [])
    st.session_state["intent"] = intent

    search_queries = []
    for nq in new_questions:
        if intent == "person_profile":
            search_queries.append(nq + " linkedin profile")
        elif intent == "company_profile":
            search_queries.append(nq + " linkedin company")
        else:
            search_queries.append(nq)

    plan = {
        "mode": "auto",
        "search_queries": search_queries
    }
    # st.write('New questions are..')
    # st.write(" ".join(new_questions))
    return {
    "plan": plan,
    "new_questions": new_questions,
    "refined_question": " ".join(new_questions)
}





# ------------------------------------------------------------
# SEARCH NODE (Tavily)
# ------------------------------------------------------------
def search_node(state):
    new_questions = state["new_questions"]
    urls = []

    for nq in new_questions:
        results = search_tool({"query": nq})

        for r in results:
            urls.append({
                "url": r["url"],
                "raw": r,
                "new_question": nq   # 🔥 this is the anchor
            })
    # st.write('Urls are..', urls)
    return {"search_urls": urls}


# ------------------------------------------------------------
# CRAWL NODE (Tavily for LinkedIn + Firecrawl for other sites)
# ------------------------------------------------------------
# def crawl_node(state):

#     plan = state["plan"]

#     # DIRECT MODE
#     if plan.get("mode") == "direct_linkedin":
#         st.markdown("🔵 **Direct LinkedIn mode — skipping all crawling**")
#         return {"crawled_pages": plan.get("pages", [])}

#     question = state["question"]
#     search_urls = state["search_urls"]

#     queue = deque([(u, 0) for u in search_urls])
#     visited = set()
#     crawled = []

#     st.markdown("### 🌐 Crawling Activity")
#     progress = st.progress(0)
#     count = 0

#     while queue and len(crawled) < MAX_PAGES:
#         url_obj, depth = queue.popleft()
#         url = url_obj["url"]
#         raw = url_obj.get("raw", {})

#         if url in visited or depth > MAX_DEPTH:
#             continue
#         visited.add(url)

#         # -------------------------------
#         # LINKEDIN — use Tavily only
#         # -------------------------------
#         if is_linkedin(url):
#             st.write(f"🔵 **LinkedIn detected — skipping Firecrawl:** {url}")

#             md = raw.get("raw_content") or raw.get("content") or ""

#             crawled.append({
#                 "url": url,
#                 "markdown": md,
#                 "raw": raw,
#                 "search_query": url_obj.get("search_query", "")
#             })

#             count += 1
#             progress.progress(count / MAX_PAGES)
#             continue

#         # -------------------------------
#         # NON-LINKEDIN — ALWAYS crawl
#         # -------------------------------
#         st.write(f"🟢 Crawling: {url}")
#         crawled_page = crawl_tool(url)

#         crawled.append({
#             "url": url,
#             "markdown": crawled_page.get("markdown", ""),
#             "raw": raw,
#             "search_query": url_obj.get("search_query", "")
#         })

#         # -------------------------------
#         # Add promising sub-links
#         # -------------------------------
#         # sublinks = crawled_page.get("links", [])[:MAX_SUBLINKS_PER_PAGE]
#         # promising_keywords = plan.get("promising_keywords", [])

#         # for sub in sublinks:
#         #     s = sub.lower()
#         #     if any(k in s for k in promising_keywords):
#         #         queue.append((
#         #             {
#         #                 "url": sub,
#         #                 "raw": {"content": ""},
#         #                 "search_query": url_obj.get("search_query", "")
#         #             },
#         #             depth + 1
#         #         ))
#         # if depth == 0:
#         #     queue.append((
#         #         {
#         #             "url": sub,
#         #             "raw": {"content": ""},
#         #             "search_query": url_obj.get("search_query", "")
#         #         },
#         #         depth + 1
#         #     ))

#         count += 1
#         progress.progress(count / MAX_PAGES)

#     return {"crawled_pages": crawled}

def crawl_node(state):

    plan = state["plan"]

    # -------------------------------------------------
    # DIRECT LINKEDIN MODE
    # -------------------------------------------------
    if plan.get("mode") == "direct_linkedin":
        st.markdown("🔵 **Direct LinkedIn mode — skipping all crawling**")
        return {"crawled_pages": plan.get("pages", [])}

    search_urls = state["search_urls"]

    crawled = []
    visited = set()

    st.markdown("### 🌐 Crawling Activity")
    progress = st.progress(0)
    count = 0

    for url_obj in search_urls:

        if len(crawled) >= MAX_PAGES:
            break

        url = url_obj["url"]
        raw = url_obj.get("raw", {})
        search_query = url_obj.get("new_question", "")

        if url in visited:
            continue
        visited.add(url)

        # -------------------------------------------------
        # LINKEDIN → use Tavily content only
        # -------------------------------------------------
        if is_linkedin(url):
            st.write(f"🔵 LinkedIn detected — skipping crawl: {url}")

            md = raw.get("raw_content") or raw.get("content") or ""

            crawled.append({
                "url": url,
                "markdown": md,
                "raw": raw,
                "new_question": url_obj.get("new_question", "")
            })

            count += 1
            progress.progress(count / MAX_PAGES)
            continue

        # -------------------------------------------------
        # CHEAP RELEVANCE GATE (EMBEDDING-BASED)
        # -------------------------------------------------
        snippet = raw.get("raw_content") or raw.get("content") or ""

        if not snippet.strip():
            continue

        # if not cheap_relevance_gate(snippet, search_query):
        #     st.write(f"⛔ Skipped (cheap relevance): {url}")
        #     continue

        # -------------------------------------------------
        # NON-LINKEDIN → CRAWL FULL PAGE
        # -------------------------------------------------
        st.write(f"🟢 Crawling: {url}")
        crawled_page = crawl_tool(url)

        crawled.append({
            "url": url,
            "markdown": crawled_page.get("markdown", ""),
            "raw": raw,
            "new_question": url_obj.get("new_question", "")
        })

        count += 1
        progress.progress(count / MAX_PAGES)

    return {"crawled_pages": crawled}


# ------------------------------------------------------------
# EXTRACTOR NODE (Identify: person / company / generic page)
# ------------------------------------------------------------
import re

def extractor_node(state):
    pages = state["crawled_pages"]
    question = state["question"].lower()

    extracted = []
    st.markdown("### 📄 Extracted Pages Summary")

    intent = detect_intent_llm(question, pages)
    # st.write('Intent of pages is..', intent)
    st.session_state['intent'] = intent


    # ====================================
    # FIXED URL CLASSIFIERS
    # ====================================
    def is_real_profile(u: str) -> bool:
        return bool(re.search(r"linkedin\.com\/in\/[^\/?#]+\/?$", u))

    def is_real_company(u: str) -> bool:
        return bool(re.search(r"linkedin\.com\/company\/[^\/?#]+\/?$", u))

    # ====================================
    # UPDATED SCORING — USE STRICT RULES
    # ====================================
    def score_url(u: str) -> int:
        u = u.lower()

        if intent == "person_profile":
            if is_real_profile(u):
                return 200   # highest score
            if "linkedin.com/posts/" in u: 
                return 20
            if "linkedin.com/company/" in u:
                return 5
            return 1

        if intent == "company_profile":
            if is_real_company(u):
                return 200
            if is_real_profile(u):
                return 10
            if "linkedin.com/posts/" in u:
                return 5
            return 1

        return 1

    pages_sorted = sorted(pages, key=lambda p: score_url(p["url"]), reverse=True)

    # ====================================
    # EXTRACTION LOOP
    # ====================================
    for p in pages_sorted:
        url = p["url"].lower()
        text = p["markdown"]
        raw = p["raw"]

        extracted.append({
            "url": p["url"],
            "is_profile": is_real_profile(url),
            "is_company": is_real_company(url),
            "is_post": ("linkedin.com/posts/" in url),
            "content": text,
            "raw": raw,
            "image_url": extract_profile_image(raw, text),
            "search_query": p.get("new_question", "")
        })

        st.write(
            f"• {p['url']} — "
            f"Profile: {is_real_profile(url)}, "
            f"Company: {is_real_company(url)}, "
            f"Post: {'linkedin.com/posts/' in url}, "
            f"Score: {score_url(url)}"
        )
    # st.write('Extracted is..', extracted)
    return {"extracted": extracted}


# ============================================================
# CHUNK 3 — Fact Extractor + Persona Engine + Synthesizer
# ============================================================


# ------------------------------------------------------------
# PERSON FACT EXTRACTOR
# ------------------------------------------------------------
def extract_person_facts(extracted_pages):
    """
    Extracts clean LinkedIn PERSON factual data.
    Only reads what is actually present.
    """
    system_prompt = """
    You are a strict LinkedIn profile parser.
    Read ALL crawled pages and extract ONLY factual information.

    Return STRICT JSON:
    {
      "name": "",
      "headline": "",
      "current_title": "",
      "current_company": "",
      "location": "",
      "followers": "",
      "total_experience_years": "",
      "past_roles": [{"title":"", "company":"", "duration":""}],
      "education": [{"school":"", "degree":"", "years":""}],
      "skills": [],
      "about": "",
      "profile_url": ""
    }

    Rules:
    - NO hallucinations.
    - Use ONLY text from LinkedIn pages.
    - Prefer linkedin.com/in/ pages over posts.
    """

    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(extracted_pages)}
    ])

    return safe_json(llm, [{"role": "user", "content": resp.content}])


# ------------------------------------------------------------
# COMPANY FACT EXTRACTOR
# ------------------------------------------------------------
def extract_company_facts(extracted_pages):
    """
    Clean LinkedIn COMPANY profile extractor.
    """
    system_prompt = """
    Extract.LinkedIn COMPANY facts ONLY.
    Return strict JSON:
    {
        "name": "",
        "industry": "",
        "followers": "",
        "employees": "",
        "headquarters": "",
        "year_of_establishment": "",
        "company_url": ""
    }

    Rules:
    - Must read raw_content from LinkedIn company pages.
    - Do NOT guess. Only extract if present.
    - If multiple values appear, pick the most authoritative one.
    """

    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(extracted_pages)}
    ])

    return safe_json(llm, [{"role": "user", "content": resp.content}])


# ------------------------------------------------------------
# PERSONA ENGINE (Big Five, DISC, Work Style, Cold Call)
# ------------------------------------------------------------
def persona_engine(extracted_pages):
    system_prompt = """
    Produce STRICT JSON persona model:

{
  "big_five": {...},
  "disc": {...},
  "work_style": {...},
  "communication_style": {...},
  "sales_guidance": {...},
  "personalization": {...},
  "cold_call": {
      "what_to_say": "",
      "what_not_to_say": "",
      "script": {
          "0_20_seconds": "",
          "20_40_seconds": "",
          "40_60_seconds": "",
          "60_80_seconds": "",
          "80_100_seconds": "",
          "100_120_seconds": ""
      }
  },
  "buyer_intent": {...},
  "confidence": 0-1
}

Rules:
- Cold call script MUST be a **2-minute script only** (NOT 5 minutes).
- Break into **6 clean segments** of 20 seconds each.
- Each segment must contain **one crisp objective**, not long paragraphs.
- Make the talk-track aligned to this specific buyer’s persona.
- Avoid repetition between segments.
- Always provide actionable guidance; no fluff.
- JSON must be strictly valid.
"""
    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(extracted_pages)}
    ])

    return safe_json(llm, [{"role": "user", "content": resp.content}])


# ------------------------------------------------------------
# FINAL HUMANTIC-STYLE SYNTHESIZER (NO duplication)
# ------------------------------------------------------------
def prioritize_for_summary(extracted, intent):
    """
    Returns extracted pages sorted so that
    PERSON → /in/ first → posts → rest
    COMPANY → /company/ first → posts → rest
    GENERAL → no ordering
    """

    def score(p):
        u = p["url"].lower()

        if intent == "person_profile":
            if "linkedin.com/in/" in u: return 300
            if "linkedin.com/posts/" in u: return 150
            return 10

        if intent == "company_profile":
            if "linkedin.com/company/" in u: return 300
            if "linkedin.com/posts/" in u: return 150
            return 10

        return 1

    return sorted(extracted, key=score, reverse=True)

def cheap_relevance_gate(raw_text, search_query, threshold=0.45):
    snippet = raw_text[:1000]

    q_emb = embed_texts([search_query])[0]
    s_emb = embed_texts([snippet])[0]

    return cosine_similarity(q_emb, s_emb) >= threshold

def synthesis_node(state):
    extracted = state["extracted"]
    question = state["question"]

    # ---------------------------------------------------------
    # 1️⃣ LET LLM DECIDE THE TRUE INTENT
    # ---------------------------------------------------------
    intent = st.session_state['intent']
    st.write('I am in synthesis node and intent is..', st.session_state['intent'])
    # returns: person_profile / company_profile / general_research

    # ---------------------------------------------------------
    # 2️⃣ FLAGS FROM CRAWLER (used only to SUPPORT logic)
    # ---------------------------------------------------------
    has_profile_page = any(x["is_profile"] for x in extracted)
    has_company_page = any(x["is_company"] for x in extracted)

    # =========================================================
    # 🔵 PERSON MODE (when LLM says it is a person)
    # =========================================================
    if intent == "person_profile":

        # Fallback safeguard → if no profile page found but task is person
        if not has_profile_page:
            # keep processing anyway → persona engine still works
            pass

        facts = extract_person_facts(extracted)
        persona = persona_engine(extracted)

        # -------------------------------
        # 2-MIN COLD CALL SCRIPT
        # -------------------------------
        # Replace the 5-minute script with 2-minute structure
        narrative_prompt = """
        Create a HUMANTIC-style narrative report using ONLY the provided facts/persona.

        DO NOT repeat JSON fields. DO NOT hallucinate. 
        KEEP the writing tight, clean, professional.

        SECTIONS REQUIRED:

        1) LinkedIn Profile Summary (factual, crisp)
        2) LinkedIn followers
        2) Professional Overview (5–7 lines)
        3) Behavioural Summary (3–5 lines)
        4) Communication & Influence Style (short)
        5) Personalized Cold Call Guidance (short)
        6) **2-Minute Cold Call Script** → Break into:
           - First 30 seconds (Opener)
           - 30–60 seconds (Value Hook)
           - 60–90 seconds (Relevance Pitch)
           - 90–120 seconds (Close / CTA)
        7) Buyer Intent Signals (short)

        DO NOT mention persona JSON. DO NOT repeat big-five or DISC numbers.
        Just synthesize.
        """

        combined = {"facts": facts, "persona": persona}
        ranked = prioritize_for_summary(extracted, intent)

        final_answer = llm.invoke([
            {"role": "system", "content": narrative_prompt},
            {"role": "user", "content": json.dumps(ranked)}
        ]).content

        img = next((x["image_url"] for x in extracted if x["image_url"]), "")

        return {
            "mode": "person",
            "answer": final_answer,
            "facts": facts,
            "persona": persona,
            "image_url": img,
            "profile_url": facts.get("profile_url", "")
        }


   # ---------------------------------------------------------
    # 🟣 COMPANY MODE  (Bullet-point Summary)
    # ---------------------------------------------------------
    if intent == "company_profile":
        facts = extract_company_facts(extracted)

        narrative_prompt = """
        You are a Company Intelligence Writer.

        Transform the provided COMPANY FACTS into a strictly formatted,
        BULLET-POINT SUMMARY.
        STRICT REQUIREMENTS:
        ⚠ ABSOLUTE NON-NEGOTIABLE RULES ⚠
        - ONLY bullet points (each line MUST start with "- ")
        - NO paragraphs
        - NO continuous text longer than one sentence
        - NO storytelling
        - NO description beyond factual wording
        - Do NOT infer anything not explicitly present
        - If a field is not found → skip it
        - ALWAYS include Followers if present
        - ALWAYS place LinkedIn URL as the LAST bullet

        REQUIRED BULLET FORMAT (example):
        - **Overview:** <one crisp factual line>
        - **Industry:** <value>
        - **Headquarters:** <value>
        - **Employees:** <value>
        - **Followers:** <value>
        - **Founded:** <value>
        - **Key Focus Areas:** <comma-separated list, extracted ONLY from given text>
        - **LinkedIn:** <url>

        DO NOT return anything except bullet points.
        DO NOT write a paragraph summary.
        """

        final_answer = llm.invoke([
            {"role": "system", "content": narrative_prompt},
            {"role": "user", "content": json.dumps(facts)}
        ]).content

        img = next((x["image_url"] for x in extracted if x["image_url"]), "")

        return {
            "mode": "company",
            "answer": final_answer,
            "facts": facts,
            "persona": {},
            "image_url": img,
            "profile_url": facts.get("company_url", "")
        }

    # =========================================================
    # 🟢 GENERAL TOPIC MODE
    # =========================================================
    # Example queries:
    # - "recent developments at Nestle India"
    # - "market updates on EV batteries"
    # - "find me latest benchmarks"
    system_prompt = """
    Summarize clearly and factually using the extracted content. 
    No persona. No LinkedIn formatting. No hallucinations. 
    Just a clean research summary.
    """
    # st.write('New intent is..', intent)
    # =========================================================
# 🟢 GENERAL RESEARCH MODE (LLM decides table vs text)
# =========================================================
    if intent == "general_research":

        # --------------------------------------------------
    # 1️⃣ Use refined question for semantic matching
    # --------------------------------------------------
        # st.write('Going in semantic filter..')
        semantic_chunks = semantic_filter_chunks_by_new_question(
        extracted_pages=extracted,
        threshold=0.4,
        top_k=25,
        max_chunks_per_page=20
    )

        if not semantic_chunks:
            return {
                "mode": "generic",
                "answer": {
                    "summary": "No highly relevant information found.",
                    "sources": {}
                }
            }
        with st.spinner("🧠 Preparing final result… please wait."):
            final_answer = llm_general_summary_with_citations(
                question=question,          # ✅ ORIGINAL USER QUESTION
                semantic_chunks=semantic_chunks
            )

        return {
            "mode": "generic",
            "answer": final_answer,
            "sources": {str(i+1): c["url"] for i, c in enumerate(semantic_chunks)}
        }
    


def deduplicate_chunks(chunks):
    seen = set()
    unique = []

    for c in chunks:
        h = hashlib.sha1(c["content"].strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(c)

    return unique
def semantic_filter_chunks_by_new_question(
    extracted_pages,
    threshold=0.4,
    top_k=30,
    max_chunks_per_page=10   # 🔒 HARD CAP
):
    all_chunks = []



    for page in extracted_pages:
        if not page.get("search_query"):
            st.error(f"❌ Missing search_query for page: {page.get('url')}")
        raw = page.get("raw", {}) or {}
        text = (
        raw.get("raw_content")
        or raw.get("content")
        or page.get("content")
        or ""
    )   
        
        # st.write('Text is..', text)
        
        url = page.get("url", "")
        new_q = page.get("search_query", "")  # ✅ correct source
        
        # st.write('New question is...', new_q)
        if not text or not new_q:
            continue

        # Embed the NEW research question once
        q_emb = embed_texts([new_q])[0]

        # ----------------------------
        # Chunk + CAP at 10 chunks
        # ----------------------------
        chunks = chunk_text(text, max_tokens=500)
        chunks = chunks[:max_chunks_per_page]   # 🔒 CAP HERE

        if not chunks:
            continue

        chunk_embs = embed_texts(chunks)

        for chunk, emb in zip(chunks, chunk_embs):
            # st.write('Chunk is..', chunk)
            
            score = cosine_similarity(q_emb, emb)

            # st.write('Score is..', score)

            if score >= threshold:
                all_chunks.append({
                    "url": url,
                    "content": chunk.strip(),
                    "similarity": round(float(score), 3),
                    "matched_question": new_q
                })

    if not all_chunks:
        return []

    # Deduplicate identical content
    all_chunks = deduplicate_chunks(all_chunks)

    # Rank globally by similarity
    all_chunks = sorted(
        all_chunks,
        key=lambda x: x["similarity"],
        reverse=True
    )

    # Global safety cap (LLM input control)
    return all_chunks[:top_k]

# def semantic_filter_chunks(extracted_pages, threshold=0.7, top_k=20):
#     """
#     Semantic filtering against SEARCH INTENT, not user question
#     """

#     all_chunks = []

#     for page in extracted_pages:
#         text = page.get("content", "")
#         url = page.get("url", "")
#         search_query = page.get("search_query", "")

#         if not text.strip() or not search_query:
#             continue

#         # 👇 Embed the SEARCH QUERY
#         query_embedding = embed_texts([search_query])[0]

#         chunks = chunk_text(text, max_tokens=500)
#         chunk_embeddings = embed_texts(chunks)

#         for chunk, emb in zip(chunks, chunk_embeddings):
#             st.write('Chunk is...', chunk)
#             st.write('Search query is..', search_query)

#             score = cosine_similarity(query_embedding, emb)

#             st.write('Cosine similarity is..', score)

#             if score >= threshold:
#                 all_chunks.append({
#                     "url": url,
#                     "content": chunk.strip(),
#                     "similarity": round(float(score), 3),
#                     "search_query": search_query
#                 })

#     if not all_chunks:
#         return []

#     all_chunks = deduplicate_chunks(all_chunks)
#     all_chunks = sorted(all_chunks, key=lambda x: x["similarity"], reverse=True)

#     return all_chunks[:top_k]



def llm_general_summary_with_citations(question, semantic_chunks):

    # Number chunks for citations
    numbered_chunks = []
    source_map = {}

    for i, c in enumerate(semantic_chunks, start=1):
        numbered_chunks.append({
            "id": i,
            "content": c["content"],
            "url": c["url"]
        })
        source_map[str(i)] = c["url"]

    system_prompt = """
You are a senior research analyst preparing an executive-ready research brief.

Your task:
- Synthesize the provided evidence into a clear, professional summary
- Explain relationships and relevance logically
- Maintain a neutral, factual tone

Formatting rules:
- Use Markdown (NOT JSON)
- Use clear section headers (##, ###)
- Prefer bullet points where helpful
- Keep paragraphs concise
- Use inline numeric citations like [1], [2]
- End with a section titled '### 🔗 Sources' listing each citation with its URL

Content rules:
- Use ONLY the provided evidence
- Do NOT invent facts or sources
- Clearly distinguish facts from inferred relevance
- If direct evidence is missing, explicitly say so

Write as if the output will be read by senior leadership.
"""


    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps({
                "question": question,
                "evidence": numbered_chunks
            })
        }
    ])

    # return safe_json(llm, [{"role": "user", "content": resp.content}])
    return resp.content


# =====================================================================
# AGENT GRAPH — DEFINES THE FULL STATE MACHINE
# =====================================================================

class ResearchState(TypedDict):
    question: str

    # Planner outputs
    plan: dict
    new_questions: list
    refined_question: str

    # Search / crawl
    search_urls: list
    crawled_pages: list
    extracted: list

    # Final outputs
    mode: str
    answer: dict | str
    facts: dict
    persona: dict
    image_url: str
    profile_url: str



# -----------------------------
# BUILD GRAPH
# -----------------------------
graph = StateGraph(ResearchState)

# Add nodes (declared in chunks 1–3)
graph.add_node("planner", planner_node)
graph.add_node("search", search_node)
graph.add_node("crawl", crawl_node)
graph.add_node("extract", extractor_node)
graph.add_node("synthesize", synthesis_node)

# Connect nodes
graph.add_edge("planner", "search")
graph.add_edge("search", "crawl")
graph.add_edge("crawl", "extract")
graph.add_edge("extract", "synthesize")

# Entry + Exit
graph.set_entry_point("planner")
graph.set_finish_point("synthesize")

# IMPORTANT: compile graph ONCE
graph = graph.compile()

# =====================================================================
# STREAMLIT UI — HYBRID HUMANTIC LAYOUT (FINAL)
# =====================================================================

st.set_page_config(page_title="GoSi - Agentic AI enabled Web App", layout="wide")
st.title("🧠 GoSi -Agentic AI enabled Web App -  Ask me Anything!!!")
st.markdown("Works on publicly available data")

# =====================================================================
# DIRECT LINKEDIN URL HANDLING
# =====================================================================

import re

def normalize_query(user_query: str):
    """
    If user enters a LinkedIn URL:
    → we DO NOT search
    → we DO NOT add keywords
    → we FORCE the system to process ONLY that URL
    """

    q = user_query.strip()

    if q.startswith("http://") or q.startswith("https://"):
        if "linkedin.com/in/" in q or "linkedin.com/company/" in q:
            # important: no search keywords added
            return q  

    return q

query = st.text_area("Enter LinkedIn URL or Search Query:", height=110)
run_btn = st.button("Analyze")

if run_btn and query.strip():

    with st.spinner("Running Deep Multi-Source Analysis…"):
        norm_query = normalize_query(query)
        result = graph.invoke({"question": norm_query})

    # --------------------------------------------------------
    # UNPACK RESULT
    # --------------------------------------------------------
    mode = result.get("mode")
    answer = result.get("answer", "")

    facts = result.get("facts", {}) or {}
    persona = result.get("persona", {}) or {}

    image_url = result.get("image_url", "")
    profile_url = result.get("profile_url", "")

    # ========================================================
    # 🧭 MODE DISPLAY
    # ========================================================
    # mode_label = {
    #     "person": "👤 PERSON PROFILE MODE",
    #     "company": "🏢 COMPANY PROFILE MODE",
    #     "generic": "🌐 GENERAL RESEARCH MODE"
    # }.get(mode, "🌐 GENERAL MODE")

    mode = st.session_state['intent']
    st.markdown(f"### {mode}")

    # ========================================================
    # 🟪 CREATE TABS
    # ========================================================
    tab_summary, tab_persona, tab_coldcall, tab_links, tab_debug = st.tabs(
        ["📄 Summary", "🧠 Persona Dashboard", "📞 Cold Call Intelligence",
         "🔗 Crawled Links", "🐞 Debug JSON"]
    )

    # ========================================================
    # 📄 TAB 1 — SUMMARY
    # ========================================================
    with tab_summary:

        if mode == "generic":

            output = answer  # this is dict now

            if isinstance(output, dict) and output.get("format") == "table":
                st.markdown("### 📊 Research Summary")
                df = pd.DataFrame(output["rows"], columns=output["columns"])
                st.dataframe(df, use_container_width=True)

            elif isinstance(output, dict) and output.get("format") == "text":
                st.markdown("### 📄 Research Summary")
                st.write(output["content"])

            else:
                # fallback safety
                st.write(answer)

        # ---------------------------
        # PERSON MODE UI
        # ---------------------------
        if mode == "person_profile":

            col1, col2 = st.columns([1, 3])

            with col1:
                if image_url:
                    st.image(image_url, width=260)
                else:
                    st.info("No profile image detected.")

            with col2:
                # DISC Badge
                disc_scores = persona.get("disc", {})
                if disc_scores:
                    st.markdown(render_disc_badge(disc_scores),
                                unsafe_allow_html=True)

                # High-level identity block
                st.markdown(f"""
                # **{facts.get("name", "Unknown Name")}**
                ### {facts.get("headline", "")}

                **{facts.get("current_title","")} — {facts.get("current_company","")}**  
                📍 {facts.get("location","unknown")}  
                ⭐ Experience: **{facts.get("total_experience_years","unknown")} years**  
                🌐 Followers: **{facts.get("followers","unknown")}**
                """)

                if profile_url:
                    st.markdown(f"🔗 **LinkedIn:** {profile_url}")

            st.markdown("---")
            st.markdown("### 📝 Humantic-Style Summary")
            st.write(answer)

        # ---------------------------
        # COMPANY MODE UI
        # ---------------------------
        elif mode == "company_profile":
            col1, col2 = st.columns([1, 3])

            with col1:
                if image_url:
                    st.image(image_url, width=250)

            with col2:
                st.markdown(f"""
                # 🏢 **{facts.get("name","Unknown Company")}**
                **Industry:** {facts.get("industry","unknown")}  
                **Followers:** {facts.get("followers","unknown")}  
                **Employees:** {facts.get("employees","unknown")}  
                **Headquarters:** {facts.get("headquarters","unknown")}  
                **Established:** {facts.get("year_of_establishment","unknown")}  
                """)

                if facts.get("company_url"):
                    st.markdown(f"🔗 **LinkedIn:** {facts.get('company_url')}")

            st.markdown("---")
            st.markdown("### 📝 Company Summary")
            st.write(answer)

        # ---------------------------
        # GENERIC MODE UI
        # ---------------------------
        else:
            st.markdown("### 📄 Research Summary")
            st.write(answer)

    # ========================================================
    # 🧠 TAB 2 — PERSONA DASHBOARD
    # ========================================================
    with tab_persona:
        if mode != "person_profile":
            st.info("Persona insights available only in PERSON MODE.")
        else:
            colA, colB = st.columns(2)

            with colA:
                st.markdown("### 🔷 Big Five Radar Chart")
                bigfive = persona.get("big_five", {})
                if bigfive:
                    fig = plot_big_five_radar(bigfive)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Big Five values missing.")

            with colB:
                st.markdown("### 🧭 DISC Wheel")
                disc = persona.get("disc", {})
                if disc:
                    fig = plot_disc_wheel(disc)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("DISC values missing.")

            st.markdown("---")
            st.markdown("### 🧩 Work Style")
            st.json(persona.get("work_style", {}))

            st.markdown("### 🗣 Communication Style")
            st.json(persona.get("communication_style", {}))

            st.markdown("### 💡 Sales Guidance")
            st.json(persona.get("sales_guidance", {}))

            st.markdown("### 🎯 Personalization Insights")
            st.json(persona.get("personalization", {}))

            st.markdown("### 🛒 Buyer Intent Signals")
            st.json(persona.get("buyer_intent", {}))

            st.markdown("### 🔐 Confidence Score")
            conf = persona.get("confidence")
            if conf is not None:
                st.write(f"{conf:.2f}")

    # ========================================================
    # 📞 TAB 3 — COLD CALL INTELLIGENCE
    # ========================================================
    with tab_coldcall:
        if mode != "person_profile":
            st.info("Cold call intelligence is available only for individuals.")
        else:
            cold = persona.get("cold_call", {})

            st.markdown("### 🟢 What to Say")
            st.write(cold.get("what_to_say", ""))

            st.markdown("### 🔴 What NOT to Say")
            st.write(cold.get("what_not_to_say", ""))

            st.markdown("### 📞 Full 5-Minute Cold Call Script")
            st.write(cold.get("script", ""))

    # ========================================================
    # 🔗 TAB 4 — CRAWLED LINKS
    # ========================================================
    with tab_links:
        st.markdown("### 🌍 All Crawled Links")
        crawled_list = [p["url"] for p in result.get("crawled_pages", [])]
        st.write("\n".join(crawled_list) if crawled_list else "No links crawled")

    # ========================================================
    # 🐞 TAB 5 — DEBUG JSON
    # ========================================================
    with tab_debug:
        st.markdown("### 🐞 Full JSON Output")
        st.json(result)





