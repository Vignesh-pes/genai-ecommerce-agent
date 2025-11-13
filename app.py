import io
import os
import re
import time
import json
import altair as alt
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from datetime import datetime

# LangChain Core (tools + SQL utilities)
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Some LangChain installs expose tool decorator from different places.
# We attempt to import; if not available we provide a noop decorator.
try:
    from langchain_core.tools import tool
except Exception:
    try:
        from langchain.tools import tool
    except Exception:
        def tool(func=None, **_):
            def _d(f):
                return f
            return _d(func) if func else _d

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "sqlite:///olist.sqlite")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")  # optional
DEFAULT_MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", 6))
TODAY = datetime.now().strftime("%Y-%m-%d")

# -----------------------------
# PAGE CONFIG & STYLING (NEW)
# -----------------------------
st.set_page_config(page_title="E-commerce Data Agent", layout="wide")

# (NEW) Custom CSS for a dashboard look
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #1E1E1E; /* Dark background */
        color: #FAFAFA;
    }

    /* Titles and headers */
    h1, h2, h3, h4, h5 {
        color: #00b4d8; /* Title color */
    }

    /* Streamlit's bordered container (for KPIs) */
    [data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
        background-color: #2D2D2D; /* Card background */
        border-radius: 10px;
        border: 1px solid #444;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Chat message containers */
    [data-testid="chat-message-container"] {
        background-color: #2D2D2D;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* User message */
    [data-testid="chat-message-container"] {
        /* ... existing styles ... */
    }
    
    /* Assistant message */
    [data-testid="chat-message-container"]:has([data-testid="chat-avatar-assistant"]) {
         background-color: #2D2D2D;
    }

    /* Chat input box */
    [data-testid="stChatInput"] {
        background-color: #2D2D2D;
    }

    /* Buttons */
    .stButton > button {
        background-color: #0077b6;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #0096c7;
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #252526;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# STREAMING UTIL (typing UX)
# -----------------------------
def stream_text(text: str, placeholder, delay: float = 0.004):
    """Type-out UX: writes text into the supplied Streamlit placeholder word-by-word."""
    if not text:
        placeholder.markdown("")
        return
    words = text.split()
    out = ""
    for i, w in enumerate(words):
        out += (" " if i > 0 else "") + w
        # use markdown to preserve simple formatting
        placeholder.markdown(out)
        time.sleep(delay)
    # final
    placeholder.markdown(out)

# -----------------------------
# CONVERSATION WINDOW MEMORY
# -----------------------------
class ConversationWindowMemory:
    """
    Keeps a window of the last K messages in session_state (user + assistant).
    """
    def __init__(self, k: int = DEFAULT_MEMORY_WINDOW):
        self.k = int(k)
        if "conv_window_messages" not in st.session_state:
            st.session_state["conv_window_messages"] = []

    def add_user(self, text: str):
        if not text:
            return
        st.session_state["conv_window_messages"].append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str):
        if not text:
            return
        st.session_state["conv_window_messages"].append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self):
        st.session_state["conv_window_messages"] = st.session_state["conv_window_messages"][-(self.k * 2):]

    def get_messages(self):
        return st.session_state.get("conv_window_messages", [])

    def clear(self):
        st.session_state["conv_window_messages"] = []

# -----------------------------
# PAGE & SIDEBAR
# -----------------------------
# Main title
st.markdown("<h1 style='text-align: center; color:#00b4d8;'>🛍 E-commerce Data Agent</h1>", unsafe_allow_html=True)
st.caption(f"Date : ({TODAY})", unsafe_allow_html=True)


st.sidebar.header("Controls")
tone_mode = st.sidebar.radio("Assistant tone", ("🧾 Professional Analyst", "💬 Friendly"))
debug = st.sidebar.checkbox("Show debug/memory", value=False)

# Safe new chat
if "new_chat_request" not in st.session_state:
    st.session_state["new_chat_request"] = False
if st.sidebar.button("🆕 New chat (clear history & memory)"):
    st.session_state["new_chat_request"] = True
    st.rerun() # Rerun to reflect cleared state

# -----------------------------
# Export conversation — only PDF (as requested)
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Export conversation")
def _export_pdf_if_possible(msgs):
    # try to generate PDF, if reportlab not installed show helpful message
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        return None, "No module named 'reportlab'"
    blob = []
    for m in msgs:
        # (FIX) Handle complex summaries (dicts) in export
        content = m['content']
        if isinstance(content, dict):
            blob.append(f"{m['role'].upper()}:\n")
            blob.append(f"  Key Finding: {content.get('key_finding', 'N/A')}\n")
            blob.append(f"  Why It Matters: {content.get('why_it_matters', 'N/A')}\n")
            blob.append(f"  Next Step: {content.get('next_step', 'N/A')}\n\n")
        else:
            blob.append(f"{m['role'].upper()}:\n{str(content)}\n\n")
    blob = "".join(blob)
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    w, h = letter
    y = h - 40
    for line in blob.splitlines():
        if y < 60:
            c.showPage()
            y = h - 40
        safe = re.sub(r"[\U00010000-\U0010FFFF]", " ", line)
        c.drawString(40, y, safe[:180])
        y -= 12
    c.save()
    buffer.seek(0)
    return buffer, None

if st.sidebar.button("Export PDF"):
    msgs = st.session_state.get("messages", [])
    pdf_buffer, errmsg = _export_pdf_if_possible(msgs)
    if pdf_buffer:
        st.sidebar.download_button("Download conversation (PDF)", data=pdf_buffer, file_name="chat_session.pdf", mime="application/pdf")
    else:
        st.sidebar.warning(f"PDF export unavailable ({errmsg}).")

st.sidebar.markdown("---")

# (REMOVED) "Quick history view" section was here. It was redundant.

# (NEW) Added a cleaner "About" section
st.sidebar.header("About")
st.sidebar.info(
    "This agentic dashboard uses Google's Gemini model to answer questions "
    "about the Olist e-commerce dataset using a suite of custom tools."
)

# -----------------------------
# DB + LLM
# -----------------------------
@st.cache_resource
def get_db_engine():
    return create_engine(DB_URL, connect_args={"check_same_thread": False})

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.22)

engine = get_db_engine()
db = SQLDatabase(engine=engine)
llm = get_llm()

# SQL toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = toolkit.get_tools()

# -----------------------------
# SMART UTILITY TOOLS (docstrings required)
# -----------------------------
@tool
def get_order_location_tool(order_id: str):
    """
    Use this tool to get the latitude/longitude and city/state for a *specific order_id*.
    The user must provide the order_id. Do not use this for general location questions.
    """
    order_id = (order_id or "").strip()
    if not order_id:
        return "No order_id provided."
    sql = """
    SELECT o.order_id, c.customer_city, c.customer_state, g.geolocation_lat, g.geolocation_lng
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    LEFT JOIN geolocation g ON c.customer_zip_code_prefix = g.geolocation_zip_code_prefix
    WHERE o.order_id = :order_id LIMIT 1;
    """
    try:
        df = pd.read_sql_query(sql, engine, params={"order_id": order_id})
        if df.empty:
            return f"Order {order_id} not found."
        row = df.iloc[0].to_dict()
        lat = row.get("geolocation_lat")
        lon = row.get("geolocation_lng")
        city = row.get("customer_city") or ""
        state = row.get("customer_state") or ""
        if lat is None or lon is None:
            return {"order_id": order_id, "address": f"{city}, {state}", "lat": None, "lon": None}
        return {"order_id": order_id, "address": f"{city}, {state}", "lat": float(lat), "lon": float(lon)}
    except Exception as e:
        return f"Error in get_order_location: {e}"

@tool
def translate_text_tool(text: str, target_language: str = "en"):
    """
    Use this tool to translate a piece of text into a target language.
    Default target language is English (en).
    """
    if not text:
        return "No text provided."
    prompt = f"Translate the following to {target_language}:\n\n{text}"
    try:
        resp = llm.invoke(prompt)
        if hasattr(resp, "content"):
            return resp.content.strip()
        return str(resp).strip()
    except Exception as e:
        return f"Translation error: {e}"

@tool
def define_term_tool(term: str):
    """
    Use this tool to get a concise, one-paragraph definition for an e-commerce term.
    """
    if not term:
        return "No term provided."
    prompt = f"Define this e-commerce term concisely: {term}\nProvide one short paragraph."
    try:
        resp = llm.invoke(prompt)
        if hasattr(resp, "content"):
            return resp.content.strip()
        return str(resp).strip()
    except Exception as e:
        return f"Definition error: {e}"

@tool
def external_search_tool(query: str):
    """
    Use this tool to find external information, such as current events, news, or general knowledge
    about topics *outside* the Olist database (e.g., "latest e-commerce trends in Brazil").
    """
    if not query:
        return "No query provided."
    
    # (FIX) Mock the API response for the demo
    # We removed the SEARCH_API_KEY check.
    # This provides a plausible, hard-coded response.
    print(f"External Search (Mock) called with query: {query}")
    
    query_lower = query.lower()
    
    if "e-commerce trends in brazil" in query_lower:
        return """
        (Mocked Search Result)
        Latest e-commerce trends in Brazil include:
        1.  **Pix Payments:** The rapid adoption of the 'Pix' instant payment system is dominating online checkouts.
        2.  **Cross-Border Commerce:** Consumers are increasingly buying from international stores.
        3.  **Mobile-First:** The majority of online shopping is now done via smartphones.
        """
    elif "e-commerce" in query_lower:
        return "(Mocked Search Result) E-commerce (electronic commerce) is the buying and selling of goods and services over the internet."
    else:
        return f"(Mocked Search Result) A top search snippet for '{query}' would appear here. This feature is currently mocked for demo purposes."

custom_tools = [get_order_location_tool, translate_text_tool, define_term_tool, external_search_tool]
tools_for_agent = sql_tools + custom_tools

# -----------------------------
# AGENT: system prompt + creation
# -----------------------------
SYSTEM_PROMPT = f"""
You are a data analyst for the Olist E-commerce dataset.
- Reply in plain text only (no raw tool traces).
- Keep answers concise and factual.
- Today's date is: {TODAY}

--- (NEW) TOOL ROUTING RULES (CRITICAL) ---
1.  **For Olist DB Questions:** Use SQL tools. These are questions about *specific* data in the database, e.g., "total sales," "top 5 products," "orders per month," "average order value."
2.  **For General Knowledge:** Use `external_search_tool`. These are questions about topics *outside* the database, e.g., "latest e-commerce trends," "news about Brazil," "what is e-commerce?"
3.  **For Definitions:** Use `define_term_tool` (e.g., "what is AOV?").
4.  **For Translations:** Use `translate_text_tool` (e.g., "translate 'olá' to English").
5.  **For Order Locations:** Use `get_order_location_tool` *only* if the user provides a *specific order_id*.

--- (NEW) SQL GENERATION RULES ---
- **(FIX) If the user asks for data, your response MUST contain *only* the valid SQL query, and nothing else.**
- **(FIX) Do NOT include any natural language, summary, or commentary in your response.** A separate function will generate the summary *after* the SQL is run.
- **(FIX) If you use a non-SQL tool (like search or define), then you *should* provide a natural language answer.**
- **Provide More Context:** When a user asks for a simple metric (e.g., 'average order value' or 'total sales'), do NOT just return the single number.
- **Instead, you MUST try to provide more context by grouping by a relevant dimension.** For example:
    - "What are total sales?" -> (Provide SQL for) "total sales, broken down by month." (GROUP BY strftime('%Y-%m', ...))
    - "What is the average order value?" -> (Provide SQL for) "average order value by product category." (GROUP BY product_category_name_english)
- **CRITICAL SQL LOGIC for 'Average Order Value'**: To calculate 'Average Order Value' (AOV), you must first SUM the `price` and `freight_value` for each `order_id` in a subquery, and then take the AVG of that sum.
- DO NOT just `AVG(price + freight_value)` directly, as this calculates average item value, not average order value.
- **(FIX) For 'Percentage Share' Queries:** To calculate percentage share (e.g., by payment type), you can simply SUM(price + freight_value) grouped by the dimension. You do *not* need to use the complex AOV subquery for this.
- **(FIX) CRITICAL:** Your SQL query must be a single, valid SQL statement. Do NOT include any natural language, commentary, or your summary *inside* the SQL query block.
- If you plan to run SQL, include *only* the SQL statement. For any computed columns (e.g., SUM, COUNT, AVG), you MUST use a clear alias (e.g., `AS TotalSales` or `AS AverageValue`).
- **(FIX) After answering, do NOT suggest follow-ups.** A separate function will handle that.
"""
agent = create_agent(model=llm, tools=tools_for_agent, system_prompt=SYSTEM_PROMPT)

# store agent & memory wrappers
if "agent" not in st.session_state:
    st.session_state["agent"] = agent
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "memory_win" not in st.session_state:
    st.session_state["memory_win"] = ConversationWindowMemory(k=DEFAULT_MEMORY_WINDOW)


# new chat handling
if st.session_state.get("new_chat_request", False):
    st.session_state["messages"] = []
    st.session_state["memory_win"].clear()
    st.session_state["pending_followup"] = None
    st.session_state["new_chat_request"] = False
    st.success("Started a new chat — history & window memory cleared.")
    st.rerun()

# -----------------------------
# UTILITIES: robust extract & humanize
# -----------------------------
def clean_sql(raw_sql):
    """Normalize SQL: strip markdown/code fences, take only the first valid statement."""
    if not raw_sql:
        return None
    cleaned = re.sub(r"```(sql)?", "", raw_sql, flags=re.IGNORECASE).strip()
    
    # (FIX) Prioritize parsing Common Table Expressions (WITH clauses)
    m_cte = re.search(r"(?si)(with\s.+select\s.+?;)", cleaned)
    if m_cte:
        cleaned = m_cte.group(1).strip()
    else:
        # Fallback to simple SELECT
        m_select = re.search(r"(?si)(select\s.+?;)", cleaned)
        if m_select:
            cleaned = m_select.group(1).strip()
        else:
             # No semicolon — still might be valid SELECT; take up to a newline/question
            m = re.search(r"(?si)(select\s.+?)(\n|$)", cleaned)
            if m:
                cleaned = m.group(1).strip()
    
    # Drop trailing commentary words commonly appended by LLMs
    cleaned = re.split(r"(?i)\b(what|why|how|note|tip|insight)\b", cleaned)[0].strip()
    
    # Keep only the first statement (no semicolons chained)
    cleaned = (cleaned.split(";")[0].strip() + ";") if cleaned else ""
    
    # --- (FIX) Stricter Check ---
    # Must be SELECT or WITH and MUST contain FROM to be valid
    if (cleaned.lower().startswith("select") or cleaned.lower().startswith("with")) and "from" in cleaned.lower():
        return cleaned
    return None # Discard junk SQL

def extract_text_and_sql(result):
    """
    Return (assistant_text, sql_str_or_None) from LangChain agent result.
    Handles nested content/list/dict that Gemini may return.
    """
    assistant_text = ""
    sql_text = None

    # If it's a simple string
    if isinstance(result, str):
        text = result.strip()
        # Extract SQL if present
        m = re.search(r"(?si)(select\s+.+|with\s+.+select\s.+)", text)
        if m:
            sql_text = clean_sql(m.group(1))
            assistant_text = (text[:m.start()].strip() or "").strip()
        else:
            assistant_text = text
        return assistant_text, sql_text

    # If object with content attribute (LangChain response)
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list):
            joined = " ".join(str(c) for c in content if isinstance(c, str))
            m = re.search(r"(?si)(select\s+.+|with\s+.+select\s.+)", joined)
            if m:
                sql_text = clean_sql(m.group(1))
            assistant_text = re.sub(r"(?si)(select\s+.+|with\s+.+select\s.+)", "", joined).strip()
            return assistant_text, sql_text
        if isinstance(content, str):
            m = re.search(r"(?si)(select\s+.+|with\s+.+select\s.+)", content)
            if m:
                sql_text = clean_sql(m.group(1))
                assistant_text = content.replace(m.group(0), "").strip()
            else:
                assistant_text = content.strip()
            return assistant_text, sql_text

    # Dict or nested structure
    if isinstance(result, dict):
        content = result.get("content") or ""
        if isinstance(content, str) and content.strip():
            return extract_text_and_sql(content)
        if "messages" in result:
            # inspect messages in reverse for last SQL or text
            for msg in reversed(result["messages"]):
                txt = None
                if isinstance(msg, dict):
                    txt = msg.get("content") or msg.get("text") or msg.get("output")
                else:
                    txt = getattr(msg, "content", None) or getattr(msg, "text", None)
                if isinstance(txt, str):
                    if "select" in txt.lower() or "with" in txt.lower():
                        maybe = clean_sql(txt)
                        if maybe:
                            sql_text = sql_text or maybe
                            # if SQL found, prefer returning SQL only
                            return "", sql_text
                    # otherwise keep as assistant_text if not empty
                    if txt.strip():
                        assistant_text = assistant_text or txt.strip()
            return assistant_text.strip(), sql_text

    # fallback - cast to str
    text = str(result)
    m = re.search(r"(?si)(select\s+.+|with\s+.+select\s.+)", text)
    if m:
        sql_text = clean_sql(m.group(1))
        assistant_text = text.replace(m.group(0), "").strip()
    else:
        assistant_text = text.strip()
    return assistant_text, sql_text

def llm_extract_text(resp):
    """Small helper to extract text content from llm.invoke result."""
    if resp is None:
        return ""
    if hasattr(resp, "content"):
        c = resp.content
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            return " ".join([str(x) for x in c if isinstance(x, str)]).strip()
    return str(resp).strip()

def sanitize_assistant_text(text: str) -> str:
    """Remove LLM artifacts like 'undefined' or repeated SQL tags."""
    if not text:
        return ""
    t = text
    # Common LLM quirks: 'undefined' inserted or 'SQL used: undefined'
    t = re.sub(r"(?i)sql used:\s*undefined", "", t)
    t = re.sub(r"\bundefined\b", "", t)
    # Remove duplicated question lines appended to SQL results
    t = re.sub(r"\n+What (was|is).*$", "", t, flags=re.IGNORECASE | re.DOTALL)
    return t.strip()

def humanize(text: str, tone: str):
    """Add small friendly or analyst prefix depending on tone."""
    if not text:
        return ""
    # remove leftover 'SQL used:' fragments
    text = sanitize_assistant_text(text)
    if tone and tone.startswith("💬"):
        return f"🙂 {text}"
    return f"📊 {text}"

# (NEW) Restored display_llm_summary for chat history replay AND main response
def display_llm_summary(placeholder, insights_dict, tone) -> bool:
    """
    Formats and displays the rich LLM summary as the main chat response.
    Returns True on success, False on failure.
    """
    if not insights_dict or not insights_dict.get('key_finding'):
        return False 
    
    prefix = "🙂" if tone.startswith("💬") else "📊"
    
    response_md = f"""
    {prefix} **LLM Summary**

    1.  **Key Finding:** {insights_dict.get('key_finding', 'N/A')}
    2.  **Why It Matters:** {insights_dict.get('why_it_matters', 'N/A')}
    3.  **Next Step:** {insights_dict.get('next_step', 'N/A')}
    """
    placeholder.markdown(response_md)
    return True

@st.cache_data(ttl=3600)
def get_llm_insights(user_query: str, sql_query: str, df_sample: str) -> dict:
    """Use LLM to generate deep insights: Key Finding, Why it Matters, Next Step."""
    if not user_query or not sql_query or not df_sample:
        return {}
    
    # (FIX) Define the JSON schema for the model
    json_schema = {
        "type": "OBJECT",
        "properties": {
            "key_finding": {"type": "STRING"},
            "why_it_matters": {"type": "STRING"},
            "next_step": {"type": "STRING"}
        },
        "required": ["key_finding", "why_it_matters", "next_step"]
    }
    
    prompt = f"""
    You are an expert data analyst. The user asked a question, a SQL query was run, and here is a sample of the data that was returned.

    User Query: "{user_query}"
    SQL Query: "{sql_query}"
    Data Sample:
    "{df_sample}"

    Please provide a deep analysis of this data, focusing on a key finding, its importance, and a logical next step.
    """
    try:
        summary_llm = get_llm() 
        
        # (FIX) Force JSON output using generationConfig
        resp = summary_llm.invoke(
            [{"role": "user", "content": prompt}],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": json_schema,
            }
        )
        
        text = llm_extract_text(resp)
        
        if text:
            # The model should now *only* return valid JSON text
            # We add our own cleaning logic just in case
            json_str = re.sub(r",\s*([\]\}])", r"\1", text)
            json_str = json_str.replace("'", '"')
            return json.loads(json_str)
        else:
            print("Insight Gen Error: No text found in LLM response.")
            return {}

    except Exception as e:
        print(f"Error generating/parsing LLM insights. Error: {e}")
        return {}

# (NEW) Robust 1-sentence summary fallback
@st.cache_data(ttl=3600)
def get_simple_summary(user_query: str, sql_query: str) -> str:
    """Generates a reliable 1-sentence summary if the rich insights fail."""
    prompt = f"""
    Based on the user's query and the SQL that was run, provide a simple, 1-sentence natural language summary.

    User Query: "{user_query}"
    SQL Query: "{sql_query}"

    Example:
    User Query: "Show me the trend of total sales per month"
    SQL Query: "SELECT strftime('%Y-%m', ...) AS Month, SUM(...) AS TotalSales ..."
    Response: Here is the trend of total sales per month.

    Your 1-sentence response:
    """
    try:
        summary_llm = get_llm()
        resp = summary_llm.invoke([{"role": "user", "content": prompt}])
        text = llm_extract_text(resp)
        return text or "Here is the data I found based on your request:"
    except Exception as e:
        print(f"Error in get_simple_summary: {e}")
        return "Here is the data I found based on your request:"


# -----------------------------
# CHART INTENT & SQL+CHART GENERATOR
# -----------------------------
def infer_chart_intent(q: str):
    q = (q or "").lower()
    
    # (FIX) More conservative trend check
    if any(w in q for w in ["month", "year", "over time", "seasonality"]) or \
       ("trend" in q and any(w in q for w in ["sales", "orders", "revenue", "customer"])):
        return "line"
    if any(w in q for w in ["compare", "vs", "compare to", "comparison"]):
        return "bar"
    if any(w in q for w in ["distribution", "histogram", "distribution of"]):
        return "hist"
    if any(w in q for w in ["relationship", "scatter", "correl", "correlation"]):
        return "scatter"
    if any(w in q for w in ["share", "proportion", "percentage of", "pie"]):
        return "pie"
    if any(w in q for w in ["state", "by state", "per state", "freight"]):
        return "bar"
    if any(w in q for w in ["category", "categories", "top", "most", "highest"]):
        return "bar"
    return None

def auto_sql_chart(query):
    """
    Best-effort fallback generator that returns (sql, df, chart_hint).
    We will prefer agent-provided SQL when available (handled elsewhere).
    """
    q = (query or "").lower()
    hint = infer_chart_intent(q)

    # Time series
    if hint == "line": # Relies on the smarter infer_chart_intent
        sql = """
        SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS Month,
               COUNT(DISTINCT o.order_id) AS Orders,
               SUM(IFNULL(oi.price,0) + IFNULL(oi.freight_value,0)) AS Sales
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        GROUP BY Month ORDER BY Month;
        """
        df = pd.read_sql_query(sql, engine)
        if not df.empty:
            try:
                df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
            except Exception:
                pass
            return sql, df, "line"

    # state
    if "state" in q or "freight" in q:
        if "freight" in q:
            sql = """
            SELECT c.customer_state AS State,
                   SUM(IFNULL(oi.freight_value,0)) AS TotalFreight,
                   COUNT(DISTINCT o.order_id) AS Orders
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN customers c ON o.customer_id = c.customer_id
            GROUP BY State ORDER BY TotalFreight DESC;
            """
            df = pd.read_sql_query(sql, engine)
            return sql, df, "bar" if not df.empty else (None, None, None)
        else:
            sql = """
            SELECT c.customer_state AS State,
                   COUNT(DISTINCT o.order_id) AS Orders,
                   SUM(IFNULL(oi.price,0)) AS TotalSales
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN customers c ON o.customer_id = c.customer_id
            GROUP BY State ORDER BY Orders DESC;
            """
            df = pd.read_sql_query(sql, engine)
            return sql, df, "bar" if not df.empty else (None, None, None)

    # category
    if "category" in q or "categories" in q or "top" in q or "most" in q:
        sql = """
        SELECT pct.product_category_name_english AS Category,
               COUNT(oi.order_id) AS Orders,
               SUM(IFNULL(oi.price,0)) AS TotalSales
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        JOIN product_category_name_translation pct
             ON p.product_category_name = pct.product_category_name
        GROUP BY Category ORDER BY Orders DESC LIMIT 30;
        """
        df = pd.read_sql_query(sql, engine)
        if not df.empty:
            return sql, df, "bar"

    # distribution
    if hint == "hist":
        if "price" in q or "value" in q or "sales" in q:
            sql = "SELECT oi.price AS price FROM order_items oi WHERE oi.price IS NOT NULL;"
            df = pd.read_sql_query(sql, engine)
            return sql, df, "hist" if not df.empty else (None, None, None)
        if "freight" in q:
            sql = "SELECT oi.freight_value AS freight FROM order_items oi WHERE oi.freight_value IS NOT NULL;"
            df = pd.read_sql_query(sql, engine)
            return sql, df, "hist" if not df.empty else (None, None, None)

    # scatter fallback
    if infer_chart_intent(q) == "scatter":
        sql = """
        SELECT oi.price AS price, oi.freight_value AS freight
        FROM order_items oi
        WHERE oi.price IS NOT NULL AND oi.freight_value IS NOT NULL LIMIT 5000;
        """
        df = pd.read_sql_query(sql, engine)
        return sql, df, "scatter" if not df.empty else (None, None, None)

    return None, None, None

def choose_and_render_chart(df: pd.DataFrame, chart_hint: str):
    """
    Smart chart renderer using Altair. Accepts dataframes returned either by the agent's SQL or fallback.
    """
    if df is None or df.empty:
        st.info("No data available to visualize.")
        return

    # --- Handle single-value metric ---
    if df.shape == (1, 1) and pd.api.types.is_numeric_dtype(df.iloc[0, 0]):
        try:
            col_name = df.columns[0]
            value = df.iloc[0, 0]
            # Use st.metric for a much nicer single-value display
            st.metric(label=str(col_name), value=f"{value:,.2f}")
            st.info(f"Here's the specific value you requested:")
            return
        except Exception:
            pass # Fall through to dataframe
    # --- (END NEW) ---

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    # --- (FIX) Re-ordered Logic: Prioritize hints FIRST ---
    
    # 1. Line/time (Hint-based)
    if chart_hint == "line" or ("Month" in df.columns or "OrderMonth" in df.columns or dt_cols):
        y_candidates = [c for c in ["Sales", "Orders", "TotalSales", "TotalFreight", "AverageOrderValue", "TotalRevenue", "DeliveredOrders", "TotalItemsSold"] if c in df.columns]
        y = y_candidates[0] if y_candidates else (num_cols[0] if num_cols else None)
        x = "Month" if "Month" in df.columns else ("OrderMonth" if "OrderMonth" in df.columns else (dt_cols[0] if dt_cols else (cat_cols[0] if cat_cols else None)))
        if x and y:
            chart = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X(x, title=x.title()), 
                y=alt.Y(y, title=y.replace("_", " ").title())
            ).properties(height=350, title=f"{y} Over Time").interactive()
            st.altair_chart(chart, use_container_width=True)
            return

    # 2. Hist (Hint-based)
    if chart_hint == "hist" and num_cols:
        x = num_cols[0]
        chart = alt.Chart(df).mark_bar().encode(
            alt.X(f"{x}:Q", bin=alt.Bin(maxbins=50), title=x.title()), 
            y='count()'
        ).properties(height=350, title=f"Distribution of {x.title()}").interactive()
        st.altair_chart(chart, use_container_width=True)
        return

    # 3. Scatter (Hint-based)
    if (chart_hint == "scatter") and len(num_cols) >= 2:
        sample = df.sample(min(len(df), 2000))
        # Use first two numeric columns for scatter
        x_col = num_cols[0]
        y_col = num_cols[1]
        chart = alt.Chart(sample).mark_circle(size=60, opacity=0.7).encode(
            x=alt.X(x_col, title=x_col.title()), 
            y=alt.Y(y_col, title=y_col.title())
        ).properties(height=350, title=f"{y_col.title()} vs. {x_col.title()}").interactive()
        st.altair_chart(chart, use_container_width=True)
        return

    # 4. Pie (Hint-based)
    if chart_hint == "pie" and cat_cols and num_cols:
        cat = cat_cols[0]
        num = num_cols[0]
        df2 = df.nlargest(10, num) if len(df) > 10 else df
        base = alt.Chart(df2).encode(
            theta=alt.Theta(field=num, type="quantitative", stack=True)
        ).properties(height=350, title=f"Share by {cat.title()}")
        
        # Calculate percentage for tooltip
        df_sum = df2[num].sum()
        if df_sum > 0:
            df2['percent'] = df2[num] / df_sum
        else:
            df2['percent'] = 0.0

        pie = base.mark_arc(outerRadius=120).encode(
            color=alt.Color(field=cat, type="nominal"),
            order=alt.Order(num, sort="descending"),
            tooltip=[cat, num, alt.Tooltip('percent', format=".1%")]
        )
        
        text = base.mark_text(radius=140).encode(
            text=alt.Text('percent', format=".1%"),
            order=alt.Order(num, sort="descending"),
            color=alt.value("#FAFAFA") # Set text color
        )
        chart = pie + text
        st.altair_chart(chart, use_container_width=True)
        return

    # --- Fallback Logic (if no specific hint matched) ---

    # 5. Categorical bar (Fallback)
    if cat_cols and (num_cols or any(c in df.columns for c in ["Orders", "TotalSales", "TotalFreight", "TotalRevenue", "TotalItemsSold"])):
        cat = cat_cols[0]
        num = next((c for c in ["Orders", "TotalSales", "TotalFreight", "AverageOrderValue", "TotalRevenue", "TotalItemsSold"] if c in df.columns), None)
        if num is None and num_cols:
            num = num_cols[0]
        if num:
            df2 = df.nlargest(30, num) if df[cat].nunique() > 30 else df
            chart = alt.Chart(df2).mark_bar().encode(
                x=alt.X(f"{cat}:N", sort='-y', title=cat.replace("_", " ").title()), 
                y=alt.Y(num, title=num.replace("_", " ").title())
            ).properties(height=350, title=f"{num} by {cat}").interactive()
            st.altair_chart(chart, use_container_width=True)
            return

    # 6. Scatter (Fallback)
    if len(num_cols) >= 2:
        sample = df.sample(min(len(df), 2000))
        x_col = num_cols[0]
        y_col = num_cols[1]
        chart = alt.Chart(sample).mark_circle(size=60, opacity=0.7).encode(
            x=alt.X(x_col, title=x_col.title()), 
            y=alt.Y(y_col, title=y_col.title())
        ).properties(height=350, title=f"{y_col.title()} vs. {x_col.title()}").interactive()
        st.altair_chart(chart, use_container_width=True)
        return
    # --- (END FIX) ---

    # fallback table
    st.info("No specific chart type detected. Displaying raw data.")
    st.dataframe(df.head(50))

# -----------------------------
# KPI DASHBOARD (NEW)
# -----------------------------
@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_kpi_data():
    """Run 4 simple queries to get headline KPIs for the dashboard."""
    db_engine = get_db_engine()
    try:
        q_sales = "SELECT SUM(oi.price + oi.freight_value) FROM order_items oi;"
        q_orders = "SELECT COUNT(DISTINCT order_id) FROM orders;"
        q_cust = "SELECT COUNT(DISTINCT customer_id) FROM customers;"
        
        sales = pd.read_sql(q_sales, db_engine).iloc[0, 0]
        orders = pd.read_sql(q_orders, db_engine).iloc[0, 0]
        cust = pd.read_sql(q_cust, db_engine).iloc[0, 0]
        
        if orders > 0:
            # (FIX) Correct AOV calculation for KPI
            q_aov = """
            SELECT AVG(ov.OrderTotal) 
            FROM (
                SELECT order_id, SUM(price + freight_value) AS OrderTotal 
                FROM order_items 
                GROUP BY order_id
            ) ov;
            """
            avg_val = pd.read_sql(q_aov, db_engine).iloc[0, 0]
        else:
            avg_val = 0
            
        return {
            "sales": sales,
            "orders": orders,
            "cust": cust,
            "avg_val": avg_val
        }
    except Exception as e:
        print(f"Error fetching KPI data: {e}")
        return {"sales": 0, "orders": 0, "cust": 0, "avg_val": 0}

# -----------------------------
# PROCESS & RESPOND (REVERTED to single column)
# -----------------------------
if "pending_followup" not in st.session_state:
    st.session_state["pending_followup"] = None

def set_pending_followup(q: str):
    st.session_state["pending_followup"] = q

def process_and_respond(user_query: str):
    """
    (REVERTED & FIXED)
    1. Gets agent response (SQL only, or text-only for non-SQL)
    2. Runs SQL (or auto_sql fallback) to get data.
    3. Tries to generate rich 3-point summary (LLM Summary)
    4. Displays LLM Summary (or robust 1-sentence fallback) as MAIN chat response.
    5. Displays all content (Chart, Data, Followups) in tabs *below* the summary.
    """
    if not user_query:
        return

    # append user + memory
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.session_state["memory_win"].add_user(user_query)

    # show user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # assistant placeholder
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        try:
            # Build messages for agent
            messages_for_agent = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state["memory_win"].get_messages() + [{"role": "user", "content": user_query}]

            # agent invoke (may call tools)
            result = st.session_state["agent"].invoke({"messages": messages_for_agent})

            # extract 1-sentence summary (assistant_text) and any SQL
            assistant_text, agent_sql = extract_text_and_sql(result)
            assistant_text = sanitize_assistant_text(assistant_text or "")
            
            # Detect agent echo
            if assistant_text.strip().lower() == user_query.strip().lower():
                assistant_text = ""
            
            # --- Now, process data ---
            executed_df = None
            executed_sql = None
            chart_hint = None
            map_data = None 
            insights = {} 

            if agent_sql:
                try:
                    sql_candidate = clean_sql(agent_sql)
                    if sql_candidate:
                        executed_df = pd.read_sql_query(sql_candidate, engine)
                        executed_sql = sql_candidate
                        chart_hint = infer_chart_intent(user_query) or infer_chart_intent(agent_sql)
                    else:
                        print(f"Agent SQL rejected by clean_sql: {agent_sql}")
                        st.warning(f"Agent provided invalid SQL. Attempting fallback...")
                        
                except Exception as e:
                    st.warning(f"⚠ Agent provided SQL that failed to execute: {e}")

            # Fallback to auto_sql_chart if agent failed
            if executed_df is None and (agent_sql or not assistant_text): # (FIX) Only fallback if agent *tried* to get SQL or said nothing
                fallback_sql, fallback_df, fallback_hint = auto_sql_chart(user_query)
                if fallback_df is not None and not fallback_df.empty:
                    executed_sql = fallback_sql
                    executed_df = fallback_df
                    chart_hint = fallback_hint
            
            # (NEW) Check for map data in agent's text response (for non-SQL tools)
            if assistant_text:
                try:
                    json_objects = re.findall(r"\{.*?\}", assistant_text, flags=re.DOTALL)
                    for j in reversed(json_objects):
                        try:
                            parsed = json.loads(j)
                            lat = parsed.get("lat") or parsed.get("latitude") or parsed.get("geolocation_lat")
                            lon = parsed.get("lon") or parsed.get("longitude") or parsed.get("geolocation_lng")
                            if lat and lon:
                                map_data = pd.DataFrame([{"lat": float(lat), "lon": float(lon)}])
                                break
                        except Exception:
                            continue
                except Exception:
                    pass

            # --- (FIX) Main Response Logic ---
            if executed_df is not None and not executed_df.empty:
                # We have data! Generate rich summary
                placeholder.markdown("Data retrieved. Generating summary...")
                df_sample_str = executed_df.head(3).to_string()
                
                insights = get_llm_insights(user_query, executed_sql, df_sample_str)
                
                # (FIX) Display summary in chat (or fallback)
                summary_success = display_llm_summary(placeholder, insights, tone_mode)
                
                if summary_success:
                    # Store rich summary dict in messages for replay
                    st.session_state["messages"].append({"role": "assistant", "content": insights})
                    st.session_state["memory_win"].add_assistant(json.dumps(insights)) 
                else:
                    # FALLBACK: Rich summary failed, generate simple 1-sentence summary
                    simple_summary_text = get_simple_summary(user_query, executed_sql)
                    final_text = humanize(simple_summary_text, tone_mode)
                    
                    placeholder.markdown(final_text)
                    store_text = final_text.strip("📊 ").strip("🙂 ")
                    st.session_state["messages"].append({"role": "assistant", "content": store_text})
                    st.session_state["memory_win"].add_assistant(store_text)

                # --- (REVERTED) Show Tabs *directly in chat* ---
                tab1, tab2, tab3 = st.tabs(["📈 Visualization", "🧮 Data & SQL", "🔁 Follow-ups"])

                with tab1:
                    st.markdown("##### 📈 Visualization")
                    choose_and_render_chart(executed_df, chart_hint)

                with tab2:
                    st.markdown("##### 🧮 Data & SQL")
                    st.code(executed_sql, language="sql")
                    st.dataframe(executed_df.head(100))
                
                with tab3:
                    st.markdown("##### 🔁 You could also ask:")
                    
                    # (FIX) Use insights and add static fallbacks
                    next_step = insights.get('next_step') # Will be None if insights failed
                    if next_step:
                        st.button(f"💬 {next_step}", key="follow_contextual", on_click=set_pending_followup, args=(next_step,), use_container_width=True)
                    
                    # Add "always-on" general queries
                    st.button("💬 What are the total sales over time?", key="follow_sales", on_click=set_pending_followup, args=("What are the total sales over time?",), use_container_width=True)
                    st.button("💬 Who are the top 5 sellers by revenue?", key="follow_sellers", on_click=set_pending_followup, args=("Who are the top 5 sellers by revenue?",), use_container_width=True)
                
                # (REVERTED) Show map data directly in chat if it exists
                if map_data is not None:
                    st.markdown("##### 🗺️ Location")
                    st.map(map_data)
            
            else:
                # No data was found. (e.g., text-only answer from search/define/translate)
                # Agent's text is the *only* response
                final_text = ""
                if assistant_text: # Agent provided an explanation
                    final_text = humanize(assistant_text, tone_mode)
                else: # Agent provided no text and no data
                    final_text = humanize("Sorry, I couldn't find an answer for that.", tone_mode)
                
                placeholder.markdown(final_text)
                store_text = final_text.strip("📊 ").strip("🙂 ")
                st.session_state["messages"].append({"role": "assistant", "content": store_text})
                st.session_state["memory_win"].add_assistant(store_text)
                
                # (FIX) Also show map if it was a location query
                if map_data is not None:
                    st.markdown("##### 🗺️ Location")
                    st.map(map_data)

        except Exception as e:
            placeholder.error(f"⚠ An unexpected error occurred: {e}")
            st.session_state["messages"].append({"role": "assistant", "content": f"Error: {e}"})

# -----------------------------
# MAIN FLOW: (REVERTED) Single-Column Layout
# -----------------------------

# --- (REVERTED) KPI DASHBOARD at the top ---
with st.container(border=True):
    st.markdown("##### 🌎 Global E-commerce Overview")
    try:
        kpi_data = get_kpi_data()
        if kpi_data["orders"] > 0: # Only show if data was loaded
            kpi_cols = st.columns(4)
            # (FIX) Corrected the f-string syntax error
            kpi_cols[0].metric("Total Sales", f"R${kpi_data['sales'] / 1_000_000:.2f}M")
            kpi_cols[1].metric("Total Orders", f"{kpi_data['orders'] / 1_000:.1f}K")
            kpi_cols[2].metric("Total Customers", f"{kpi_data['cust'] / 1_000:.1f}K")
            kpi_cols[3].metric("Avg. Order Value", f"R${kpi_data['avg_val']:.2f}")
        else:
            st.warning("Waiting for data... (If this persists, check DB connection)")
    except Exception as e:
        st.error(f"Could not load KPI dashboard: {e}")

st.markdown("---")
st.subheader("💬 Agent Chat")

# --- (REVERTED) Replay chat history in main column ---
for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        # (FIX) Handle rendering rich summary dicts from history
        content = msg.get("content")
        if isinstance(content, dict):
            # Render the rich summary
            display_llm_summary(st.empty(), content, tone_mode)
        else:
            # Render simple text
            st.markdown(str(content))

# Handle pending followup (if clicked)
if st.session_state.get("pending_followup"):
    q = st.session_state.get("pending_followup")
    st.session_state["pending_followup"] = None
    process_and_respond(q)

# Always show the chat input (placed at the bottom)
user_input = st.chat_input("Ask your question about Olist dataset...")
if user_input:
    process_and_respond(user_input)


# -----------------------------
# DEBUG / Memory
# -----------------------------
if debug:
    st.sidebar.subheader("Window Memory (last messages)")
    st.sidebar.write(st.session_state.get("conv_window_messages", []))
    st.sidebar.subheader("Full Messages")
    st.sidebar.write(st.session_state.get("messages", []))

# End of file