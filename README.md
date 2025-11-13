# 🛍️ E-Commerce GenAI Agent — README

A fully agentic, conversational, and intelligent data analysis system built for the **Maersk GenAI Agentic System Assignment** using the **Olist Brazilian E‑Commerce Dataset**.

This README covers:

* How to run the app
* Architecture overview
* Design decisions
* Features implemented
* What can be improved if more time were available

---

# 📌 1. Project Overview

This project implements a **GenAI‑powered agentic system** capable of:

* Understanding natural‑language questions
* Planning steps autonomously
* Generating SQL queries
* Executing them on the Olist dataset
* Producing answers enriched with charts, insights, and summaries
* Using extra tools such as definitions, translations, and order‑location lookup

The entire system is packaged inside a **modern Streamlit UI** with conversational memory, charts, KPIs, and multi‑tab responses.

---

# 🚀 2. How to Run the Application

### **Step 1 — Clone the Repository**

```
git clone <your‑repo‑url>
cd <repo>
```

### **Step 2 — Create Virtual Environment (optional but recommended)**

```
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### **Step 3 — Install Dependencies**

```
pip install -r requirements.txt
```

### **Step 4 — Add Your API Keys**

Create a `.env` file:

```
GEMINI_API_KEY=your_key_here
DATABASE_URL=sqlite:///olist.sqlite
MEMORY_WINDOW=6
```

> ⚠️ **Do NOT commit this file.** The repo includes `.gitignore` to protect API keys.

### **Step 5 — Run the App**

```
streamlit run app.py
```

Your browser will open automatically.

---

# 🧠 3. High-Level Architecture

```
                 ┌──────────────────────────────────────────┐
                 │           Streamlit Frontend             │
                 │ Chat UI • KPIs • Charts • Map            │
                 └──────────────────────────────────────────┘
                                │
                                ▼
                 ┌──────────────────────────────────────────┐
                 │         Agent Orchestration Layer        │
                 │  (LangChain ReAct-style agent)           │
                 └──────────────────────────────────────────┘
                                │
               ┌────────────────┼─────────────────────────┐
               ▼                ▼                         ▼
   ┌──────────────────┐  ┌───────────────────┐   ┌────────────────────┐
   │ SQLDatabaseTool  │  │ Utility Tools     │   │ LLM (Gemini Flash) │
   │ Text → SQL → DB  │  │ Translation       │   │ Planning + Reason  │
   │ Olist Queries     │  │ Definitions       │   │ Summary Generation │
   └──────────────────┘  │ Order Location    │   └────────────────────┘
                          │ External Search   │
                          └───────────────────┘
                                │
                                ▼
                 ┌──────────────────────────────────────────┐
                 │            SQLite Olist Database          │
                 └──────────────────────────────────────────┘
```

---

# 🧩 4. Key Features

### ✅ **1. Agentic Reasoning (ReAct-style)**

The agent:

* Parses natural language
* Chooses tools automatically
* Generates safe SQL
* Adds context (e.g., group by month/category)
* Returns results + visualizations

### ✅ **2. Conversational Memory**

Solves the *forgetfulness* issue:

* Tracks last N conversation turns
* Understands follow-up queries

### ✅ **3. Smart Utilities (Breadth)**

| Utility                     | Purpose                                 |
| --------------------------- | --------------------------------------- |
| **translate_text_tool**     | Translate any text to any language      |
| **define_term_tool**        | Explain e-commerce terms                |
| **get_order_location_tool** | Map order_id → customer city + lat/long |
| **external_search_tool**    | Add outside knowledge to analysis       |

### ✅ **4. Automatic Charts & Insight Summaries**

* Detects intent (trend, distribution, category analysis, etc.)
* Generates Altair charts automatically
* Renders 3-point insights:

  * **Key Finding**
  * **Why it Matters**
  * **Next Step**

### ✅ **5. Dashboard KPIs**

* Total Sales
* Total Orders
* Total Customers
* Correct Average Order Value

### ✅ **6. Modern UI/UX**

* Dark themed dashboard
* Chat bubbles
* Tabs for chart, SQL, and follow-ups
* PDF export of full conversation

---

# 🧱 5. Design Decisions

### **1. Streamlit for Rapid, Beautiful UI**

Chosen for:

* Speed → 7-day deadline constraint
* Built-in components
* Easy charts + chat interface

### **2. Gemini Flash for Real-Time Reasoning**

Why?

* Fast
* Excellent at Tool Calling
* Stable JSON output for insights

### **3. LangChain SQLDatabaseToolkit**

Benefits:

* Safe SQL execution
* Automatic schema awareness

### **4. ConversationWindowMemory**

Custom memory:

* Prevents model from rambling
* Efficient (keeps only recent messages)

### **5. Auto-SQL Fallback Engine**

When agent SQL fails → system auto-generates SQL by heuristics.
This ensures reliability during demo.

---

# 📈 6. What I Would Do With More Time

### **1. Proactive Insights (Innovation++)**

Agent could:

* Detect anomalies or trends
* Notify user automatically
* Suggest follow-up questions

### **2. Sentiment Analysis on Review Text**

Using LLM or embeddings to:

* Identify product issues
* Extract key complaint themes

### **3. Hybrid DB: SQLite + DuckDB**

Accelerate analytical queries.

### **4. Fine-Tuned SQL Agent**

Train small model on:

* Olist schema
* Real SQL examples

### **5. Real External Search API**

Replace mocked search with:

* Google Custom Search
* SerpAPI

### **6. Deploy App on Streamlit Cloud**

Make publicly accessible.

---

# 🎥 7. Demo Video (Add link here)

Example:

```
https://youtu.be/your-demo-video
```

---

# 📚 8. Folder Structure

```
├── app.py
├── README.md
├── requirements.txt
├── olist.sqlite
├── utils/
│   ├── tools.py
│   ├── memory.py
│   └── charts.py
├── .env.example
└── .gitignore
```

---

# 🏁 9. Final Notes

This project was built in the spirit of the **hacker‑builder mindset**:

* Focus on outcomes
* Ship fast, iterate fast
* Build features that directly align with **Depth**, **Breadth**, **Innovation**, **UX**, and **Communication** scoring.

If you're reviewing this project, thank you! Happy to walk through the architecture or the code in detail.
