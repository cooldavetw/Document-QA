import asyncio
import json
import os
from typing import List

import streamlit as st
from openai import OpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------
# DB CONFIG (keep in sync with loader if you change it there)
# ---------------------------------------------------------------------
PG_HOST = "pgvector-db"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "sEgMa6"
PG_DATABASE = "postgres"

PAGE_CONTENT_SCHEMA = "page_content"

EMBEDDING_MODEL = "qwen3"  # OpenAI embedding model
OPENAI_DEFAULT_BASE_URL = "http://192.168.66.26:4000/v1"


# ---------------------------------------------------------------------
# Helpers (duplicated here to keep this app independent)
# ---------------------------------------------------------------------
def sanitize_table_name(name: str) -> str:
    """
    Allow letters, digits, underscore, spaces, and CJK characters.
    Disallow quotes/semicolons to avoid SQL injection; return stripped name.
    """
    import re

    if not name:
        raise ValueError("Table name cannot be empty")

    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Table name cannot be empty")
    if '"' in cleaned or ";" in cleaned:
        raise ValueError('Table name cannot contain quotes or semicolons')

    allowed_pattern = r'^[A-Za-z0-9_\s\u4e00-\u9fff]+$'
    if not re.match(allowed_pattern, cleaned):
        raise ValueError(
            "Table name may contain letters, digits, underscore, spaces, and Chinese characters only"
        )
    return cleaned


def quote_table_name(name: str) -> str:
    """Return a safely double-quoted table identifier."""
    return f'"{sanitize_table_name(name)}"'


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    driver_candidates = ["psycopg2", "psycopg"]
    engine = None
    last_error = None

    for driver in driver_candidates:
        url = f"postgresql+{driver}://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
        try:
            engine = create_engine(url, future=True)
            break
        except ImportError as exc:
            last_error = exc
            continue

    if engine is None:
        raise ImportError(
            "No PostgreSQL driver available. Install `psycopg2-binary` (preferred) or `psycopg`."
        ) from last_error

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {PAGE_CONTENT_SCHEMA}"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))

    return engine


def list_page_content_tables(engine: Engine) -> List[str]:
    sql = text(
        """
        SELECT tablename
        FROM pg_catalog.pg_tables
        WHERE schemaname = :schema
        ORDER BY tablename
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"schema": PAGE_CONTENT_SCHEMA}).fetchall()
    return [r[0] for r in rows]


def embed_texts(api_key: str, model: str, base_url: str, texts: List[str]) -> List[List[float]]:
    client = OpenAI(api_key=api_key, base_url=base_url or OPENAI_DEFAULT_BASE_URL)
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def vector_to_pg_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def fetch_top_chunks(
    engine, table_name: str, query_embedding: List[float], top_k: int
) -> List[dict]:
    """
    Retrieve the top_k chunks ordered by vector distance.
    """
    quoted_table = quote_table_name(table_name)
    query_vec = vector_to_pg_literal(query_embedding)
    sql = (
        text(
            f"""
            SELECT
                id,
                "pageContent" AS page_content,
                metadata,
                embedding <=> CAST(:query_vec AS vector) AS distance
            FROM {PAGE_CONTENT_SCHEMA}.{quoted_table}
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:query_vec AS vector)
            LIMIT :top_k
            """
        )
        .bindparams()
    )

    with engine.connect() as conn:
        rows = conn.execute(sql, {"query_vec": query_vec, "top_k": top_k}).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "id": row.id,
                "content": row.page_content,
                "metadata": row.metadata,
                "distance": float(row.distance),
            }
        )
    return results


def build_agent(
    llm_api_key: str, model_name: str, base_url: str = OPENAI_DEFAULT_BASE_URL
) -> Agent[OpenAIChatModel]:
    os.environ["OPENAI_API_KEY"] = llm_api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    llm = OpenAIChatModel(model_name)
    return Agent(
        model=llm,
        system_prompt=(
            "You are a helpful assistant that answers user questions using the provided context. "
            "Stick to the context; if the answer is not contained there, say you do not know."
        ),
    )


def format_context(chunks: List[dict]) -> str:
    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = json.dumps(chunk.get("metadata") or {}, ensure_ascii=False)
        parts.append(
            f"[Chunk {idx} | dist={chunk['distance']:.4f} | id={chunk['id']}]\n"
            f"Metadata: {meta}\n"
            f"Content:\n{chunk['content']}\n"
        )
    return "\n".join(parts)


def main():
    st.title("AI文件問答小幫手")

    st.sidebar.header("LLM Settings")
    llm_api_key = st.sidebar.text_input(
        "LLM API key",
        type="password",
        help="API key for completions.",
        value="abcd"
    )
    llm_base_url = st.sidebar.text_input(
        "LLM base URL",
        value=OPENAI_DEFAULT_BASE_URL,
        help="Base URL for OpenAI-compatible LLM endpoint.",
    )
    llm_model = st.sidebar.text_input(
        "LLM model name",
        value="gpt-oss",
        help="Model name for answering questions.",
    )

    st.sidebar.header("Embedding Settings")
    embed_api_key = st.sidebar.text_input(
        "Embedding API key",
        type="password",
        help="API key for embeddings (can match LLM key).",
        value="abcd"
    )
    embed_base_url = st.sidebar.text_input(
        "Embedding base URL",
        value=OPENAI_DEFAULT_BASE_URL,
        help="Base URL for OpenAI-compatible embedding endpoint.",
    )
    embed_model = st.sidebar.text_input(
        "Embedding model name",
        value=EMBEDDING_MODEL,
        help="Model name for generating embeddings.",
    )

    engine = get_engine()
    tables = list_page_content_tables(engine)
    st.subheader("1. 選擇文件")
    if tables:
        selected_table = st.selectbox("Table", tables)
    else:
        st.error("No page_content tables found. Load PDFs first.")
        return

    st.subheader("2. 輸入你的問題")
    user_query = st.text_area("Your question", height=120)
    top_k = st.slider("Top K passages", min_value=1, max_value=10, value=2)
    run = st.button("進行AI問答")

    if run:
        if not user_query.strip():
            st.error("輸入你的問題")
            return
        if not embed_api_key:
            st.error("OpenAI API key for embeddings is required.")
            return
        if not llm_api_key:
            st.error("OpenAI API key for the LLM is required.")
            return
        if not embed_model.strip():
            st.error("Embedding model name is required.")
            return
        if not llm_model.strip():
            st.error("LLM model name is required.")
            return
        try:
            table = sanitize_table_name(selected_table)
        except ValueError as exc:
            st.error(f"Invalid table name: {exc}")
            return

        with st.spinner("Embedding query..."):
            try:
                query_embedding = embed_texts(embed_api_key, embed_model, embed_base_url, [user_query])[0]
            except Exception as exc:
                st.error(f"Failed to generate embedding: {exc}")
                return

        with st.spinner("Retrieving context from pgvector..."):
            try:
                chunks = fetch_top_chunks(engine, table, query_embedding, top_k)
            except Exception as exc:
                st.error(f"Failed to fetch context: {exc}")
                return

        if not chunks:
            st.warning("No results found in the selected table.")
            return

        context_text = format_context(chunks)
        agent = build_agent(llm_api_key, llm_model, llm_base_url or OPENAI_DEFAULT_BASE_URL)
        prompt = (
            "Use the following context to answer the question.\n\n"
            f"{context_text}\n\nQuestion: {user_query}"
        )

        with st.spinner("Running agent..."):
            try:
                result = asyncio.run(agent.run(prompt))
            except Exception as exc:
                st.error(f"Agent call failed: {exc}")
                return

        answer = getattr(result, "output", None) or getattr(result, "data", None) or str(result)
        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved context"):
            st.write(context_text)

        with st.expander("Raw result"):
            raw_dict = None
            if hasattr(result, "model_dump"):
                raw_dict = result.model_dump()
            elif hasattr(result, "to_dict"):
                raw_dict = result.to_dict()
            if raw_dict is not None:
                st.json(raw_dict, expanded=False)
            else:
                st.write(result)


if __name__ == "__main__":
    main()
