import os
import json
import time
import uuid
import re
import hashlib

import boto3
import streamlit as st
from dotenv import load_dotenv

######################################
# ログイン（Auth0）
######################################
if not getattr(st.user, "is_logged_in", False):
    st.login("auth0")
    st.stop()

st.success(f"Hello {st.user.name}")

######################################
# 環境変数
######################################
load_dotenv()
REGION = os.getenv("AWS_REGION") or "us-west-2"
AGENT_RUNTIME_ARN = os.getenv("AGENT_RUNTIME_ARN")
MEMORY_ID = os.getenv("MEMORY_ID")

if not AGENT_RUNTIME_ARN or not MEMORY_ID:
    st.error("環境変数 AGENT_RUNTIME_ARN / MEMORY_ID が未設定です")
    st.stop()

######################################
# AgentCore Client
######################################
agent_core_client = boto3.client(
    "bedrock-agentcore",
    region_name=REGION,
)

######################################
# actor_id 正規化
######################################
ACTOR_ID_ALLOWED = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9\-_\/]*(?::[a-zA-Z0-9\-_\/]+)*[a-zA-Z0-9\-_\/]*$"
)

def get_actor_id_from_auth0() -> str:
    u = st.user
    return (
        str(getattr(u, "sub", "")).strip()
        or str(getattr(u, "id", "")).strip()
        or str(getattr(u, "email", "")).strip()
        or str(getattr(u, "name", "")).strip()
        or "anonymous"
    )

def normalize_actor_id(raw: str) -> str:
    raw = raw or "anonymous"
    raw = raw.strip()
    if raw.startswith("actor:"):
        raw = raw[len("actor:"):]
    safe = re.sub(r"[^a-zA-Z0-9\-_\/:]", "_", raw)
    if not re.match(r"^[a-zA-Z0-9]", safe):
        safe = "a_" + safe
    safe = safe.rstrip(":")
    candidate = f"actor:{safe}"
    if ACTOR_ID_ALLOWED.match(candidate):
        return candidate
    digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"actor:{digest}"

actor_id = normalize_actor_id(get_actor_id_from_auth0())

######################################
# Session ID（33文字以上）
######################################
def generate_session_id() -> str:
    return f"{int(time.time())}_{uuid.uuid4().hex}"

if "session_id" not in st.session_state:
    st.session_state["session_id"] = generate_session_id()

######################################
# 共通：SSE(JSON) iterator
######################################
def iter_sse_json(response):
    body = response.get("response")
    if body is None:
        return

    for raw in body.iter_lines(chunk_size=1024):
        if not raw:
            continue

        line = raw.decode("utf-8", errors="ignore").strip()
        if not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        if not payload:
            continue

        try:
            yield json.loads(payload)
        except Exception:
            continue

######################################
# AgentCore message 正規化
######################################
def unwrap_messages(obj):
    if isinstance(obj, list):
        out = []
        for x in obj:
            if isinstance(x, dict) and isinstance(x.get("message"), dict):
                out.append(x["message"])
            elif isinstance(x, dict) and "role" in x and "content" in x:
                out.append(x)
        return out

    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        return unwrap_messages(obj["messages"])

    if isinstance(obj, dict) and isinstance(obj.get("message"), dict):
        return [obj["message"]]

    return []

def extract_text_from_message(m: dict) -> str:
    content = m.get("content", [])
    if isinstance(content, list):
        return "\n".join(
            c["text"]
            for c in content
            if isinstance(c, dict) and isinstance(c.get("text"), str)
        )
    if isinstance(content, str):
        return content
    return ""

######################################
# ストリーミング（完全耐性版）
######################################
def streaming(response):
    for obj in iter_sse_json(response):

        # list が来るケース
        if isinstance(obj, list):
            msgs = unwrap_messages(obj)
            for m in msgs:
                t = extract_text_from_message(m)
                if t:
                    yield t
            continue

        if not isinstance(obj, dict):
            continue

        ev = obj.get("event")
        if isinstance(ev, dict):
            # 通常の delta
            text = (
                ev.get("contentBlockDelta", {})
                  .get("delta", {})
                  .get("text", "")
            )
            if isinstance(text, str) and text:
                yield text
                continue

            # message / messages 形式
            if isinstance(ev.get("message"), dict):
                t = extract_text_from_message(ev["message"])
                if t:
                    yield t
            if isinstance(ev.get("messages"), list):
                for m in unwrap_messages(ev):
                    t = extract_text_from_message(m)
                    if t:
                        yield t
            continue

        # トップレベル message
        if isinstance(obj.get("message"), dict):
            t = extract_text_from_message(obj["message"])
            if t:
                yield t

######################################
# UI
######################################
st.title("インフルエンサー検索エージェント")
st.write("YouTube / Instagram のインフルエンサーを検索します")

######################################
# メッセージ履歴
######################################
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    role = msg.get("role", "assistant")
    with st.chat_message(role):
        st.write(msg.get("content", ""))

######################################
# チャット入力
######################################
prompt = st.chat_input()
if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    response = agent_core_client.invoke_agent_runtime(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        runtimeSessionId=st.session_state["session_id"],
        payload=json.dumps(
            {
                "action": "chat",
                "prompt": prompt,              # ← ★重要
                "memory_id": MEMORY_ID,
                "user_id": actor_id,
                "session_id": st.session_state["session_id"],
            },
            ensure_ascii=False,
        ),
        qualifier="DEFAULT",
    )

    with st.chat_message("assistant"):
        assistant_text = st.write_stream(streaming(response)) or ""

    assistant_text = str(assistant_text)

    st.session_state["messages"].append(
        {"role": "user", "content": prompt}
    )
    st.session_state["messages"].append(
        {"role": "assistant", "content": assistant_text}
    )

######################################
# サイドバー
######################################
with st.sidebar:
    st.text_input("Session ID", st.session_state["session_id"], disabled=True)
    if st.button("new thread", type="primary"):
        st.session_state["session_id"] = generate_session_id()
        st.session_state["messages"] = []
