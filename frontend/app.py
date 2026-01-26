import os
import json
import uuid
import time
import re
import hashlib

import boto3
import streamlit as st
from dotenv import load_dotenv

######################################
# ログイン
######################################
if not getattr(st.user, "is_logged_in", False):
    st.login("auth0")
    st.stop()

st.success(f"Hello {st.user.name}")

######################################
# Env / Client
######################################
load_dotenv()

REGION = os.getenv("AWS_REGION") or "us-west-2"
AGENT_RUNTIME_ARN = os.getenv("AGENT_RUNTIME_ARN")
MEMORY_ID = os.getenv("MEMORY_ID")

agent_core_client = boto3.client("bedrock-agentcore", region_name=REGION)

######################################
# actor_id 正規化（AgentCore Memory制約用）
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
    if not raw:
        raw = "anonymous"
    raw = str(raw).strip()
    if raw.startswith("actor:"):
        raw = raw[len("actor:"):]
    safe = re.sub(r"[^a-zA-Z0-9\-_\/:]", "_", raw)
    if not re.match(r"^[a-zA-Z0-9]", safe):
        safe = "a_" + safe
    safe = safe.rstrip(":")
    candidate = f"actor:{safe}"
    if ACTOR_ID_ALLOWED.match(candidate):
        return candidate
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"actor:{digest}"

######################################
# Session ID（サンプル踏襲：33文字以上）
######################################
def generate_session_id():
    # 33文字以上ないとエラーになる、という前提に合わせる
    return str(int(time.time())) + "_" + str(uuid.uuid4()).replace("-", "")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = generate_session_id()

######################################
# messages の形式をサンプルに統一する
# - [{"role":"user"/"assistant", "content":[{"text":"..."}]}]
######################################
def ensure_messages():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

def append_user_message(text: str):
    ensure_messages()
    st.session_state["messages"].append(
        {"role": "user", "content": [{"text": text}]}
    )

def append_assistant_message(text: str):
    ensure_messages()
    st.session_state["messages"].append(
        {"role": "assistant", "content": [{"text": text}]}
    )

######################################
# ストリーム処理（サンプルを踏襲）
# → contentBlockDelta.delta.text だけ拾う
######################################
def streaming(response):
    for line in response["response"].iter_lines(chunk_size=10):
        if not line:
            continue
        s = line.decode("utf-8")
        if not s.startswith("data: "):
            continue
        payload = s[6:]  # remove "data: "

        # keep-alive等で json.loads できないものは捨てる
        try:
            obj = json.loads(payload)
        except Exception:
            continue

        text = (
            obj.get("event", {})
              .get("contentBlockDelta", {})
              .get("delta", {})
              .get("text", "")
        )

        if isinstance(text, str) and text:
            yield text

######################################
# thread 切り替え（サンプル踏襲）
######################################
def set_session_id(session_id: str):
    st.session_state["session_id"] = session_id
    if "messages" in st.session_state:
        del st.session_state["messages"]

######################################
# UI
######################################
st.title("インフルエンサー検索エージェント")
st.write("Youtube, Instagramのインフルエンサーの情報を収集します！")
st.write("「あなたは何ができますか？」と聞いてみてください。")

if not AGENT_RUNTIME_ARN or not MEMORY_ID:
    st.error("環境変数 AGENT_RUNTIME_ARN と MEMORY_ID が未設定です。")
    st.stop()

actor_id = normalize_actor_id(get_actor_id_from_auth0())

######################################
# 1) 初回のみ履歴ロード（サンプル踏襲）
######################################
ensure_messages()

if len(st.session_state["messages"]) == 0:
    resp = agent_core_client.invoke_agent_runtime(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        runtimeSessionId=st.session_state["session_id"],
        payload=json.dumps(
            {
                "action": "get_message_list",
                "memory_id": MEMORY_ID,
                "user_id": actor_id,
                "session_id": st.session_state["session_id"],
            },
            ensure_ascii=False,
        ),
        qualifier="DEFAULT",
    )

    # サンプル同様：返却の "data:" を json.loads した結果が
    # 「messages配列そのもの」で返ってくる前提
    for line in resp["response"].iter_lines(chunk_size=10):
        if not line:
            continue
        s = line.decode("utf-8")
        if not s.startswith("data: "):
            continue
        payload = s[6:]
        try:
            obj = json.loads(payload)
        except Exception:
            continue

        # obj が list（=messages）ならそれを採用
        # dictの場合は {"messages":[...]} も許容
        if isinstance(obj, list):
            st.session_state["messages"] = obj
        elif isinstance(obj, dict) and isinstance(obj.get("messages"), list):
            st.session_state["messages"] = obj["messages"]

######################################
# 2) 履歴描画（サンプル踏襲）
######################################
for msg in st.session_state["messages"]:
    role = str(msg.get("role", "assistant")).lower()
    if role not in ("user", "assistant"):
        role = "assistant"

    with st.chat_message(role):
        content_list = msg.get("content", [])
        if isinstance(content_list, list):
            for c in content_list:
                if isinstance(c, dict) and isinstance(c.get("text"), str):
                    st.write(c["text"])

######################################
# 3) チャット入力 → invoke → ストリーミング描画（サンプル踏襲）
######################################
prompt = st.chat_input()
if prompt:
    # UIに即表示
    with st.chat_message("user"):
        st.write(prompt)

    # Memoryへも投げる（AgentCore側の仕様に合わせる）
    resp = agent_core_client.invoke_agent_runtime(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        runtimeSessionId=st.session_state["session_id"],
        payload=json.dumps(
            {
                "memory_id": MEMORY_ID,
                "user_id": actor_id,
                "session_id": st.session_state["session_id"],
                # ✅ 英語寄り防止（必要なら外してOK）
                "prompt": f"必ず日本語で回答してください。\n\n{prompt}",
            },
            ensure_ascii=False,
        ),
        qualifier="DEFAULT",
    )

    # assistant をストリーム描画
    with st.chat_message("assistant"):
        assistant_text = st.write_stream(streaming(resp))

    # messages はサンプル形式に統一して append
    append_user_message(prompt)
    append_assistant_message(assistant_text)

    # session_id_list に自分を追加（サンプル踏襲）
    if "session_id_list" not in st.session_state:
        st.session_state["session_id_list"] = []
    if st.session_state["session_id"] not in st.session_state["session_id_list"]:
        st.session_state["session_id_list"].append(st.session_state["session_id"])

######################################
# 4) サイドバー：スレッド管理（サンプル踏襲）
######################################
with st.sidebar:
    st.text_input(label="Session ID", value=st.session_state["session_id"], disabled=True)

    st.button(
        "new thread",
        on_click=set_session_id,
        args=[generate_session_id()],
        type="primary",
    )

    # session_id_list を初回取得
    if "session_id_list" not in st.session_state:
        with st.spinner():
            resp = agent_core_client.invoke_agent_runtime(
                agentRuntimeArn=AGENT_RUNTIME_ARN,
                runtimeSessionId=st.session_state["session_id"],
                payload=json.dumps(
                    {
                        "action": "get_session_id_list",
                        "memory_id": MEMORY_ID,
                        "user_id": actor_id,
                    },
                    ensure_ascii=False,
                ),
                qualifier="DEFAULT",
            )

            for line in resp["response"].iter_lines(chunk_size=10):
                if not line:
                    continue
                s = line.decode("utf-8")
                if not s.startswith("data: "):
                    continue
                payload = s[6:]
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue

                # 期待: [{"sessionId":"..."}...]
                if isinstance(obj, list):
                    st.session_state["session_id_list"] = [x["sessionId"] for x in obj if isinstance(x, dict) and x.get("sessionId")]

    # session_id_list のボタン表示
    if "session_id_list" in st.session_state:
        for sid in st.session_state["session_id_list"]:
            st.button(sid, on_click=set_session_id, args=[sid])
