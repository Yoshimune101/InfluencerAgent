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

######################################
# AgentCore Client
######################################
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
# 重要：AgentCoreの返却（wrapper）を剥がす
######################################
def unwrap_messages(obj):
    """
    AgentCoreから返るログの揺れを吸収して
    messages=[{"role": "...", "content":[{"text":"..."}]}] の形に寄せる。

    想定される返却:
    - [{"message": {...}, "created_at":...}, ...]  ← あなたが見てるJSON
    - {"messages":[...]}
    - [{"role":"user","content":[{"text":"..."}]}, ...]
    """
    if isinstance(obj, list):
        out = []
        for x in obj:
            if isinstance(x, dict) and isinstance(x.get("message"), dict):
                out.append(x["message"])  # ✅ wrapper を剥がす
            elif isinstance(x, dict) and ("role" in x and "content" in x):
                out.append(x)
        return out

    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        return unwrap_messages(obj["messages"])

    if isinstance(obj, dict) and isinstance(obj.get("message"), dict):
        return [obj["message"]]

    return []

def _extract_text_from_message_obj(m: dict) -> str:
    """
    message={"role":..,"content":[{"text":"..."}]} 形式から、テキストを結合して返す
    """
    content = m.get("content", [])
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and isinstance(c.get("text"), str):
                parts.append(c["text"])
        return "\n".join([p for p in parts if p])
    if isinstance(content, str):
        return content
    return ""


def normalize_display_text(s: str) -> str:
    """
    画面表示用の最終正規化。

    - s が JSON 文字列なら json.loads して wrapper を剥がし、本文だけ抜く
    - \uXXXX は json.loads できれば自動で復元される
    - それでも無理なら生文字列で返す
    """
    if not isinstance(s, str):
        return str(s)

    raw = s.strip()
    # すでに普通の文章っぽければそのまま返す（無駄なパースを避ける）
    if not raw:
        return ""
    if not (raw.startswith("{") or raw.startswith("[")):
        return s

    try:
        obj = json.loads(raw)  # ← \uXXXX をここで復元できる
    except Exception:
        return s

    # obj が {"message":{...}} や {"messages":[...]} や [ ... ] の可能性があるので既存ロジックで吸収
    msgs = unwrap_messages(obj)
    if not msgs:
        # たとえば {"text":"..."} みたいな単純構造の場合
        if isinstance(obj, dict) and isinstance(obj.get("text"), str):
            return obj["text"]
        return s

    # 1件でも本文が取れたらそれを表示用に返す（複数なら結合）
    texts = []
    for m in msgs:
        if isinstance(m, dict):
            t = _extract_text_from_message_obj(m)
            if t:
                texts.append(t)
    return "\n\n".join(texts) if texts else s


######################################
# ストリーミング（サンプル踏襲）
######################################
def streaming(response):
    """
    invoke_agent_runtime のストリームを処理。
    「テキスト delta」だけ表示する（JSONは表示しない）。
    """
    for line in response["response"].iter_lines(chunk_size=10):
        if not line:
            continue

        s = line.decode("utf-8")
        if not s.startswith("data: "):
            continue

        payload = s[6:]  # remove "data: "
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
# 初回：messagesをロード（サンプル踏襲）
######################################
if "messages" not in st.session_state:
    st.session_state["messages"] = []

    response = agent_core_client.invoke_agent_runtime(
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

    # ✅ ここが核心：wrapper を剥がして messages 本体だけ保存
    latest_msgs = None

    for line in response["response"].iter_lines(chunk_size=10):
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

        msgs = unwrap_messages(obj)
        if msgs:
            latest_msgs = msgs  # ✅最後に取れた有効なmessagesを保持

    if latest_msgs is not None:
        st.session_state["messages"] = latest_msgs


######################################
# 履歴描画（サンプル踏襲）
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
                    st.write(normalize_display_text(c["text"]))  # ✅ここが肝
        elif isinstance(content_list, str):
            st.write(normalize_display_text(content_list))      # ✅ここも

######################################
# チャット入力 → invoke → ストリーミング
######################################
prompt = st.chat_input()
if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner():
        response = agent_core_client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            runtimeSessionId=st.session_state["session_id"],
            payload=json.dumps(
                {
                    "memory_id": MEMORY_ID,
                    "user_id": actor_id,
                    "session_id": st.session_state["session_id"],
                },
                ensure_ascii=False,
            ),
            qualifier="DEFAULT",
        )

        with st.chat_message("assistant"):
            assistant_message = st.write_stream(streaming(response))

    # ✅ messagesはサンプル形式に統一して追記
    st.session_state["messages"].append({"role": "user", "content": [{"text": prompt}]})
    st.session_state["messages"].append(
        {"role": "assistant", "content": [{"text": assistant_message}]}
    )

    # session_id_list に自分を追加（サンプル踏襲）
    if "session_id_list" not in st.session_state:
        st.session_state["session_id_list"] = []
    if st.session_state["session_id"] not in st.session_state["session_id_list"]:
        st.session_state["session_id_list"].append(st.session_state["session_id"])

######################################
# サイドバー：スレッド管理（サンプル踏襲）
######################################
with st.sidebar:
    st.text_input(label="Session ID", value=st.session_state["session_id"], disabled=True)

    st.button(
        "new thread",
        on_click=set_session_id,
        args=[generate_session_id()],
        type="primary",
    )

    # session_id_list 初回取得
    if "session_id_list" not in st.session_state:
        with st.spinner():
            response = agent_core_client.invoke_agent_runtime(
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

            for line in response["response"].iter_lines(chunk_size=10):
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
                    st.session_state["session_id_list"] = [
                        x["sessionId"]
                        for x in obj
                        if isinstance(x, dict) and x.get("sessionId")
                    ]

    if "session_id_list" in st.session_state:
        for sid in st.session_state["session_id_list"]:
            st.button(sid, on_click=set_session_id, args=[sid])
