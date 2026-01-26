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
    st.error("環境変数 AGENT_RUNTIME_ARN と MEMORY_ID が未設定です。")
    st.stop()

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

actor_id = normalize_actor_id(get_actor_id_from_auth0())

######################################
# Session ID（33文字以上想定）
######################################
def generate_session_id() -> str:
    return str(int(time.time())) + "_" + str(uuid.uuid4()).replace("-", "")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = generate_session_id()

######################################
# 重要：AgentCoreの返却（wrapper）を剥がす
######################################
def unwrap_messages(obj):
    """
    messages=[{"role": "...", "content":[{"text":"..."}]}] に寄せる
    """
    if isinstance(obj, list):
        out = []
        for x in obj:
            if isinstance(x, dict) and isinstance(x.get("message"), dict):
                out.append(x["message"])
            elif isinstance(x, dict) and ("role" in x and "content" in x):
                out.append(x)
        return out

    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        return unwrap_messages(obj["messages"])

    if isinstance(obj, dict) and isinstance(obj.get("message"), dict):
        return [obj["message"]]

    return []

def _extract_text_from_message_obj(m: dict) -> str:
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
    if not isinstance(s, str):
        return str(s)

    raw = s.strip()
    if not raw:
        return ""
    if not (raw.startswith("{") or raw.startswith("[")):
        return s

    try:
        obj = json.loads(raw)
    except Exception:
        return s

    msgs = unwrap_messages(obj)
    if not msgs:
        if isinstance(obj, dict) and isinstance(obj.get("text"), str):
            return obj["text"]
        return s

    texts = []
    for m in msgs:
        if isinstance(m, dict):
            t = _extract_text_from_message_obj(m)
            if t:
                texts.append(t)
    return "\n\n".join(texts) if texts else s

######################################
# ストリーム行読み取り（requests/botocore両対応）
######################################
def _iter_lines_any(body, chunk_size: int = 10):
    """
    - requests.Response.iter_lines(...) でも
    - botocore.response.StreamingBody.iter_lines(...) でも
    どちらでも読めるようにする互換層
    """
    # requests は decode_unicode を受けるが botocore は受けない
    try:
        it = body.iter_lines(chunk_size=chunk_size, decode_unicode=False)
    except TypeError:
        it = body.iter_lines(chunk_size=chunk_size)

    for line in it:
        if line is None:
            continue
        if isinstance(line, bytes):
            s = line.decode("utf-8", errors="ignore")
        else:
            s = str(line)
        yield s

######################################
# SSE (data: ...) パース
######################################
def _iter_sse_json(response: dict, chunk_size: int = 10):
    """
    AgentCoreの返却ストリームが
      data: {...}\n
    のような SSE 形式で流れてくる想定で JSON を取り出す
    """
    body = response.get("response")
    if body is None:
        return

    for s in _iter_lines_any(body, chunk_size=chunk_size):
        if not s:
            continue
        s = s.strip()
        if not s.startswith("data:"):
            continue

        payload = s[5:].strip()  # remove "data:"
        if not payload:
            continue

        try:
            obj = json.loads(payload)
            yield obj
        except Exception:
            # 途中フレーム/ゴミは捨てる
            continue

######################################
# ストリーミング（テキスト delta のみ）
######################################
def streaming(invoke_response: dict):
    """
    invoke_agent_runtime のストリームを処理し、テキストだけ yield
    """
    for obj in _iter_sse_json(invoke_response, chunk_size=10):
        text = (
            obj.get("event", {})
              .get("contentBlockDelta", {})
              .get("delta", {})
              .get("text", "")
        )
        if isinstance(text, str) and text:
            yield text

######################################
# thread 切り替え
######################################
def set_session_id(session_id: str):
    st.session_state["session_id"] = session_id
    # 表示用ログをクリア（必要なら保持戦略に変えてOK）
    st.session_state["messages"] = []

######################################
# UI
######################################
st.title("インフルエンサー検索エージェント")
st.write("Youtube, Instagramのインフルエンサーの情報を収集します！")
st.write("「あなたは何ができますか？」と聞いてみてください。")

######################################
# 初回：messagesをロード（過去履歴）
######################################
if "messages" not in st.session_state:
    st.session_state["messages"] = []

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

    latest_msgs = None
    for obj in _iter_sse_json(resp, chunk_size=10):
        msgs = unwrap_messages(obj)
        if msgs:
            latest_msgs = msgs

    if latest_msgs is not None:
        st.session_state["messages"] = latest_msgs

######################################
# 履歴描画
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
                    st.write(normalize_display_text(c["text"]))
        elif isinstance(content_list, str):
            st.write(normalize_display_text(content_list))

######################################
# チャット入力 → invoke → ストリーミング
######################################
prompt = st.chat_input()
if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("AgentCore実行中..."):
        # ここで prompt を payload に載せる（あなた側のRuntime実装に合わせてキー名調整可）
        resp = agent_core_client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            runtimeSessionId=st.session_state["session_id"],
            payload=json.dumps(
                {
                    "prompt": prompt,          # ← 重要：ユーザ入力
                    "memory_id": MEMORY_ID,
                    "user_id": actor_id,
                    "session_id": st.session_state["session_id"],
                },
                ensure_ascii=False,
            ),
            qualifier="DEFAULT",
        )

        with st.chat_message("assistant"):
            assistant_message = st.write_stream(streaming(resp))

    # 表示ログに追記（role/content 形式を維持）
    st.session_state["messages"].append({"role": "user", "content": [{"text": prompt}]})
    st.session_state["messages"].append(
        {"role": "assistant", "content": [{"text": assistant_message or ""}]}
    )

    # session_id_list に自分を追加
    if "session_id_list" not in st.session_state:
        st.session_state["session_id_list"] = []
    if st.session_state["session_id"] not in st.session_state["session_id_list"]:
        st.session_state["session_id_list"].append(st.session_state["session_id"])

######################################
# サイドバー：スレッド管理
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
        st.session_state["session_id_list"] = []
        with st.spinner("セッション一覧取得中..."):
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

            # 期待: [{"sessionId":"..."}...]
            for obj in _iter_sse_json(resp, chunk_size=10):
                if isinstance(obj, list):
                    st.session_state["session_id_list"] = [
                        x["sessionId"]
                        for x in obj
                        if isinstance(x, dict) and x.get("sessionId")
                    ]

    # セッションボタン
    for sid in st.session_state.get("session_id_list", []):
        st.button(sid, on_click=set_session_id, args=[sid])
