import os
import boto3
import json
import uuid
import re
import hashlib

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
# 環境変数とクライアント
######################################
load_dotenv()

REGION = os.getenv("AWS_REGION") or "us-west-2"
AGENT_RUNTIME_ARN = os.getenv("AGENT_RUNTIME_ARN")
MEMORY_ID = os.getenv("MEMORY_ID")

@st.cache_resource
def get_agentcore_client(region: str):
    return boto3.client("bedrock-agentcore", region_name=region)

agentcore = get_agentcore_client(REGION)

######################################
# actor_id / thread_id 正規化
######################################
def get_actor_id_from_auth0() -> str:
    u = st.user
    return (
        str(getattr(u, "sub", "")).strip()
        or str(getattr(u, "id", "")).strip()
        or str(getattr(u, "email", "")).strip()
        or str(getattr(u, "name", "")).strip()
        or "anonymous"
    )

ACTOR_ID_ALLOWED = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9\-_\/]*(?::[a-zA-Z0-9\-_\/]+)*[a-zA-Z0-9\-_\/]*$"
)
THREAD_ID_ALLOWED = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9\-_]*$")

def normalize_actor_id(raw: str) -> str:
    """
    'actor:' が既に付いている場合は二重付与しない
    """
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

def normalize_thread_id(raw: str) -> str:
    """
    thread_id(session_id) は ':' '/' を入れない。
    'sess_' が既に付いている場合は二重付与しない
    """
    if not raw:
        raw = "default"
    raw = str(raw).strip()
    if raw.startswith("sess_"):
        raw = raw[len("sess_"):]

    safe = re.sub(r"[^a-zA-Z0-9\-_]", "_", raw)
    if not re.match(r"^[a-zA-Z0-9]", safe):
        safe = "t_" + safe

    candidate = f"sess_{safe}"
    if THREAD_ID_ALLOWED.match(candidate):
        return candidate

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"sess_{digest}"

######################################
# セッション状態
######################################
if "messages" not in st.session_state:
    # UI表示用: [{"role":"user"/"assistant","content":"..."}]
    st.session_state.messages = []

if "runtime_session_id" not in st.session_state:
    st.session_state.runtime_session_id = str(uuid.uuid4())

if "thread_id" not in st.session_state:
    st.session_state.thread_id = normalize_thread_id(st.session_state.runtime_session_id)

if "loaded_thread_id" not in st.session_state:
    st.session_state.loaded_thread_id = None

def start_new_thread():
    st.session_state.messages = []
    st.session_state.runtime_session_id = str(uuid.uuid4())
    st.session_state.thread_id = normalize_thread_id(st.session_state.runtime_session_id)
    st.session_state.loaded_thread_id = None
    st.rerun()

def switch_thread(thread_id: str):
    st.session_state.messages = []
    st.session_state.thread_id = thread_id
    st.session_state.runtime_session_id = thread_id
    st.session_state.loaded_thread_id = None
    st.rerun()

######################################
# 正規化（表示用）
######################################
def normalize_message_item(item):
    """
    AgentCore側の返却揺れを吸収し、必ず {role, content(str)} を返す。
    対応:
    - {"role":"assistant","content":[{"text":"..."}]}
    - {"message":{"role":"assistant","content":[{"text":"..."}]}, ...}
    - {"message":{"role":"user","content":[{"toolResult": {...}}]}, ...}
    """
    if not isinstance(item, dict):
        return {"role": "assistant", "content": str(item)}

    core = item.get("message") if isinstance(item.get("message"), dict) else item

    role = str(core.get("role", "assistant")).lower()
    if role not in ("user", "assistant"):
        role = "assistant"

    content = core.get("content")

    # list: [{"text":...}] / [{"toolResult":...}]
    if isinstance(content, list):
        parts = []
        for c in content:
            if not isinstance(c, dict):
                continue

            # text
            t = c.get("text")
            if isinstance(t, str) and t:
                parts.append(t)
                continue

            # toolResult
            tr = c.get("toolResult")
            if isinstance(tr, dict):
                tr_contents = tr.get("content")
                if isinstance(tr_contents, list):
                    tr_text = "".join(
                        (x.get("text", "") if isinstance(x, dict) else str(x))
                        for x in tr_contents
                    )
                else:
                    tr_text = str(tr_contents or "")

                tool_name = tr.get("toolName") or tr.get("name") or "tool"
                status = tr.get("status") or "unknown"
                parts.append(f"【ツール結果: {tool_name} / {status}】\n{tr_text}")
                continue

        return {"role": role, "content": "".join(parts).strip()}

    # string content
    if isinstance(content, str):
        return {"role": role, "content": content}

    return {"role": role, "content": ""}

def extract_text_from_data_string(s: str) -> str:
    """
    ストリーム event["data"] に JSON文字列が混ざるケースの保険。
    JSONなら parse して text だけ抜く。JSONでないなら捨てる。
    """
    if not s:
        return ""
    ss = s.strip()
    if ss.startswith("{") or ss.startswith("["):
        try:
            obj = json.loads(ss)
            if isinstance(obj, dict):
                m = normalize_message_item(obj)
                return m["content"]
            if isinstance(obj, list):
                parts = []
                for it in obj:
                    m = normalize_message_item(it)
                    if m["content"]:
                        parts.append(m["content"])
                return "\n".join(parts).strip()
            return ""
        except Exception:
            return ""
    return ""

######################################
# UI
######################################
st.title("インフルエンサー検索エージェント")
st.write("Youtube, Instagramのインフルエンサーの情報を収集します！")
st.write("「あなたは何ができますか？」と聞いてみてください。")

######################################
# 前提チェック
######################################
if not AGENT_RUNTIME_ARN or not MEMORY_ID:
    st.error("環境変数 AGENT_RUNTIME_ARN と MEMORY_ID が未設定です。")
    st.stop()

actor_id = normalize_actor_id(get_actor_id_from_auth0())

def invoke_agentcore(payload_dict: dict):
    body = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")
    return agentcore.invoke_agent_runtime(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        runtimeSessionId=st.session_state.runtime_session_id,
        payload=body,
        qualifier="DEFAULT",
    )

######################################
# 1) 履歴ロード（threadごとに1回）
######################################
if st.session_state.loaded_thread_id != st.session_state.thread_id:
    try:
        resp = invoke_agentcore(
            {
                "action": "get_message_list",
                "memory_id": MEMORY_ID,
                "user_id": actor_id,
                "session_id": st.session_state.thread_id,
                "k": 100,
            }
        )

        loaded = []

        for line in resp["response"].iter_lines():
            if not line:
                continue

            s = line.decode("utf-8")
            if not s.startswith("data: "):
                continue

            data = s[6:]
            if data.startswith('"') or data.startswith("'"):
                continue

            obj = json.loads(data)

            if isinstance(obj, list):
                for item in obj:
                    m = normalize_message_item(item)
                    if m["content"]:
                        loaded.append(m)
            elif isinstance(obj, dict):
                # {"messages":[...]} 形式も吸収
                if isinstance(obj.get("messages"), list):
                    for item in obj["messages"]:
                        m = normalize_message_item(item)
                        if m["content"]:
                            loaded.append(m)
                else:
                    m = normalize_message_item(obj)
                    if m["content"]:
                        loaded.append(m)

        st.session_state.messages = loaded
        st.session_state.loaded_thread_id = st.session_state.thread_id
        st.rerun()

    except Exception as e:
        st.warning(f"履歴ロードに失敗しました: {e}")
        st.session_state.loaded_thread_id = st.session_state.thread_id

######################################
# 2) 表示（必ず正規化してから描画）
######################################
for raw in st.session_state.messages:
    m = normalize_message_item(raw)
    if not m["content"]:
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

######################################
# 3) チャット入力 → ストリーミング（JSON混入根治）
######################################
prompt = st.chat_input("メッセージを入力してね")
if prompt:
    # 表示用に先に積む
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 日本語固定（英語寄りの抑止）
        req_payload = {
            "prompt": f"必ず日本語で回答してください。\n\n{prompt}",
            "actor_id": actor_id,
            "session_id": st.session_state.thread_id,
        }

        resp = invoke_agentcore(req_payload)

        container = st.container()
        text_holder = container.empty()
        buffer = ""

        for line in resp["response"].iter_lines():
            if not line:
                continue

            s = line.decode("utf-8")
            if not s.startswith("data: "):
                continue

            data = s[6:]
            if data.startswith('"') or data.startswith("'"):
                continue

            event = json.loads(data)

            # toolUse開始
            if "event" in event and "contentBlockStart" in event["event"]:
                start = event["event"]["contentBlockStart"].get("start", {})
                tool = start.get("toolUse")
                if tool:
                    if buffer:
                        text_holder.markdown(buffer)
                        buffer = ""
                    tool_name = tool.get("name", "unknown")
                    container.info(f"🔍 {tool_name} ツールを利用しています")
                    text_holder = container.empty()
                continue

            # ✅ ここだけ採用（assistantのテキストdelta）
            if "event" in event and "contentBlockDelta" in event["event"]:
                delta = event["event"]["contentBlockDelta"].get("delta", {})
                t = delta.get("text")
                if isinstance(t, str) and t:
                    buffer += t
                    text_holder.markdown(buffer)
                continue

            # event["data"] は原則捨てる（JSON混入の根源）
            # ただしJSON文字列ならテキストだけ抽出して採用（保険）
            if "data" in event and isinstance(event["data"], str):
                extracted = extract_text_from_data_string(event["data"])
                if extracted:
                    buffer += extracted
                    text_holder.markdown(buffer)
                continue

        text_holder.markdown(buffer)

    # 表示用履歴に積む
    st.session_state.messages.append({"role": "assistant", "content": buffer})

######################################
# サイドバー：スレッド管理
######################################
with st.sidebar:
    st.markdown("### Thread")
    st.text_input("Current thread_id", value=st.session_state.thread_id, disabled=True)
    st.text_input("runtime_session_id", value=st.session_state.runtime_session_id, disabled=True)

    st.button("new thread", on_click=start_new_thread, type="primary")

    st.markdown("---")
    st.markdown("### Past threads")

    if "thread_id_list" not in st.session_state:
        st.session_state.thread_id_list = []

    if st.button("refresh threads"):
        try:
            resp = invoke_agentcore(
                {
                    "action": "get_session_id_list",
                    "memory_id": MEMORY_ID,
                    "user_id": actor_id,
                }
            )

            ids = []
            for line in resp["response"].iter_lines():
                if not line:
                    continue

                s = line.decode("utf-8")
                if not s.startswith("data: "):
                    continue

                data = s[6:]
                if data.startswith('"') or data.startswith("'"):
                    continue

                obj = json.loads(data)
                if isinstance(obj, list):
                    for x in obj:
                        if isinstance(x, dict) and x.get("sessionId"):
                            ids.append(x["sessionId"])

            st.session_state.thread_id_list = sorted(set([i for i in ids if isinstance(i, str) and i]))

        except Exception as e:
            st.warning(f"スレッド一覧の取得に失敗しました: {e}")

    for tid in st.session_state.thread_id_list:
        st.button(tid, on_click=switch_thread, args=(tid,))
