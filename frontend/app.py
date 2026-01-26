import os, boto3, json, uuid, re, hashlib, random
from datetime import datetime, timezone
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
# 環境変数
######################################
load_dotenv()
REGION = os.getenv("AWS_REGION")
AGENT_RUNTIME_ARN = os.getenv("AGENT_RUNTIME_ARN")
MEMORY_ID = os.getenv("MEMORY_ID")

if not REGION:
    st.error("AWS_REGION が未設定です")
    st.stop()
if not AGENT_RUNTIME_ARN:
    st.error("AGENT_RUNTIME_ARN が未設定です")
    st.stop()
if not MEMORY_ID:
    st.error("MEMORY_ID が未設定です（AgentCore Memory のIDを環境変数に設定してください）")
    st.stop()

######################################
# ✅ actorId / sessionId（Memory用）
######################################
ACTOR_ID_ALLOWED = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9-/]*:[a-zA-Z0-9-/]+$")
SESSION_ID_ALLOWED = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9\-_]*$")


def normalize_actor_id(raw: str) -> str:
    if not raw:
        raw = "anonymous"
    safe = raw.strip().replace(":", "-")
    safe = re.sub(r"[^a-zA-Z0-9-/]", "-", safe)
    safe = re.sub(r"-{2,}", "-", safe).strip("-/")
    if not safe:
        safe = "anonymous"
    if not re.match(r"^[a-zA-Z0-9]", safe):
        safe = "a" + safe
    candidate = f"actor:{safe}".rstrip("-/")
    if candidate.endswith("actor:"):
        candidate = "actor:anonymous"
    if ACTOR_ID_ALLOWED.match(candidate):
        return candidate
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"actor:{digest}"


def normalize_session_id(raw: str) -> str:
    """
    ✅ idempotent：何回呼んでも sess_ が増殖しない
    - 既に sess_ で始まる場合：prefixを剥がして正規化してから付け直す
    - 最終的に SESSION_ID_ALLOWED を満たす sess_<safe> を返す
    """
    if not raw:
        raw = "default"

    s = str(raw).strip()

    # 既に sess_ が付いてるなら剥がす（増殖防止）
    while s.startswith("sess_"):
        s = s[len("sess_") :]

    if not s:
        s = "default"

    safe = re.sub(r"[^a-zA-Z0-9\-_]", "_", s)
    if not re.match(r"^[a-zA-Z0-9]", safe):
        safe = "s_" + safe

    candidate = f"sess_{safe}"
    if SESSION_ID_ALLOWED.match(candidate):
        return candidate

    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]
    return f"sess_{digest}"


def generate_new_session_id(actor_id: str) -> str:
    actor_digest = hashlib.sha256(actor_id.encode("utf-8")).hexdigest()[:8]
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rnd = f"{random.randint(0, 9999):04d}"
    return normalize_session_id(f"{actor_digest}_{ts}_{rnd}")

######################################
# ✅ runtimeSessionId（Runtime用：正規化しない）
######################################
RUNTIME_SID_ALLOWED = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-]{7,}$")


def ensure_runtime_session_id(raw: str | None) -> str:
    if raw and isinstance(raw, str) and RUNTIME_SID_ALLOWED.match(raw):
        return raw
    # さらに保守的にするなら: return "rt" + uuid.uuid4().hex
    return f"rt-{uuid.uuid4()}"

######################################
# actor_id
######################################
def get_actor_id_from_auth0() -> str:
    u = st.user
    raw = (
        str(getattr(u, "sub", "")).strip()
        or str(getattr(u, "id", "")).strip()
        or str(getattr(u, "email", "")).strip()
        or str(getattr(u, "name", "")).strip()
        or "anonymous"
    )
    return normalize_actor_id(raw)

######################################
# AgentCore クライアント
######################################
@st.cache_resource
def get_agentcore_client(region: str):
    return boto3.client("bedrock-agentcore", region_name=region)

agentcore = get_agentcore_client(REGION)

######################################
# ✅ state
######################################
if "messages" not in st.session_state:
    st.session_state.messages = []

if "actor_id" not in st.session_state:
    st.session_state.actor_id = get_actor_id_from_auth0()

if "memory_session_id" not in st.session_state:
    st.session_state.memory_session_id = normalize_session_id("default")
else:
    # 念のため：増殖しない版なので安全
    st.session_state.memory_session_id = normalize_session_id(st.session_state.memory_session_id)

if "runtime_session_id" not in st.session_state:
    st.session_state.runtime_session_id = ensure_runtime_session_id(None)
else:
    st.session_state.runtime_session_id = ensure_runtime_session_id(st.session_state.runtime_session_id)

######################################
# Memory：ListSessions
######################################
def list_memory_sessions(memory_id: str, actor_id: str, max_results: int = 100) -> list[dict]:
    items: list[dict] = []
    next_token = None

    while True:
        kwargs = {"memoryId": memory_id, "actorId": actor_id, "maxResults": min(max_results, 100)}
        if next_token:
            kwargs["nextToken"] = next_token

        try:
            res = agentcore.list_sessions(**kwargs)
        except Exception as e:
            # ✅ actor未作成時は ResourceNotFoundException で落ちることがある → 空扱い
            msg = str(e)
            if "ResourceNotFoundException" in msg and "Actor" in msg and "not found" in msg:
                return []
            raise

        items.extend(res.get("sessionSummaries", []))
        next_token = res.get("nextToken")
        if not next_token or len(items) >= max_results:
            break

    def _ts(x):
        v = x.get("createdAt")
        if isinstance(v, datetime):
            return v
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    items.sort(key=_ts, reverse=True)
    return items[:max_results]

######################################
# Memory：ListEvents
######################################
def load_memory_history(memory_id: str, actor_id: str, session_id: str, max_events: int = 300) -> list[dict]:
    session_id = normalize_session_id(session_id)

    events: list[dict] = []
    next_token = None

    while True:
        kwargs = {
            "memoryId": memory_id,
            "actorId": actor_id,
            "sessionId": session_id,
            "includePayloads": True,
            "maxResults": 50,
        }
        if next_token:
            kwargs["nextToken"] = next_token

        res = agentcore.list_events(**kwargs)
        events.extend(res.get("events", []))
        next_token = res.get("nextToken")
        if not next_token or len(events) >= max_events:
            break

    def _ets(e):
        v = e.get("eventTimestamp")
        if isinstance(v, datetime):
            return v
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    events.sort(key=_ets)

    messages: list[dict] = []
    for e in events:
        payload_list = e.get("payload") or []
        for p in payload_list:
            conv = (p or {}).get("conversational")
            if not conv:
                continue
            role = conv.get("role")
            text = (((conv.get("content") or {}).get("text")) or "").strip()
            if not text:
                continue
            if role == "USER":
                messages.append({"role": "user", "content": text})
            elif role == "ASSISTANT":
                messages.append({"role": "assistant", "content": text})
    return messages

######################################
# UI
######################################
st.title("インフルエンサー検索エージェント")
st.write("Youtube, Instagramのインフルエンサーの情報を収集します！")
st.write("あなたは何ができますか？ と聞いてみてください。")

######################################
# Sidebar
######################################
with st.sidebar:
    st.caption("セッション（AgentCore Memory）")
    actor_id = st.session_state.actor_id
    if not ACTOR_ID_ALLOWED.match(actor_id):
        st.error(f"actor_id invalid: {actor_id}")
        st.stop()

    st.caption("debug")
    st.write("actor_id:", repr(actor_id))
    st.write("memory_session_id:", repr(st.session_state.memory_session_id))
    st.write("runtime_session_id:", repr(st.session_state.runtime_session_id))

    if "session_list" not in st.session_state:
        try:
            st.session_state.session_list = list_memory_sessions(MEMORY_ID, actor_id)
        except Exception as e:
            st.error(f"list_sessions failed: {e}")
            st.session_state.session_list = []

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 一覧更新"):
            try:
                st.session_state.session_list = list_memory_sessions(MEMORY_ID, actor_id)
                st.rerun()
            except Exception as e:
                st.error(f"list_sessions failed: {e}")

    with col2:
        if st.button("➕ 新規"):
            new_sid = generate_new_session_id(actor_id)
            st.session_state.memory_session_id = new_sid
            st.session_state.messages = []
            st.rerun()

    sessions = st.session_state.session_list or []
    ids = [s["sessionId"] for s in sessions] if sessions else [st.session_state.memory_session_id]
    labels = [f'{s["sessionId"]}  ({s.get("createdAt","")})' for s in sessions] if sessions else [st.session_state.memory_session_id]

    current_sid = st.session_state.memory_session_id if st.session_state.memory_session_id in ids else ids[0]
    current_index = ids.index(current_sid)

    selected_index = st.selectbox(
        "過去セッション",
        options=list(range(len(ids))),
        format_func=lambda i: labels[i],
        index=current_index,
    )

    if st.button("📥 開く"):
        target_session_id = normalize_session_id(ids[selected_index])
        st.session_state.memory_session_id = target_session_id
        try:
            st.session_state.messages = load_memory_history(MEMORY_ID, actor_id, target_session_id)
            st.rerun()
        except Exception as e:
            st.error(f"load history failed: {e}")

    st.caption("現在のセッションID（Memory）")
    st.code(st.session_state.memory_session_id, language="text")

######################################
# 過去メッセージ描画
######################################
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

######################################
# Chat
######################################
if prompt := st.chat_input("メッセージを入力してね"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        payload_obj = {
            "op": "chat",
            "prompt": prompt,
            "actor_id": st.session_state.actor_id,
            "session_id": st.session_state.memory_session_id,
            "memory_id": MEMORY_ID,
        }

        response = agentcore.invoke_agent_runtime(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            runtimeSessionId=st.session_state.runtime_session_id,  # ✅固定
            payload=json.dumps(payload_obj).encode("utf-8"),
        )

        container = st.container()
        text_holder = container.empty()
        buffer = ""

        for line in response["response"].iter_lines():
            if not line:
                continue
            s = line.decode("utf-8")
            if not s.startswith("data: "):
                continue

            data = s[6:]
            if data.startswith('"') or data.startswith("'"):
                continue

            event = json.loads(data)

            if isinstance(event, dict) and event.get("type") == "meta":
                sid = event.get("session_id")
                if sid:
                    st.session_state.memory_session_id = normalize_session_id(sid)
                continue

            if isinstance(event, dict) and event.get("type") == "error":
                st.error(event.get("message", "unknown error"))
                break

            if "data" in event and isinstance(event["data"], str):
                buffer += event["data"]
                text_holder.markdown(buffer)
            elif "event" in event and "contentBlockDelta" in event["event"]:
                buffer += event["event"]["contentBlockDelta"]["delta"].get("text", "")
                text_holder.markdown(buffer)

        text_holder.markdown(buffer)

    st.session_state.messages.append({"role": "assistant", "content": buffer})
