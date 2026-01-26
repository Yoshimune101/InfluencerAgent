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
    """
    if not raw:
        raw = "default"

    s = str(raw).strip()
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
    st.session_state.memory_session_id = normalize_session_id(st.session_state.memory_session_id)

if "runtime_session_id" not in st.session_state:
    st.session_state.runtime_session_id = ensure_runtime_session_id(None)
else:
    st.session_state.runtime_session_id = ensure_runtime_session_id(st.session_state.runtime_session_id)

# ✅ 既知 sessionId（actorごと）
if "known_session_ids" not in st.session_state:
    st.session_state.known_session_ids = {}  # { actor_id: set([...]) }

actor_id = st.session_state.actor_id
st.session_state.known_session_ids.setdefault(actor_id, set())
st.session_state.known_session_ids[actor_id].add(st.session_state.memory_session_id)

# ✅ debug: 最後に開いたセッションの events 数/サンプル
if "debug_opened_session" not in st.session_state:
    st.session_state.debug_opened_session = {"sessionId": "", "eventsCount": None, "sample": None}

######################################
# Memory：ListSessions（本命、揺れ吸収）
######################################
def list_memory_sessions_official(memory_id: str, actor_id: str, max_results: int = 100) -> list[dict]:
    items: list[dict] = []
    next_token = None

    while True:
        kwargs = {"memoryId": memory_id, "actorId": actor_id, "maxResults": min(max_results, 100)}
        if next_token:
            kwargs["nextToken"] = next_token

        try:
            res = agentcore.list_sessions(**kwargs)
        except Exception as e:
            msg = str(e)
            if "ResourceNotFoundException" in msg and "not found" in msg:
                return []
            raise

        raw_list = (res.get("sessionSummaries") or res.get("items") or res.get("sessions") or [])

        for s in raw_list:
            sid = s.get("sessionId") or s.get("session_id")
            if not sid:
                continue
            sid = normalize_session_id(sid)

            created_at = s.get("createdAt") or s.get("created_at") or ""
            updated_at = s.get("updatedAt") or s.get("updated_at") or created_at or ""

            items.append(
                {
                    "sessionId": sid,
                    "createdAt": str(created_at) if created_at else "",
                    "updatedAt": str(updated_at) if updated_at else "",
                }
            )

        next_token = res.get("nextToken")
        if not next_token or len(items) >= max_results:
            break

    items.sort(key=lambda x: x.get("updatedAt", "") or x.get("createdAt", ""), reverse=True)
    return items[:max_results]

######################################
# Memory：ListEvents（sessionId必須の環境）
######################################
def list_events_for_session(memory_id: str, actor_id: str, session_id: str, include_payloads: bool, max_events: int) -> list[dict]:
    session_id = normalize_session_id(session_id)

    events: list[dict] = []
    next_token = None

    while True:
        kwargs = {
            "memoryId": memory_id,
            "actorId": actor_id,
            "sessionId": session_id,
            "includePayloads": include_payloads,
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
    return events[:max_events]

def _infer_role_from_event_name(name: str) -> str | None:
    n = (name or "").lower()
    if "user" in n:
        return "user"
    if "assistant" in n:
        return "assistant"
    if "tool" in n:
        return "assistant"
    return None

def _extract_texts_from_event(ev: dict) -> list[tuple[str, str]]:
    """
    できるだけ多様な形式から user/assistant のテキストを抽出する。
    戻り値: [(role, text), ...]
    """
    out: list[tuple[str, str]] = []

    # 1) payload.conversational (想定)
    payload_list = ev.get("payload") or []
    for p in payload_list:
        if not isinstance(p, dict):
            continue
        conv = p.get("conversational")
        if isinstance(conv, dict):
            role = conv.get("role")
            text = (((conv.get("content") or {}).get("text")) or "").strip()
            if role in ("USER", "ASSISTANT") and text:
                out.append(("user" if role == "USER" else "assistant", text))

        # 2) payload 直下に content/text があるケース（環境差分）
        #    例: {"role":"USER","content":{"text":"..."}} など
        role_guess = p.get("role") or p.get("type")
        content = p.get("content")
        if isinstance(content, dict):
            text = (content.get("text") or content.get("message") or "").strip() if isinstance(content.get("text") or content.get("message"), str) else ""
            if text:
                if role_guess in ("USER", "user"):
                    out.append(("user", text))
                elif role_guess in ("ASSISTANT", "assistant"):
                    out.append(("assistant", text))

        # 3) payload 内に文字列が入るケース
        for k in ("text", "message", "data"):
            v = p.get(k)
            if isinstance(v, str) and v.strip():
                role = "assistant"
                if role_guess in ("USER", "user"):
                    role = "user"
                out.append((role, v.strip()))

    # 4) attributes.content 互換（旧実装）
    attrs = ev.get("attributes") or {}
    content = attrs.get("content")
    if content:
        text = None
        try:
            obj = json.loads(content) if isinstance(content, str) else content
            if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "text" in obj[0]:
                text = "".join([x.get("text", "") for x in obj if isinstance(x, dict)]).strip()
            elif isinstance(obj, str):
                text = obj.strip()
        except Exception:
            text = content.strip() if isinstance(content, str) else None

        if text:
            role = _infer_role_from_event_name(ev.get("name") or ev.get("eventName") or "")
            if role:
                out.append((role, text))

    return out

def load_memory_history(memory_id: str, actor_id: str, session_id: str, max_events: int = 300) -> list[dict]:
    events = list_events_for_session(memory_id, actor_id, session_id, include_payloads=True, max_events=max_events)
    msgs: list[dict] = []
    for ev in events:
        pairs = _extract_texts_from_event(ev)
        for role, text in pairs:
            msgs.append({"role": role, "content": text})

    # 同じ連続メッセージの重複を軽く除去（payload+attributesで二重に拾うことがある）
    deduped: list[dict] = []
    last = None
    for m in msgs:
        key = (m["role"], m["content"])
        if key == last:
            continue
        deduped.append(m)
        last = key

    return deduped

######################################
# ✅ セッション一覧取得（official優先、fallbackで既知ID）
######################################
def get_session_list(memory_id: str, actor_id: str) -> list[dict]:
    official = list_memory_sessions_official(memory_id, actor_id, max_results=100)
    if official:
        for s in official:
            st.session_state.known_session_ids[actor_id].add(s["sessionId"])
        return official

    known = sorted(list(st.session_state.known_session_ids.get(actor_id, set())))
    return [{"sessionId": sid, "createdAt": "", "updatedAt": ""} for sid in known]

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

    # 最後に「開く」した時の events 情報
    dbg = st.session_state.debug_opened_session
    if dbg.get("sessionId"):
        st.caption("debug(open)")
        st.write("opened sessionId:", dbg.get("sessionId"))
        st.write("events count:", dbg.get("eventsCount"))
        if dbg.get("sample"):
            st.json(dbg.get("sample"))

    if "session_list" not in st.session_state:
        try:
            st.session_state.session_list = get_session_list(MEMORY_ID, actor_id)
        except Exception as e:
            st.error(f"list sessions failed: {e}")
            st.session_state.session_list = [{"sessionId": st.session_state.memory_session_id, "createdAt": "", "updatedAt": ""}]

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 一覧更新"):
            try:
                st.session_state.session_list = get_session_list(MEMORY_ID, actor_id)
                st.rerun()
            except Exception as e:
                st.error(f"list sessions failed: {e}")

    with col2:
        if st.button("➕ 新規"):
            new_sid = generate_new_session_id(actor_id)
            st.session_state.memory_session_id = new_sid
            st.session_state.known_session_ids[actor_id].add(new_sid)
            st.session_state.messages = []
            st.rerun()

    sessions = st.session_state.session_list or []
    ids = [s["sessionId"] for s in sessions] if sessions else [st.session_state.memory_session_id]

    # ✅ () が出ないラベルにする：日時が無いなら sessionId だけ
    labels = []
    for s in sessions:
        sid = s["sessionId"]
        ts = (s.get("updatedAt") or s.get("createdAt") or "").strip()
        labels.append(f"{sid}  ({ts})" if ts else sid)
    if not labels:
        labels = [st.session_state.memory_session_id]

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
        st.session_state.known_session_ids[actor_id].add(target_session_id)

        try:
            # debug(open): まず events が存在するか確認（payloadは不要）
            evs = list_events_for_session(MEMORY_ID, actor_id, target_session_id, include_payloads=False, max_events=50)
            st.session_state.debug_opened_session = {
                "sessionId": target_session_id,
                "eventsCount": len(evs),
                "sample": {
                    "name": (evs[0].get("name") if evs else None),
                    "sessionId": (evs[0].get("sessionId") if evs else None),
                    "eventTimestamp": str(evs[0].get("eventTimestamp")) if evs else None,
                } if evs else None
            }

            # その後、履歴をロード（payload込み）
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
            runtimeSessionId=st.session_state.runtime_session_id,
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
                    sid = normalize_session_id(sid)
                    st.session_state.memory_session_id = sid
                    st.session_state.known_session_ids[st.session_state.actor_id].add(sid)
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

    # チャット後に一覧更新
    try:
        st.session_state.session_list = get_session_list(MEMORY_ID, st.session_state.actor_id)
    except Exception:
        pass
