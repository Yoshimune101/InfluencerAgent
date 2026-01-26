import os, boto3, json, uuid
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
# 環境変数と認証の設定
######################################
load_dotenv()
REGION = os.getenv("AWS_REGION")
AGENT_RUNTIME_ARN = os.getenv("AGENT_RUNTIME_ARN")
MEMORY_ID = os.getenv("MEMORY_ID")  # ★追加

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
# actor_idの設定
######################################
def get_actor_id_from_auth0() -> str:
    """
    Auth0/Streamlit の st.user から安定して一意なIDを引く。
    優先順位: sub > id > email > name
    """
    u = st.user
    return (
        str(getattr(u, "sub", "")).strip()
        or str(getattr(u, "id", "")).strip()
        or str(getattr(u, "email", "")).strip()
        or str(getattr(u, "name", "")).strip()
        or "anonymous"
    )

######################################
# AgentCore クライアント
######################################
@st.cache_resource
def get_agentcore_client(region: str):
    return boto3.client("bedrock-agentcore", region_name=region)

agentcore = get_agentcore_client(REGION)

######################################
# ✅ セッション状態
######################################
if "messages" not in st.session_state:
    st.session_state.messages = []

# “memory_session_id” を会話スレッドIDとして扱う
if "memory_session_id" not in st.session_state:
    st.session_state.memory_session_id = "default"

# runtimeSessionId は「空だと即死」なので、必ず非空にする
if "runtime_session_id" not in st.session_state or not st.session_state.runtime_session_id:
    st.session_state.runtime_session_id = st.session_state.memory_session_id or f"rt_{uuid.uuid4().hex}"

######################################
# AgentCore Runtime 呼び出し（チャット専用）
######################################
def invoke_agentcore_stream(payload_obj: dict) -> tuple[list[dict], str]:
    """
    AgentCore Runtime を呼び出し、ストリームから
    - meta/error を回収
    - assistantの最終テキストを返す
    """
    runtime_session_id = st.session_state.get("runtime_session_id")
    if not runtime_session_id:
        runtime_session_id = f"rt_{uuid.uuid4().hex}"
        st.session_state.runtime_session_id = runtime_session_id

    response = agentcore.invoke_agent_runtime(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        runtimeSessionId=runtime_session_id,
        payload=json.dumps(payload_obj).encode("utf-8"),
    )

    controls: list[dict] = []
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

        # control(dict) を拾う（meta/error）
        if isinstance(event, dict) and event.get("type") in ("meta", "error"):
            controls.append(event)
            continue

        # テキスト抽出（最小）
        if isinstance(event, dict) and "data" in event and isinstance(event["data"], str):
            buffer += event["data"]
        elif isinstance(event, dict) and "event" in event and "contentBlockDelta" in event["event"]:
            buffer += event["event"]["contentBlockDelta"]["delta"].get("text", "")

    return controls, buffer

######################################
# AgentCore Memory：セッション一覧取得
######################################
def list_memory_sessions(memory_id: str, actor_id: str, max_results: int = 100) -> list[dict]:
    """
    list_sessions はページングがあるので全件取得して返す
    """
    items: list[dict] = []
    next_token = None

    while True:
        kwargs = {"memoryId": memory_id, "actorId": actor_id, "maxResults": min(max_results, 100)}
        if next_token:
            kwargs["nextToken"] = next_token

        res = agentcore.list_sessions(**kwargs)  # ★公式API :contentReference[oaicite:0]{index=0}
        items.extend(res.get("sessionSummaries", []))
        next_token = res.get("nextToken")
        if not next_token or len(items) >= max_results:
            break

    # createdAt 新しい順
    items.sort(key=lambda x: x.get("createdAt", datetime(1970, 1, 1, tzinfo=timezone.utc)), reverse=True)
    return items[:max_results]

######################################
# AgentCore Memory：会話履歴取得（sessionId単位）
######################################
def load_memory_history(memory_id: str, actor_id: str, session_id: str, max_events: int = 200) -> list[dict]:
    """
    list_events -> payload.conversational から user/assistant のメッセージ配列を復元する
    """
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

        res = agentcore.list_events(**kwargs)  # ★公式API :contentReference[oaicite:1]{index=1}
        events.extend(res.get("events", []))
        next_token = res.get("nextToken")
        if not next_token or len(events) >= max_events:
            break

    # 古い順に並べる
    events.sort(key=lambda e: e.get("eventTimestamp", datetime(1970, 1, 1, tzinfo=timezone.utc)))

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

            # AgentCore Memory の role は USER/ASSISTANT なので Streamlit 表示用に寄せる
            if role == "USER":
                messages.append({"role": "user", "content": text})
            elif role == "ASSISTANT":
                messages.append({"role": "assistant", "content": text})
            # TOOL/OTHER はここでは捨てる（必要なら後で可視化）

    return messages

######################################
# Streamlit UI
######################################
st.title("インフルエンサー検索エージェント")
st.write("Youtube, Instagramのインフルエンサーの情報を収集します！")
st.write("あなたは何ができますか？ と聞いてみてください。")

######################################
# Sidebar（セッション一覧・切替）
######################################
with st.sidebar:
    st.caption("セッション（AgentCore Memory）")
    actor_id = get_actor_id_from_auth0()

    # 初回ロード
    if "session_list" not in st.session_state:
        try:
            st.session_state.session_list = list_memory_sessions(MEMORY_ID, actor_id)
        except Exception as e:
            st.error(f"list_sessions failed: {e}")
            st.session_state.session_list = []

    col1, col2 = st.columns(2)

    # 一覧更新
    with col1:
        if st.button("🔄 一覧更新"):
            try:
                st.session_state.session_list = list_memory_sessions(MEMORY_ID, actor_id)
                st.rerun()
            except Exception as e:
                st.error(f"list_sessions failed: {e}")

    # 新規セッション作成：Memory側は「勝手に作られる」こともあるので、
    # ここではクライアント側で新しい session_id を採番して切替（確実に動く）
    with col2:
        if st.button("➕ 新規"):
            new_sid = f"sess_{uuid.uuid4().hex}"
            st.session_state.memory_session_id = new_sid
            st.session_state.runtime_session_id = new_sid  # 同期
            st.session_state.messages = []
            st.rerun()

    sessions = st.session_state.session_list or []
    ids = [s["sessionId"] for s in sessions] if sessions else ["default"]
    labels = [
        f'{s["sessionId"]}  ({s.get("createdAt","")})' for s in sessions
    ] if sessions else ["default"]

    current_sid = st.session_state.memory_session_id if st.session_state.memory_session_id in ids else ids[0]
    current_index = ids.index(current_sid)

    selected_index = st.selectbox(
        "過去セッション",
        options=list(range(len(ids))),
        format_func=lambda i: labels[i],
        index=current_index,
    )

    if st.button("📥 開く"):
        target_session_id = ids[selected_index]
        st.session_state.memory_session_id = target_session_id
        st.session_state.runtime_session_id = target_session_id  # 同期

        try:
            st.session_state.messages = load_memory_history(MEMORY_ID, actor_id, target_session_id)
            st.rerun()
        except Exception as e:
            st.error(f"load history failed: {e}")

    st.caption("現在のセッションID")
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
        actor_id = get_actor_id_from_auth0()

        payload_obj = {
            "op": "chat",
            "prompt": prompt,
            "actor_id": actor_id,
            "session_id": st.session_state.memory_session_id,
        }

        response = agentcore.invoke_agent_runtime(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            runtimeSessionId=st.session_state.runtime_session_id or f"rt_{uuid.uuid4().hex}",
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

            # meta / error
            if isinstance(event, dict) and event.get("type") == "meta":
                sid = event.get("session_id")
                if sid:
                    st.session_state.memory_session_id = sid
                    st.session_state.runtime_session_id = sid
                continue
            if isinstance(event, dict) and event.get("type") == "error":
                st.error(event.get("message", "unknown error"))
                break

            # ツール利用検出（既存仕様）
            if "event" in event and "contentBlockStart" in event["event"]:
                if "toolUse" in event["event"]["contentBlockStart"].get("start", {}):
                    if buffer:
                        text_holder.markdown(buffer)
                        buffer = ""

                    tool_name = event["event"]["contentBlockStart"]["start"]["toolUse"].get("name", "unknown")
                    container.info(f"🔍 {tool_name} ツールを利用しています")
                    text_holder = container.empty()

            # テキスト
            if "data" in event and isinstance(event["data"], str):
                buffer += event["data"]
                text_holder.markdown(buffer)
            elif "event" in event and "contentBlockDelta" in event["event"]:
                buffer += event["event"]["contentBlockDelta"]["delta"].get("text", "")
                text_holder.markdown(buffer)

        text_holder.markdown(buffer)

    st.session_state.messages.append({"role": "assistant", "content": buffer})
