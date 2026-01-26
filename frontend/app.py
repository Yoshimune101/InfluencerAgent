import os, boto3, json
import streamlit as st
from dotenv import load_dotenv

######################################
# ログイン
#######################################
if not st.user.is_logged_in:
    st.login("auth0")
    st.stop()

st.success(f"Hello {st.user.name}")

######################################
# 環境変数と認証の設定
######################################
load_dotenv()
REGION = os.getenv("AWS_REGION")
AGENT_RUNTIME_ARN = os.getenv("AGENT_RUNTIME_ARN")

######################################
# actor_idの設定
#######################################
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
# AgentCore クライアント（先に作る）
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

if "memory_session_id" not in st.session_state:
    st.session_state.memory_session_id = "default"

if "runtime_session_id" not in st.session_state:
    # runtime は memory と揃える（ズレを無くす）
    st.session_state.runtime_session_id = st.session_state.memory_session_id

######################################
# AgentCore 呼び出し（control message 回収）
######################################
def invoke_agentcore_stream(payload_obj: dict) -> tuple[list[dict], str]:
    """
    AgentCore を呼び出し、ストリームから
    - control messages（type=meta/sessions/history/error）を回収
    - assistantの最終テキストを返す
    """
    response = agentcore.invoke_agent_runtime(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        runtimeSessionId=st.session_state.runtime_session_id,
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
            # 文字列コンテンツは既存仕様に合わせてスキップ
            continue

        event = json.loads(data)

        # control(dict) を拾う（AgentCore側が明示的に返す type）
        if isinstance(event, dict) and event.get("type") in ("meta", "sessions", "history", "error"):
            controls.append(event)
            continue

        # テキスト抽出（最小）
        if "data" in event and isinstance(event["data"], str):
            buffer += event["data"]
        elif "event" in event and "contentBlockDelta" in event["event"]:
            buffer += event["event"]["contentBlockDelta"]["delta"].get("text", "")

    return controls, buffer

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
    st.caption("セッション")
    actor_id = get_actor_id_from_auth0()

    # 初回は自動で一覧取得
    if "session_list" not in st.session_state:
        controls, _ = invoke_agentcore_stream({
            "op": "list_sessions",
            "actor_id": actor_id,
        })
        err = next((c for c in controls if c.get("type") == "error"), None)
        if err:
            st.error(f"list_sessions error: {err.get('message')}")
            st.session_state.session_list = []
        else:
            sess = next((c for c in controls if c.get("type") == "sessions"), None)
            st.session_state.session_list = sess.get("items", []) if sess else []

    col1, col2 = st.columns(2)

    # 一覧更新
    with col1:
        if st.button("🔄 一覧更新"):
            controls, _ = invoke_agentcore_stream({
                "op": "list_sessions",
                "actor_id": actor_id,
            })
            err = next((c for c in controls if c.get("type") == "error"), None)
            if err:
                st.error(f"list_sessions error: {err.get('message')}")
            else:
                sess = next((c for c in controls if c.get("type") == "sessions"), None)
                st.session_state.session_list = sess.get("items", []) if sess else []
                st.rerun()

    # 新規セッション作成
    with col2:
        if st.button("➕ 新規"):
            controls, _ = invoke_agentcore_stream({
                "op": "new_session",
                "actor_id": actor_id,
            })
            err = next((c for c in controls if c.get("type") == "error"), None)
            if err:
                st.error(f"new_session error: {err.get('message')}")
            else:
                meta = next((c for c in controls if c.get("type") == "meta"), None)
                if meta and meta.get("session_id"):
                    st.session_state.memory_session_id = meta["session_id"]
                    st.session_state.runtime_session_id = meta["session_id"]
                    st.session_state.messages = []
                    st.rerun()

    options = st.session_state.session_list
    ids = [x["session_id"] for x in options] if options else ["default"]
    labels = [f'{x["session_id"]}  ({x.get("updatedAt","")})' for x in options] if options else ["default"]

    # 現在選択中を維持
    current_index = ids.index(st.session_state.memory_session_id) if st.session_state.memory_session_id in ids else 0

    selected_index = st.selectbox(
        "過去セッション",
        options=list(range(len(ids))),
        format_func=lambda i: labels[i],
        index=current_index,
    )

    if st.button("📥 開く"):
        target_session_id = ids[selected_index]
        st.session_state.memory_session_id = target_session_id
        st.session_state.runtime_session_id = target_session_id

        controls, _ = invoke_agentcore_stream({
            "op": "get_session",
            "actor_id": actor_id,
            "target_session_id": target_session_id,
        })
        err = next((c for c in controls if c.get("type") == "error"), None)
        if err:
            st.error(f"get_session error: {err.get('message')}")
        else:
            hist = next((c for c in controls if c.get("type") == "history"), None)
            st.session_state.messages = hist.get("messages", []) if hist else []
            st.rerun()

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

            # ✅ meta / error を拾って同期（ここが重要）
            if isinstance(event, dict) and event.get("type") == "meta":
                sid = event.get("session_id")
                if sid:
                    st.session_state.memory_session_id = sid
                    st.session_state.runtime_session_id = sid
                continue
            if isinstance(event, dict) and event.get("type") == "error":
                st.error(event.get("message", "unknown error"))
                break

            # ツール利用を検出（既存仕様）
            if "event" in event and "contentBlockStart" in event["event"]:
                if "toolUse" in event["event"]["contentBlockStart"].get("start", {}):
                    if buffer:
                        text_holder.markdown(buffer)
                        buffer = ""

                    tool_name = event["event"]["contentBlockStart"]["start"]["toolUse"].get("name", "unknown")
                    container.info(f"🔍 {tool_name} ツールを利用しています")
                    text_holder = container.empty()

            # テキストコンテンツを検出
            if "data" in event and isinstance(event["data"], str):
                buffer += event["data"]
                text_holder.markdown(buffer)
            elif "event" in event and "contentBlockDelta" in event["event"]:
                buffer += event["event"]["contentBlockDelta"]["delta"].get("text", "")
                text_holder.markdown(buffer)

        text_holder.markdown(buffer)

    st.session_state.messages.append({"role": "assistant", "content": buffer})
