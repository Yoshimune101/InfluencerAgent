import os, boto3, json, uuid, re, hashlib
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
MEMORY_ID = os.getenv("MEMORY_ID")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

agent_core_client = boto3.client("bedrock-agentcore", region_name="us-west-2")

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

ACTOR_ID_ALLOWED = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9\-_\/]*(?::[a-zA-Z0-9\-_\/]+)*[a-zA-Z0-9\-_\/]*$"
)
SESSION_ID_ALLOWED = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9\-_]*$"
)

def normalize_actor_id(raw: str) -> str:
    if not raw:
        raw = "anonymous"
    safe = re.sub(r"[^a-zA-Z0-9\-_\/:]", "_", raw)
    if not re.match(r"^[a-zA-Z0-9]", safe):
        safe = "a_" + safe
    safe = safe.rstrip(":")
    # actor: を付けておく（: は actorId では許可される）
    candidate = f"actor:{safe}"
    if ACTOR_ID_ALLOWED.match(candidate):
        return candidate
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"actor:{digest}"

def normalize_session_id(raw: str) -> str:
    """
    ListEvents の sessionId 制約に合わせる：
    - 先頭英数字
    - 以降は [a-zA-Z0-9-_] のみ
    => ':' '/' は絶対に入れない
    """
    if not raw:
        raw = "default"
    # 許可文字以外は '_' に
    safe = re.sub(r"[^a-zA-Z0-9\-_]", "_", raw)
    # 先頭英数字制約
    if not re.match(r"^[a-zA-Z0-9]", safe):
        safe = "s_" + safe
    candidate = f"sess_{safe}"  # ':' を使わない
    if SESSION_ID_ALLOWED.match(candidate):
        return candidate
    # 最終手段：sha256（英数字のみ）
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"sess_{digest}"


######################################
# ✅ セッション（会話履歴 + AgentCoreセッションID）を保持
######################################
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"/"assistant","content":"..."}]

if "runtime_session_id" not in st.session_state:
    # AgentCore に同一会話として扱わせるID（このブラウザセッション中は固定）
    st.session_state.runtime_session_id = str(uuid.uuid4())

def set_session_id():
    """ボタンクリック時の処理。セッションIDをセットし、メッセージをクリアする。"""
    st.session_state["messages"]
    st.session_state.messages = []
    st.session_state.runtime_session_id = str(uuid.uuid4())
    st.rerun()

######################################
# StreamlitアプリのUI構築
######################################
st.title("インフルエンサー検索エージェント")
st.write("Youtube, Instagramのインフルエンサーの情報を収集します！")
st.write("あなたは何ができますか？ と聞いてみてください。")

# ✅ 過去メッセージを先に描画（これで「前回の表示が消える」を解消）
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

@st.cache_resource
def get_agentcore_client(region: str):
    # クライアント生成をキャッシュ（毎回作らない）
    return boto3.client("bedrock-agentcore", region_name=region)

agentcore = get_agentcore_client(REGION)


if AGENT_RUNTIME_ARN and MEMORY_ID:
    actor_id = normalize_actor_id(get_actor_id_from_auth0())

    ##############
    # 過去メッセージを取得して表示 
    ##############
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
                }
            ),
            qualifier="DEFAULT",
        )

        for line in response["response"].iter_lines(chunk_size=10):
            if line:
                line = line.decode("utf-8")
                line = line[6:]  # 先頭の`data: `を除去
                line = json.loads(line)
                st.session_state["messages"] = line

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"].lower()):
            for content in message["content"]:
                st.write(content["text"])

    ##############
    # チャット入力UIを構築 
    ##############
    # チャットボックスを描画
    if prompt := st.chat_input("メッセージを入力してね"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            actor_id = get_actor_id_from_auth0()

            payload = json.dumps({
                "prompt": prompt,
                "actor_id": actor_id,
                "session_id": st.session_state.runtime_session_id,
            })

            response = agentcore.invoke_agent_runtime(
                agentRuntimeArn=AGENT_RUNTIME_ARN,
                runtimeSessionId=st.session_state.runtime_session_id,  # AgentCoreの会話継続用
                payload=payload.encode("utf-8"),
            )

            ### ここから下はストリーミングレスポンスの処理 ------------------------------------------
            container = st.container()
            text_holder = container.empty()
            buffer = ""

            for line in response["response"].iter_lines():
                if line and line.decode("utf-8").startswith("data: "):
                    data = line.decode("utf-8")[6:]

                    # 文字列コンテンツの場合は無視
                    if data.startswith('"') or data.startswith("'"):
                        continue

                    event = json.loads(data)

                    # ツール利用を検出
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
            ### ここから上はストリーミングレスポンスの処理 ------------------------------------------

        # ✅ 返答を履歴に追加（次回の rerun で残る）
        st.session_state.messages.append({"role": "assistant", "content": buffer})


    ##############
    # サイドバーでセッション切り替えUIを構築
    ##############
    with st.sidebar:
        st.text_input(
            label="Session ID", value=st.session_state.runtime_session_id, disabled=True
        )

        st.button(
            "new thread",
            on_click=set_session_id(),
            type="primary",
        )

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
                        }
                    ),
                    qualifier="DEFAULT",
                )

                for line in response["response"].iter_lines(chunk_size=10):
                    if line:
                        line = line.decode("utf-8")
                        line = line[6:]  # 先頭の`data: `を除去
                        line = json.loads(line)

                        st.session_state["session_id_list"] = list(
                            map(lambda x: x["sessionId"], line)
                        )

        if "session_id_list" in st.session_state:
            for session_id in st.session_state["session_id_list"]:
                st.button(session_id, on_click=set_session_id, args=[session_id])
