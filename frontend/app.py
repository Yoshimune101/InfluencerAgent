import os, boto3, json, uuid
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
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

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
# ✅ セッション（会話履歴 + AgentCoreセッションID）を保持
######################################
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"/"assistant","content":"..."}]

if "runtime_session_id" not in st.session_state:
    # AgentCore に同一会話として扱わせるID（このブラウザセッション中は固定）
    st.session_state.runtime_session_id = str(uuid.uuid4())

######################################
# StreamlitアプリのUI構築
######################################
st.title("インフルエンサー検索エージェント")
st.write("Youtube, Instagramのインフルエンサーの情報を収集します！")
st.write("あなたは何ができますか？ と聞いてみてください。")

# ✅ リセット（必要なら）
with st.sidebar:
    st.caption("セッション")
    st.code(st.session_state.runtime_session_id, language="text")
    if st.button("会話をリセット"):
        st.session_state.messages = []
        st.session_state.runtime_session_id = str(uuid.uuid4())
        st.rerun()

# ✅ 過去メッセージを先に描画（これで「前回の表示が消える」を解消）
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

@st.cache_resource
def get_agentcore_client(region: str):
    # クライアント生成をキャッシュ（毎回作らない）
    return boto3.client("bedrock-agentcore", region_name=region)

agentcore = get_agentcore_client(REGION)

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
