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
REGION = os.getenv("AWS_REGION", "us-west-2")
AGENT_RUNTIME_ARN = os.getenv("AGENT_RUNTIME_ARN")

# ✅ Consoleで作成したMemory ID（例: influencer_agent_memory-6u1h9b8Hgk）
MEMORY_ID = os.getenv("AGENTCORE_MEMORY_ID")

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
    # ✅ sessionId は Memory側制約が厳しいので、英数字/ハイフン/アンスコのみで安全な形式にしておく
    # uuid4() は先頭英数字で hyphen あり → OK
    st.session_state.runtime_session_id = str(uuid.uuid4())

if "session_id_list" not in st.session_state:
    st.session_state.session_id_list = []

@st.cache_resource
def get_agentcore_client(region: str):
    return boto3.client("bedrock-agentcore", region_name=region)

agentcore = get_agentcore_client(REGION)

######################################
# ✅ AgentCoreから「セッション一覧」「メッセージ一覧」を取る
######################################
def fetch_session_id_list(actor_id: str) -> list[str]:
    """
    AgentCore側entrypointが action=get_session_id_list を処理して返す前提
    戻り: [{"sessionId": "..."}] or ["..."] のどちらでも吸収
    """
    resp = agentcore.invoke_agent_runtime(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        runtimeSessionId=st.session_state.runtime_session_id,
        payload=json.dumps(
            {
                "action": "get_session_id_list",
                "memory_id": MEMORY_ID,
                "actor_id": actor_id,
            }
        ).encode("utf-8"),
        qualifier="DEFAULT",
    )

    # data: ... のストリームから最終JSONを拾う（1回返し想定）
    last = None
    for line in resp["response"].iter_lines(chunk_size=10):
        if not line:
            continue
        s = line.decode("utf-8")
        if not s.startswith("data: "):
            continue
        data = s[6:]
        if data.startswith('"') or data.startswith("'"):
            continue
        last = json.loads(data)

    if last is None:
        return []

    # ["id", ...] 形式
    if isinstance(last, list) and (len(last) == 0 or isinstance(last[0], str)):
        return last

    # [{"sessionId":"..."}, ...] 形式
    if isinstance(last, list) and len(last) > 0 and isinstance(last[0], dict):
        return [x.get("sessionId") for x in last if x.get("sessionId")]

    return []

def fetch_message_list(actor_id: str, session_id: str) -> list[dict]:
    """
    AgentCore側entrypointが action=get_message_list を処理して返す前提
    戻り: [{"role":"user","content":[{"text":"..."}]}, ...] を想定
    """
    resp = agentcore.invoke_agent_runtime(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        runtimeSessionId=session_id,  # ←ここは「見たいセッション」に合わせる
        payload=json.dumps(
            {
                "action": "get_message_list",
                "memory_id": MEMORY_ID,
                "actor_id": actor_id,
                "session_id": session_id,
            }
        ).encode("utf-8"),
        qualifier="DEFAULT",
    )

    last = None
    for line in resp["response"].iter_lines(chunk_size=10):
        if not line:
            continue
        s = line.decode("utf-8")
        if not s.startswith("data: "):
            continue
        data = s[6:]
        if data.startswith('"') or data.startswith("'"):
            continue
        last = json.loads(data)

    if isinstance(last, list):
        return last
    return []

def set_session_id(session_id: str, actor_id: str):
    """
    サイドバーのボタン押下時：
    - セッションを切り替え
    - そのセッションの過去メッセージを復元
    """
    st.session_state.runtime_session_id = session_id
    st.session_state.messages = []  # 画面用の表示バッファもクリア

    # Memoryから復元
    messages = fetch_message_list(actor_id, session_id)
    # UI描画用に平坦化（contentがlistで来る想定）
    restored = []
    for m in messages:
        role = (m.get("role") or "").lower()
        if role not in ("user", "assistant"):
            continue
        content = m.get("content", "")
        # content=[{"text": "..."}] 形式を想定
        if isinstance(content, list):
            text = "".join([c.get("text", "") for c in content if isinstance(c, dict)])
        elif isinstance(content, str):
            text = content
        else:
            text = ""
        restored.append({"role": role, "content": text})

    st.session_state.messages = restored
    st.rerun()

def refresh_session_list(actor_id: str):
    new_list = fetch_session_id_list(actor_id)  # AgentCoreへ問い合わせ
    current = st.session_state.runtime_session_id

    # ✅ 今のセッションがMemory側にまだ出てこない場合でも、UIから消えないように保持
    if current and current not in new_list:
        new_list.insert(0, current)

    st.session_state.session_id_list = new_list

######################################
# UI
######################################
st.title("インフルエンサー検索エージェント")
st.write("Youtube, Instagramのインフルエンサーの情報を収集します！")
st.write("「あなたは何ができますか？」と聞いてみてください。")

actor_id = get_actor_id_from_auth0()

# ✅ サイドバー：過去セッション一覧＋切替
with st.sidebar:
    st.caption("セッション管理")

    # 初期化（ここで runtime_session_id を作るのは既に上でやっている前提）
    if "session_id_list" not in st.session_state:
        st.session_state.session_id_list = []
    if "session_list_loaded" not in st.session_state:
        st.session_state.session_list_loaded = False

    def refresh_session_list(actor_id: str):
        new_list = fetch_session_id_list(actor_id)
        current = st.session_state.runtime_session_id
        if current and current not in new_list:
            new_list.insert(0, current)
        st.session_state.session_id_list = new_list

    # ✅ 初回ロードだけ自動取得
    if not st.session_state.session_list_loaded:
        refresh_session_list(actor_id)
        st.session_state.session_list_loaded = True

    st.text_input("Current Session ID", value=st.session_state.runtime_session_id, disabled=True)

    if st.button("🔄 セッション一覧を更新"):
        refresh_session_list(actor_id)
        st.rerun()

    if st.button("🆕 new thread", type="primary"):
        new_id = str(uuid.uuid4())
        st.session_state.runtime_session_id = new_id
        st.session_state.messages = []
        # UI上すぐ出す（Memoryに書かれる前でも見える）
        if new_id not in st.session_state.session_id_list:
            st.session_state.session_id_list.insert(0, new_id)
        st.rerun()

    st.divider()
    st.caption("過去セッション")
    for sid in st.session_state.session_id_list:
        st.button(
            sid,
            key=f"sid_{sid}",  # ✅ key を付けて再描画のブレを抑える
            on_click=set_session_id,
            args=(sid, actor_id),
            use_container_width=True,
        )

# ✅ 過去メッセージを描画
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

######################################
# チャット送信
######################################
if prompt := st.chat_input("メッセージを入力してね"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        payload = json.dumps(
            {
                "prompt": prompt,
                "actor_id": actor_id,
                "session_id": st.session_state.runtime_session_id,  # ← Memoryのsessionにも使う
            }
        )

        response = agentcore.invoke_agent_runtime(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            runtimeSessionId=st.session_state.runtime_session_id,  # AgentCoreの会話継続用
            payload=payload.encode("utf-8"),
            qualifier="DEFAULT",
        )

        container = st.container()
        text_holder = container.empty()
        buffer = ""

        for line in response["response"].iter_lines():
            if line and line.decode("utf-8").startswith("data: "):
                data = line.decode("utf-8")[6:]

                if data.startswith('"') or data.startswith("'"):
                    continue

                event = json.loads(data)

                # ツール利用検出
                if "event" in event and "contentBlockStart" in event["event"]:
                    if "toolUse" in event["event"]["contentBlockStart"].get("start", {}):
                        if buffer:
                            text_holder.markdown(buffer)
                            buffer = ""
                        tool_name = event["event"]["contentBlockStart"]["start"]["toolUse"].get("name", "unknown")
                        container.info(f"🔍 {tool_name} ツールを利用しています")
                        text_holder = container.empty()

                # テキスト検出
                if "data" in event and isinstance(event["data"], str):
                    buffer += event["data"]
                    text_holder.markdown(buffer)
                elif "event" in event and "contentBlockDelta" in event["event"]:
                    buffer += event["event"]["contentBlockDelta"]["delta"].get("text", "")
                    text_holder.markdown(buffer)

        text_holder.markdown(buffer)

    st.session_state.messages.append({"role": "assistant", "content": buffer})

    # ✅ セッション一覧に追加（未登録なら）
    if st.session_state.runtime_session_id not in st.session_state.session_id_list:
        st.session_state.session_id_list.insert(0, st.session_state.runtime_session_id)
