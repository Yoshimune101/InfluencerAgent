import os, boto3, json, uuid, re, hashlib
import streamlit as st
from dotenv import load_dotenv

######################################
# ログイン
######################################
# st.user は secrets.toml の auth 設定が前提。未設定なら st.user.is_logged_in が無いことがある。
if not getattr(st.user, "is_logged_in", False):
    st.login("auth0")
    st.stop()

st.success(f"Hello {st.user.name}")

######################################
# 環境変数と認証の設定
######################################
load_dotenv()

REGION = os.getenv("AWS_REGION") or "us-west-2"
AGENT_RUNTIME_ARN = os.getenv("AGENT_RUNTIME_ARN")
MEMORY_ID = os.getenv("MEMORY_ID")

# NOTE: Streamlit Cloud / ECS ではアクセスキー直指定より IAM Role 推奨
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

@st.cache_resource
def get_agentcore_client(region: str):
    return boto3.client("bedrock-agentcore", region_name=region)

agentcore = get_agentcore_client(REGION)

######################################
# actor_id / thread_id の正規化
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

ACTOR_ID_ALLOWED = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9\-_\/]*(?::[a-zA-Z0-9\-_\/]+)*[a-zA-Z0-9\-_\/]*$"
)
THREAD_ID_ALLOWED = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9\-_]*$")

def normalize_actor_id(raw: str) -> str:
    if not raw:
        raw = "anonymous"
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
    """
    if not raw:
        raw = "default"
    safe = re.sub(r"[^a-zA-Z0-9\-_]", "_", raw)
    if not re.match(r"^[a-zA-Z0-9]", safe):
        safe = "t_" + safe
    candidate = f"sess_{safe}"
    if THREAD_ID_ALLOWED.match(candidate):
        return candidate
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"sess_{digest}"

######################################
# ✅ セッション保持（表示用 messages / 実行用 runtime_session_id / Memory用 thread_id）
######################################
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"/"assistant","content":"..."}]

if "runtime_session_id" not in st.session_state:
    # AgentCore invoke の会話継続用（Streamlitブラウザセッション中は固定でもOK）
    st.session_state.runtime_session_id = str(uuid.uuid4())

if "thread_id" not in st.session_state:
    # Memory側の「スレッドID」（あなたのバックエンド仕様に合わせて session_id として渡す想定）
    st.session_state.thread_id = normalize_thread_id(st.session_state.runtime_session_id)

if "loaded_thread_id" not in st.session_state:
    # 「今表示している thread の履歴をロード済みか」を判定するためのフラグ
    st.session_state.loaded_thread_id = None

def start_new_thread():
    """新規スレッドを開始（表示履歴もクリア）"""
    st.session_state.messages = []
    st.session_state.runtime_session_id = str(uuid.uuid4())
    st.session_state.thread_id = normalize_thread_id(st.session_state.runtime_session_id)
    st.session_state.loaded_thread_id = None
    st.rerun()

def switch_thread(thread_id: str):
    """過去スレッドへ切り替え（表示履歴をクリアして、次の描画でロードさせる）"""
    st.session_state.messages = []
    st.session_state.thread_id = thread_id
    # runtime_session_id は「推論継続」用途。過去スレッドを続けたいなら thread_id 由来に寄せる
    st.session_state.runtime_session_id = thread_id
    st.session_state.loaded_thread_id = None
    st.rerun()

######################################
# UI
######################################
st.title("インフルエンサー検索エージェント")
st.write("Youtube, Instagramのインフルエンサーの情報を収集します！")
st.write("「あなたは何ができますか？」と聞いてみてください。")

# ✅ 先に表示履歴を描画（rerunしても消えない）
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

######################################
# 前提チェック
######################################
if not AGENT_RUNTIME_ARN or not MEMORY_ID:
    st.error("環境変数 AGENT_RUNTIME_ARN と MEMORY_ID が未設定です。")
    st.stop()

actor_id = normalize_actor_id(get_actor_id_from_auth0())

######################################
# 1) 初回だけ：Memoryから履歴ロード（thread_id単位）
######################################
def invoke_json(action: str, payload_dict: dict):
    """
    invoke_agent_runtime を JSON/bytes 前提で統一して呼ぶヘルパー
    """
    body = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")
    return agentcore.invoke_agent_runtime(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        runtimeSessionId=st.session_state.runtime_session_id,
        payload=body,
        qualifier="DEFAULT",
    )

if st.session_state.loaded_thread_id != st.session_state.thread_id:
    # 履歴ロード（あなたのバックエンドが "get_message_list" を実装している前提）
    try:
        resp = invoke_json(
            "get_message_list",
            {
                "action": "get_message_list",
                "memory_id": MEMORY_ID,
                "user_id": actor_id,
                "session_id": st.session_state.thread_id,
            },
        )

        loaded_messages = []
        for line in resp["response"].iter_lines():
            if not line:
                continue
            s = line.decode("utf-8")
            if not s.startswith("data: "):
                continue
            data = s[6:]

            # 文字列だけの keep-alive を無視
            if data.startswith('"') or data.startswith("'"):
                continue

            obj = json.loads(data)

            # 期待形式:
            # [{"role":"USER"/"ASSISTANT","content":[{"text":"..."}]} ...]
            # or [{"role":"user","content":"..."} ...] など混在しても吸収
            if isinstance(obj, list):
                for item in obj:
                    role = str(item.get("role", "")).lower()
                    if role in ("user", "USER"):
                        role = "user"
                    elif role in ("assistant", "ASSISTANT"):
                        role = "assistant"

                    content = item.get("content")
                    if isinstance(content, list):
                        # [{"text": "..."}] を連結
                        text = "".join([c.get("text", "") for c in content if isinstance(c, dict)])
                    else:
                        text = str(content or "")

                    if role in ("user", "assistant") and text:
                        loaded_messages.append({"role": role, "content": text})

        st.session_state.messages = loaded_messages
        st.session_state.loaded_thread_id = st.session_state.thread_id
        st.rerun()

    except Exception as e:
        st.warning(f"履歴ロードに失敗しました: {e}")
        st.session_state.loaded_thread_id = st.session_state.thread_id  # 無限リトライ防止

######################################
# 2) チャット入力 → AgentCore invoke（ストリーミング描画）
######################################
if prompt := st.chat_input("メッセージを入力してね"):
    # 表示用に先に積む
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        req_payload = {
            "prompt": prompt,
            "actor_id": actor_id,
            "session_id": st.session_state.thread_id,
        }

        resp = agentcore.invoke_agent_runtime(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            runtimeSessionId=st.session_state.runtime_session_id,
            payload=json.dumps(req_payload, ensure_ascii=False).encode("utf-8"),
            qualifier="DEFAULT",
        )

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
            # keep-alive
            if data.startswith('"') or data.startswith("'"):
                continue

            event = json.loads(data)

            # ツール利用の表示（任意）
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

            # テキスト delta の吸収（Event形式の揺れを吸収）
            if "data" in event and isinstance(event["data"], str):
                buffer += event["data"]
                text_holder.markdown(buffer)
            elif "event" in event and "contentBlockDelta" in event["event"]:
                buffer += event["event"]["contentBlockDelta"]["delta"].get("text", "")
                text_holder.markdown(buffer)

        text_holder.markdown(buffer)

    # 表示用履歴に積む
    st.session_state.messages.append({"role": "assistant", "content": buffer})

######################################
# 3) サイドバー：スレッド管理
######################################
with st.sidebar:
    st.markdown("### Thread")
    st.text_input("Current thread_id", value=st.session_state.thread_id, disabled=True)
    st.text_input("runtime_session_id", value=st.session_state.runtime_session_id, disabled=True)

    st.button("new thread", on_click=start_new_thread, type="primary")

    st.markdown("---")
    st.markdown("### Past threads")

    # セッション一覧（あなたのバックエンドが "get_session_id_list" を実装している前提）
    if "thread_id_list" not in st.session_state:
        st.session_state.thread_id_list = []

    if st.button("refresh threads"):
        try:
            resp = invoke_json(
                "get_session_id_list",
                {
                    "action": "get_session_id_list",
                    "memory_id": MEMORY_ID,
                    "user_id": actor_id,
                },
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
                # 期待: [{"sessionId": "..."} , ...]
                if isinstance(obj, list):
                    ids.extend([x.get("sessionId") for x in obj if isinstance(x, dict) and x.get("sessionId")])

            # 重複除去＆自己防衛
            ids = [i for i in ids if isinstance(i, str) and i]
            st.session_state.thread_id_list = sorted(list(set(ids)))

        except Exception as e:
            st.warning(f"スレッド一覧の取得に失敗しました: {e}")

    if st.session_state.thread_id_list:
        for tid in st.session_state.thread_id_list:
            st.button(
                tid,
                on_click=switch_thread,
                args=(tid,),
            )
