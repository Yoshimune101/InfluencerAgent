import os, boto3, json, uuid, re, hashlib
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

REGION = os.getenv("AWS_REGION") or "us-west-2"
AGENT_RUNTIME_ARN = os.getenv("AGENT_RUNTIME_ARN")
MEMORY_ID = os.getenv("MEMORY_ID")

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
    ✅ FIX: 'actor:' が既に付いている場合は二重付与しない
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
# ✅ セッション保持（表示用 messages / 実行用 runtime_session_id / Memory用 thread_id）
######################################
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"/"assistant","content":"..."}]

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
# invoke helper
######################################
def invoke_json(payload_dict: dict):
    body = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")
    return agentcore.invoke_agent_runtime(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        runtimeSessionId=st.session_state.runtime_session_id,
        payload=body,
        qualifier="DEFAULT",
    )

######################################
# ✅ 履歴要素の正規化（JSON丸出し＆日本語化け対策）
######################################
def normalize_message_item(item):
    """
    AgentCore側が返す形の揺れを吸収する。
    - {"role":"assistant","content":[{"text":"..."}]}
    - {"message":{"role":"assistant","content":[{"text":"..."}]}, ...}
    - content が string の場合も吸収
    """
    if not isinstance(item, dict):
        return {"role": "assistant", "content": str(item)}

    core = item.get("message") if isinstance(item.get("message"), dict) else item

    role = str(core.get("role", "assistant")).lower()
    if role not in ("user", "assistant"):
        role = "assistant"

    content = core.get("content")
    if isinstance(content, list):
        text = "".join(
            (c.get("text", "") if isinstance(c, dict) else str(c))
            for c in content
        )
    else:
        text = str(content or "")

    return {"role": role, "content": text}

######################################
# 1) 初回だけ：Memoryから履歴ロード（thread_id単位）
######################################
if st.session_state.loaded_thread_id != st.session_state.thread_id:
    try:
        resp = invoke_json(
            {
                "action": "get_message_list",
                "memory_id": MEMORY_ID,
                "user_id": actor_id,
                "session_id": st.session_state.thread_id,
            }
        )

        loaded_messages = []

        for line in resp["response"].iter_lines():
            if not line:
                continue
            s = line.decode("utf-8")
            if not s.startswith("data: "):
                continue
            data = s[6:]

            # keep-alive を無視
            if data.startswith('"') or data.startswith("'"):
                continue

            obj = json.loads(data)

            # obj が list / dict どちらでも処理
            if isinstance(obj, list):
                for item in obj:
                    m = normalize_message_item(item)
                    if m["content"]:
                        loaded_messages.append(m)
            elif isinstance(obj, dict):
                m = normalize_message_item(obj)
                if m["content"]:
                    loaded_messages.append(m)

        st.session_state.messages = loaded_messages
        st.session_state.loaded_thread_id = st.session_state.thread_id
        st.rerun()

    except Exception as e:
        st.warning(f"履歴ロードに失敗しました: {e}")
        st.session_state.loaded_thread_id = st.session_state.thread_id

######################################
# 2) チャット入力 → AgentCore invoke（ストリーミング描画）
######################################
if prompt := st.chat_input("メッセージを入力してね"):
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
            if data.startswith('"') or data.startswith("'"):
                continue

            event = json.loads(data)

            # ツール利用表示
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

            # テキスト delta 吸収
            if "data" in event and isinstance(event["data"], str):
                buffer += event["data"]
                text_holder.markdown(buffer)
            elif "event" in event and "contentBlockDelta" in event["event"]:
                buffer += event["event"]["contentBlockDelta"]["delta"].get("text", "")
                text_holder.markdown(buffer)

        text_holder.markdown(buffer)

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

    if "thread_id_list" not in st.session_state:
        st.session_state.thread_id_list = []

    if st.button("refresh threads"):
        try:
            resp = invoke_json(
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
                    ids.extend(
                        [x.get("sessionId") for x in obj if isinstance(x, dict) and x.get("sessionId")]
                    )

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
