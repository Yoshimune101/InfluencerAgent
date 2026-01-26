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
# ✅ 履歴要素の正規化（JSON丸出し＆日本語化け対策）
######################################
def normalize_message_item(item):
    """
    AgentCore側の返却揺れを吸収し、必ず {role, content} (string) に正規化する。
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

    # contentが list の場合
    if isinstance(content, list):
        parts = []
        for c in content:
            if not isinstance(c, dict):
                parts.append(str(c))
                continue

            # text
            if "text" in c and isinstance(c["text"], str):
                parts.append(c["text"])
                continue

            # toolResult (Strandsの会話ログでよく混ざる)
            tr = c.get("toolResult")
            if isinstance(tr, dict):
                # toolResult.content: [{"text":"..."}] を連結
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
                # 画面にJSONを出さないよう、テキストだけに落とす
                parts.append(f"【ツール結果: {tool_name} / {status}】\n{tr_text}")
                continue

            # その他 dict は無視（JSON露出防止）
            # parts.append(str(c))  ← これをやるとJSONが出るのでやらない

        text = "".join(parts).strip()
        return {"role": role, "content": text}

    # contentが string / dict の場合
    if isinstance(content, str):
        return {"role": role, "content": content}

    # dict等はJSON露出防止で空にする
    return {"role": role, "content": ""}

######################################
# UI
######################################
st.title("インフルエンサー検索エージェント")
st.write("Youtube, Instagramのインフルエンサーの情報を収集します！")
st.write("「あなたは何ができますか？」と聞いてみてください。")

# ✅ 先に表示履歴を描画（rerunしても消えない）
for raw in st.session_state.messages:
    m = normalize_message_item(raw)
    if not m["content"]:
        continue
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
                if isinstance(obj.get("messages"), list):
                    for item in obj["messages"]:
                        m = normalize_message_item(item)
                        if m["content"]:
                            loaded_messages.append(m)
                else:
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
