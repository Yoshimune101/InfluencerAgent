import os
import json
import time
import uuid
import re
import hashlib
from typing import Any, Dict, List, Optional

import boto3
import streamlit as st
from dotenv import load_dotenv

######################################
# ログイン（Auth0）
######################################
if not getattr(st.user, "is_logged_in", False):
    st.login("auth0")
    st.stop()

st.success(f"Hello {st.user.name}")

######################################
# 環境変数
######################################
load_dotenv()
REGION = os.getenv("AWS_REGION") or "us-west-2"
AGENT_RUNTIME_ARN = os.getenv("AGENT_RUNTIME_ARN")
MEMORY_ID = os.getenv("MEMORY_ID")

if not AGENT_RUNTIME_ARN or not MEMORY_ID:
    st.error("環境変数 AGENT_RUNTIME_ARN と MEMORY_ID が未設定です。")
    st.stop()

######################################
# AgentCore Client
######################################
agent_core_client = boto3.client("bedrock-agentcore", region_name=REGION)

######################################
# actor_id 正規化（AgentCore Memory制約用）
######################################
ACTOR_ID_ALLOWED = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9\-_\/]*(?::[a-zA-Z0-9\-_\/]+)*[a-zA-Z0-9\-_\/]*$"
)


def get_actor_id_from_auth0() -> str:
    u = st.user
    return (
        str(getattr(u, "sub", "")).strip()
        or str(getattr(u, "id", "")).strip()
        or str(getattr(u, "email", "")).strip()
        or str(getattr(u, "name", "")).strip()
        or "anonymous"
    )


def normalize_actor_id(raw: str) -> str:
    if not raw:
        raw = "anonymous"
    raw = str(raw).strip()

    if raw.startswith("actor:"):
        raw = raw[len("actor:") :]

    safe = re.sub(r"[^a-zA-Z0-9\-_\/:]", "_", raw)
    if not re.match(r"^[a-zA-Z0-9]", safe):
        safe = "a_" + safe
    safe = safe.rstrip(":")

    candidate = f"actor:{safe}"
    if ACTOR_ID_ALLOWED.match(candidate):
        return candidate

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"actor:{digest}"


actor_id = normalize_actor_id(get_actor_id_from_auth0())

######################################
# Session ID（33文字以上）
######################################
def generate_session_id() -> str:
    return str(int(time.time())) + "_" + str(uuid.uuid4()).replace("-", "")


if "session_id" not in st.session_state:
    st.session_state["session_id"] = generate_session_id()

######################################
# SSE / AgentCoreレスポンスの吸収
######################################
def _sse_json(line: bytes) -> Optional[Any]:
    if not line:
        return None
    s = line.decode("utf-8")
    if not s.startswith("data: "):
        return None
    payload = s[6:]
    try:
        return json.loads(payload)
    except Exception:
        return None


def unwrap_messages(obj: Any) -> List[Dict[str, Any]]:
    """
    messages=[{"role": "...", "content":[{"text":"..."}]}] の形に寄せる

    想定される返却:
    - [{"message": {...}, "created_at":...}, ...]
    - {"messages":[...]}
    - {"message": {...}}
    - [{"role":"user","content":[{"text":"..."}]}, ...]
    """
    if isinstance(obj, list):
        out: List[Dict[str, Any]] = []
        for x in obj:
            if isinstance(x, dict) and isinstance(x.get("message"), dict):
                out.append(x["message"])
            elif isinstance(x, dict) and ("role" in x and "content" in x):
                out.append(x)
        return out

    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        return unwrap_messages(obj["messages"])

    if isinstance(obj, dict) and isinstance(obj.get("message"), dict):
        return [obj["message"]]

    return []


def extract_messages_from_event_obj(obj: Any) -> List[Dict[str, Any]]:
    """
    SSEの1イベントから message/messages を取り出す
    - {"event": {...}} 包装を剥がす
    - event.payload が JSON文字列のパターンも吸収
    """
    if not isinstance(obj, dict):
        return []

    ev = obj.get("event", obj)

    # payload に本体が入るケース
    payload = ev.get("payload")
    if isinstance(payload, str):
        try:
            payload_obj = json.loads(payload)
        except Exception:
            payload_obj = None
        if payload_obj is not None:
            msgs = unwrap_messages(payload_obj)
            if msgs:
                return msgs

    # 直接 message/messages が入るケース
    msgs = unwrap_messages(ev)
    if msgs:
        return msgs

    # さらに一段深いケースを拾う（保険）
    if isinstance(ev, dict):
        for k in ("data", "result", "response"):
            if k in ev:
                msgs = unwrap_messages(ev.get(k))
                if msgs:
                    return msgs

    return []


######################################
# 表示系：toolResult / JSONテキストの整形
######################################
def try_parse_jsonish_text(s: str) -> Optional[Any]:
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None

    # 素直に
    try:
        return json.loads(t)
    except Exception:
        pass

    # よくある救済: 余計なダブルクォートで包まれている
    # 例: "\"{\\\"ok\\\": true}\"" や "\"{...}\""
    if (t.startswith('"{') and t.endswith('}"')) or (t.startswith('"[') and t.endswith(']"')):
        t2 = t[1:-1].replace('\\"', '"')
        try:
            return json.loads(t2)
        except Exception:
            pass

    return None


def normalize_display_text(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        return str(s)

    raw = s.strip()
    if not raw:
        return ""

    # JSONっぽくないならそのまま
    if not (raw.startswith("{") or raw.startswith("[")):
        return s

    # JSONなら「message wrapper」を剥がして本文だけ抜く
    obj = try_parse_jsonish_text(raw)
    if obj is None:
        return s

    msgs = unwrap_messages(obj)
    if not msgs:
        if isinstance(obj, dict) and isinstance(obj.get("text"), str):
            return obj["text"]
        return json.dumps(obj, ensure_ascii=False, indent=2)

    texts: List[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        content = m.get("content", [])
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip():
                    texts.append(c["text"])
        elif isinstance(content, str) and content.strip():
            texts.append(content)

    return "\n\n".join(texts) if texts else json.dumps(obj, ensure_ascii=False, indent=2)


def is_tool_only_message(msg: Dict[str, Any]) -> bool:
    content = msg.get("content", [])
    if not isinstance(content, list) or not content:
        return False
    has_tool = any(isinstance(x, dict) and "toolResult" in x for x in content)
    has_text = any(
        isinstance(x, dict) and isinstance(x.get("text"), str) and x.get("text", "").strip()
        for x in content
    )
    return has_tool and not has_text


def render_message_content(content_list: Any) -> None:
    """
    content の中身を人間が読める形で描画する。
    - {"text": "..."} を表示
    - {"toolResult": {...}} は toolResult.content[].text を抽出して
      JSONなら st.json、そうでなければ markdown
    """
    if isinstance(content_list, list):
        for c in content_list:
            if not isinstance(c, dict):
                continue

            # 1) 通常テキスト
            if isinstance(c.get("text"), str):
                txt = c["text"]
                parsed = try_parse_jsonish_text(txt)
                if parsed is not None:
                    st.json(parsed)
                else:
                    st.markdown(normalize_display_text(txt))
                continue

            # 2) toolResult
            tr = c.get("toolResult")
            if isinstance(tr, dict):
                tr_contents = tr.get("content", [])
                if isinstance(tr_contents, list):
                    for tc in tr_contents:
                        if isinstance(tc, dict) and isinstance(tc.get("text"), str):
                            txt = tc["text"]
                            parsed = try_parse_jsonish_text(txt)
                            if parsed is not None:
                                st.json(parsed)
                            else:
                                st.markdown(normalize_display_text(txt))
                continue

            # 3) それ以外
            st.markdown(normalize_display_text(json.dumps(c, ensure_ascii=False)))

    elif isinstance(content_list, str):
        parsed = try_parse_jsonish_text(content_list)
        if parsed is not None:
            st.json(parsed)
        else:
            st.markdown(normalize_display_text(content_list))
    else:
        st.markdown(normalize_display_text(str(content_list)))


######################################
# thread 切り替え
######################################
def set_session_id(session_id: str):
    st.session_state["session_id"] = session_id
    # UI上のメッセージキャッシュを切り替え
    st.session_state.pop("messages", None)


######################################
# UI
######################################
st.title("インフルエンサー検索エージェント")
st.write("Youtube, Instagramのインフルエンサーの情報を収集します！")
st.write("「あなたは何ができますか？」と聞いてみてください。")

######################################
# 初回：messagesをロード（get_message_list）
######################################
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
            },
            ensure_ascii=False,
        ),
        qualifier="DEFAULT",
    )

    latest_msgs: Optional[List[Dict[str, Any]]] = None
    for line in response["response"].iter_lines(chunk_size=10):
        obj = _sse_json(line)
        if not obj:
            continue

        msgs = extract_messages_from_event_obj(obj)
        if msgs:
            latest_msgs = msgs

    if latest_msgs is not None:
        st.session_state["messages"] = latest_msgs


######################################
# 履歴描画
######################################
for msg in st.session_state["messages"]:
    if not isinstance(msg, dict):
        continue

    # toolResult-only を UI に出さない（ノイズ対策）
    if is_tool_only_message(msg):
        continue

    role = str(msg.get("role", "assistant")).lower()
    if role not in ("user", "assistant"):
        role = "assistant"

    with st.chat_message(role):
        render_message_content(msg.get("content", []))


######################################
# ストリーミング（assistantのdeltaだけ表示）
######################################
def streaming_text_deltas(response):
    for line in response["response"].iter_lines(chunk_size=10):
        obj = _sse_json(line)
        if not obj or not isinstance(obj, dict):
            continue

        text = (
            obj.get("event", {})
            .get("contentBlockDelta", {})
            .get("delta", {})
            .get("text", "")
        )
        if isinstance(text, str) and text:
            yield text


######################################
# チャット入力 → invoke → ストリーミング
######################################
prompt = st.chat_input()
if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner():
        # ★重要：prompt を payload に入れる（runtime側の期待キーに合わせる）
        response = agent_core_client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            runtimeSessionId=st.session_state["session_id"],
            payload=json.dumps(
                {
                    "prompt": prompt,  # ← ここが抜けてると挙動が崩れる
                    "memory_id": MEMORY_ID,
                    "user_id": actor_id,
                    "session_id": st.session_state["session_id"],
                },
                ensure_ascii=False,
            ),
            qualifier="DEFAULT",
        )

        with st.chat_message("assistant"):
            assistant_text = st.write_stream(streaming_text_deltas(response))

    # UI用に自前でも履歴を追記（get_message_list を正にするまでの暫定）
    st.session_state["messages"].append({"role": "user", "content": [{"text": prompt}]})
    st.session_state["messages"].append({"role": "assistant", "content": [{"text": assistant_text}]})

    # session_id_list に自分を追加
    if "session_id_list" not in st.session_state:
        st.session_state["session_id_list"] = []
    if st.session_state["session_id"] not in st.session_state["session_id_list"]:
        st.session_state["session_id_list"].append(st.session_state["session_id"])


######################################
# サイドバー：スレッド管理
######################################
with st.sidebar:
    st.text_input(label="Session ID", value=st.session_state["session_id"], disabled=True)

    st.button(
        "new thread",
        on_click=set_session_id,
        args=[generate_session_id()],
        type="primary",
    )

    # session_id_list 初回取得
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
                    },
                    ensure_ascii=False,
                ),
                qualifier="DEFAULT",
            )

            session_ids: List[str] = []
            for line in response["response"].iter_lines(chunk_size=10):
                obj = _sse_json(line)
                if not obj:
                    continue

                # event.payload に list が入る / 直接 list が返る、両方吸収
                ev = obj.get("event", obj) if isinstance(obj, dict) else obj
                payload = ev.get("payload") if isinstance(ev, dict) else None

                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        payload = None

                candidate = payload if payload is not None else ev
                if isinstance(candidate, list):
                    for x in candidate:
                        if isinstance(x, dict) and x.get("sessionId"):
                            session_ids.append(x["sessionId"])

            st.session_state["session_id_list"] = session_ids

    # セッション一覧
    for sid in st.session_state.get("session_id_list", []):
        st.button(sid, on_click=set_session_id, args=[sid])
