# 必要なライブラリをインポート
import os, boto3, json
import streamlit as st

# secrets.toml が読めてないと st.user に属性が生えないので、先に検知して落とす
if "auth" not in st.secrets:
    st.error("`.streamlit/secrets.toml` が読み込めていません（[auth] が見つからない）。配置場所とファイル名を確認してください。")
    st.stop()

# 旧APIは捨てて st.user を使う
if not getattr(st.user, "is_logged_in", False):
    st.login("auth0") 
    st.stop()

st.success(f"Hello, {st.user.name}!")
if st.button("Log out"):
    st.logout()


REGION = os.getenv("AWS_REGION")
agent_runtime_arn = os.getenv("AGENT_RUNTIME_ARN")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# タイトルを描画
st.title("インフルエンサー検索エージェント")
st.write("Youtube APIを使用してインフルエンサーの情報収集します！")
st.write("「登録者数〇万人から〇万人までの〇〇系Youtuberを検索して」といったリクエストに対応します。")
st.write("例：ビールを売るために、登録者数1万人から5万人までのグルメ系Youtuberを教えて")

# チャットボックスを描画
if prompt := st.chat_input("メッセージを入力してね"):
    # ユーザーのプロンプトを表示
    with st.chat_message("user"):
        st.markdown(prompt)

    # エージェントの回答を表示
    with st.chat_message("assistant"):
        # AgentCoreランタイムを呼び出し
        agentcore = boto3.client('bedrock-agentcore', region_name=REGION)
        payload = json.dumps({"prompt": prompt})
        response = agentcore.invoke_agent_runtime(
            agentRuntimeArn=agent_runtime_arn,
            payload=payload.encode()
        )

        ### ここから下はストリーミングレスポンスの処理 ------------------------------------------
        container = st.container()
        text_holder = container.empty()
        buffer = ""

        # レスポンスを1行ずつチェック
        for line in response["response"].iter_lines():
            if line and line.decode("utf-8").startswith("data: "):
                data = line.decode("utf-8")[6:]

                # 文字列コンテンツの場合は無視
                if data.startswith('"') or data.startswith("'"):
                    continue

                # 読み込んだ行をJSONに変換
                event = json.loads(data)

                # ツール利用を検出
                if "event" in event and "contentBlockStart" in event["event"]:
                    if "toolUse" in event["event"]["contentBlockStart"].get("start", {}):
                        # 現在のテキストを確定
                        if buffer:
                            text_holder.markdown(buffer)
                            buffer = ""

                        # ツールステータスを表示
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

        # 最後に残ったテキストを表示
        text_holder.markdown(buffer)
        ### ------------------------------------------------------------------------------

