import streamlit as st
from google import genai
import traceback
import datetime
import json
import os

# --- API 키 설정 (여러 방법으로 시도) ---
def get_api_key():
    # 1. 직접 환경변수에서 가져오기
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # 2. st.secrets에서 가져오기
    if not api_key and hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    
    return api_key

# API 키 설정 (초기화 단계에서는 확인만)
api_key = get_api_key()

# --- Page Config ---
st.set_page_config(page_title="Gemini Chatbot", layout="wide")

# --- Custom CSS (Light/Dark Mode 지원) ---
def apply_custom_css(dark_mode=False):
    if dark_mode:
        user_bg = "#2E7D32"   # 진한 초록
        assistant_bg = "#424242"  # 진한 회색
        text_color = "white"
        container_bg = "#212121"
    else:
        user_bg = "#DCF8C6"   # WhatsApp 스타일 초록
        assistant_bg = "#F1F0F0"
        text_color = "black"
        container_bg = "#fafafa"

    st.markdown(
        f"""
        <style>
        .chat-message {{
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 12px;
            max-width: 80%;
            color: {text_color};
        }}
        .user-message {{
            background-color: {user_bg};
            margin-left: auto;
            margin-right: 0;
        }}
        .assistant-message {{
            background-color: {assistant_bg};
            margin-right: auto;
            margin-left: 0;
        }}
        .chat-container {{
            height: 65vh;
            overflow-y: auto;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 12px;
            background-color: {container_bg};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Helper functions ---
def init_client():
    api_key = get_api_key()
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY가 환경변수나 secrets.toml에 설정되지 않았습니다.")
        st.info("API 키를 설정하는 방법: (1) 환경변수 GEMINI_API_KEY 설정 또는 (2) .streamlit/secrets.toml 파일에 GEMINI_API_KEY = '당신의_API_키' 추가")
        st.stop()
    
    # genai 라이브러리 설정
    genai.configure(api_key=api_key)
    return genai

def append_history(role, text):
    st.session_state.history.append(
        {"role": role, "text": text, "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    )

def render_history():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.history:
        role = msg["role"]
        text = msg["text"]
        time_str = msg["time"]
        css_class = "user-message" if role == "user" else "assistant-message"
        avatar = "🧑" if role == "user" else "🤖"
        st.markdown(
            f'<div class="chat-message {css_class}">'
            f"<b>{avatar} {role.capitalize()}</b> <small>{time_str}</small><br>{text}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

def safe_get_text_from_resp(resp):
    if isinstance(resp, str):
        return resp
    if hasattr(resp, "text"):
        return resp.text
    if isinstance(resp, dict):
        return resp.get("text", "")
    return ""

# --- Initialize state ---
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "assistant", "text": "안녕하세요! Gemini 2.5 Flash 챗봇입니다 😊 무엇을 도와드릴까요?", "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
    ]

# --- Sidebar Settings ---
with st.sidebar:
    st.title("⚙️ 설정")
    model = st.text_input("모델", value="gemini-2.5-flash")
    max_tokens = st.number_input("Max tokens", 128, 8192, 1024, step=64)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.8, 0.05)
    dark_mode = st.toggle("🌗 다크 모드", value=False)

    st.markdown("---")
    st.subheader("📥 대화 내보내기")
    if st.button("💾 Save as TXT"):
        text_data = "\n\n".join([f"[{m['time']}] {m['role'].capitalize()}: {m['text']}" for m in st.session_state.history])
        st.download_button("다운로드 TXT", text_data, file_name="chat_history.txt")
    if st.button("💾 Save as JSON"):
        json_data = json.dumps(st.session_state.history, indent=2, ensure_ascii=False)
        st.download_button("다운로드 JSON", json_data, file_name="chat_history.json")

# Apply CSS
apply_custom_css(dark_mode)

# --- Main Layout ---
st.title("💬 Gemini Chatbot (gemini-2.5-flash)")

# Render chat history
render_history()

# --- Input Area ---
with st.container():
    user_input = st.text_area("메시지를 입력하세요", height=80, key="user_input", placeholder="Shift+Enter 로 줄바꿈")
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        send_btn = st.button("전송 🚀")

# --- On Send ---
if send_btn and user_input.strip():
    append_history("user", user_input)
    client = init_client()

    try:
        # genai 2.0 API 사용
        model_obj = client.GenerativeModel(model)
        response = model_obj.generate_content(
            user_input,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        text = safe_get_text_from_resp(response)
        append_history("assistant", text)
    except Exception as e:
        error_msg = f"❌ 오류 발생: {e}\n\n```\n{traceback.format_exc()}\n```"
        append_history("assistant", error_msg)
