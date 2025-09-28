import streamlit as st
import os
import datetime
import json
import traceback
from google import genai

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
def get_api_key():
    """API 키를 여러 소스에서 가져오기 시도"""
    # 1. 환경 변수에서 확인
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # 2. streamlit secrets에서 확인
    if not api_key and hasattr(st, 'secrets'):
        api_key = st.secrets.get("GEMINI_API_KEY")
    
    return api_key

def init_gemini_model(model_name):
    """Gemini 모델 초기화"""
    api_key = get_api_key()
    
    if not api_key:
        st.error("⚠️ API 키가 설정되지 않았습니다. 환경변수 GEMINI_API_KEY 또는 secrets.toml 파일을 확인해주세요.")
        st.info("API 키 설정 방법: (1) 환경변수 GEMINI_API_KEY='키값' 설정 또는 (2) ~/.streamlit/secrets.toml 파일에 GEMINI_API_KEY = '키값' 추가")
        st.stop()
    
    # genai 설정
    genai.configure(api_key=api_key)
    
    # 모델 반환
    return genai.GenerativeModel(model_name)

def append_history(role, text):
    """채팅 기록에 메시지 추가"""
    st.session_state.history.append({
        "role": role, 
        "text": text, 
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def render_history():
    """채팅 기록 렌더링"""
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
    """응답에서 안전하게 텍스트 추출"""
    try:
        if isinstance(resp, str):
            return resp
        if hasattr(resp, "text"):
            return resp.text
        if isinstance(resp, dict):
            return resp.get("text", "")
        if hasattr(resp, "candidates") and resp.candidates:
            return resp.candidates[0].content.parts[0].text
        return str(resp)
    except Exception as e:
        return f"응답 파싱 오류: {str(e)}"

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

    # API 키 상태 표시
    api_key = get_api_key()
    if api_key:
        st.success("✅ API 키가 설정되어 있습니다")
    else:
        st.error("⚠️ API 키가 설정되지 않았습니다")
    
    st.markdown("---")
    st.subheader("📥 대화 내보내기")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 TXT로 저장"):
            text_data = "\n\n".join([f"[{m['time']}] {m['role'].capitalize()}: {m['text']}" for m in st.session_state.history])
            st.download_button("TXT 다운로드", text_data, file_name="chat_history.txt")
    
    with col2:
        if st.button("💾 JSON으로 저장"):
            json_data = json.dumps(st.session_state.history, indent=2, ensure_ascii=False)
            st.download_button("JSON 다운로드", json_data, file_name="chat_history.json")

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
