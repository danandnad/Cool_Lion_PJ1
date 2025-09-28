# app.py
import streamlit as st
from google import genai
from google.genai import types
import time
import traceback

st.set_page_config(page_title="Gemini Chat (gemini-2.5-flash)", layout="wide")

# --- Helpers ---
def init_client():
    # Read API key from Streamlit secrets, fallback to environment variable
    api_key = st.secrets.get("GEMINI_API_KEY") if st.secrets else None
    if not api_key:
        api_key = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else None  # defensive
    if not api_key:
        st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it to .streamlit/secrets.toml")
        raise RuntimeError("Missing GEMINI_API_KEY")
    # The genai client will accept api_key param
    client = genai.Client(api_key=api_key)
    return client

def append_history(role, text):
    st.session_state.history.append({"role": role, "text": text})

def render_history():
    # Left column: conversation list
    for i, msg in enumerate(st.session_state.history):
        role = msg["role"]
        text = msg["text"]
        # Simple styling
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Assistant:** {text}")

def safe_get_text_from_resp(resp):
    """
    Try many plausible attributes from SDK objects / dicts to extract the partial/full text.
    This is defensive: different SDK versions/streaming events expose text differently.
    """
    try:
        # if it's a string
        if isinstance(resp, str):
            return resp
        # if it's a response object with .text
        if hasattr(resp, "text"):
            return getattr(resp, "text")
        # some streaming chunks may have 'delta' or 'candidates'
        if isinstance(resp, dict):
            # REST-like structure
            # candidates -> content -> parts -> [ { "text": ... }]
            cands = resp.get("candidates")
            if cands:
                try:
                    return cands[0]["content"]["parts"][0]["text"]
                except Exception:
                    pass
            # fallback fields
            return resp.get("text") or resp.get("delta") or ""
        # generic object with nested attributes
        if hasattr(resp, "candidates") and resp.candidates:
            try:
                return resp.candidates[0].content.parts[0].text
            except Exception:
                pass
        return ""
    except Exception:
        return ""

# --- App UI ---
st.title("📬 Gemini Chat — gemini-2.5-flash (Streamlit)")
st.caption("Uses Streamlit secrets for the GEMINI_API_KEY. Tries streaming first, falls back to normal generate if unavailable. See docs: Quickstart & Streaming. ")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "assistant", "text": "안녕하세요! 무엇을 도와드릴까요? (Gemini 2.5 Flash 사용)"},
    ]

# Sidebar: settings
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", value="gemini-2.5-flash")
    max_output_tokens = st.number_input("Max output tokens", min_value=64, max_value=8192, value=1024, step=64)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.2, 0.05)
    use_streaming_toggle = st.checkbox("Try streaming (if SDK supports it)", value=True)
    st.markdown("---")
    st.markdown("**Notes**")
    st.markdown("- API key via Streamlit secrets: `GEMINI_API_KEY`.")
    st.markdown("- If streaming method is unsupported by your sdk version, the app will fall back to synchronous generation.")

# Main layout: history + input
history_col, input_col = st.columns([3, 1])

with history_col:
    st.subheader("Conversation")
    render_history()

with input_col:
    st.subheader("Send")
    user_input = st.text_area("Your message", value="", height=200)
    send_btn = st.button("Send")

# On send: call Gemini
if send_btn and user_input.strip():
    append_history("user", user_input)
    client = None
    placeholder = history_col.empty()
    try:
        client = init_client()
    except Exception as e:
        st.error("Failed to initialize Gemini client: " + str(e))
        st.stop()

    # Show the conversation up to now
    placeholder.markdown("### Conversation (updating...)")
    render_history()

    # We'll put the assistant partial text in a placeholder we can update
    assistant_slot = history_col.empty()
    assistant_slot.markdown("**Assistant:** _thinking..._")

    # Attempt streaming if available
    streamed_text = ""
    try:
        # Try to find a streaming method on the client models object (dynamic)
        models_obj = getattr(client, "models", None) or getattr(client, "Models", None)
        stream_fn = None
        if models_obj:
            # possible method names across SDK versions:
            for name in ("generate_content_stream", "stream_generate_content", "generate_stream", "generate_content_stream_with_options"):
                if hasattr(models_obj, name):
                    stream_fn = getattr(models_obj, name)
                    break

        if use_streaming_toggle and stream_fn:
            # Call stream function and iterate chunks
            # The streaming API varies by SDK; we try to support common patterns
            assistant_slot.markdown("**Assistant:** ")  # clear initial text
            # Prepare contents per SDK expectations:
            # Some SDKs accept simple string `contents="text"`, others expect `contents=[types.Content(...)]`
            contents_single = user_input
            # Try common streaming call signatures
            try:
                # 1) generator style: for chunk in stream_fn(model=model, contents=contents_single, config=...):
                stream_iter = stream_fn(model=model,
                                       contents=contents_single,
                                       config=types.GenerateContentConfig(
                                           max_output_tokens=int(max_output_tokens),
                                           temperature=float(temperature)
                                       ))
            except TypeError:
                # 2) maybe expects positional args or different arg names
                stream_iter = stream_fn(model, contents_single)
            except Exception as e:
                raise

            # Iterate and update UI live
            for chunk in stream_iter:
                # chunk might be str, object, dict, etc.
                delta = safe_get_text_from_resp(chunk) or ""
                if delta:
                    streamed_text += delta
                    assistant_slot.markdown(f"**Assistant:** {streamed_text}")
            # final
            append_history("assistant", streamed_text.strip())
        else:
            # Fallback: synchronous generate_content
            assistant_slot.markdown("**Assistant:** _waiting for response (non-stream)_")
            response = client.models.generate_content(
                model=model,
                contents=user_input,
                config=types.GenerateContentConfig(
                    max_output_tokens=int(max_output_tokens),
                    temperature=float(temperature)
                ),
            )
            text = safe_get_text_from_resp(response) or getattr(response, "text", "") or ""
            append_history("assistant", text.strip())
            assistant_slot.markdown(f"**Assistant:** {text}")

    except Exception as e:
        # show error and traceback for debugging
        err_msg = f"Error during generation: {e}\n\n{traceback.format_exc()}"
        assistant_slot.markdown(f"**Assistant:** _failed to generate response._\n\n```\n{err_msg}\n```")
        append_history("assistant", "[Generation failed — see error above]")

    # Re-render final history
    placeholder.markdown("### Conversation (final)")
    render_history()
    # Clear input area (client-side text area cannot be directly cleared; just show message)
    st.success("Response received (or failed). Scroll conversation on the left.")
