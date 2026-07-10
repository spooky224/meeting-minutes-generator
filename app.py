import streamlit as st
from dotenv import load_dotenv
import threading
import tempfile
import time
import os
from src.auth import _users_col, create_user, find_user_by_email, verify_user, issue_jwt, decode_jwt, find_user_by_id, is_valid_email

load_dotenv()

from src.loader   import load_transcript
from src.agents   import create_qa_agent, search_meeting_content
from src.pipeline import run_artifact_pipeline
from src.schema   import MeetingMinutes
from src.rag      import query_faiss, delete_artifact_vectors, rebuild_index_from_artifacts
from src import db
from src.db       import DEFAULT_USER_ID

from streamlit_oauth import OAuth2Component
import os
import jwt        # PyJWT library
import uuid       # Python’s built-in UUID module

# ── Page config ───────────────────────────────────────────────────────────────

client_id = os.getenv("GOOGLE_CLIENT_ID")
client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

authorize_url = "https://accounts.google.com/o/oauth2/v2/auth"
token_url = "https://oauth2.googleapis.com/token"
redirect_uri = "http://localhost:8501"  # adjust if deployed

oauth2 = OAuth2Component(client_id, client_secret, authorize_url, token_url, redirect_uri)


st.set_page_config(
    page_title="Meeting Minutes",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #1e2535; }
[data-testid="stMainBlockContainer"] { max-width: 900px; padding: 2rem 2rem 4rem; }

h1 { font-size: 1.5rem !important; font-weight: 600 !important; color: #e2e8f0 !important; letter-spacing: -0.01em; }
h2 { font-size: 1.05rem !important; font-weight: 500 !important; color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2rem !important; }
h3 { font-size: 0.95rem !important; font-weight: 500 !important; color: #cbd5e1 !important; }
p, li, label { color: #94a3b8 !important; font-size: 0.9rem; line-height: 1.7; }

.card { background:#161b27; border:1px solid #1e2535; border-radius:10px; padding:1.5rem; margin-bottom:1rem; }
.card-accent { background:#0d1b1e; border:1px solid #134e4a; border-radius:10px; padding:1.5rem; margin-bottom:1rem; }
.summary-text { font-size:0.95rem; line-height:1.8; color:#cbd5e1 !important; }

.pill-positive { background:#0d2d28; color:#5eead4; border:1px solid #134e4a; border-radius:999px; padding:3px 14px; font-size:0.78rem; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; display:inline-block; }
.pill-neutral  { background:#1a1f2e; color:#94a3b8; border:1px solid #1e2535; border-radius:999px; padding:3px 14px; font-size:0.78rem; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; display:inline-block; }
.pill-negative { background:#2d1215; color:#fda4af; border:1px solid #4c1d25; border-radius:999px; padding:3px 14px; font-size:0.78rem; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; display:inline-block; }

.decision-row { padding:10px 14px; border-left:2px solid #1d4ed8; background:#131929; border-radius:0 6px 6px 0; margin-bottom:8px; font-size:0.88rem; color:#cbd5e1 !important; line-height:1.6; }
.action-row { display:flex; gap:12px; align-items:flex-start; padding:12px 14px; background:#161b27; border:1px solid #1e2535; border-radius:8px; margin-bottom:8px; }
.action-owner { font-size:0.78rem; font-weight:600; color:#5eead4 !important; background:#0d2d28; border:1px solid #134e4a; border-radius:5px; padding:2px 9px; white-space:nowrap; flex-shrink:0; margin-top:1px; }
.action-task  { font-size:0.88rem; color:#cbd5e1 !important; line-height:1.6; }
.action-deadline { font-size:0.76rem; color:#64748b !important; margin-top:3px; }

.bubble-user { background:#1e2535; border:1px solid #2a3447; border-radius:12px 12px 2px 12px; padding:12px 16px; margin:8px 0 8px 15%; font-size:0.88rem; color:#cbd5e1 !important; line-height:1.7; }
.bubble-assistant { background:#0d1b1e; border:1px solid #134e4a; border-radius:12px 12px 12px 2px; padding:12px 16px; margin:8px 15% 8px 0; font-size:0.88rem; color:#a7f3d0 !important; line-height:1.7; }
.bubble-role { font-size:0.72rem; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; color:#475569 !important; margin-bottom:4px; }

.conv-item { padding:8px 10px; border-radius:7px; cursor:pointer; margin-bottom:2px; }
.conv-item:hover { background:#1e2535; }
.conv-item.active { background:#0d2d28; border-left:2px solid #5eead4; }

[data-testid="stFileUploader"] { background:#161b27 !important; border:1.5px dashed #1e2535 !important; border-radius:10px !important; }
.stButton > button { background:#0d4f47 !important; color:#5eead4 !important; border:1px solid #134e4a !important; border-radius:7px !important; font-weight:500 !important; font-size:0.88rem !important; padding:0.5rem 1.4rem !important; }
.stButton > button:hover { background:#0f6b61 !important; }
.info-row { font-size:0.78rem; color:#475569 !important; }
.section-label { font-size:0.72rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#475569 !important; margin-bottom:10px; }
hr { border-color:#1e2535 !important; margin:1.5rem 0 !important; }
[data-testid="stSpinner"] p { color:#5eead4 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "qa_agent":                 None,   # created once, reused across the whole session
        "active_conv_id":           None,
        "jwt_token": None,
        "active_messages":          [],
        "thread_config":            {"configurable": {"thread_id": "default"}},
        "pending_modal_artifact_id": None,  # artifact whose minutes modal should be shown
        "modal_artifact_id": None,           # artifact displayed in modal
        "show_modal": False, 
        "last_upload_signature":    None,   # (name, size) of the last file we started processing
        "uploader_key":             0,      # bumped to reset the file_uploader widget after each upload
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

user_id = decode_jwt(st.session_state.get("jwt_token"))
user = None
if user_id:
    user = find_user_by_id(user_id)


if user_id is None:
    st.markdown("""
    <style>
    .auth-title {
        font-size: 3rem;
        font-weight: 600;
        color: #5eead4;
        margin-bottom: 0.5rem;
    }
    .separator {
        border-left: 2px solid #1e2535;
        height: 220px;
        margin: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header row: logo | separator | title
    col1, col_sep, col2 = st.columns([2, 0.1, 3])
    with col1:
        st.image("proxym.png", width=700)
    with col_sep:
        st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='auth-title'>Welcome to the Meeting Minutes Generator AI Chatbot</div>", unsafe_allow_html=True)

    # Forms stacked underneath
    tab_login, tab_signup = st.tabs(["Login", "Sign up"])

    # ── Login ────────────────────────────────────────────────────────────────
    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", use_container_width=True):
            uid = verify_user(email, password)
            if uid:
                st.session_state.jwt_token = issue_jwt(uid)
                st.rerun()
            else:
                st.error("Invalid credentials")

        # Google login button
        st.markdown("---")
        result = oauth2.authorize_button("Login with Google", redirect_uri, "openid email profile")



        if result:
            id_token = result.get("id_token")
            user_info = jwt.decode(id_token, options={"verify_signature": False})
            email = user_info.get("email")
            name = user_info.get("name")

            existing = find_user_by_email(email)
            if existing:
                uid = existing["_id"]
            else:
                uid = create_user(name, email, uuid.uuid4().hex)  # random password
            st.session_state.jwt_token = issue_jwt(uid)
            st.rerun()


    # ── Signup ───────────────────────────────────────────────────────────────
    with tab_signup:
        username = st.text_input("Username", key="signup_username")
        if username:
            if _users_col().find_one({"username": username}):
                st.error("Username already taken")
            else:
                st.success("Username available ✅")

        email = st.text_input("Email", key="signup_email")
        if email:
            if not is_valid_email(email):
                st.error("Invalid email format")
            elif find_user_by_email(email):
                st.error("Email already taken")
            else:
                st.success("Email available ✅")

        password = st.text_input("Password", type="password", key="signup_password")
        if password:
            st.markdown("**Password requirements:**")
            st.markdown(f"- {'✅' if len(password) >= 8 else '❌'} At least 8 characters")
            st.markdown(f"- {'✅' if any(c.isupper() for c in password) else '❌'} One uppercase letter")
            st.markdown(f"- {'✅' if any(c.islower() for c in password) else '❌'} One lowercase letter")
            st.markdown(f"- {'✅' if any(c.isdigit() for c in password) else '❌'} One number")
            st.markdown(f"- {'✅' if any(c in '!@#$%^&*' for c in password) else '❌'} One special character (!@#$%^&*)")

        can_signup = (
            username and email and password
            and is_valid_email(email)
            and not find_user_by_email(email)
            and not _users_col().find_one({"username": username})
            and len(password) >= 8
            and any(c.isupper() for c in password)
            and any(c.islower() for c in password)
            and any(c.isdigit() for c in password)
            and any(c in "!@#$%^&*" for c in password)
        )

        if st.button("Sign up", use_container_width=True, disabled=not can_signup):
            uid = create_user(username, email, password)
            st.session_state.jwt_token = issue_jwt(uid)
            st.rerun()

    st.stop()







if st.session_state.qa_agent is None:
    st.session_state.qa_agent = create_qa_agent()


# ── Helpers ───────────────────────────────────────────────────────────────────

def switch_to_conversation(conv_id: str) -> None:
    doc = db.load_conversation(conv_id)
    if doc is None:
        return
    st.session_state.active_conv_id  = conv_id
    st.session_state.active_messages = doc.get("messages", [])
    st.session_state.thread_config   = {"configurable": {"thread_id": conv_id}}


def _fmt_time(iso: str) -> str:
    try:
        from datetime import datetime, timezone
        dt   = datetime.fromisoformat(iso)
        now  = datetime.now(timezone.utc)
        diff = (now - dt).total_seconds()
        if diff < 60:
            return "just now"
        if diff < 3600:
            return f"{int(diff // 60)}m ago"
        if diff < 86400:
            return f"{int(diff // 3600)}h ago"
        return f"{int(diff // 86400)}d ago"
    except Exception:
        return ""


def delete_artifact_cascade(artifact_id: str) -> None:
    """Delete an artifact's Mongo record and its FAISS vectors. Conversations
    are never touched by this."""
    doc = db.delete_artifact(artifact_id)
    if doc is None:
        return

    chunk_ids = doc.get("chunk_ids", [])
    try:
        delete_artifact_vectors(chunk_ids)
    except Exception:
        # This FAISS/langchain_community version doesn't support targeted
        # id deletion for this index — fall back to a full rebuild from
        # whatever artifacts are left.
        remaining = [
            db.get_artifact(a["_id"])
            
            for a in db.list_artifacts(user_id)
            if a["status"] == "ready"
        ]
        rebuild_index_from_artifacts([r for r in remaining if r is not None])

    if st.session_state.get("pending_modal_artifact_id") == artifact_id:
        st.session_state.pending_modal_artifact_id = None


def render_minutes_body(minutes: MeetingMinutes) -> None:
    """Shared rendering used inside the modal."""
    sentiment_cls = f"pill-{minutes.sentiment}"
    st.markdown(f"<span class='{sentiment_cls}'>{minutes.sentiment}</span>", unsafe_allow_html=True)

    st.markdown("<div class='section-label' style='margin-top:1rem'>Summary</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card'><p class='summary-text'>{minutes.summary}</p></div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='section-label'>Decisions &nbsp;<span style='color:#1d4ed8'>({len(minutes.decisions)})</span></div>",
        unsafe_allow_html=True,
    )
    if minutes.decisions:
        st.markdown(
            "<div>" + "".join(f"<div class='decision-row'>{d}</div>" for d in minutes.decisions) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<p class='info-row'>No decisions recorded.</p>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='section-label' style='margin-top:1.2rem'>Action items &nbsp;<span style='color:#5eead4'>({len(minutes.action_items)})</span></div>",
        unsafe_allow_html=True,
    )
    if minutes.action_items:
        for a in minutes.action_items:
            st.markdown(f"""
            <div class='action-row'>
                <div><span class='action-owner'>{a.owner}</span></div>
                <div>
                    <div class='action-task'>{a.task}</div>
                    <div class='action-deadline'>🗓 {a.deadline}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p class='info-row'>No action items recorded.</p>", unsafe_allow_html=True)


@st.dialog("Meeting processed ✅", width="large")
def show_minutes_modal(artifact: dict) -> None:
    minutes = MeetingMinutes(**artifact["minutes"])
    st.markdown(f"**{artifact['file_name']}**")
    render_minutes_body(minutes)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Close", use_container_width=True):
        st.session_state.show_modal = False
        st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    if user:

        st.markdown("### 👤 Account")

        st.write(f"**{user['username']}**")
        st.caption(user["email"])

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.jwt_token = None

            # Optional cleanup
            st.session_state.active_conv_id = None
            st.session_state.active_messages = []
            st.session_state.qa_agent = None

            st.rerun()

        st.markdown("---")

    st.markdown("### 📋 Meeting Minutes")
    st.markdown("<p style='color:#475569;font-size:0.8rem'>Upload a transcript — it's processed automatically.</p>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader(
        "Transcript file",
        type=["txt", "vtt"],
        help="Supports .txt and .vtt formats. Processing starts as soon as you upload.",
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded is not None:
        signature = (uploaded.name, uploaded.size)
        if signature != st.session_state.last_upload_signature:
            st.session_state.last_upload_signature = signature

            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            text = None
            try:
                text = load_transcript(tmp_path)
            except Exception as e:
                st.error(f"Could not read file: {e}")
            finally:
                os.unlink(tmp_path)

            if text:
                artifact_id = db.create_artifact(uploaded.name, text, user_id)
                threading.Thread(
                    target=run_artifact_pipeline,
                    args=(artifact_id, uploaded.name, text),
                    daemon=True,
                ).start()
                st.session_state.pending_modal_artifact_id = artifact_id
                st.session_state.uploader_key += 1   # reset the widget for the next upload
                st.rerun()

    st.markdown("---")

    # ── New chat ────────────────────────────────────────────────────────────
    if st.button("＋ New chat", use_container_width=True):
        cid = db.create_conversation(user_id)
        switch_to_conversation(cid)
        st.rerun()

    # ── Conversations (user-scoped, independent of files) ─────────────────────
    st.markdown("<div class='section-label' style='margin-top:1rem'>Conversations</div>", unsafe_allow_html=True)

    try:
        convs = db.list_conversations(user_id)
    except Exception as e:
        st.warning(f"MongoDB unavailable: {e}")
        convs = []

    if not convs:
        st.markdown("<p class='info-row'>No conversations yet. Start a new chat.</p>", unsafe_allow_html=True)
    else:
        for conv in convs:
            is_active = conv["id"] == st.session_state.active_conv_id
            msg_count = conv["message_count"]
            badge = f"{msg_count // 2} msg{'s' if msg_count > 2 else ''}" if msg_count else "empty"

            c1, c2 = st.columns([5, 1])
            with c1:
                if st.button(
                    conv["title"],
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                    help=f"{badge} · {_fmt_time(conv['updated_at'])}",
                ):
                    switch_to_conversation(conv["id"])
                    st.rerun()
            with c2:
                if st.button("✕", key=f"del_{conv['id']}", help="Delete this conversation"):
                    db.delete_conversation(conv["id"])
                    if st.session_state.active_conv_id == conv["id"]:
                        st.session_state.active_conv_id  = None
                        st.session_state.active_messages = []
                    st.rerun()

    st.markdown("---")

    # ── Artifacts (uploaded files, independent of conversations) ─────────────
    st.markdown("<div class='section-label'>Artifacts</div>", unsafe_allow_html=True)

    try:
        artifacts = db.list_artifacts(user_id)
    except Exception as e:
        st.warning(f"MongoDB unavailable: {e}")
        artifacts = []

    if not artifacts:
        st.markdown("<p class='info-row'>No files uploaded yet.</p>", unsafe_allow_html=True)
    else:
        status_icon = {"processing": "⏳", "ready": "📄", "failed": "⚠️"}
        for art in artifacts:
            icon = status_icon.get(art["status"], "📄")
            c1, c2 = st.columns([5, 1])
            with c1:
                if st.button(
                    f"{icon} {art['file_name']}",
                    key=f"art_{art['_id']}",
                    use_container_width=True,
                    disabled=art["status"] != "ready",
                    help="View minutes" if art["status"] == "ready" else art["status"],
                ):
                    st.session_state.modal_artifact_id = art["_id"]
                    st.session_state.show_modal = True
                    st.rerun()
            with c2:
                if st.button("✕", key=f"delart_{art['_id']}", help="Delete this file & its embeddings"):
                    delete_artifact_cascade(art["_id"])
                    st.rerun()

    st.markdown("---")

    with st.expander("⚙️ Settings", expanded=False):
        st.checkbox("Show processing log (console only)", value=True, key="show_debug")

    st.markdown("""
    <p class='info-row'>Powered by LangChain agents +<br>Gemini 2.0 Flash · FAISS RAG · MongoDB</p>
    """, unsafe_allow_html=True)


# ── Minutes modal (auto-opens on completion, or on artifact click) ────────────

# Waiting for processing to finish
if st.session_state.pending_modal_artifact_id:

    art = db.get_artifact(
        st.session_state.pending_modal_artifact_id
    )

    if art is None:
        st.session_state.pending_modal_artifact_id = None

    elif art["status"] == "ready":
        st.session_state.modal_artifact_id = art["_id"]
        st.session_state.show_modal = True
        st.session_state.pending_modal_artifact_id = None
        st.rerun()

    elif art["status"] == "failed":
        st.toast(
            f"Processing failed for {art['file_name']}",
            icon="⚠️",
        )
        st.session_state.pending_modal_artifact_id = None

# Actually display the modal
if (
    st.session_state.show_modal
    and st.session_state.modal_artifact_id
):
    art = db.get_artifact(
        st.session_state.modal_artifact_id
    )

    if art:
        show_minutes_modal(art)

# ── Live polling while something is processing ─────────────────────────────────
# Streamlit has no push updates, so while a background job is running we
# rerun the script every couple seconds to refresh sidebar status badges.
# Skipped while a modal is open so it doesn't flicker.

_any_processing = any(a["status"] == "processing" for a in artifacts)

# Only stop polling once the pending artifact is actually ready and
# its modal is being displayed.
_modal_open = False

pending_id = st.session_state.pending_modal_artifact_id
if pending_id:
    pending_artifact = db.get_artifact(pending_id)
    _modal_open = (
        pending_artifact is not None
        and pending_artifact["status"] == "ready"
    )

if _any_processing and not _modal_open:
    time.sleep(2)
    st.rerun()


# ── Main content: chat, full width ─────────────────────────────────────────────

st.markdown("<h1>💬 Meeting Assistant</h1>", unsafe_allow_html=True)

ready_artifacts = [a for a in artifacts if a["status"] == "ready"] if artifacts else []

scope_options = {"All artifacts": None}
for a in ready_artifacts:
    scope_options[a["file_name"]] = a["_id"]

col_scope, _ = st.columns([2, 3])
with col_scope:
    scope_label = st.selectbox(
        "Ask about",
        options=list(scope_options.keys()),
        label_visibility="collapsed",
    )
scope_artifact_id = scope_options[scope_label]

if st.session_state.active_conv_id is None:
    st.markdown("""
    <div style='text-align:center;padding:4rem 0;color:#334155'>
        <p style='font-size:2.5rem;margin-bottom:0.5rem'>💬</p>
        <p style='color:#334155 !important;font-size:0.9rem'>
        Click <strong style='color:#5eead4'>＋ New chat</strong> in the sidebar to get started,<br>
        or select an existing conversation.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    messages = st.session_state.active_messages
    chat_container = st.container(height=520, border=False)
    with chat_container:
        if not messages:
            st.markdown("""
            <div style='text-align:center;padding:3rem 0;color:#334155'>
                <p style='font-size:2rem;margin-bottom:0.5rem'>💬</p>
                <p style='color:#334155 !important;font-size:0.85rem'>
                Ask anything about your uploaded meetings.<br>
                <em>"Who owns the frontend task?"<br>
                "Was the deadline extended?"</em>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in messages:
                role_label = "You" if msg["role"] == "user" else "Assistant"
                css_class  = "bubble-user" if msg["role"] == "user" else "bubble-assistant"
                st.markdown(f"""
                <div class='{css_class}'>
                    <div class='bubble-role'>{role_label}</div>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)

    with st.form("qa_form", clear_on_submit=True):
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            question = st.text_input(
                "question",
                placeholder="Ask a question about your meetings…",
                label_visibility="collapsed",
            )
        with btn_col:
            ask_btn = st.form_submit_button("Ask", use_container_width=True)

    if ask_btn and question.strip():
        q = question.strip()
        with st.spinner("Searching…"):
            try:
                semantic_chunks = query_faiss(q, k=5, artifact_id=scope_artifact_id)

                refined = []
                for chunk in semantic_chunks:
                    match = search_meeting_content.func(query=q, transcript=chunk)
                    if match and "No directly matching" not in match:
                        refined.append(match)

                if not semantic_chunks:
                    context = "No transcripts have been uploaded yet, or none match this question."
                else:
                    context = "\n\n".join(refined) if refined else "No exact matches found in transcript."

                result = st.session_state.qa_agent.invoke(
                    {"messages": [{"role": "user", "content": f"Q: {q}\n\nContext:\n{context}"}]},
                    st.session_state.thread_config,
                )
                answer = result["messages"][-1].content

                db.append_messages(st.session_state.active_conv_id, q, answer, scope_artifact_id)
                st.session_state.active_messages.append({"role": "user",      "content": q})
                st.session_state.active_messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                err_msg = f"⚠️ Error: {str(e)[:300]}"
                db.append_messages(st.session_state.active_conv_id, q, err_msg, scope_artifact_id)
                st.session_state.active_messages.append({"role": "user",      "content": q})
                st.session_state.active_messages.append({"role": "assistant", "content": err_msg})

        st.rerun()