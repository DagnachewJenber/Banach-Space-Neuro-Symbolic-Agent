import contextlib
import io
import sys
import openai
from openai import OpenAI
import streamlit as st
import sympy as sp

# Try importing hyperon, with a clean fallback error if it's not installed
try:
    from hyperon import MeTTa
except ImportError:
    st.error(
        "The `hyperon` library is missing. Please run `pip install hyperon` in your terminal to enable MeTTa capability.",
        icon="⚠️",
    )
    st.stop()

# ==========================================
# 1. Page Settings & Layout
# ==========================================
st.set_page_config(page_title="Banach Logic Agent", page_icon="🧠", layout="wide")
st.title("🧠 Banach Space Neuro-Symbolic Agent")
st.write(
    "A hybrid AI framework leveraging LLM reasoning (`OpenRouter`), "
    "Axiomatic Logic (`MeTTa`), and Symbolic Algebra (`SymPy`)."
)

# ==========================================
# 2. Key Verification
# ==========================================
if "OPENROUTER_API_KEY" not in st.secrets:
    st.error(
        "Missing `OPENROUTER_API_KEY` in Streamlit Secrets (`.streamlit/secrets.toml`)!",
        icon="🚨",
    )
    st.stop()


# ==========================================
# 3. Client & Cognitive Engine Initialization
# ==========================================
@st.cache_resource
def get_openrouter_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"],
        default_headers={
            "HTTP-Referer": "https://localhost:8501",
            "X-OpenRouter-Title": "Banach Space Logic Agent",
        },
    )


@st.cache_resource
def initialize_metta_engine():
    """Initializes the MeTTa runtime and seeds it with base Banach space axioms."""
    metta = MeTTa()
    # Seeding the AtomSpace knowledge graph with basic Functional Analysis ontology
    metta.run("""
        ;; Define Types
        (: VectorSpace Type)
        (: BanachSpace Type)
        (: Complete Property)
        (: Normed Property)

        ;; Define Logical Implications
        (= (is-banach $space) (and (has-property $space Normed) (has-property $space Complete)))
        
        ;; Seed Axiomatic Ground Truths
        (: L2 VectorSpace)
        (has-property L2 Normed)
        (has-property L2 Complete)
        
        (: Q-Space VectorSpace)
        (has-property Q-Space Normed)
        ;; Note: Q-Space is incomplete, so (is-banach Q-Space) will evaluate to false/empty
    """)
    return metta


client = get_openrouter_client()
metta_engine = initialize_metta_engine()

# ==========================================
# 4. Connection Testing (Cached via Session State)
# ==========================================
if "api_verified" not in st.session_state:
    with st.spinner("Verifying connection to OpenRouter free cluster..."):
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            if isinstance(response, str):
                st.error(f"OpenRouter returned a raw text error: {response}")
            elif hasattr(response, "choices") and response.choices:
                st.session_state.api_verified = True
                st.success("Connection to OpenRouter verified successfully!")
            else:
                st.warning(f"Unexpected response format: {str(response)}")
        except Exception as e:
            st.error(f"Initialization Connection Error: {str(e)}")

# ==========================================
# 5. Core Engine Execution Helpers
# ==========================================
def execute_symbolic_math(code_string):
    """Executes pure python code using SymPy and captures printed output safely."""
    output_buffer = io.StringIO()
    local_env = {"sp": sp, "sympy": sp}
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code_string, {}, local_env)
        return output_buffer.getvalue()
    except Exception as e:
        return f"Execution Error: {str(e)}"


def execute_metta_reasoning(query_string):
    """Evaluates symbolic logic strings in the local MeTTa AtomSpace environment."""
    try:
        result = metta_engine.run(query_string)
        return str(result)
    except Exception as e:
        return f"MeTTa Core Error: {str(e)}"


# ==========================================
# 6. Prompts, Guards, and History
# ==========================================
BANACH_DEVELOPER_PROMPT = (
    "You are an elite reasoning agent specializing in Banach Space Theory and Functional Analysis.\n"
    "CRITICAL GUARDRAIL: You must strictly only answer questions within this mathematical domain.\n\n"
    "AXIOMATIC REASONING: Walk step-by-step through core axioms and theorems (Hahn-Banach, Open Mapping, Baire Category).\n\n"
    "NEURO-SYMBOLIC COGNITION: You are paired with a MeTTa logic interpreter. If the user asks about properties "
    "like completeness, vector layouts, or standard functional relations, think structurally like an AGI framework."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "developer", "content": BANACH_DEVELOPER_PROMPT}
    ]

# ==========================================
# 7. Main Interface Layout (Split UI)
# ==========================================
col_chat, col_info = st.columns([2, 1])

with col_chat:
    st.subheader("💬 Mathematical Discourse")
    # Render chat logs (Omitting developer instructions)
    for msg in st.session_state.messages:
        if msg["role"] not in ["developer", "system"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # Chat submission loop
    if user_query := st.chat_input("Ask a proof, theorem, or structural question..."):
        with st.chat_message("user"):
            st.write(user_query)

        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                stream = client.chat.completions.create(
                    model="openrouter/free",
                    messages=st.session_state.messages,
                    stream=True,
                )

                if isinstance(stream, str):
                    st.error(f"Streaming error payload: {stream}")
                else:
                    for chunk in stream:
                        if hasattr(chunk, "choices") and chunk.choices:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, "content") and delta.content:
                                full_response += delta.content
                                response_placeholder.write(
                                    full_response + "▌"
                                )
                    response_placeholder.write(full_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
            except Exception as e:
                st.error(f"API Streaming Error: {str(e)}")

with col_info:
    st.subheader("📚 Quick Reference")
    st.info(
        "**MeTTa** evaluates hypergraphs without relying on standard execution traps. "
        "Try evaluating if spaces in your environment are complete or satisfy Banach conditions using the engine on the left!"
    )

# ==========================================
# 8. Sidebar Tool Configuration (The Hybrid Engines)
# ==========================================
st.sidebar.header("🧮 Hybrid Cognitive Sandboxes")
st.sidebar.write(
    "Use these engines to execute deterministic calculations and strict logic queries."
)

tab_metta, tab_sympy = st.sidebar.tabs(["MeTTa Logic Space", "SymPy Engine"])

with tab_metta:
    st.markdown("### MeTTa Meta-Language Engine")
    sample_metta = (
        ";; Ask MeTTa if space L2 satisfies Banach properties\n!(is-banach L2)"
    )
    user_metta = st.text_area(
        "Input MeTTa Code:", value=sample_metta, height=200, key="metta_sandbox"
    )

    if st.button("Evaluate MeTTa Logic"):
        with st.spinner("Processing atom graphs..."):
            metta_result = execute_metta_reasoning(user_metta)
            st.code(metta_result, language="lisp")

with tab_sympy:
    st.markdown("### SymPy Computer Algebra")
    sample_sympy = (
        "import sympy as sp\nx = sp.Symbol('x')\n"
        "f = sp.exp(-x)\n"
        "integral = sp.integrate(f, (x, 0, sp.oo))\n"
        'print(f"Integral value: {integral}")'
    )
    user_sympy = st.text_area(
        "Input Python/SymPy:",
        value=sample_sympy,
        height=200,
        key="sympy_sandbox",
    )

    if st.button("Run Symbolic Engine"):
        with st.spinner("Calculating symbols..."):
            sympy_result = execute_symbolic_math(user_sympy)
            st.code(
                sympy_result
                if sympy_result.strip()
                else "Executed successfully (No stdout print generated)."
            )
