"""Streamlit demo: Claude + tool calling with rich Langfuse tracing.

Workshop layout: one hosted Streamlit app. Each attendee enters their own
Langfuse credentials in the sidebar — their traces go to their own Langfuse
account so they can build LLM-as-a-judge evaluators against real data they
just produced.

The Anthropic key is supplied by the host via Streamlit secrets, with a
per-session rate limit to control cost.

Stable JSON shape (per trace) — predictable paths for downstream evaluators:

  trace.input  = {"user_message": str, "history_turns": int}
  trace.output = {"reply": str, "stop_reason": str}
  trace.metadata = {
      "app":      {"name", "version", "environment"},
      "session":  {"id"},
      "user":     {"id", "tier", "locale", "channel"},
      "request":  {"id"},
      "model":    {"name", "temperature", "max_tokens"},
      "system_prompt": str,
      "summary":  {"llm_calls", "tool_calls", "tool_errors",
                   "input_tokens", "output_tokens", "latency_ms",
                   "stopped_because"},
      "tools_used": [
          {"name", "input": {...}, "output": {...}, "ok": bool, "latency_ms"}
      ],
  }

  generation.input  = {"system": str, "messages": [...]}
  generation.output = {"blocks": [...]}
  generation.metadata = {"iteration", "stop_reason", "response_id",
                         "latency_ms", "model_returned",
                         "tools_available": [...]}

  tool_span.input  = {"args": {...}}
  tool_span.output = {"result": {...}, "ok": bool}
  tool_span.metadata = {"tool_name", "tool_kind", "latency_ms", "ok"}
"""
from __future__ import annotations

import os
import time
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Host-provided secret on Streamlit Cloud (and locally via .env).
if "ANTHROPIC_API_KEY" in st.secrets and not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

from anthropic import Anthropic  # noqa: E402
from langfuse import Langfuse    # noqa: E402

from tools import TOOL_REGISTRY, TOOL_SCHEMA  # noqa: E402


# --- Config -----------------------------------------------------------------

APP_NAME = "rc-support-bot"
APP_VERSION = "0.1.0"
ENVIRONMENT = os.environ.get("APP_ENV", "workshop")

AVAILABLE_MODELS = [
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-haiku-4-5",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful customer-support assistant for an electronics retailer. "
    "Use the provided tools to answer questions about orders, products, pricing "
    "and discounts. Always cite the values returned by tools rather than guessing. "
    "Keep replies concise and friendly."
)

MAX_TOOL_ITERATIONS = 5
SESSION_MESSAGE_LIMIT = 25  # per-attendee cost guardrail


@st.cache_resource
def get_anthropic_client() -> Anthropic:
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# --- Per-session Langfuse client -------------------------------------------

def build_langfuse(public_key: str, secret_key: str, host: str) -> Langfuse:
    """Instantiate a Langfuse client scoped to this attendee's credentials."""
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def trace_url(host: str, trace_id: str) -> str:
    return f"{host.rstrip('/')}/trace/{trace_id}"


# --- Agent loop -------------------------------------------------------------

def _run_tool(langfuse: Langfuse, name: str, args: dict, registry: dict) -> dict:
    started = time.perf_counter()
    with langfuse.start_as_current_span(
        name=f"tool:{name}",
        input={"args": args},
        metadata={"tool_name": name, "tool_kind": "function"},
    ) as span:
        fn = registry.get(name)
        ok = True
        if fn is None:
            result = {"error": f"Unknown or disabled tool '{name}'."}
            ok = False
        else:
            try:
                result = fn(**args)
                if isinstance(result, dict) and "error" in result:
                    ok = False
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
                ok = False
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        span.update(
            output={"result": result, "ok": ok},
            metadata={"tool_name": name, "tool_kind": "function",
                      "latency_ms": latency_ms, "ok": ok},
            level="ERROR" if not ok else "DEFAULT",
        )
    return {"name": name, "input": args, "output": result, "ok": ok, "latency_ms": latency_ms}


def run_agent(
    *,
    langfuse: Langfuse,
    user_message: str,
    history: list[dict],
    session_id: str,
    user_id: str,
    user_attrs: dict,
    system_prompt: str,
    model: str,
    model_params: dict,
    enabled_tools: list[str],
) -> dict:
    """One chat turn. Creates a root trace via a chat-turn span."""
    client = get_anthropic_client()
    tool_schema = [t for t in TOOL_SCHEMA if t["name"] in enabled_tools]
    tool_registry = {k: v for k, v in TOOL_REGISTRY.items() if k in enabled_tools}

    request_id = str(uuid.uuid4())
    base_metadata = {
        "app": {"name": APP_NAME, "version": APP_VERSION, "environment": ENVIRONMENT},
        "session": {"id": session_id},
        "user": {
            "id": user_id,
            "tier": user_attrs.get("customer_tier"),
            "locale": user_attrs.get("locale"),
            "channel": user_attrs.get("channel"),
        },
        "request": {"id": request_id},
        "model": {"name": model, **model_params},
        "system_prompt": system_prompt,
    }

    messages = history + [{"role": "user", "content": user_message}]
    tools_used: list[dict] = []
    tool_errors = 0
    turn_started = time.perf_counter()
    total_input_tokens = 0
    total_output_tokens = 0
    llm_calls = 0
    final_text = ""
    stop_reason = "max_iterations"

    with langfuse.start_as_current_span(
        name="chat-turn",
        input={"user_message": user_message, "history_turns": len(history) // 2},
        metadata=base_metadata,
    ) as root:
        trace_id = langfuse.get_current_trace_id()
        langfuse.update_current_trace(
            name="customer-support-turn",
            session_id=session_id,
            user_id=user_id,
            input={"user_message": user_message, "history_turns": len(history) // 2},
            tags=[
                f"env:{ENVIRONMENT}",
                f"app:{APP_NAME}",
                f"model:{model}",
                f"tier:{user_attrs.get('customer_tier', 'unknown')}",
                f"channel:{user_attrs.get('channel', 'web')}",
                f"locale:{user_attrs.get('locale', 'unknown')}",
            ],
            metadata=base_metadata,
        )

        for iteration in range(MAX_TOOL_ITERATIONS):
            gen_started = time.perf_counter()
            with langfuse.start_as_current_generation(
                name="anthropic.messages.create",
                model=model,
                model_parameters=model_params,
                input={"system": system_prompt, "messages": messages},
                metadata={"iteration": iteration,
                          "tools_available": [t["name"] for t in tool_schema]},
            ) as gen:
                response = client.messages.create(
                    model=model,
                    system=system_prompt,
                    tools=tool_schema,
                    messages=messages,
                    **model_params,
                )
                latency_ms = round((time.perf_counter() - gen_started) * 1000, 2)
                llm_calls += 1
                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens
                gen.update(
                    output={"blocks": [b.model_dump() for b in response.content]},
                    usage_details={"input": response.usage.input_tokens,
                                   "output": response.usage.output_tokens},
                    metadata={
                        "iteration": iteration,
                        "tools_available": [t["name"] for t in tool_schema],
                        "stop_reason": response.stop_reason,
                        "response_id": response.id,
                        "latency_ms": latency_ms,
                        "model_returned": response.model,
                    },
                )

            messages.append({"role": "assistant",
                             "content": [b.model_dump() for b in response.content]})
            stop_reason = response.stop_reason

            if stop_reason != "tool_use":
                final_text = "".join(b.text for b in response.content if b.type == "text").strip()
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                record = _run_tool(langfuse, block.name, block.input, tool_registry)
                if not record["ok"]:
                    tool_errors += 1
                tools_used.append(record)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(record["output"]),
                    "is_error": not record["ok"],
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            final_text = "Sorry — I couldn't finish that within the tool-call limit."

        turn_latency = round((time.perf_counter() - turn_started) * 1000, 2)
        summary = {
            "llm_calls": llm_calls,
            "tool_calls": len(tools_used),
            "tool_errors": tool_errors,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "latency_ms": turn_latency,
            "stopped_because": stop_reason,
        }
        full_metadata = {**base_metadata, "summary": summary, "tools_used": tools_used}
        root.update(
            output={"reply": final_text, "stop_reason": stop_reason},
            metadata=full_metadata,
        )
        langfuse.update_current_trace(
            output={"reply": final_text, "stop_reason": stop_reason},
            metadata=full_metadata,
        )

    return {
        "reply": final_text,
        "tools_used": tools_used,
        "trace_id": trace_id,
        "latency_ms": turn_latency,
        "tokens": {"input": total_input_tokens, "output": total_output_tokens},
        "stop_reason": stop_reason,
    }


# --- Streamlit UI -----------------------------------------------------------

st.set_page_config(page_title="Langfuse Tracing Workshop", page_icon="🔎", layout="wide")
st.title("🔎 Langfuse Tracing Workshop")
st.caption("Each attendee uses their own Langfuse account. The host supplies the Claude API key.")

# --- Session state defaults ---
ss = st.session_state
ss.setdefault("session_id", f"sess-{uuid.uuid4().hex[:12]}")
ss.setdefault("attendee_suffix", uuid.uuid4().hex[:6])
ss.setdefault("history", [])
ss.setdefault("display", [])
ss.setdefault("message_count", 0)
ss.setdefault("langfuse", None)
ss.setdefault("langfuse_host", "https://cloud.langfuse.com")
ss.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)
ss.setdefault("model", AVAILABLE_MODELS[0])
ss.setdefault("temperature", 0.2)
ss.setdefault("max_tokens", 1024)
ss.setdefault("enabled_tools", [t["name"] for t in TOOL_SCHEMA])

# --- Sidebar: Langfuse connection ---
with st.sidebar:
    st.subheader("1️⃣ Connect your Langfuse account")
    st.caption("Get keys at Langfuse → Settings → API Keys")

    host_choice = st.selectbox(
        "Langfuse region",
        ["https://cloud.langfuse.com", "https://us.cloud.langfuse.com", "Other / self-hosted"],
        index=0,
    )
    if host_choice == "Other / self-hosted":
        host = st.text_input("Langfuse host", value=ss.langfuse_host)
    else:
        host = host_choice

    pk = st.text_input("Public key (pk-lf-…)", type="password")
    sk = st.text_input("Secret key (sk-lf-…)", type="password")

    col_a, col_b = st.columns(2)
    if col_a.button("Connect", use_container_width=True):
        if not (pk and sk):
            st.error("Both keys are required.")
        else:
            try:
                lf = build_langfuse(pk, sk, host)
                if not lf.auth_check():
                    st.error("Authentication failed. Check your keys and region.")
                else:
                    ss.langfuse = lf
                    ss.langfuse_host = host
                    st.success("Connected ✅")
            except Exception as exc:
                st.error(f"Could not connect: {exc}")
    if col_b.button("Disconnect", use_container_width=True):
        ss.langfuse = None
        st.rerun()

    if ss.langfuse:
        st.success(f"Connected to {ss.langfuse_host}")
    else:
        st.warning("Not connected — traces will not be recorded.")

    st.divider()
    st.subheader("2️⃣ Configure the agent")
    ss.system_prompt = st.text_area(
        "System prompt",
        value=ss.system_prompt,
        height=180,
        help="This is sent as the system message on every turn.",
    )
    if st.button("Reset to default prompt"):
        ss.system_prompt = DEFAULT_SYSTEM_PROMPT
        st.rerun()

    ss.model = st.selectbox("Model", AVAILABLE_MODELS,
                            index=AVAILABLE_MODELS.index(ss.model))
    ss.temperature = st.slider("Temperature", 0.0, 1.0, ss.temperature, 0.05)
    ss.max_tokens = st.slider("Max tokens", 256, 4096, ss.max_tokens, 64)

    ss.enabled_tools = st.multiselect(
        "Tools available to the agent",
        options=[t["name"] for t in TOOL_SCHEMA],
        default=ss.enabled_tools,
        help="Disable a tool to see how the agent copes without it.",
    )

    st.divider()
    st.subheader("3️⃣ Your identity (logged to your Langfuse)")
    user_id = st.text_input("User id", value=f"attendee-{ss.attendee_suffix}")
    customer_tier = st.selectbox("Customer tier", ["free", "pro", "enterprise"], index=1)
    locale = st.selectbox("Locale", ["en-US", "en-GB", "de-DE", "ja-JP"], index=0)
    channel = st.selectbox("Channel", ["web", "mobile", "email"], index=0)

    st.divider()
    st.subheader("Session")
    st.code(ss.session_id, language=None)
    st.caption(f"Messages used: {ss.message_count} / {SESSION_MESSAGE_LIMIT}")
    if st.button("Reset conversation"):
        ss.session_id = f"sess-{uuid.uuid4().hex[:12]}"
        ss.history = []
        ss.display = []
        ss.message_count = 0
        st.rerun()

    st.divider()
    with st.expander("Try asking…"):
        st.markdown(
            "- *What's the status of order ORD-1001?*\n"
            "- *How much is a laptop, and what's the price after a 15% discount?*\n"
            "- *Is the keyboard in stock? When would order ORD-1002 arrive?*"
        )


user_attrs = {"customer_tier": customer_tier, "locale": locale, "channel": channel}
model_params = {"temperature": ss.temperature, "max_tokens": ss.max_tokens}


def render_assistant_turn(turn: dict) -> None:
    st.markdown(turn["text"])
    cols = st.columns(4)
    cols[0].metric("Latency", f"{turn.get('latency_ms', 0):.0f} ms")
    tok = turn.get("tokens") or {}
    cols[1].metric("Input tok", tok.get("input", 0))
    cols[2].metric("Output tok", tok.get("output", 0))
    cols[3].metric("Tools", len(turn.get("tools", [])))

    if turn.get("tools"):
        with st.expander(f"🔧 Tool calls ({len(turn['tools'])})"):
            for t in turn["tools"]:
                status = "✅" if t.get("ok", True) else "⚠️"
                st.markdown(f"{status} **{t['name']}** — {t.get('latency_ms', 0)} ms")
                st.json({"input": t["input"], "output": t["output"]})

    tid = turn.get("trace_id")
    host = turn.get("trace_host")
    if tid and host:
        st.caption(f"trace `{tid}` — [open in Langfuse]({trace_url(host, tid)})")


for turn in ss.display:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant":
            render_assistant_turn(turn)
        else:
            st.markdown(turn["text"])


# --- Chat input ---
prompt = st.chat_input(
    "Ask about an order, product, or discount…",
    disabled=(ss.langfuse is None or ss.message_count >= SESSION_MESSAGE_LIMIT
              or not ss.enabled_tools and False),  # tools optional
)

if ss.langfuse is None:
    st.info("👈 Connect your Langfuse account in the sidebar to start chatting.")
elif ss.message_count >= SESSION_MESSAGE_LIMIT:
    st.warning(f"Per-session message limit ({SESSION_MESSAGE_LIMIT}) reached. "
               f"Use **Reset conversation** in the sidebar to start a new session.")

if prompt:
    ss.display.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = run_agent(
                    langfuse=ss.langfuse,
                    user_message=prompt,
                    history=ss.history,
                    session_id=ss.session_id,
                    user_id=user_id or f"attendee-{ss.attendee_suffix}",
                    user_attrs=user_attrs,
                    system_prompt=ss.system_prompt,
                    model=ss.model,
                    model_params=model_params,
                    enabled_tools=ss.enabled_tools,
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()
            finally:
                if ss.langfuse:
                    ss.langfuse.flush()

        ss.message_count += 1
        ss.history.append({"role": "user", "content": prompt})
        ss.history.append({"role": "assistant", "content": result["reply"]})
        ss.display.append({
            "role": "assistant",
            "text": result["reply"],
            "tools": result["tools_used"],
            "trace_id": result["trace_id"],
            "trace_host": ss.langfuse_host,
            "latency_ms": result["latency_ms"],
            "tokens": result["tokens"],
        })
        render_assistant_turn(ss.display[-1])
