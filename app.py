"""Streamlit demo: Claude + tool calling with Langfuse tracing.

Pattern proven in production at GTP-Rag-System:
- Langfuse v3 SDK initialized ONCE per process with OTel batch tuning.
- AnthropicInstrumentor auto-captures every Anthropic SDK call (tokens,
  model, latency) as a generation span.
- @observe wraps custom functions (chat-turn root span, per-tool spans).
- propagate_attributes attaches session_id / user_id / metadata / tags to
  the current trace.

Multi-tenant note for the workshop: the Langfuse client + OTel tracer
provider are process-global, so the "currently active" Langfuse account is
whoever clicked Connect last. For simultaneous attendees, each should run
their own copy. For a single-presenter workshop with one shared hosted app,
this is fine.
"""
from __future__ import annotations

import os
import time
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

# IMPORTANT: set OTel batch env vars BEFORE importing langfuse, so they apply
# the very first time the tracer provider initialises.
os.environ.setdefault("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "32")
os.environ.setdefault("OTEL_BSP_SCHEDULE_DELAY_MILLIS", "2000")
os.environ.setdefault("OTEL_BSP_MAX_QUEUE_SIZE", "4096")

from anthropic import Anthropic  # noqa: E402
from langfuse import Langfuse, get_client, observe  # noqa: E402
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor  # noqa: E402

from tools import TOOL_REGISTRY, TOOL_SCHEMA  # noqa: E402


# --- Config -----------------------------------------------------------------

APP_NAME = "rc-support-bot"
APP_VERSION = "0.1.0"
ENVIRONMENT = os.environ.get("APP_ENV", "workshop")
LANGFUSE_HOSTS = [
    "https://cloud.langfuse.com",
    "https://us.cloud.langfuse.com",
]
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
SESSION_MESSAGE_LIMIT = 25


@st.cache_resource
def get_anthropic_client() -> Anthropic:
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# --- Langfuse init ----------------------------------------------------------

def _probe_host(public_key: str, secret_key: str, host: str) -> tuple[bool, str]:
    import urllib.request, urllib.error, base64
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/public/projects",
        headers={"Authorization": f"Basic {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return (200 <= resp.status < 300), f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def init_langfuse_session(public_key: str, secret_key: str, host_choice: str,
                          custom_host: str) -> tuple[bool, str]:
    """Validate creds via HTTP, then (re)configure the global Langfuse client.

    Returns (ok, message). On success the global langfuse client points at
    the resolved host and the Anthropic instrumentor is active.
    """
    if host_choice == "Other / self-hosted":
        candidates = [custom_host.strip()] if custom_host else []
    elif host_choice == "Auto-detect":
        candidates = list(LANGFUSE_HOSTS)
    else:
        candidates = [host_choice] + [h for h in LANGFUSE_HOSTS if h != host_choice]

    resolved = None
    last = "no candidates"
    for h in candidates:
        if not h:
            continue
        ok, detail = _probe_host(public_key, secret_key, h)
        if ok:
            resolved = h
            break
        last = f"{h} → {detail}"
    if resolved is None:
        return False, last

    # Push creds into env so Langfuse picks them up and so any later
    # `get_client()` returns a client pointing at the right host.
    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    os.environ["LANGFUSE_HOST"] = resolved
    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = ENVIRONMENT

    # (Re)initialise — Langfuse v3 caches the singleton, but constructing
    # again with new creds updates the active config.
    Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=resolved,
        timeout=30,
    )

    # Instrument Anthropic once per process.
    if not st.session_state.get("_anthropic_instrumented"):
        try:
            AnthropicInstrumentor().instrument()
            st.session_state["_anthropic_instrumented"] = True
        except Exception as e:
            return False, f"Anthropic instrumentation failed: {e}"

    # Final sanity check using the SDK client.
    client = get_client()
    if not client.auth_check():
        return False, "SDK auth_check failed after init"

    return True, resolved


# --- Agent loop -------------------------------------------------------------

@observe(name="tool-call")
def _run_tool(name: str, args: dict, registry: dict) -> dict:
    import traceback
    started = time.perf_counter()
    fn = registry.get(name)
    ok = True
    error_kind: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None

    if fn is None:
        result = {"error": f"Unknown or disabled tool '{name}'."}
        ok = False
        error_kind = "UnknownTool"
        error_message = result["error"]
    else:
        try:
            result = fn(**args)
            if isinstance(result, dict) and "error" in result:
                # Tool ran but returned a domain-level error (e.g. "order not found").
                # We treat that as a WARNING, not ERROR — useful signal but not a bug.
                ok = False
                error_kind = "ToolReturnedError"
                error_message = str(result.get("error"))
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
            ok = False
            error_kind = type(exc).__name__
            error_message = str(exc)
            error_traceback = traceback.format_exc()

    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    # Choose level: real exceptions are ERROR, domain errors (tool returned
    # {"error": ...}) are WARNING, success is unset (DEFAULT).
    if error_traceback is not None:
        level = "ERROR"
    elif error_kind:
        level = "WARNING"
    else:
        level = None

    span_metadata = {
        "tool_name": name,
        "tool_kind": "function",
        "latency_ms": latency_ms,
        "ok": ok,
    }
    if error_kind:
        span_metadata["error"] = {
            "kind": error_kind,
            "message": error_message,
            "traceback": error_traceback,
        }

    update_kwargs = {
        "name": f"tool:{name}",
        "input": {"args": args},
        "output": {"result": result, "ok": ok},
        "metadata": span_metadata,
    }
    if level is not None:
        update_kwargs["level"] = level
        update_kwargs["status_message"] = f"{error_kind}: {error_message}"

    try:
        get_client().update_current_span(**update_kwargs)
    except Exception as upd_exc:
        # Don't swallow silently — surface in stdout so it shows up in Streamlit Cloud logs.
        print(f"[langfuse] update_current_span failed: {upd_exc}")

    return {
        "name": name, "input": args, "output": result, "ok": ok,
        "latency_ms": latency_ms,
        "error": ({"kind": error_kind, "message": error_message} if error_kind else None),
    }


@observe(name="chat-turn")
def run_agent(
    *,
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
    client = get_anthropic_client()
    lf = get_client()
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
    tags = [
        f"env:{ENVIRONMENT}",
        f"app:{APP_NAME}",
        f"model:{model}",
        f"tier:{user_attrs.get('customer_tier', 'unknown')}",
        f"channel:{user_attrs.get('channel', 'web')}",
        f"locale:{user_attrs.get('locale', 'unknown')}",
    ]

    # Set trace-level attributes early so they apply to all child spans.
    lf.update_current_trace(
        name="customer-support-turn",
        session_id=session_id,
        user_id=user_id,
        tags=tags,
        input={"user_message": user_message, "history_turns": len(history) // 2},
        metadata=base_metadata,
    )
    lf.update_current_span(
        input={"user_message": user_message, "history_turns": len(history) // 2},
        metadata=base_metadata,
    )
    if True:

        messages = history + [{"role": "user", "content": user_message}]
        tools_used: list[dict] = []
        tool_errors = 0
        turn_started = time.perf_counter()
        total_input_tokens = 0
        total_output_tokens = 0
        llm_calls = 0
        final_text = ""
        stop_reason = "max_iterations"

        for iteration in range(MAX_TOOL_ITERATIONS):
            response = client.messages.create(
                model=model,
                system=system_prompt,
                tools=tool_schema,
                messages=messages,
                **model_params,
            )
            llm_calls += 1
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

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
                record = _run_tool(block.name, block.input, tool_registry)
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
        lf.update_current_span(
            output={"reply": final_text, "stop_reason": stop_reason},
            metadata=full_metadata,
        )
        lf.update_current_trace(
            name="customer-support-turn",
            input={"user_message": user_message, "history_turns": len(history) // 2},
            output={"reply": final_text, "stop_reason": stop_reason},
            metadata=full_metadata,
            session_id=session_id,
            user_id=user_id,
            tags=tags,
        )
        trace_id = lf.get_current_trace_id()
        try:
            url = lf.get_trace_url(trace_id=trace_id)
        except Exception:
            url = None

    return {
        "reply": final_text,
        "tools_used": tools_used,
        "trace_id": trace_id,
        "trace_url": url,
        "latency_ms": turn_latency,
        "tokens": {"input": total_input_tokens, "output": total_output_tokens},
        "stop_reason": stop_reason,
    }


# --- Streamlit UI -----------------------------------------------------------

st.set_page_config(page_title="Langfuse Tracing Workshop", page_icon="🔎", layout="wide")
st.title("🔎 Langfuse Tracing Workshop")
st.caption("Claude + tool calling, traced into your own Langfuse account.")

ss = st.session_state
ss.setdefault("session_id", f"sess-{uuid.uuid4().hex[:12]}")
ss.setdefault("attendee_suffix", uuid.uuid4().hex[:6])
ss.setdefault("history", [])
ss.setdefault("display", [])
ss.setdefault("message_count", 0)
ss.setdefault("langfuse_ready", False)
ss.setdefault("langfuse_host", "")
ss.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)
ss.setdefault("model", AVAILABLE_MODELS[0])
ss.setdefault("temperature", 0.2)
ss.setdefault("max_tokens", 1024)
ss.setdefault("enabled_tools", [t["name"] for t in TOOL_SCHEMA])

with st.sidebar:
    st.subheader("1️⃣ Connect your Langfuse account")
    st.caption("Get keys at Langfuse → Settings → API Keys")

    host_choice = st.selectbox(
        "Langfuse region",
        ["Auto-detect", "https://cloud.langfuse.com", "https://us.cloud.langfuse.com", "Other / self-hosted"],
        index=0, key="host_choice",
        help="Auto-detect tries both EU and US Langfuse Cloud regions.",
    )
    custom_host = st.text_input("Langfuse host", value="", key="custom_host") \
        if host_choice == "Other / self-hosted" else ""

    pk = st.text_input("Public key (pk-lf-…)", type="password", key="pk_input")
    sk = st.text_input("Secret key (sk-lf-…)", type="password", key="sk_input")

    col_a, col_b = st.columns(2)
    if col_a.button("Connect", use_container_width=True):
        if not (pk and sk):
            st.error("Both keys are required.")
        else:
            ok, info = init_langfuse_session(pk.strip(), sk.strip(), host_choice, custom_host)
            if ok:
                ss.langfuse_ready = True
                ss.langfuse_host = info
                st.success(f"Connected ✅ ({info})")
            else:
                ss.langfuse_ready = False
                st.error(f"Could not connect: {info}")
    if col_b.button("Disconnect", use_container_width=True):
        ss.langfuse_ready = False
        st.rerun()

    if ss.langfuse_ready:
        st.success(f"Connected to {ss.langfuse_host}")
    else:
        st.warning("Not connected — traces will not be recorded.")

    st.divider()
    st.subheader("2️⃣ Configure the agent")
    ss.system_prompt = st.text_area("System prompt", value=ss.system_prompt, height=180)
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
    )

    st.divider()
    st.subheader("3️⃣ Your identity (logged to Langfuse)")
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
    url = turn.get("trace_url")
    if tid:
        link = f" — [open in Langfuse]({url})" if url else ""
        st.caption(f"trace `{tid}`" + link)


for turn in ss.display:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant":
            render_assistant_turn(turn)
        else:
            st.markdown(turn["text"])

prompt = st.chat_input(
    "Ask about an order, product, or discount…",
    disabled=(not ss.langfuse_ready or ss.message_count >= SESSION_MESSAGE_LIMIT),
)

if not ss.langfuse_ready:
    st.info("👈 Connect your Langfuse account in the sidebar to start chatting.")
elif ss.message_count >= SESSION_MESSAGE_LIMIT:
    st.warning(f"Per-session message limit ({SESSION_MESSAGE_LIMIT}) reached.")

if prompt:
    ss.display.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = run_agent(
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
                try:
                    get_client().flush()
                except Exception:
                    pass

        ss.message_count += 1
        ss.history.append({"role": "user", "content": prompt})
        ss.history.append({"role": "assistant", "content": result["reply"]})
        ss.display.append({
            "role": "assistant",
            "text": result["reply"],
            "tools": result["tools_used"],
            "trace_id": result["trace_id"],
            "trace_url": result.get("trace_url"),
            "latency_ms": result["latency_ms"],
            "tokens": result["tokens"],
        })
        render_assistant_turn(ss.display[-1])
