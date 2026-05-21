# Langfuse Tracing Workshop — Claude + Tool Calling

Interactive Streamlit app for workshops. Host deploys it once on Streamlit
Cloud; each attendee opens the same URL, pastes their **own** Langfuse keys
in the sidebar, and chats with the agent. Their traces land in their own
Langfuse account, where they can build LLM-as-a-judge evaluators against
real data they just produced.

## What attendees can do in the UI

- Connect their own Langfuse account (region + public + secret key) with a one-click test.
- Edit the **system prompt** and watch how the agent's behaviour changes.
- Switch **Claude model** and **temperature**.
- Toggle which **tools** the agent has access to.
- Set their **user_id / customer tier / locale / channel** — all logged to Langfuse for filtering.
- Click straight from each reply into the corresponding trace in Langfuse.

## What's logged per turn (stable JSON paths)

```
trace.input  = {"user_message": str, "history_turns": int}
trace.output = {"reply": str, "stop_reason": str}
trace.metadata = {
  app:      {name, version, environment},
  session:  {id},
  user:     {id, tier, locale, channel},
  request:  {id},
  model:    {name, temperature, max_tokens},
  system_prompt: str,
  summary:  {llm_calls, tool_calls, tool_errors,
             input_tokens, output_tokens, latency_ms, stopped_because},
  tools_used: [{name, input, output, ok, latency_ms}, ...],
}
```

Each LLM call is a child generation; each tool execution is its own span,
also with stable input/output/metadata shapes.

## Local setup (for the host)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY (only)
streamlit run app.py
```

Attendees do **not** need a `.env` — they enter their Langfuse keys directly
in the running app's sidebar.

## Streamlit Cloud deployment (host workflow)

1. Push this repo to GitHub.
2. Create a new app at https://share.streamlit.io pointing at `app.py`.
3. In **Settings → Secrets**, add only the host-supplied key:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

4. Share the deployed URL with attendees.

There is a per-session message cap (default 25 in `app.py`) to keep your
Anthropic spend bounded across the workshop.

## Suggested workshop flow

1. Attendees open the app URL.
2. Sign up / log in to https://cloud.langfuse.com (free tier).
3. Create a project → Settings → API Keys → copy public + secret keys.
4. Paste keys into the app sidebar → click **Connect** (green check = ready).
5. Chat with the agent (try the suggested prompts). Click "open in Langfuse"
   to inspect each trace.
6. In Langfuse: **Evaluation → New evaluator** to add an LLM-as-a-judge.
   The stable JSON paths above (e.g. `$.input.user_message`, `$.output.reply`,
   `$.metadata.summary.tool_calls`) make selectors easy to write.
7. Re-run chats to see scores attach automatically.

## Tools

See `tools.py` for the mock data (orders, products, discount math). Edit it
to fit your audience's domain.
