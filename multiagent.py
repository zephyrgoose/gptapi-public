"""
High-Level LLM Multi-Agent Architecture

Flow Sequence:

1. **CLI Entry**
   - User runs `multiagent.py --schedule <path> "<prompt>"`.

2. **Schedule Load**
   - `load_schedule()` reads YAML: lists of worker agents, meta agent, and limits.

3. **Profile Initialization**
   - For each agent name (workers + meta), `load_profile()` loads model, prompts, params, schema from `profiles/`.

4. **Context Setup**
   - `messages` ← [{role: user, content: prompt}].
   - `conversation_log`, `meta_log`, and counters set to zero.

5. **Main Loop (until max_rounds or max_meta_iterations):**
   a) **Worker Pass** (for each worker in order):
      i. `call_structured()` sends full `messages` to model with function schema → blocks until LLM returns.
      ii. Parse JSON output ({goals, reasoning, output}).
      iii. Append to `messages` (assistant), `conversation_log` and increment agent count.

   b) **Meta Decision**:
      i. `call_structured()` on meta profile with updated `messages` → returns `{next_action, rationale, final_output_context}`.
      ii. Append to `meta_log` and increment meta count.
      iii. Evaluate `next_action`:
         - **output_final_result**: break loop.
         - **question_user**:
             • Display `final_output_context` to user.
             • `input()` user reply, append as user message and to `conversation_log`.
             • **Continue** loop without incrementing round counter.
         - **work** (or unspecified): increment `round_count` and `meta_count`, repeat.

6. **Metrics Compilation**
   - Build `metrics`: rounds_executed, agent_message_counts, total_messages.
   
7. **Result Output**
   - Print JSON: `{conversation_log, meta_log, metrics}`.

Key Architecture Points:
- **Function-Call Schema**: Enforces strict JSON per-agent via OpenAI function-calling.
- **Separation of Concerns**: Workers handle domain reasoning; meta coordinates control flow.
- **Dynamic User Interaction**: Meta can interrupt for clarifications (`question_user`).
- **Configurable Pipeline**: Agents and limits defined externally in YAML.
- **Traceable Execution**: Detailed DEBUG logs and timing metrics for each call.
- **Extensibility**: Add, reorder, or replace agents by editing schedule.

Third-Party Components:
- **profiles/*.yaml**: Agent definitions (system_prompt, schema).
- **gptapi_core.py**: Core wrapper for loading profiles and invoking LLM via function calls.
"""
# multiagent.py
# Written by Dan W + GPT4

import argparse
import os
import json
import yaml
import logging
import time
from gptapi_core import load_profile, call_structured

def load_schedule(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True,
                help="Path to schedule YAML (e.g. schedules/example-schema.yaml)")
    ap.add_argument("prompt", help="User prompt (quoted)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.debug(f"Starting multiagent with schedule={args.schedule} and prompt='{args.prompt}'")

    try:
        sched = load_schedule(args.schedule)
        logging.debug(f"Loaded schedule: {sched}")
    except Exception as e:
        logging.error(f"Failed to load schedule: {e}")
        return

    BASE = os.path.join(os.path.dirname(__file__), "profiles")

    workers = sched.get("workers", [])
    meta = sched.get("meta")
    limits = sched.get("limits", {})
    max_r = limits.get("max_rounds", 6)
    max_mi = limits.get("max_meta_iterations", 8)
    logging.debug(f"Workers: {workers}, Meta: {meta}, max_rounds: {max_r}, max_meta_iterations: {max_mi}")

    profiles = {}
    for name in workers + [meta]:
        try:
            profiles[name] = load_profile(name, BASE)
            logging.debug(f"Loaded profile '{name}'")
        except Exception as e:
            logging.error(f"Failed to load profile '{name}': {e}")
            return

    messages = [{"role": "user", "content": args.prompt}]
    conversation = []
    meta_log = []

    agent_counts = {name: 0 for name in workers + [meta]}
    rounds_executed = 0

    round_count = 0
    meta_count = 0

    while round_count < max_r and meta_count < max_mi:
        logging.debug(f"Starting round {round_count + 1}")

        # worker pass
        for w in workers:
            logging.debug(f"Calling worker '{w}' with {len(messages)} messages in context")
            start = time.time()
            try:
                out = call_structured(profiles[w], messages)
            except Exception as e:
                logging.exception(f"Error in call_structured for '{w}': {e}")
                return
            duration = time.time() - start
            logging.debug(f"Worker '{w}' returned in {duration:.2f}s: {out}")

            msg_content = json.dumps(out, separators=",:")
            messages.append({"role": "assistant", "name": w, "content": msg_content})
            conversation.append({"agent": w, "message": out})
            agent_counts[w] += 1

        logging.debug(f"Calling meta '{meta}' for decision")
        start = time.time()
        try:
            meta_out = call_structured(profiles[meta], messages)
        except Exception as e:
            logging.exception(f"Error in call_structured for meta '{meta}': {e}")
            return

        # Log and increment
        duration = time.time() - start
        logging.debug(f"Meta '{meta}' returned in {duration:.2f}s: {meta_out}")

        meta_log.append(meta_out)
        agent_counts[meta] += 1

        action = meta_out.get("next_action")
        logging.debug(f"Meta next_action: {action}")

        # ----- Decision handling: Tool invocation and other actions -----
        if action == "output_final_result":
            logging.debug("Termination requested by meta")
            break
        elif action == "question_user":
            question = meta_out.get("final_output_context", "").strip()
            user_reply = input(question + "\n> ")
            messages.append({"role": "user", "content": user_reply})
            conversation.append({"agent": "user", "message": user_reply})
            continue
        elif action == "web_search":
            # ------------------------------------------------------------
            # 1)  `final_output_context` from meta is the query string.
            # 2)  Pass it straight to run_web_search()
            # 3)  Feed the returned web info back into the conversation.
            # ------------------------------------------------------------
            query_prompt = meta_out.get("final_output_context", "").strip()
            if not query_prompt:
                logging.error("Meta requested web_search but supplied no query.")
                break

            try:
                from gptapi_core import run_web_search
                web_info = run_web_search(query_prompt)
            except Exception as e:
                logging.exception(f"web_search tool call failed: {e}")
                web_info = input("Web-search failed; paste info manually:\n> ").strip()

            info_agent = "web_info"
            messages.append({
                "role": "assistant",
                "name": info_agent,
                "content": web_info
            })
            conversation.append({"agent": info_agent, "message": web_info})
            agent_counts.setdefault(info_agent, 0)
            agent_counts[info_agent] += 1

            continue

        round_count += 1
        meta_count += 1
        rounds_executed += 1

    metrics = {
        "rounds_executed": rounds_executed,
        "agent_message_counts": agent_counts,
        "total_messages": sum(agent_counts.values())
    }
    logging.debug(f"Metrics: {metrics}")

    result = {
        "conversation_log": conversation,
        "meta_log": meta_log,
        "metrics": metrics
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
