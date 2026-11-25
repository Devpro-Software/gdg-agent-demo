"""Chat loop with time, math, and book-meeting tools."""

import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in ISO format",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_meeting",
            "description": "Book a meeting with a given name and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name of the meeting"},
                    "time": {"type": "string", "description": "The current time. Use the get_current_time tool to get the current time."},
                },
                "required": ["name", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "eval_math",
            "description": "Evaluate a Python arithmetic expression. Use only digits and + - * / ** ( ). Use * for multiplication, never the letter x.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Python arithmetic, e.g. 2 + 3 * 4 or 10 * 10"},
                },
                "required": ["expression"],
            },
        },
    },
]


def math_expression(expression: str) -> str:
    """Turn a casual math string into a Python arithmetic expression."""
    expression = expression.strip()
    expression = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
    expression = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expression)
    expression = re.sub(r"\s+", "", expression)
    if not expression or not re.fullmatch(r"[0-9+\-*/().]+", expression):
        raise ValueError(expression)
    return expression


def run_tool(name: str, arguments: dict) -> str:
    """Run a tool and return the result"""
    if name == "get_current_time":
        d = datetime.now()
        return d.isoformat()
    if name == "eval_math":
        raw = arguments.get("expression", "0")
        try:
            expression = math_expression(raw)
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception:
            return "Could not evaluate that expression. Use digits and + - * / ** ( )."
    if name == "book_meeting":
        return f"Meeting booked with {arguments.get('name', 'Unknown')} at {arguments.get('time', 'Unknown')}"
    return json.dumps({"error": f"Unknown tool: {name}"})


def chat_loop():
    """Chat loop"""
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    messages = [{
        "role": "system",
        "content": "Your name is Jimmy. You are a helpful assistant that can use tools to get the current time, evaluate math expressions, and book meetings. Only output text, no special characters or parentheses.",
    }]

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue

        messages.append({"role": "user", "content": user})

        count = 0
        while True:
            count += 1
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            if len(resp.choices) == 0:
                print(f"[{count}] No response from model. Error: {resp.error}")
                break

            choice = resp.choices[0]
            msg = choice.message

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    print(f"[{count}] Running tool {name} with arguments {args}")
                    result = run_tool(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue

            if msg.content:
                print("Bot:", msg.content)
            messages.append({"role": "assistant", "content": msg.content or ""})
            break


if __name__ == "__main__":
    chat_loop()
