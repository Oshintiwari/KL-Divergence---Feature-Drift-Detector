import os
import json
import anthropic
from tasks.data_drift_detection import grader

def run_drift_task():
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # read the problem prompt
    with open("tasks/data_drift_detection/drift_prompt.md", "r", encoding="utf-8") as f:
        prompt = f.read()

    print("=== Sending drift detection task to Claude ===\n")

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1500,
        temperature=0,
        system="You are a helpful AI assistant who writes clean, production-quality Python code for ML engineers.",
        messages=[
            {"role": "user", "content": prompt}
        ],
        tools=[
            {
                "name": "python_expression",
                "description": "Write the Python code implementing detect_drift() in detect_drift.py",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Python code string for detect_drift()"
                        }
                    },
                    "required": ["expression"],
                },
            }
        ],
    )

    # Extract model text content (compatible with new Anthropic SDK)
    text_parts = []
    for block in message.content:
        # In the latest SDK, message.content is a list of TextBlock objects
        if hasattr(block, "type") and block.type == "text":
            text_parts.append(block.text)
        # Backward compatibility for older SDK versions (dict-based)
        elif isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))

    full_text = "\n".join(text_parts)


    # extract code between ```python ... ```
    import re
    code_blocks = re.findall(r"```python(.*?)```", full_text, re.DOTALL)
    if not code_blocks:
        print("⚠️ No Python code found in model response.")
        return

    # write first code block to detect_drift.py
    code = code_blocks[0].strip()
    with open("tasks/data_drift_detection/detect_drift.py", "w", encoding="utf-8") as f:
        f.write(code)

    print("✅ detect_drift.py updated. Running grader...\n")

    result = grader.grade()
    print("=== Grader Result ===")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_drift_task()
