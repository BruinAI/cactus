# Gemma 3 270M Tool Use

## Execution Steps

- Credit: [https://gemini.google.com/app/6d9c7c2d443584aa](https://gemini.google.com/app/6d9c7c2d443584aa)

### 1\. Define Function Calling Format

**IMPORTANT:** Gemma 3 only supports `user` and `model` roles. The `system` role is NOT supported.
System instructions and tool definitions must be prepended to the first user message.

Reference: [Gemma 3 Formatting Guide](https://ai.google.dev/gemma/docs/formatting)

#### Tool Definitions

- Simple instruction: `"Available tools:\n"`
- Followed by JSON list of all tools wrapped by `<tools>`, `</tools>`
- Prepended to the **first user message** (NOT in a separate system turn)
- Uses Qwen-style XML tags for compatibility

#### Tool Calling

- Model provides JSON dict describing each tool wrapped by `<tool_call>`, `</tool_call>`
- Format: `{"name": "<function-name>", "args": {...}}`
- Can generate multiple tool calls in sequence
- May include reasoning text before/after tool calls

#### Tool Responding

- Use `user` chat role with results wrapped by `<tool_response>`, `</tool_response>`
- Format: `{"name": "<function-name>", "result": {...}}`
- Return JSON dict inside XML element

#### Complete Example

```
<bos><start_of_turn>user
Here are the available tools that you can use:
<tools>
[
  {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "The city and state, e.g. San Francisco, CA"
        }
      },
      "required": ["location"]
    }
  },
  {
    "name": "get_top_news",
    "description": "Get the top news headline for a given location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "The city and state, e.g. Boston, MA"
        }
      },
      "required": ["location"]
    }
  }
]
</tools>

What's the weather in Boston and what's the top news headline there?<end_of_turn>
<start_of_turn>model
Searching up Boston's Weather and News
<tool_call>
{
  "name": "get_current_weather",
  "args": {
    "location": "Boston, MA"
  }
}
</tool_call>
<tool_call>
{
  "name": "get_top_news",
  "args": {
    "location": "Boston, MA"
  }
}
</tool_call><end_of_turn>
<start_of_turn>user
<tool_response>
{
  "name": "get_current_weather",
  "result": {
    "temperature": 72,
    "condition": "Sunny"
  }
}
</tool_response>
<tool_response>
{
  "name": "get_top_news",
  "result": {
    "headline": "Red Sox Win 9-0 against the Phillies!",
    "source": "Boston Globe"
  }
}
</tool_response><end_of_turn>
<start_of_turn>model
In Boston, the weather is 72°F and the Red Sox just won 9-0 against the Phillies!<end_of_turn>
```

#### With System Instructions

If a system prompt is provided (e.g., from BFCL test cases), prepend it to the first user message:

```
<bos><start_of_turn>user
You are a helpful weather assistant.

Available tools:
<tools>
[...]
</tools>

What's the weather in Paris?<end_of_turn>
<start_of_turn>model
<tool_call>
{
  "name": "get_current_weather",
  "args": {"location": "Paris, France"}
}
</tool_call><end_of_turn>
```

**Key Points:**
- System instructions come FIRST in the user message (if present)
- Then "Available tools:" instruction
- Then tools definition in `<tools>` tags
- Then actual user query
- All within a single `<start_of_turn>user` ... `<end_of_turn>` block

### 2\. Setup Eval with BFCL

### 3\. Find data for 1-3 tasks and SFT on that

### 4\. Find data on more tasks and SFT on that

- Ideally from same dataset as stage 2

### 5\. Make RL data setup and GRPO gemma (optional)

