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
- Format: `{"name":"<function-name>","args":{...}}` (compact JSON, no spaces after `:` or `,`)
- No newlines inside the `<tool_call>` tags - only newline after `</tool_call>`
- Can generate multiple tool calls in sequence
- May include reasoning text before/after tool calls

**Important Formatting Rules:**
- Use compact JSON with `separators=(',', ':')` - no extra whitespace
- Opening tag format: `<tool_call>` (no newline after)
- Closing tag format: `</tool_call>\n` (newline after closing tag only)
- This reduces token usage and eliminates ambiguity for the model

**Implementation Note:**
In Python, use `json.dumps(call_data, separators=(',', ':'))` to generate compact JSON.
Do NOT use default `json.dumps()` as it adds spaces after `:` and `,` which:
1. Increases token count unnecessarily
2. Creates ambiguity for the model (should it add spaces or not?)
3. Makes the format harder to learn consistently

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
<tool_call>{"name":"get_current_weather","args":{"location":"Boston, MA"}}</tool_call>
<tool_call>{"name":"get_top_news","args":{"location":"Boston, MA"}}</tool_call>
<end_of_turn>
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
<tool_call>{"name":"get_current_weather","args":{"location":"Paris, France"}}</tool_call>
<end_of_turn>
```

**Key Points:**
- System instructions come FIRST in the user message (if present)
- Then "Available tools:" instruction
- Then tools definition in `<tools>` tags
- Then actual user query
- All within a single `<start_of_turn>user` ... `<end_of_turn>` block

### 2\. Setup Eval with BFCL

**BFCL Simple Test Characteristics**

Analyzed the BFCL simple test suite to understand baseline evaluation requirements:

**Test Counts:**
- Simple Python: 400 tests
- Simple Java: 100 tests
- Simple JavaScript: 50 tests

**Key Characteristics:**
- **Exactly 1 function available** per test (no distractors, no tool selection complexity)
- **Single-turn only** (no multi-step conversations)
- **Pure function calling** - tests the model's ability to generate correct function calls with proper arguments
- **Zero tool selection complexity** - the model only needs to decide whether to call the function and format the call correctly

Based on `toucan_analysis_phase1.csv` analysis, the Toucan SFT subset provides excellent training progressions:

### 3\. Initial SFT: Limited Tool Vocabulary, Simple Tasks

**Dataset**: [Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M)

**Approach**: Start with a reduced tool vocabulary to minimize selection complexity
- Use the **SFT subset** (119k rows) as the primary source - it's pre-filtered for high quality (≥4-5 on Likert scales)
- Further filter to **single-turn-original** examples only (simpler, core pipeline output)
- **Key simplification: Limit to ~50-100 distinct tools** with non-overlapping functionality
  - Select tools with simple schemas (few parameters, clear types)
  - Ensure tools have distinct purposes (no overlapping functionality)
  - Include tasks requiring 1-3 tool calls to teach sequencing
- Focus on **sequential tool calls** (avoid parallel execution complexity initially)
- Filter for high quality scores and complete tool utilization
- Target: ~10-30k examples for this stage

**Rationale**: Research shows "fewer tools equals less confusion" - the main challenge is tool selection from 2,000+ options, not the number of calls per task. Starting with a limited, non-overlapping tool vocabulary builds selection competence while still teaching multi-tool coordination. Tool diversity within the limited set improves zero-shot capability (ToolACE, 2024).

**Alternative approach**: If tool vocabulary limiting proves difficult, fall back to filtering for tasks requiring only 1-2 tools maximum while keeping full tool vocabulary.

### 4\. Extended SFT: Multi-turn, Complex Tool Use

**Dataset**: [Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M)

**Approach**: Expand to full agentic capabilities using curriculum learning
- Start with remaining **SFT subset** data (all 119k examples)
- Progressively add complexity in this order (following paper's ablation study):
  1. **Single-turn with irrelevance** (40k from Ext.1) - teaches tool selection with distractors
  2. **Diversified single-turn** (15.8k from Ext.2) - same target tools, varied queries
  3. **Multi-turn conversations** (35.2k from Ext.3) - complex multi-step interactions
- Include **parallel tool execution** examples (~20% of dataset)
- No limit on number of tools per task
- Target: Full SFT subset (119k) + potentially sample from larger subsets (Kimi-K2: 519k, Qwen3: 552k, OSS: 457k)

**Rationale**: Progressive complexity matches the dataset's design. Multi-turn and parallel execution represent real-world agentic behavior. The ablation study shows sequential addition of extensions improves performance.

**Note**: All 1.5M trajectories span 495 real-world MCPs with 2,000+ tools. Can sample from full subsets if model capacity allows.

### 5\. Make RL data setup and GRPO gemma (optional, future)
