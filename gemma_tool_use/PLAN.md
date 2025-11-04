# Gemma 3 270M Tool Use

## Execution Steps

- Credit: [https://gemini.google.com/app/6d9c7c2d443584aa](https://gemini.google.com/app/6d9c7c2d443584aa)

### 1\. Define Function Calling Format

#### Tool Definitions

- Json list in system prompt of dict of all tools wrapped by \<tools\>, \</tools\>

#### Tool Calling

- Model provided json dict describing each tool wrapped by \<tool\_call\>, \</tool\_call?  
- NOTHING ELSE ALLOWED IN THAT PROMPT

#### Tool Responding

- Use user chat role wrapped by \<tool\_response\>, \</tool\_response\>  
- Return json Dict inside of xml element

#### Altogether

- Just copied what Qwen does for tool calling

| \<start\_of\_turn\>system \<tools\> \[   {     "name": "get\_current\_weather",     "description": "Get the current weather in a given location",     "parameters": {       "type": "OBJECT",       "properties": {         "location": {           "type": "STRING",           "description": "The city and state, e.g. San Francisco, CA"         }       },       "required": \["location"\]     }   },   {     "name": "get\_top\_news",     "description": "Get the top news headline for a given location",     "parameters": {       "type": "OBJECT",       "properties": {         "location": {           "type": "STRING",           "description": "The city and state, e.g. Boston, MA"         }       },       "required": \["location"\]     }   } \] \</tools\>\<end\_of\_turn\> \<start\_of\_turn\>user What's the weather in Boston and what's the top news headline there?\<end\_of\_turn\> \<start\_of\_turn\>model Searching up Boston’s Weather and News \<tool\_call\> {   "name": "get\_current\_weather",   "args": {     "location": "Boston, MA"   } } \</tool\_call\> \<tool\_call\> {   "name": "get\_top\_news",   "args": {     "location": "Boston, MA"   } } \</tool\_call\>\<end\_of\_turn\> \<start\_of\_turn\>user \<tool\_response\> {   "name": "get\_current\_weather",   "result": {     "temperature": 72,     "condition": "Sunny"   } } \</tool\_response\> \<tool\_response\> {   "name": "get\_top\_news",   "result": {     "headline": "Red Sox Win 9-0 against the Phillies\!",     "source": "Boston Globe"   } } \</tool\_response\>\<end\_of\_turn\> \<start\_of\_turn\>model In Boston, the weather is 72°F and the Red Sox just won 9-0 against the Phillies\!\<end\_of\_turn\>  |
| :---- |

- Credit: [https://gemini.google.com/app/7c7315d5163cbb74](https://gemini.google.com/app/7c7315d5163cbb74)

### 2\. Setup Eval with BFCL

### 3\. Find data for 1-3 tasks and SFT on that

### 4\. Find data on more tasks and SFT on that

- Ideally from same dataset as stage 2

### 5\. Make RL data setup and GRPO gemma (optional)

