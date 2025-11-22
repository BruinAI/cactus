"""
High-level Python API for Cactus models
"""

import json
from typing import List, Dict, Optional, Callable, Any

from . import bindings


class CactusModel:
    """
    High-level interface to Cactus AI models.

    Example:
        >>> model = CactusModel("weights/lfm2-350m", context_size=2048)
        >>> response = model.complete([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(response["response"])
    """

    def __init__(self, model_path: str, context_size: int = 2048):
        """
        Initialize a Cactus model.

        Args:
            model_path: Path to the model directory containing weights and config
            context_size: Maximum context size in tokens (default: 2048)

        Raises:
            RuntimeError: If model initialization fails
        """
        self.model_path = model_path
        self.context_size = context_size
        self._model = bindings.cactus_init(model_path, context_size)

    def complete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stream_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Generate a completion with optional tool calling.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                     Roles can be "system", "user", "assistant", or "tool".
            tools: Optional list of tool definitions in OpenAI format
            max_tokens: Maximum number of tokens to generate (default: 256)
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling parameter (default: 0.9)
            top_k: Top-k sampling parameter (default: 40)
            stream_callback: Optional callback(token: str, token_id: int) for streaming

        Returns:
            Dictionary with response and optional function_calls:
            {
                "response": "text response",
                "function_calls": [  # Only present if tools were called
                    {
                        "name": "tool_name",
                        "arguments": "{...}"
                    }
                ]
            }

        Example:
            >>> tools = [{
            ...     "function": {
            ...         "name": "get_weather",
            ...         "description": "Get weather for a location",
            ...         "parameters": {
            ...             "type": "object",
            ...             "properties": {
            ...                 "location": {"type": "string"}
            ...             },
            ...             "required": ["location"]
            ...         }
            ...     }
            ... }]
            >>> response = model.complete(
            ...     messages=[{"role": "user", "content": "Weather in SF?"}],
            ...     tools=tools
            ... )
            >>> if "function_calls" in response:
            ...     print(response["function_calls"])
        """
        # Prepare messages JSON
        messages_json = json.dumps(messages)

        # Prepare options JSON
        options = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }
        options_json = json.dumps(options)

        # Prepare tools JSON if provided
        tools_json = json.dumps(tools) if tools else None

        # Call the low-level binding
        response_str = bindings.cactus_complete(
            model=self._model,
            messages_json=messages_json,
            response_buffer_size=8192,
            options_json=options_json,
            tools_json=tools_json,
            callback=stream_callback
        )

        # Parse and return response
        return json.loads(response_str)

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.

        Args:
            text: Text to embed

        Returns:
            List of embedding values

        Example:
            >>> embeddings = model.embed("Hello world")
            >>> print(len(embeddings))  # Embedding dimension
        """
        return bindings.cactus_embed(self._model, text)

    def reset(self):
        """
        Reset model state (clear KV cache).

        Useful when you want to start a fresh conversation without
        reloading the model.
        """
        bindings.cactus_reset(self._model)

    def stop(self):
        """
        Stop ongoing generation.

        Can be called from another thread to interrupt generation.
        """
        bindings.cactus_stop(self._model)

    def __del__(self):
        """Clean up model resources."""
        if hasattr(self, '_model') and self._model:
            bindings.cactus_destroy(self._model)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.__del__()

    def __repr__(self):
        return f"CactusModel(model_path='{self.model_path}', context_size={self.context_size})"


class ToolRegistry:
    """
    Helper class for managing tools and executing them.

    Example:
        >>> registry = ToolRegistry()
        >>>
        >>> @registry.register
        ... def get_weather(location: str) -> str:
        ...     '''Get weather for a location'''
        ...     return f"Weather in {location}: Sunny, 72F"
        >>>
        >>> tools = registry.get_tool_definitions()
        >>> # Use tools with model.complete(...)
    """

    def __init__(self):
        self.tools = {}

    def register(self, func: Callable) -> Callable:
        """
        Register a function as a tool.

        The function's docstring and type hints are used to generate
        the tool definition.

        Args:
            func: Function to register as a tool

        Returns:
            The same function (decorator pattern)
        """
        self.tools[func.__name__] = func
        return func

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a registered tool.

        Args:
            name: Tool name
            arguments: Tool arguments as a dictionary

        Returns:
            Tool result

        Raises:
            KeyError: If tool is not registered
        """
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not registered")

        return self.tools[name](**arguments)

    def get_tool_definitions(self) -> List[Dict]:
        """
        Get tool definitions for all registered tools.

        Returns:
            List of tool definitions in OpenAI format
        """
        # For now, return empty list
        # TODO: Auto-generate from function signatures and docstrings
        return []
