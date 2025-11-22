"""
Low-level ctypes bindings for Cactus C FFI
"""

import ctypes
import os
import platform
from typing import Optional, Callable

# Determine library extension based on platform
_LIB_EXT = {
    "Darwin": "dylib",
    "Linux": "so",
    "Windows": "dll"
}.get(platform.system(), "so")

# Try to find the library
def _find_library():
    """Find the Cactus shared library"""
    # Common search paths
    search_paths = [
        # Relative to this file (python/cactus/bindings.py -> cactus/cactus/build/)
        os.path.join(os.path.dirname(__file__), "..", "..", "cactus", "build", f"libcactus.{_LIB_EXT}"),
        # Relative to this file with lib subdirectory
        os.path.join(os.path.dirname(__file__), "..", "..", "cactus", "build", "lib", f"libcactus.{_LIB_EXT}"),
        # In build directory from cwd
        os.path.join(os.getcwd(), "cactus", "cactus", "build", f"libcactus.{_LIB_EXT}"),
        # In build directory with lib subdirectory
        os.path.join(os.getcwd(), "cactus", "build", "lib", f"libcactus.{_LIB_EXT}"),
        # System library path
        f"libcactus.{_LIB_EXT}",
    ]

    for path in search_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path

    raise FileNotFoundError(
        f"Could not find libcactus.{_LIB_EXT}. "
        "Make sure to build the shared library first using build.sh"
    )

# Load the library
_lib_path = _find_library()
_lib = ctypes.CDLL(_lib_path)

# Type definitions
cactus_model_t = ctypes.c_void_p
cactus_token_callback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)

# Function: cactus_init
_lib.cactus_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
_lib.cactus_init.restype = cactus_model_t

def cactus_init(model_path: str, context_size: int) -> cactus_model_t:
    """
    Initialize a Cactus model.

    Args:
        model_path: Path to the model directory
        context_size: Maximum context size in tokens

    Returns:
        Opaque pointer to the model

    Raises:
        RuntimeError: If model initialization fails
    """
    model = _lib.cactus_init(model_path.encode('utf-8'), context_size)
    if not model:
        raise RuntimeError(f"Failed to initialize model from {model_path}")
    return model

# Function: cactus_complete
_lib.cactus_complete.argtypes = [
    cactus_model_t,           # model
    ctypes.c_char_p,          # messages_json
    ctypes.c_char_p,          # response_buffer
    ctypes.c_size_t,          # buffer_size
    ctypes.c_char_p,          # options_json
    ctypes.c_char_p,          # tools_json
    cactus_token_callback,    # callback
    ctypes.c_void_p           # user_data
]
_lib.cactus_complete.restype = ctypes.c_int

def cactus_complete(
    model: cactus_model_t,
    messages_json: str,
    response_buffer_size: int = 8192,
    options_json: Optional[str] = None,
    tools_json: Optional[str] = None,
    callback: Optional[Callable[[str, int], None]] = None
) -> str:
    """
    Generate a completion with optional tool calling.

    Args:
        model: Model pointer from cactus_init
        messages_json: JSON string of chat messages
        response_buffer_size: Size of response buffer (default 8192)
        options_json: Optional JSON string with generation options
        tools_json: Optional JSON string with tool definitions
        callback: Optional callback function for streaming tokens

    Returns:
        JSON string with response and optional function calls

    Raises:
        RuntimeError: If completion fails
    """
    # Create response buffer
    response_buffer = ctypes.create_string_buffer(response_buffer_size)

    # Convert Python callback to C callback if provided
    c_callback = None
    if callback:
        def wrapper(token: bytes, token_id: int, user_data):
            callback(token.decode('utf-8'), token_id)
        c_callback = cactus_token_callback(wrapper)

    # Call the C function
    # Note: Use cast(None, callback_type) to pass NULL pointer when callback is not provided
    result = _lib.cactus_complete(
        model,
        messages_json.encode('utf-8'),
        response_buffer,
        response_buffer_size,
        options_json.encode('utf-8') if options_json else None,
        tools_json.encode('utf-8') if tools_json else None,
        c_callback if c_callback else ctypes.cast(None, cactus_token_callback),
        None  # user_data
    )

    # C++ returns result > 0 for success, <= 0 for failure
    if result <= 0:
        raise RuntimeError(f"cactus_complete failed with error code {result}")

    return response_buffer.value.decode('utf-8')

# Function: cactus_embed
_lib.cactus_embed.argtypes = [
    cactus_model_t,           # model
    ctypes.c_char_p,          # text
    ctypes.POINTER(ctypes.c_float),  # embeddings_buffer
    ctypes.c_size_t,          # buffer_size
    ctypes.POINTER(ctypes.c_size_t)  # embedding_dim
]
_lib.cactus_embed.restype = ctypes.c_int

def cactus_embed(model: cactus_model_t, text: str, max_dim: int = 4096) -> list:
    """
    Generate embeddings for text.

    Args:
        model: Model pointer from cactus_init
        text: Text to embed
        max_dim: Maximum embedding dimension (default 4096)

    Returns:
        List of embedding values

    Raises:
        RuntimeError: If embedding generation fails
    """
    # Create buffer for embeddings
    embeddings_buffer = (ctypes.c_float * max_dim)()
    embedding_dim = ctypes.c_size_t()

    result = _lib.cactus_embed(
        model,
        text.encode('utf-8'),
        embeddings_buffer,
        max_dim,
        ctypes.byref(embedding_dim)
    )

    # C++ returns result > 0 for success, <= 0 for failure
    if result <= 0:
        raise RuntimeError(f"cactus_embed failed with error code {result}")

    # Convert to Python list
    return list(embeddings_buffer[:embedding_dim.value])

# Function: cactus_reset
_lib.cactus_reset.argtypes = [cactus_model_t]
_lib.cactus_reset.restype = None

def cactus_reset(model: cactus_model_t):
    """
    Reset model state (clear KV cache).

    Args:
        model: Model pointer from cactus_init
    """
    _lib.cactus_reset(model)

# Function: cactus_stop
_lib.cactus_stop.argtypes = [cactus_model_t]
_lib.cactus_stop.restype = None

def cactus_stop(model: cactus_model_t):
    """
    Stop ongoing generation.

    Args:
        model: Model pointer from cactus_init
    """
    _lib.cactus_stop(model)

# Function: cactus_destroy
_lib.cactus_destroy.argtypes = [cactus_model_t]
_lib.cactus_destroy.restype = None

def cactus_destroy(model: cactus_model_t):
    """
    Free model resources.

    Args:
        model: Model pointer from cactus_init
    """
    if model:
        _lib.cactus_destroy(model)
