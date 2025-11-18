"""
Possible parameter values for each tool and argument.

This dictionary defines the possible values for each parameter of each tool.
- For discrete parameters with a limited set of values, lists the possible values (including None/null)
- For free-form text parameters, uses "free-text" to indicate any generated text is valid
"""

import random
from typing import Dict, Any, Optional


# Persona variations for text generation diversity
PERSONAS = [
    "a busy parent managing family schedules",
    "a college student tracking assignments and deadlines",
    "a working professional organizing their day",
    "a freelancer juggling multiple projects",
    "someone planning personal errands and tasks",
    "a health-conscious person tracking habits",
    "someone managing household chores and maintenance",
    "a person coordinating social events and meetups",
    "someone with a packed daily routine",
    "a casual user making quick personal notes"
]

# Natural length variations for implicit length control
LENGTH_INSTRUCTIONS = [
    "Write a quick note (3-5 words)",
    "Write a brief message (1 short sentence)",
    "Write 1-2 basic sentences",
]

# Language/tone variations for instruction diversity
TONE_STYLES = [
    "professional",
    "casual",
    "slang",
    "abbreviated - minimal/skipped words (like a text message)",
]


def _generate_valid_dates() -> list[str]:
    """
    Generate all valid MM-DD date strings for a non-leap year.

    Returns:
        List of date strings in MM-DD format
    """
    days_in_month = {
        1: 31,   # January
        2: 28,   # February (non-leap year)
        3: 31,   # March
        4: 30,   # April
        5: 31,   # May
        6: 30,   # June
        7: 31,   # July
        8: 31,   # August
        9: 30,   # September
        10: 31,  # October
        11: 30,  # November
        12: 31   # December
    }

    dates = []
    for month, max_day in days_in_month.items():
        for day in range(1, max_day + 1):
            dates.append(f"{month:02d}-{day:02d}")

    return dates

possible_params = {
    "create_note": {
        "text": "free-text"  # Any user message/quote
    },

    "set_alarm": {
        "time_hours": list(range(0, 24)),  # 0-23 (24-hour format)
        "time_minutes": list(range(0, 60))  # 0-59
    },

    "set_timer_absolute": {
        "day_offset": [
            "today",
            "tomorrow",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "1", "2", "3", "4", "5", "6", "7",  # Days from now (as strings)
        ],
        "time_hours": list(range(0, 24)),  # 0-23 (24-hour format)
        "time_minutes": list(range(0, 60))  # 0-59
    },

    "set_timer": {
        "time_hours": list(range(0, 24)),  # 0-23 hours
        "time_minutes": list(range(0, 60)),  # 0-59 minutes
        "time_seconds": list(range(0, 60))  # 0-59 seconds
    },

    "reminder_absolute": {
        "day_offset": [
            "today",
            "tomorrow",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "1", "2", "3", "4", "5", "6", "7",  # Days from now (as strings)
            None  # Can be null if date_month_day is specified
        ],
        "absolute_time_hour": list(range(0, 24)),  # 0-23 (24-hour format)
        "absolute_time_minute": list(range(0, 60)),  # 0-59
        "date_month_day": [None] + _generate_valid_dates(),  # None if using day_offset, or any valid MM-DD date
        "date_year": [
            None,  # Can be null
            2025, 2026, 2027, 2030  # Reasonable year range
        ],
        "message": "free-text"  # Any reminder message
    },

    "create_reminder_relative": {
        "relative_time": list(range(1, 20)),  # 1-20 (reasonable range)
        "time_unit": [
            "seconds",
            "minutes",
            "hours",
            "days",
            "weeks",
            "months",
            "years"
        ],
        "message": "free-text"  # Any reminder message
    }
}


def get_param_values(tool_name: str, param_name: str):
    """
    Get possible values for a specific parameter of a tool.

    Args:
        tool_name: The name of the tool
        param_name: The name of the parameter

    Returns:
        List of possible values, "free-text" string, or None if tool/param not found
    """
    if tool_name not in possible_params:
        return None
    if param_name not in possible_params[tool_name]:
        return None
    return possible_params[tool_name][param_name]


def is_free_text(tool_name: str, param_name: str) -> bool:
    """
    Check if a parameter accepts free-form text.

    Args:
        tool_name: The name of the tool
        param_name: The name of the parameter

    Returns:
        True if the parameter is free-text, False otherwise
    """
    values = get_param_values(tool_name, param_name)
    return values == "free-text"


def get_all_tools():
    """Get list of all tool names."""
    return list(possible_params.keys())


def validate_tool_parameters(tool_name: str, params: Dict[str, Any]) -> bool:
    """
    Validate that sampled parameters meet tool-specific constraints.

    Args:
        tool_name: The name of the tool
        params: Dictionary of parameter values (may include metadata fields like _persona)

    Returns:
        True if parameters are valid, False otherwise

    Tool-specific validation rules:
        - reminder_absolute: Either day_offset OR (date_month_day AND date_year) must be non-null

    Note: Metadata fields (prefixed with _) are ignored during validation.
    """
    if tool_name == "reminder_absolute":
        # Either day_offset is set, OR both date_month_day and date_year are set
        day_offset = params.get("day_offset")
        date_month_day = params.get("date_month_day")
        date_year = params.get("date_year")

        # Valid if day_offset is not None, OR both date fields are not None
        has_day_offset = day_offset is not None
        has_full_date = (date_month_day is not None) and (date_year is not None)

        return has_day_offset or has_full_date

    # All other tools have no special validation
    return True


def sample_tool_parameters(
    tool_name: str,
    seed: Optional[int] = None,
    validate: bool = True,
    max_attempts: int = 100
) -> Dict[str, Any]:
    """
    Randomly sample parameter values for a given tool.

    For discrete parameters, randomly selects from possible values.
    For free-text parameters, returns the string "free-text" as a placeholder.
    Also includes "_persona", "_length_instruction", and "_tone_style" metadata for text generation.

    Args:
        tool_name: The name of the tool to sample parameters for
        seed: Optional random seed for reproducibility
        validate: If True, ensure parameters pass tool-specific validation
        max_attempts: Maximum sampling attempts when validation is enabled

    Returns:
        Dictionary mapping parameter names to sampled values.
        Free-text parameters will have the value "free-text".
        Also includes "_persona", "_length_instruction", and "_tone_style" keys for generation metadata.

    Raises:
        ValueError: If tool_name is not found in possible_params
        RuntimeError: If unable to generate valid parameters after max_attempts

    Example:
        >>> sample_tool_parameters("set_alarm", seed=42)
        {'time_hours': 15, 'time_minutes': 32, '_persona': '...', '_length_instruction': '...', '_tone_style': '...'}

        >>> sample_tool_parameters("create_note")
        {'text': 'free-text', '_persona': '...', '_length_instruction': '...', '_tone_style': '...'}

        >>> # reminder_absolute will always have valid date constraints
        >>> params = sample_tool_parameters("reminder_absolute")
        >>> # Either day_offset is set, OR both date_month_day and date_year are set
    """
    if tool_name not in possible_params:
        raise ValueError(f"Tool '{tool_name}' not found in possible_params")

    if seed is not None:
        random.seed(seed)

    for attempt in range(max_attempts):
        sampled_params = {}

        for param_name, param_values in possible_params[tool_name].items():
            if param_values == "free-text":
                # Return placeholder for free-text parameters
                sampled_params[param_name] = "free-text"
            else:
                # Randomly select from discrete values
                sampled_params[param_name] = random.choice(param_values)

        # Add generation metadata (persona, length, and tone)
        sampled_params["_persona"] = random.choice(PERSONAS)
        sampled_params["_length_instruction"] = random.choice(LENGTH_INSTRUCTIONS)
        sampled_params["_tone_style"] = random.choice(TONE_STYLES)

        # Check if validation is needed and passes
        if not validate or validate_tool_parameters(tool_name, sampled_params):
            return sampled_params

    # If we get here, validation failed after max_attempts
    raise RuntimeError(
        f"Failed to generate valid parameters for '{tool_name}' "
        f"after {max_attempts} attempts"
    )


def sample_multiple_tools(
    tool_name: str,
    count: int,
    seed: Optional[int] = None,
    validate: bool = True
) -> list[Dict[str, Any]]:
    """
    Generate multiple random parameter samples for a tool.

    Args:
        tool_name: The name of the tool to sample parameters for
        count: Number of samples to generate
        seed: Optional random seed for reproducibility
        validate: If True, ensure all parameters pass tool-specific validation

    Returns:
        List of dictionaries, each containing sampled parameters

    Example:
        >>> samples = sample_multiple_tools("set_alarm", 3, seed=42)
        >>> len(samples)
        3

        >>> # All reminder_absolute samples will have valid date constraints
        >>> samples = sample_multiple_tools("reminder_absolute", 10)
        >>> all(validate_tool_parameters("reminder_absolute", s) for s in samples)
        True
    """
    if seed is not None:
        random.seed(seed)

    samples = []
    for _ in range(count):
        samples.append(sample_tool_parameters(tool_name, seed=None, validate=validate))

    return samples


if __name__ == "__main__":
    # Example usage
    print("All tools:", get_all_tools())
    print("\nExample parameter values:")
    print(f"set_alarm.time_hours: {get_param_values('set_alarm', 'time_hours')[:5]}... (truncated)")
    print(f"create_note.text is free-text: {is_free_text('create_note', 'text')}")
    print(f"\ncreate_reminder_relative.time_unit: {get_param_values('create_reminder_relative', 'time_unit')}")

    print("\n" + "="*60)
    print("Random parameter sampling examples:")
    print("="*60)

    # Sample each tool
    for tool in get_all_tools():
        print(f"\n{tool}:")
        sample = sample_tool_parameters(tool)
        for param, value in sample.items():
            print(f"  {param}: {value}")

    print("\n" + "="*60)
    print("Multiple samples for set_alarm:")
    print("="*60)
    samples = sample_multiple_tools("set_alarm", 5, seed=42)
    for i, sample in enumerate(samples, 1):
        print(f"{i}. {sample}")

    print("\n" + "="*60)
    print("Validation test for reminder_absolute:")
    print("="*60)
    print("Generating 10 samples with validation enabled...")
    reminder_samples = sample_multiple_tools("reminder_absolute", 10)

    valid_count = 0
    for i, sample in enumerate(reminder_samples, 1):
        is_valid = validate_tool_parameters("reminder_absolute", sample)
        valid_count += is_valid

        # Show validation details
        day_offset = sample.get("day_offset")
        date_month_day = sample.get("date_month_day")
        date_year = sample.get("date_year")

        has_day = day_offset is not None
        has_full_date = (date_month_day is not None) and (date_year is not None)

        print(f"\n{i}. Valid: {is_valid}")
        print(f"   day_offset: {day_offset} (set: {has_day})")
        print(f"   date_month_day: {date_month_day}, date_year: {date_year} (both set: {has_full_date})")
        print(f"   → Constraint satisfied: {has_day or has_full_date}")

    print(f"\n✓ All {valid_count}/{len(reminder_samples)} samples are valid!")
