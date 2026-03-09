TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "initialize_search_space",
            "description": (
                "Define the initial hyperparameter search space on the first heartbeat. "
                "Each parameter should include type, range or options, scale when relevant, and notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parameters": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["continuous", "categorical"],
                                },
                                "min": {"type": "number"},
                                "max": {"type": "number"},
                                "scale": {
                                    "type": "string",
                                    "enum": ["log", "linear"],
                                },
                                "options": {"type": "array"},
                                "notes": {"type": "string"},
                            },
                            "required": ["type", "notes"],
                        },
                    }
                },
                "required": ["parameters"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_search_space",
            "description": (
                "Modify the current search space. You can narrow ranges, add parameters, "
                "or remove parameters by setting them to null."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "updates": {"type": "object"},
                    "changelog_entry": {"type": "string"},
                },
                "required": ["updates", "changelog_entry"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "launch_run",
            "description": (
                "Start a new training run with hyperparameters that fit inside the current search space."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "params": {"type": "object"},
                },
                "required": ["params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_strategy",
            "description": (
                "Persist observations and next steps to the strategy file. Start the content "
                "with a single-line `Insight: ...` summary of what changed this heartbeat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                },
                "required": ["content"],
            },
        },
    },
]
