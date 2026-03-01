TOOL_DEFINITIONS = [
    {
        "name": "initialize_search_space",
        "description": (
            "Define the initial hyperparameter search space. Call this on the "
            "first heartbeat after reading the model description in config.json. "
            "Each parameter should specify type, range/options, scale, and notes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "parameters": {
                    "type": "object",
                    "description": (
                        "Map of parameter names to their definitions. Each value "
                        "should have 'type' ('continuous' or 'categorical'), and "
                        "either 'min'/'max'/'scale' for continuous or 'options' "
                        "for categorical, plus a 'notes' string."
                    ),
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
                            "options": {
                                "type": "array",
                            },
                            "notes": {"type": "string"},
                        },
                        "required": ["type", "notes"],
                    },
                },
            },
            "required": ["parameters"],
        },
    },
    {
        "name": "update_search_space",
        "description": (
            "Modify the search space based on experimental results. You can add "
            "new parameters, remove parameters (set to null), or change ranges. "
            "Always provide a changelog entry explaining what changed and why."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "object",
                    "description": (
                        "Map of parameter names to updated definitions. Set a "
                        "parameter to null to remove it. Provide full definition "
                        "to add a new parameter or partial updates for existing."
                    ),
                },
                "changelog_entry": {
                    "type": "string",
                    "description": "Description of what changed and why.",
                },
            },
            "required": ["updates", "changelog_entry"],
        },
    },
    {
        "name": "launch_run",
        "description": (
            "Start a new training run with specified hyperparameters. Parameters "
            "should be within the current search space ranges. Will fail if max "
            "concurrent runs is reached."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "description": "Hyperparameter dict for the training run.",
                },
            },
            "required": ["params"],
        },
    },
    {
        "name": "update_strategy",
        "description": (
            "Write your current observations and plan to the strategy file. "
            "This is your persistent memory across heartbeats. Update it "
            "whenever you learn something new or change your plan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Markdown text for the strategy file.",
                },
            },
            "required": ["content"],
        },
    },
]
