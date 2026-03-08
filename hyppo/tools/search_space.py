from hyppo.state import WorkspaceState, now_iso


def execute_initialize_search_space(parameters: dict, state: WorkspaceState) -> dict:
    if state.search_space_exists():
        return {"error": "Search space already initialized. Use update_search_space."}

    search_space = {
        "version": 1,
        "created_at": now_iso(),
        "last_updated": now_iso(),
        "parameters": parameters,
        "changelog": [
            {
                "version": 1,
                "timestamp": now_iso(),
                "description": f"Initial search space: {', '.join(parameters.keys())}",
            }
        ],
    }
    state.write_search_space(search_space)
    return {"status": "created", "parameter_count": len(parameters)}


def execute_update_search_space(
    updates: dict, changelog_entry: str, state: WorkspaceState
) -> dict:
    space = state.read_search_space()
    if not space:
        return {"error": "No search space exists. Use initialize_search_space first."}

    for param, value in updates.items():
        if value is None:
            space["parameters"].pop(param, None)
        elif param in space["parameters"]:
            space["parameters"][param].update(value)
        else:
            space["parameters"][param] = value

    space["version"] += 1
    space["last_updated"] = now_iso()
    space["changelog"].append(
        {
            "version": space["version"],
            "timestamp": now_iso(),
            "description": changelog_entry,
        }
    )
    state.write_search_space(space)

    return {
        "status": "updated",
        "version": space["version"],
        "parameter_count": len(space["parameters"]),
    }
