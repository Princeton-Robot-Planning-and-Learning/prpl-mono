"""Environment object types."""


def get_object_types(env_name: str) -> tuple[str, ...]:
    """Get object types for environment."""
    if env_name == "TwoPileNim":
        return ("tpn.EMPTY", "tpn.TOKEN", "None")
    if env_name == "CheckmateTactic":
        return (
            "ct.EMPTY",
            "ct.HIGHLIGHTED_WHITE_QUEEN",
            "ct.BLACK_KING",
            "ct.HIGHLIGHTED_WHITE_KING",
            "ct.WHITE_KING",
            "ct.WHITE_QUEEN",
            "None",
        )
    if env_name == "StopTheFall":
        return (
            "stf.EMPTY",
            "stf.FALLING",
            "stf.RED",
            "stf.STATIC",
            "stf.ADVANCE",
            "stf.DRAWN",
            "None",
        )
    if env_name == "Chase":
        return (
            "ec.EMPTY",
            "ec.TARGET",
            "ec.AGENT",
            "ec.WALL",
            "ec.DRAWN",
            "ec.LEFT_ARROW",
            "ec.RIGHT_ARROW",
            "ec.UP_ARROW",
            "ec.DOWN_ARROW",
            "None",
        )
    if env_name == "ReachForTheStar":
        return (
            "rfts.EMPTY",
            "rfts.AGENT",
            "rfts.STAR",
            "rfts.DRAWN",
            "rfts.LEFT_ARROW",
            "rfts.RIGHT_ARROW",
            "None",
        )

    raise ValueError(f"Unknown environment: {env_name}")
