import argparse
import json


def _strip_json_comments(text):
    result = []
    in_string = False
    escape = False
    in_line_comment = False
    in_block_comment = False
    i = 0

    while i < len(text):
        char = text[i]
        next_char = text[i + 1] if i + 1 < len(text) else ""

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                result.append(char)
            i += 1
            continue

        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                i += 2
                continue
            if char == "\n":
                result.append(char)
            i += 1
            continue

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue

        if char == "/" and next_char == "/":
            in_line_comment = True
            i += 2
            continue

        if char == "/" and next_char == "*":
            in_block_comment = True
            i += 2
            continue

        if char == "#":
            in_line_comment = True
            i += 1
            continue

        result.append(char)
        i += 1

    return "".join(result)


def parse_args_with_json_config(parser, args=None, allow_unknown_config_keys=False):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    preliminary_args, remaining = pre_parser.parse_known_args(args=args)
    config_path = getattr(preliminary_args, "config", None)
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.loads(_strip_json_comments(f.read()))
        if not isinstance(config, dict):
            raise ValueError(f"Config file must contain a JSON object: {config_path}")

        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config.keys()) - valid_keys)
        if unknown_keys and not allow_unknown_config_keys:
            raise ValueError(
                f"Unknown config keys in {config_path}: {', '.join(unknown_keys)}"
            )
        if unknown_keys:
            config = {key: value for key, value in config.items() if key in valid_keys}

        for action in parser._actions:
            if action.dest in config:
                action.required = False

        parser.set_defaults(**config)
    parser.set_defaults(config=config_path)
    return parser.parse_args(remaining)
