import os


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def mocap_json_path(mocap_sequence, mocap_data_dir="mocap_data"):
    return os.path.join(str(mocap_data_dir), f"{mocap_sequence}.json")


def mocap_targets_path(mocap_sequence, mocap_data_dir="mocap_data"):
    return os.path.join(str(mocap_data_dir), f"{mocap_sequence}_targets.csv")


def mocap_summary_path(mocap_sequence, mocap_data_dir="mocap_data"):
    return os.path.join(str(mocap_data_dir), f"{mocap_sequence}_targets_summary.json")


def resolve_mocap_templates(args, fields=None):
    """Expand {mocap_sequence} and {mocap_data_dir} placeholders on argparse args."""
    mocap_sequence = getattr(args, "mocap_sequence", None)
    if mocap_sequence is None:
        return args

    mocap_data_dir = getattr(args, "mocap_data_dir", "mocap_data")
    mapping = _SafeFormatDict(
        mocap_sequence=str(mocap_sequence),
        mocap_data_dir=str(mocap_data_dir),
        mocap_json=mocap_json_path(mocap_sequence, mocap_data_dir),
        mocap_targets=mocap_targets_path(mocap_sequence, mocap_data_dir),
        mocap_summary=mocap_summary_path(mocap_sequence, mocap_data_dir),
    )
    if fields is None:
        fields = ("path", "modelname", "occupancy_map_file", "output_json")

    for field in fields:
        value = getattr(args, field, None)
        if isinstance(value, str):
            setattr(args, field, value.format_map(mapping))
    return args
