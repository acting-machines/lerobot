EPS = 1e-6


def get_range_regex(min_val: int, max_val: int) -> str:
    if min_val == 0 and max_val == 1024:
        # 0-9 | 10-99 | 100-999 | 1000-1019 | 1020-1023
        return '([0-9] | [1-9] [0-9] | [1-9] [0-9] [0-9] | "10" [0-1] [0-9] | "102" [0-3])'
    elif min_val == 0 and max_val == 512:
        # 0-9 | 10-99 | 100-499 | 500-509 | 510-511
        return '([0-9] | [1-9] [0-9] | [1-4] [0-9] [0-9] | "50" [0-9] | "51" [0-1])'
    elif min_val == 0 and max_val == 256:
        # 0-9 | 10-99 | 100-199 | 200-249 | 250-255
        return '([0-9] | [1-9] [0-9] | "1" [0-9] [0-9] | "2" [0-4] [0-9] | "25" [0-5])'
    else:
        raise ValueError(f"Range {min_val}:{max_val} is not supported.")


def build_exact_n_numbers_grammar(n_numbers: int, min_val: int, max_val: int) -> str:
    """
    Constructs an EBNF grammar that enforces exactly `n_numbers` integers.
    """
    int_pattern = get_range_regex(min_val, max_val)
    base_rules = f"""
    integer ::= {int_pattern}
    space ::= " "
    """

    # Build the exact sequence string: integer space integer space integer ...
    # We construct "integer " * (N-1) + "integer"
    sequence_parts = ["integer"] * n_numbers
    sequence_rule = "root ::= space " + " space ".join(sequence_parts)

    return base_rules + sequence_rule
