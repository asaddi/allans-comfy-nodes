from enum import Enum, auto
from typing import Any


class WildParserState(Enum):
    LITERAL = auto()
    ESCAPE = auto()
    CHOICES = auto()
    CHOICES_ESCAPE = auto()


def _flush_lit(buf: list[str], target: list) -> None:
    # Append buffered chars to target as one LITERAL token, then empty buf
    if buf:
        target.append((WildParserState.LITERAL, "".join(buf)))
        buf.clear()


def _append_char(lit: list[str], c: str, pending_optional: bool) -> bool:
    # Append a literal char, materializing a pending `*` first; the `*` is
    # consumed, so the updated pending_optional is always False
    if pending_optional:
        lit.append("*")
    lit.append(c)
    return False


def _push_context(
    stack: list,
    is_top_level: bool,
    pending_optional: bool,
    choice_parts: list,
    choices: list,
) -> tuple[bool, list, list]:
    # Save the current choice context on the stack; return the reset loop
    # variables (pending_optional, choice_parts, choices) for the new choice
    stack.append((is_top_level, pending_optional, choice_parts, choices))
    return False, [], []


def _pop_frame(
    stack: list,
    choices: list,
    output: list,
    apply_optional: bool,
    finalize_option: bool,
) -> tuple[bool, bool, list, list]:
    # Pop a choice context and attach the completed CHOICES token.
    # Returns (is_top_level, pending_optional, choice_parts, choices) to restore.
    # apply_optional: emit the empty choice for a pending `*` (done when a `}`
    # closed the choice, not at end of input). finalize_option: also close the
    # parent's in-progress option (only needed when unwinding at end of input).
    (
        is_top_level,
        prev_pending_optional,
        prev_choice_parts,
        prev_choices,
    ) = stack.pop()
    if apply_optional and prev_pending_optional:
        # Handle optional: add empty choice for *{...}
        choices.append([(WildParserState.LITERAL, "")])
    choice_token = (WildParserState.CHOICES, choices)
    if is_top_level:
        # This was a top-level choice, add to output
        output.append(choice_token)
        return is_top_level, False, [], []
    # This was nested choice, add to parent's current choice option
    prev_choice_parts.append(choice_token)
    if finalize_option and prev_choice_parts:
        prev_choices.append(prev_choice_parts)
    pending = False if apply_optional else prev_pending_optional
    return is_top_level, pending, prev_choice_parts, prev_choices


def parse_wildcards(input: str) -> list[tuple[WildParserState, Any]]:
    # Use a stack for nested choice contexts
    # Stack frame: (is_top_level, prev_pending_optional, prev_choice_parts, prev_choices)
    # is_top_level: True if this was entered from LITERAL state (not nested in another choice)

    output = []  # Final output for top level
    stack = []  # Stack of (is_top_level, pending_optional, choice_parts, choices)
    lit = []  # Current literal chars being accumulated
    pending_optional = False
    choice_parts = []  # Parts of current choice option (list of tokens)
    choices = []  # List of completed choice options (each is list of tokens)
    state = WildParserState.LITERAL

    for c in input:
        if state == WildParserState.LITERAL:
            if c == "\\":
                state = WildParserState.ESCAPE
            elif c == "{":
                # Flush any pending literal to output, start a top-level choice
                _flush_lit(lit, output)
                pending_optional, choice_parts, choices = _push_context(
                    stack, True, pending_optional, [], []
                )
                state = WildParserState.CHOICES
            elif c == "*":
                pending_optional = True
            else:
                pending_optional = _append_char(lit, c, pending_optional)

        elif state == WildParserState.ESCAPE:
            lit.append(c)
            pending_optional = False
            state = WildParserState.LITERAL

        elif state == WildParserState.CHOICES:
            if c == "\\":
                state = WildParserState.CHOICES_ESCAPE
            elif c == "}":
                # End of choice: flush lit to choice_parts, complete the option
                _flush_lit(lit, choice_parts)
                if choice_parts:
                    choices.append(choice_parts)
                choice_parts = []
                if stack:
                    is_top, pending_optional, choice_parts, choices = _pop_frame(
                        stack,
                        choices,
                        output,
                        apply_optional=True,
                        finalize_option=False,
                    )
                    state = (
                        WildParserState.LITERAL if is_top else WildParserState.CHOICES
                    )
                else:
                    # Stack underflow - shouldn't happen, but handle gracefully
                    output.append((WildParserState.CHOICES, choices))
                    pending_optional = False
                    state = WildParserState.LITERAL
            elif c == "{":
                # Nested choice start: flush lit to choice_parts, save context
                _flush_lit(lit, choice_parts)
                pending_optional, choice_parts, choices = _push_context(
                    stack, False, pending_optional, choice_parts, choices
                )
            elif c == "|":
                # Complete current choice option and start new one
                _flush_lit(lit, choice_parts)
                if choice_parts:
                    choices.append(choice_parts)
                choice_parts = []
            elif c == "*":
                pending_optional = True
            else:
                pending_optional = _append_char(lit, c, pending_optional)

        elif state == WildParserState.CHOICES_ESCAPE:
            lit.append(c)
            pending_optional = False
            state = WildParserState.CHOICES

    # Clean up at end of input
    if state == WildParserState.LITERAL:
        _flush_lit(lit, output)
    elif state == WildParserState.ESCAPE:
        # Escape at end of string, preserve the backslash
        lit.append("\\")
        _flush_lit(lit, output)
    elif state in (WildParserState.CHOICES, WildParserState.CHOICES_ESCAPE):
        if state == WildParserState.CHOICES_ESCAPE:
            # Escape at end of choice, preserve the backslash
            lit.append("\\")
        _flush_lit(lit, choice_parts)
        if choice_parts:
            choices.append(choice_parts)
        # Unwind stack
        while stack:
            _, pending_optional, choice_parts, choices = _pop_frame(
                stack, choices, output, apply_optional=False, finalize_option=True
            )
        if choices:
            output.append((WildParserState.CHOICES, choices))

    return output


def test_basic():
    """Test parsing a simple literal string without any special characters."""
    parsed = parse_wildcards("Hello there")
    assert len(parsed) == 1
    assert parsed[0][0] == WildParserState.LITERAL
    assert parsed[0][1] == "Hello there"


def test_escape():
    """Test escaped brace is treated as literal, not choice delimiter."""
    parsed = parse_wildcards(r"Hello \{there")
    assert len(parsed) == 1
    assert parsed[0][0] == WildParserState.LITERAL
    assert parsed[0][1] == r"Hello {there"


def test_invalid_escape():
    """Test invalid escape sequence (not followed by special char) is handled gracefully."""
    parsed = parse_wildcards(r"Hello \there")
    assert len(parsed) == 1
    assert parsed[0][0] == WildParserState.LITERAL
    assert parsed[0][1] == r"Hello there"


def test_escape_eos():
    """Test backslash at end of string is preserved as literal."""
    parsed = parse_wildcards("Hello there\\")
    assert len(parsed) == 1
    assert parsed[0][0] == WildParserState.LITERAL
    assert parsed[0][1] == "Hello there\\"


def test_single_choice():
    """Test a choice with a single option (no | separator)."""
    parsed = parse_wildcards(
        "This comes before. {This is the one choice.} This comes after."
    )
    expected = [
        (WildParserState.LITERAL, "This comes before. "),
        (
            WildParserState.CHOICES,
            [[(WildParserState.LITERAL, "This is the one choice.")]],
        ),
        (WildParserState.LITERAL, " This comes after."),
    ]
    assert parsed == expected


def test_single_escape_choice():
    """Test escaped pipe character within a choice is treated as literal."""
    parsed = parse_wildcards(
        "This comes before. {This is the \\|one choice.} This comes after."
    )
    expected = [
        (WildParserState.LITERAL, "This comes before. "),
        (
            WildParserState.CHOICES,
            [[(WildParserState.LITERAL, "This is the |one choice.")]],
        ),
        (WildParserState.LITERAL, " This comes after."),
    ]
    assert parsed == expected


def test_multiple_choice():
    """Test a choice with multiple options separated by |."""
    parsed = parse_wildcards(
        "This comes before. {This is the one choice.|This is another.} This comes after."
    )
    expected = [
        (WildParserState.LITERAL, "This comes before. "),
        (
            WildParserState.CHOICES,
            [
                [(WildParserState.LITERAL, "This is the one choice.")],
                [(WildParserState.LITERAL, "This is another.")],
            ],
        ),
        (WildParserState.LITERAL, " This comes after."),
    ]
    assert parsed == expected


def test_incomplete_choice():
    """Test gracefully handling a choice that is not closed (missing closing brace)."""
    parsed = parse_wildcards(
        "This comes before. {This is the one choice.|This is an incomplete"
    )
    expected = [
        (WildParserState.LITERAL, "This comes before. "),
        (
            WildParserState.CHOICES,
            [
                [(WildParserState.LITERAL, "This is the one choice.")],
                [(WildParserState.LITERAL, "This is an incomplete")],
            ],
        ),
    ]
    assert parsed == expected


def test_empty_choice():
    """Test that an empty option before | is discarded."""
    parsed = parse_wildcards(
        "This comes before. {|This is the one choice.} This comes after."
    )
    expected = [
        (WildParserState.LITERAL, "This comes before. "),
        (
            WildParserState.CHOICES,
            [[(WildParserState.LITERAL, "This is the one choice.")]],
        ),
        (WildParserState.LITERAL, " This comes after."),
    ]
    assert parsed == expected


def test_optional_choice():
    """Test optional choice marker (*) adds an empty string option."""
    parsed = parse_wildcards(
        "This comes before. *{This is one choice.|This is another.} This comes after."
    )
    expected = [
        (WildParserState.LITERAL, "This comes before. "),
        (
            WildParserState.CHOICES,
            [
                [(WildParserState.LITERAL, "This is one choice.")],
                [(WildParserState.LITERAL, "This is another.")],
                [(WildParserState.LITERAL, "")],
            ],
        ),
        (WildParserState.LITERAL, " This comes after."),
    ]
    assert parsed == expected


def test_escaped_optional_choice():
    """Test escaped * is treated as literal star, not optional marker."""
    parsed = parse_wildcards(
        "This comes before. \\*{This is one choice.|This is another.} This comes after."
    )
    expected = [
        (WildParserState.LITERAL, "This comes before. *"),
        (
            WildParserState.CHOICES,
            [
                [(WildParserState.LITERAL, "This is one choice.")],
                [(WildParserState.LITERAL, "This is another.")],
            ],
        ),
        (WildParserState.LITERAL, " This comes after."),
    ]
    assert parsed == expected


def test_nested_choice():
    """Test nested choice where one option contains another choice."""
    parsed = parse_wildcards(
        "This comes before. {This is the first choice.|This is {sub-choice one|sub-choice two}.} This comes after."
    )
    expected = [
        (WildParserState.LITERAL, "This comes before. "),
        (
            WildParserState.CHOICES,
            [
                [(WildParserState.LITERAL, "This is the first choice.")],
                [
                    (WildParserState.LITERAL, "This is "),
                    (
                        WildParserState.CHOICES,
                        [
                            [(WildParserState.LITERAL, "sub-choice one")],
                            [(WildParserState.LITERAL, "sub-choice two")],
                        ],
                    ),
                    (WildParserState.LITERAL, "."),
                ],
            ],
        ),
        (WildParserState.LITERAL, " This comes after."),
    ]
    assert parsed == expected


def test_deeply_nested_choice():
    """Test three levels of nesting."""
    parsed = parse_wildcards("{a|{b|{c|d}}}")
    expected = [
        (
            WildParserState.CHOICES,
            [
                [(WildParserState.LITERAL, "a")],
                [
                    (
                        WildParserState.CHOICES,
                        [
                            [(WildParserState.LITERAL, "b")],
                            [
                                (
                                    WildParserState.CHOICES,
                                    [
                                        [(WildParserState.LITERAL, "c")],
                                        [(WildParserState.LITERAL, "d")],
                                    ],
                                )
                            ],
                        ],
                    )
                ],
            ],
        )
    ]
    assert parsed == expected


def test_multiple_nested_choices_sequence():
    """Test multiple nested choices in sequence."""
    parsed = parse_wildcards("{a|{b|c}}{d|{e|f}}")
    expected = [
        (
            WildParserState.CHOICES,
            [
                [(WildParserState.LITERAL, "a")],
                [
                    (
                        WildParserState.CHOICES,
                        [
                            [(WildParserState.LITERAL, "b")],
                            [(WildParserState.LITERAL, "c")],
                        ],
                    )
                ],
            ],
        ),
        (
            WildParserState.CHOICES,
            [
                [(WildParserState.LITERAL, "d")],
                [
                    (
                        WildParserState.CHOICES,
                        [
                            [(WildParserState.LITERAL, "e")],
                            [(WildParserState.LITERAL, "f")],
                        ],
                    )
                ],
            ],
        ),
    ]
    assert parsed == expected


def test_optional_outer_with_nested():
    """Test optional marker on outer choice containing nested choice."""
    parsed = parse_wildcards("*{a|{b|c}}")
    expected = [
        (
            WildParserState.CHOICES,
            [
                [(WildParserState.LITERAL, "a")],
                [
                    (
                        WildParserState.CHOICES,
                        [
                            [(WildParserState.LITERAL, "b")],
                            [(WildParserState.LITERAL, "c")],
                        ],
                    )
                ],
                [(WildParserState.LITERAL, "")],
            ],
        )
    ]
    assert parsed == expected


def test_optional_inner_nested():
    """Test optional marker on inner nested choice."""
    parsed = parse_wildcards("{a|*{b|c}}")
    expected = [
        (
            WildParserState.CHOICES,
            [
                [(WildParserState.LITERAL, "a")],
                [
                    (
                        WildParserState.CHOICES,
                        [
                            [(WildParserState.LITERAL, "b")],
                            [(WildParserState.LITERAL, "c")],
                            [(WildParserState.LITERAL, "")],
                        ],
                    )
                ],
            ],
        )
    ]
    assert parsed == expected


def test_nested_choice_with_escape():
    """Test escape sequences within nested choices."""
    parsed = parse_wildcards("{a|{b\\|c|d}}")
    expected = [
        (
            WildParserState.CHOICES,
            [
                [(WildParserState.LITERAL, "a")],
                [
                    (
                        WildParserState.CHOICES,
                        [
                            [(WildParserState.LITERAL, "b|c")],
                            [(WildParserState.LITERAL, "d")],
                        ],
                    )
                ],
            ],
        )
    ]
    assert parsed == expected


def test_nested_choice_with_escaped_brace():
    """Test escaped braces within nested choices."""
    parsed = parse_wildcards("{a|{b\\}|c}}")
    expected = [
        (
            WildParserState.CHOICES,
            [
                [(WildParserState.LITERAL, "a")],
                [
                    (
                        WildParserState.CHOICES,
                        [
                            [(WildParserState.LITERAL, "b}")],
                            [(WildParserState.LITERAL, "c")],
                        ],
                    )
                ],
            ],
        )
    ]
    assert parsed == expected


def test_unclosed_nested_choice():
    """Test graceful handling of unclosed nested choice."""
    parsed = parse_wildcards("{a|{b|c")
    expected = [
        (
            WildParserState.CHOICES,
            [
                [(WildParserState.LITERAL, "a")],
                [
                    (
                        WildParserState.CHOICES,
                        [
                            [(WildParserState.LITERAL, "b")],
                            [(WildParserState.LITERAL, "c")],
                        ],
                    )
                ],
            ],
        )
    ]
    assert parsed == expected


def test_nested_choice_complex():
    """Test complex nesting with multiple levels and options."""
    parsed = parse_wildcards("Start {A|{B1|B2}|C{D|E}} End")
    expected = [
        (WildParserState.LITERAL, "Start "),
        (
            WildParserState.CHOICES,
            [
                [(WildParserState.LITERAL, "A")],
                [
                    (
                        WildParserState.CHOICES,
                        [
                            [(WildParserState.LITERAL, "B1")],
                            [(WildParserState.LITERAL, "B2")],
                        ],
                    )
                ],
                [
                    (WildParserState.LITERAL, "C"),
                    (
                        WildParserState.CHOICES,
                        [
                            [(WildParserState.LITERAL, "D")],
                            [(WildParserState.LITERAL, "E")],
                        ],
                    ),
                ],
            ],
        ),
        (WildParserState.LITERAL, " End"),
    ]
    assert parsed == expected


if __name__ == "__main__":
    test_basic()
    test_escape()
    test_invalid_escape()
    test_escape_eos()
    test_single_choice()
    test_single_escape_choice()
    test_multiple_choice()
    test_incomplete_choice()
    test_empty_choice()
    test_optional_choice()
    test_escaped_optional_choice()
    test_nested_choice()
    test_deeply_nested_choice()
    test_multiple_nested_choices_sequence()
    test_optional_outer_with_nested()
    test_optional_inner_nested()
    test_nested_choice_with_escape()
    test_nested_choice_with_escaped_brace()
    test_unclosed_nested_choice()
    test_nested_choice_complex()

    print("All tests passed!")
