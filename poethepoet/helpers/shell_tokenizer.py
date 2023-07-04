"""
This is an alternative to shlex from the standard library for tokenizing posix
style command line statements.


On top of the word-splitting and quote removal functionality of shlex, this
library also makes it possilbe to tell which glob characters are escaped, and
preserves semantics for variable & arithmetic expansions.
- command substitution
- redirect operations
- control operations


This implementation is somewhat over-generalized for what is currently required,
but it may be more useful in future.


Out of scope:
- history expansion
- Content validation for expansions/substitutions
- shell keywords
- multiline constructs/multi-token grammar: control flow, functions, multiline groups
- tokenizing the content of command or arithmetic expansions
- advanced parameter expansion or whatever you call it, like  ${param:-}
- arrays
- groups
- advanced glob patterns (classes and pattern lists)
"""

from collections import namedtuple, deque
from enum import Enum
from io import StringIO
from typing import IO, List, Optional, Tuple
import re

# characters that are escaped by glob:escape
GLOB_INIT_CHARS = "*?["
PREFIX_CHARS = "\\$"
METACHARACTERS = "|;&<>(){}"
WHITESPACE = " \t\r\n"
NON_RANGE_CHARS = WHITESPACE + "!`'\";&|()<>"
VAR_INIT_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
DIGITS = "0123456789"
VAR_CHARS = VAR_INIT_CHARS + DIGITS
SPECIAL_VARS = DIGITS + "!@#$*-?"


def tokenize(cmd_line: str):
    return list(CommandLineTokenizer(StringIO(cmd_line)))


class TokenType(Enum):
    WORD = 0
    CONTROL_OPERATOR = 1
    REDIRECT_OPERATOR = 2
    COMMENT = 3


class TokenFeatureType(str, Enum):
    VARIABLE = "var" # TODO: rename PARAMETER
    COMMAND = "cmd"
    ARITHMETIC = "arth"
    TILDE = "tld"
    GROUP = "grp"


class TokenFeature(
    namedtuple(
        "TokenFeature",
        ("type", "content", "start", "end", "quoted"),
    )
):
    type: TokenFeatureType
    content: str
    start: int
    end: int
    quoted: bool


class Token(
    namedtuple(
        "Token",
        (
            "content",
            "includes_glob",
            "escaped_glob_indices",
            "expansions",
            "type",
        ),
    )
):
    content: str
    includes_glob: bool
    escaped_glob_indices: Tuple[int, ...]
    expansions: Tuple[TokenFeature, ...]
    type: TokenType

    @staticmethod
    def operator(content: str):
        return Token(content, False, tuple(), tuple(), TokenType.CONTROL_OPERATOR)


_expansion_types = {
    "(": (")", TokenFeatureType.GROUP),
    "{": ("}", TokenFeatureType.GROUP),
    "${": ("}", TokenFeatureType.VARIABLE),
    "$((": ("))", TokenFeatureType.ARITHMETIC),
    "$(": (")", TokenFeatureType.COMMAND),
    "`": ("`", TokenFeatureType.COMMAND),
    "$": ("", TokenFeatureType.VARIABLE),
    "~": ("", TokenFeatureType.TILDE),
}


def includes_glob(input_stream: IO[str]) -> bool: # TODO: unit test me
    """
    Check if the input stream includes are valid glob sequence, such as:
    - *
    - ?
    - a valid simple range expression
    Escaping of glob characters is not considered.
    """
    next_char = None
    range_sequence_len = 0
    escape_active = False

    while next_char := input_stream.read(1):
        if not next_char:
            return False

        if next_char in "*?":
            return True

        if range_sequence_len:
            if not escape_active:
                if next_char == "]":
                    if range_sequence_len == 1:
                        range_sequence_len++
                    else:
                        return True

                if next_char == '"':
                    escape_active = True

            if next_char in NON_RANGE_CHARS:
                range_sequence_len = 0
                continue

            range_sequence_len++

        if next_char == "[":
            range_sequence_len = 1


class RedirectParseState:
    """
    Here-document and here-string redirects are not supported.
    """

    left_fd: str = ""
    operator: str = ""
    right_fd: str = ""

    def __init__(self, left_fd: str = "", operator: str = ""):
        self.left_fd = left_fd
        self.operator = operator
        self.right_fd = ""

    @property
    def valid(self):
        if self.left_fd and not self.left_fd.isdigit():
            return False

        if not re.match(r"^\d*\-?$", self.right_fd):
            return False

        if self.left_fd and not (self.operator or self.right_fd):
            return True

        if self.operator in ("&>", "&>>") and not (self.left_fd or self.right_fd):
            return True

        if self.operator in (">", ">>", ">|", "<", "<>") and not self.right_fd:
            return True

        if self.operator in (">&", "<&"):
            return True

        return False

    def read(self) -> str:
        return self.left_fd + self.operator + self.right_fd

    def append(self, next_char: str) -> bool:
        """
        This method assumes self will always be initialized with left_fd OR
        with operator as one of: "<" or ">" or "&>"

        If next_char is successfully appended to this token then return empty string.
        Otherwise return the most recent chars that are cannot be appended to this
        token. Usually this is just next_char but it can also more some digits.
        """

        rejected = ""

        if self.operator in ("<&", ">&"):
            if not self.right_fd.endswith("-") and (
                next_char.isdigit() or next_char == "-"
            ):
                self.right_fd += next_char
            elif next_char in " \t\n|;&<>":
                rejected = next_char
            else:
                # what we thought was closing digits is actually the next word
                rejected = self.right_fd + next_char
                self.right_fd = ""

        elif self.operator == ">":
            if next_char in ">&|":
                self.operator += next_char
            else:
                rejected = next_char

        elif self.operator == "<":
            if next_char in ">&":
                self.operator += next_char
            else:
                rejected = next_char

        elif self.operator == "&>":
            if next_char == ">":
                self.operator += next_char
            elif next_char in "|;&<>()":
                raise CommandLineTokenizeError(
                    f"Syntax error near redirect operator {self.read()!r}"
                )

            else:
                rejected = next_char

        elif not self.operator:
            if next_char.isdigit():
                self.left_fd += next_char
            elif next_char in "<>":
                self.operator += next_char
            else:
                rejected = next_char

        else:
            rejected = next_char

        return rejected


class TokenParseState:
    quote_active: str
    range_active: bool
    active_prefix: str
    scope_terminator: str
    redirect: Optional[RedirectParseState]
    paren_depth: int
    includes_glob: bool
    _expansion_scope_content: List[str]
    _parts: List[str]
    _escaped_glob_indices: List[int]
    _expansions: List[TokenFeature]
    _expansion_scope_start_index: Optional[int] = None
    expansion_type: Optional[str] = None

    def __init__(self):
        self.quote_active = ""
        self.range_active = False
        self.active_prefix = ""
        self.scope_terminator = ""
        self.redirect = None
        self.paren_depth = 0
        self.includes_glob = False
        self._expansion_scope_content = []
        self._parts = []
        self._escaped_glob_indices = []
        self._expansions = []

    def read(self):
        if self.redirect:
            return self.redirect.read()

        # TODO: keep count of length in append instead of calling read just to measure?
        if self.is_empty:
            return ""

        if len(self._parts) > 1:
            self._consolidate()

        return self._parts[0]

    def get_token(self) -> Optional[Token]:
        if self.is_empty:
            return None

        print("T:", self.read())

        return Token(
            content=self.read(),
            includes_glob=self.includes_glob,
            escaped_glob_indices=tuple(self._escaped_glob_indices),
            expansions=tuple(self._expansions),
            type=self._get_token_type(),
        )

    def _get_token_type(self) -> TokenType:
        if self.redirect:
            return TokenType.REDIRECT_OPERATOR
        if self.is_comment:
            return TokenType.COMMENT
        return TokenType.WORD

    def _consolidate(self):
        self._parts = ["".join(self._parts)]

    @property
    def is_empty(self) -> bool:
        return (
            not self._parts
            and not self.redirect
            and not self.active_prefix
            and not self.quote_active
            and not self.scope_terminator
        )

    @property
    def escape_active(self) -> bool:
        return self.active_prefix == "\\"

    @property
    def is_comment(self) -> bool:
        return self.active_prefix == "#"

    @property
    def in_single_quotes(self) -> bool:
        return self.quote_active == "'"

    @property
    def in_double_quotes(self) -> bool:
        return self.quote_active == '"'

    @property
    def in_expansion(self) -> bool:
        return self.expansion_type is not None

    def init_redirect(self, digits: str = "", operator: str = ""):
        self.redirect = RedirectParseState(digits, operator)

    def cancel_redirect(self):
        """
        This can be called when we parsed some digits then realized it's not a redirect.
        """
        self.append(self.redirect.read())
        self.redirect = None

    def start_expansion(self, opener: str, no_container: bool = False):
        assert not self.in_expansion
        self.append(opener, force_in_expansion=no_container)
        expansion_type = _expansion_types.get(opener, None)
        if expansion_type is None:
            raise TokenizerInternalError(f"Unknown scope opener {opener!r}")
        self.scope_terminator, self.expansion_type = expansion_type
        self._expansion_scope_start_index = len(self.read()) - len(opener)

    def end_expansion(self):
        assert self.in_expansion
        expansion_content = "".join(self._expansion_scope_content)
        self.append(self.scope_terminator)

        self._expansions.append(
            TokenFeature(
                type=self.expansion_type,
                content=expansion_content,
                start=self._expansion_scope_start_index,
                end=len(self.read()),
                quoted=self.in_double_quotes,
            )
        )

        self._expansion_scope_start_index = None
        self._expansion_scope_content = []
        self.expansion_type = None
        self.scope_terminator = ""

    def cancel_expansion(self):
        assert self.in_expansion
        self._expansion_scope_start_index = None
        self._expansion_scope_content = []
        self.expansion_type = None
        self.scope_terminator = ""

    def close_quotes(self):
        self.quote_active = ""
        # This just ensures we don't consider this token to be empty anymore
        self.append("")

    def append(
        self, char: str, escaped_glob: bool = False, force_in_expansion: bool = False
    ):
        if self.in_expansion or force_in_expansion:
            self._expansion_scope_content.append(char)
        self._parts.append(char)
        if escaped_glob:
            self._escaped_glob_indices.append(len(self.read()) - 1)


class CommandLineTokenizer:
    pushback: List[str] = []
    lookback: List[str] = []

    def __init__(self, input_stream: IO[str]):
        self.input_stream = input_stream

    def parse_token(self):
        state = TokenParseState()
        next_char = None

        while True:
            if self.pushback:
                next_char = self.pushback.pop()
                print("p:", next_char)
            else:
                next_char = self.input_stream.read(1)

            if not next_char:
                # end of input
                if state.escape_active:
                    raise CommandLineTokenizeError("No escaped character")

                if state.in_single_quotes or state.in_double_quotes:
                    raise CommandLineTokenizeError("No closing quotation")

                if state.scope_terminator:
                    raise CommandLineTokenizeError(
                        f"No closing bracket in token {state.read()!r}"
                    )

                if state.in_expansion:
                    for char in self.lookback:
                        state.append(char)
                    self.lookback.clear()
                    state.end_expansion()

                if state.redirect:
                    if state.redirect.valid:
                        if not state.redirect.operator:
                            state.cancel_redirect()
                    else:
                        raise CommandLineTokenizeError(
                            f"Unexpected end of input in redirect: {state.read()!r}"
                        )

                if state.range_active:
                    state.append("[")

                if self.lookback:
                    self._cancel_lookback()

                if state.active_prefix:
                    # TODO: this should vary depending on the prefix
                    state.append(state.active_prefix)
                    state.active_prefix = ""

                break

            if state.is_comment:
                if next_char == "\n":
                    break
                state.append(next_char)

            elif state.range_active:
                if state.escape_active:
                    self.lookback.append(next_char)
                    state.escape_active = False

                elif next_char == "]" and self.lookback:
                    # range expression complete
                    state.includes_glob = True
                    state.append("[")
                    for char in self.lookback:
                        state.append(char)
                    self.lookback.clear()
                    state.append(next_char)
                    state.range_active = False

                elif next_char in NON_RANGE_CHARS:
                    # range expression invalidated, pretend it never happened
                    state.append("[", escaped_glob=True)
                    self._cancel_lookback(next_char)
                    state.range_active = False

                else:
                    self.lookback.append(next_char)

            elif state.escape_active:
                if state.in_double_quotes and not next_char in '"$':
                    state.append("\\")
                elif next_char in GLOB_INIT_CHARS:
                    state.append(next_char, escaped_glob=True)
                else:
                    state.append(next_char)
                state.active_prefix = ""

            elif state.in_single_quotes:
                if next_char == "'":
                    state.close_quotes()
                elif next_char in GLOB_INIT_CHARS:
                    state.append(next_char, escaped_glob=True)
                else:
                    state.append(next_char)

            elif state.in_expansion:
                if state.scope_terminator == "}":
                    if next_char == "}":
                        state.end_expansion()
                    else:
                        state.append(next_char)

                elif state.scope_terminator == ")":
                    # TODO: to do this properly we need to parse the sub command to tell
                    #       whether the closing bracket is quoted or escaped
                    if next_char == ")":
                        state.end_expansion()
                    else:
                        state.append(next_char)

                elif state.scope_terminator == "`":
                    # N.B. we don't respect escaping of backticks inside backticks
                    if next_char == "`":
                        state.end_expansion()
                    else:
                        state.append(next_char)

                elif state.scope_terminator == "))":
                    if next_char == ")":
                        if state.paren_depth:
                            state.paren_depth -= 1
                            state.append(next_char)
                        elif self.lookback:
                            if self.lookback[0] == ")":
                                assert len(self.lookback) == 1
                                state.end_expansion()
                                self.lookback.clear()
                            else:
                                self._cancel_lookback(next_char)
                        else:
                            self.lookback.append(next_char)
                            continue
                    elif next_char == "(":
                        state.paren_depth += 1
                        state.append(next_char)

                    elif next_char == "\\":
                        state.active_prefix = next_char

                    else:
                        # TODO: verify if next_char is valid here?
                        state.append(next_char)

                elif state.expansion_type == TokenFeatureType.VARIABLE:
                    # This must be a variable name
                    if next_char in VAR_CHARS:
                        state.append(next_char)
                    else:
                        # We've gone off the end of the variable name, so we need to
                        # start the loop again with the same same char.
                        state.end_expansion()
                        self.pushback.append(next_char)
                        continue

                elif state.expansion_type == TokenFeatureType.TILDE:
                    if next_char in "/ \n\t\n|;&<>(){}":
                        # Reached the end of valid looking tilde expansion
                        for char in self.lookback:
                            state.append(char)
                        self.lookback.clear()
                        state.end_expansion()
                        self.pushback.append(next_char)

                    elif self.lookback:
                        if self.lookback[0] in "+-":
                            if next_char.isdigit():
                                self.lookback.append(next_char)
                            else:
                                self._cancel_lookback(next_char)
                                state.cancel_expansion()
                        elif next_char in VAR_CHARS or next_char in "-.":
                            self.lookback.append(next_char)

                    elif next_char in "+-" or next_char in VAR_CHARS:
                        self.lookback.append(next_char)

                    else:
                        self._cancel_lookback(next_char)
                        state.cancel_expansion()

                else:
                    raise TokenizerInternalError(
                        f"Unexpected expansion_type {state.expansion_type!r}"
                    )

            elif state.active_prefix == "$":
                if next_char == "(":
                    state.active_prefix = "$("
                else:
                    state.active_prefix = ""
                    if next_char == "{":
                        state.start_expansion("${")
                    elif next_char in SPECIAL_VARS:
                        state.start_expansion("$")
                        state.append(next_char)
                        state.end_expansion()
                    elif next_char in VAR_INIT_CHARS:
                        state.start_expansion("$")
                        state.append(next_char)
                    else:
                        # this $ has no meaning
                        state.append("$")
                        self.pushback.append(next_char)
                        continue

            elif state.active_prefix == "$(":
                state.active_prefix = ""
                if next_char == "(":
                    state.start_expansion("$((")
                else:
                    state.start_expansion("$(")
                    self.pushback.append(next_char)
                    continue

            elif state.redirect:
                if rejected := state.redirect.append(next_char):
                    if not state.redirect.operator:
                        state.cancel_redirect()
                        self.pushback.append(next_char)
                        continue
                    elif state.redirect.valid:
                        for char in reversed(rejected):
                            self.pushback.append(char)
                        return state.get_token()
                    else:
                        raise CommandLineTokenizeError(
                            f"Syntax error near redirect operator {state.read()!r}"
                        )

            elif state.active_prefix == "|":
                state.active_prefix = ""
                if next_char == "|":
                    return Token.operator("||")
                elif next_char == "&":
                    return Token.operator("|&")
                else:
                    self.pushback.append(next_char)
                    return Token.operator("|")

            elif state.active_prefix == "&":
                state.active_prefix = ""
                if next_char == "&":
                    return Token.operator("&&")
                elif next_char == ">":
                    state.init_redirect(operator="&>")
                else:
                    self.pushback.append(next_char)
                    return Token.operator("&")

            elif state.active_prefix == ";":
                if next_char == ";":
                    state.active_prefix = ";;"
                elif next_char == "&":
                    return Token.operator(";&")
                else:
                    self.pushback.append(next_char)
                    return Token.operator(";")

            elif state.active_prefix == ";;":
                if next_char == "&":
                    return Token.operator(";;&")
                else:
                    self.pushback.append(next_char)
                    return Token.operator(";;")

            elif next_char == "`":
                state.start_expansion(next_char)

            elif state.in_double_quotes:
                if next_char == '"':
                    state.close_quotes()
                elif next_char in PREFIX_CHARS:
                    state.active_prefix = next_char
                elif next_char in GLOB_INIT_CHARS:
                    state.append(next_char, escaped_glob=True)
                else:
                    state.append(next_char)

            elif next_char in "><":
                if state.is_empty:
                    state.init_redirect(operator=next_char)
                else:
                    self.pushback.append(next_char)
                    break

            elif next_char in "&|":
                if state.is_empty:
                    # need to check the next char to know what this means
                    state.active_prefix = next_char
                else:
                    self.pushback.append(next_char)
                    break

            elif next_char in WHITESPACE:
                if state.is_empty:
                    # ignore leading whitespace
                    continue
                else:
                    # token complete
                    break

            elif next_char in "'\"":
                # Open quotes and drop quote char
                state.quote_active = next_char

            elif next_char == "[":
                # This could be a range expression, but we need to look ahead
                # to parse the whole expression to be sure. This means validating
                # that there is a closing square bracked with valid contents.
                state.range_active = True

            elif next_char in "*?":
                state.includes_glob = True
                state.append(next_char)

            elif next_char in PREFIX_CHARS:
                state.active_prefix = next_char

            elif next_char == ";":
                if not state.is_empty:
                    self.pushback.append(next_char)
                    break
                state.active_prefix = ";"

            elif next_char == "~":
                if state.is_empty:
                    # need to check the next char to know what this means
                    state.start_expansion("~", no_container=True)
                else:
                    state.append(next_char)

            elif next_char == "#":
                if not state.is_empty:
                    self.pushback.append(next_char)
                    break
                state.active_prefix = next_char
                state.append(next_char)

            elif next_char.isdigit() and state.is_empty:
                state.init_redirect(digits=next_char)

            else:
                # TODO: validate next_char is allowed out in the open like this
                #           ... why wouldn't it be?
                # like () is usually a problem unless at start of token/line
                #           ... need to count tokens?

                state.append(next_char)

        return state.get_token()

    def _cancel_lookback(self, next_char: str = ""):
        if next_char:
            self.pushback.append(next_char)
        for char in reversed(self.lookback):
            self.pushback.append(char)
        self.lookback.clear()

    def __iter__(self):
        return self

    def __next__(self):
        if token := self.parse_token():
            return token
        raise StopIteration


class CommandLineTokenizeError(RuntimeError):
    # TODO: keep track of where we are through the whole source for errors!!
    pass

    @property
    def message(self) -> str:
        return self.args[0] if self.args else "Failed to tokenize command line"


class TokenizerInternalError(Exception):
    pass


"""

Test cases:

- escapes inside double or single quotes
- dollar or backslash at end of line


Other edge cases:
Unescaped and unmatched } is an error

"""

# In posix shells, only the quote itself or the escape
# character may be escaped within quotes.
