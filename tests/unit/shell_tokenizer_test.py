import pytest
import math
from poethepoet.helpers.shell_tokenizer import (
    tokenize,
    TokenType,
    TokenFeature,
    TokenFeatureType,
    CommandLineTokenizeError,
)

VARIABLE = TokenFeatureType.VARIABLE
COMMAND = TokenFeatureType.COMMAND
ARITHMETIC = TokenFeatureType.ARITHMETIC
TILDE = TokenFeatureType.TILDE


def shout(content, decoration="🍍", decoration_width=2, padding=2):
    try:
        lines = [
            f"⭐️ {content} ⭐️",
            "",
            *(
                "   ".join(f"{v!r}" for k, v in token._asdict().items())
                for token in tokenize(content)
            ),
        ]
    except:
        print("☠️ ", content)
        raise

    content_width = max(math.ceil(max(len(line) for line in lines) / 2) * 2, 78)
    outer_width = int(
        (content_width + padding * 2) / decoration_width + decoration_width
    )

    horiz_rule = decoration * outer_width
    empty_line = decoration + (" " * content_width + (" " * padding * 2)) + decoration

    print(horiz_rule)
    print(empty_line)

    for line in lines:
        print(
            decoration
            + (" " * padding)
            + line
            + (" " * (content_width - len(line)))
            + (" " * padding)
            + decoration
        )

    print(empty_line)
    print(horiz_rule)


def man_test_tokenize():
    # shout("poetry run 'p${}yte\"st;) ' -vvv tests/unit/shell_tokenizer_test.py") # FIXME: this should error

    # shout("echo hello 'wo'r()ld!")
    # shout("""echo "and then$${FOO}'inside'"'addon' $OK! ${OK}! $☠️  $+ """)
    # shout("echo x$(boom)y")
    # shout("echo `subcommand! or \\$${whatever}`")
    # shout("echo x$((boom))y echo $(( ((1 + 1))))")
    # shout("echo hi! 2>&99 &>99lol < out")
    # shout(
    #     r"""ls ./**/* \*.txt '*.txt' 1?2 \? \[ "n*pe" '*?' '*'? "*"? *\? \*\? '*'? """
    # )

    # test glob spotting & escaping

    # What next? Write real tests or implement more?

    # shout(
    #     "echo brace expansion a{d,c,b}e a{d,}e a{d?.*,}e and no: a{d,c, b}e a{d}e a{d, }e a{, d}e a{d,\\}e a{d\\,}e a\\{d,}e "
    # )
    # shout("echo BAD brace expansion a{-,}e")
    # A correctly-formed brace expansion must contain unquoted opening and closing braces, and at least one unquoted comma or a valid sequence expression

    # BUGS #

    shout("ls [ [] A[B]C")  # first two instances of [ should not be tagged
    # should instead only recognise [] containg non-metachars: #!@&;\s\n\t()


r"""

FIXME:
- () must be caught


!!! brace expansion probably implies the need to backtrack more than one char !!!

echo hello 'wo'r()ld!   => syntax error near unexpected token `('

cat <(ls) # <= OK


brace expansion: echo a{d,c,b}e
- tricky that globs are evaluated after brace expansions!


Edge cases:
- end line with prefix


#################
# TO IMPLEMENT: #
#################

- history expansion
    - first understand how it works!

- inline (subshell groups)
    - valid as first word in statement (start of line or after redirect)

!history expansion character is also treated as quoted if it immediately precedes the closing double quote in a double-quoted string.



"""


def test_parse_empty_token():
    tokens = tokenize("echo ''1 '' 2")

    assert_tokens(
        tokens,
        TokenDesc("echo"),
        TokenDesc("1"),
        TokenDesc(""),
        TokenDesc("2"),
    )


def test_parse_comments():
    tokens = tokenize(
        """
        echo hi! # la de da

        # |;&<>(){} # @!2
        #
        ##

        # OK ;) \t
        """
    )

    assert_tokens(
        tokens,
        TokenDesc("echo"),
        TokenDesc("hi!"),
        TokenDesc("# la de da", type=TokenType.COMMENT),
        TokenDesc("# |;&<>(){} # @!2", type=TokenType.COMMENT),
        TokenDesc("#", type=TokenType.COMMENT),
        TokenDesc("##", type=TokenType.COMMENT),
        TokenDesc("# OK ;) \t", type=TokenType.COMMENT),
    )


def test_parse_glob_tokens():
    # label unescaped ? *
    # label unescaped [ with valid content before ]

    tokens = tokenize(
        "echo h?llo/**/*.py * thing[123].txt "
        """ "h?llo/**/*.py" "*" "thing[123].txt" """
        "'h?llo/**/*.py' '*' 'thing[123].txt' "
        "h\\?llo/\\*\\*/\\*.py \\* thing\\[123].txt "
        " h?ll'*' "
        " foo[ld] o[]r lo][l 2 "
        "[[] [[yep] [nope!] [still!*] "
        "[ha&]"
    )

    # assert len(tokens) == 4
    assert_tokens(
        tokens,
        TokenDesc("echo"),
        TokenDesc("h?llo/**/*.py", includes_glob=True),
        TokenDesc("*", includes_glob=True),
        TokenDesc("thing[123].txt", includes_glob=True),
        TokenDesc(
            "h?llo/**/*.py",
            includes_glob=False,
            escaped_glob_indices=(
                1,
                6,
                7,
                9,
            ),
        ),
        TokenDesc("*", includes_glob=False, escaped_glob_indices=(0,)),
        TokenDesc("thing[123].txt", includes_glob=False, escaped_glob_indices=(5,)),
        TokenDesc(
            "h?llo/**/*.py",
            includes_glob=False,
            escaped_glob_indices=(
                1,
                6,
                7,
                9,
            ),
        ),
        TokenDesc("*", includes_glob=False, escaped_glob_indices=(0,)),
        TokenDesc("thing[123].txt", includes_glob=False, escaped_glob_indices=(5,)),
        TokenDesc(
            "h?llo/**/*.py",
            includes_glob=False,
            escaped_glob_indices=(
                1,
                6,
                7,
                9,
            ),
        ),
        TokenDesc("*", includes_glob=False, escaped_glob_indices=(0,)),
        TokenDesc("thing[123].txt", includes_glob=False, escaped_glob_indices=(5,)),
        TokenDesc("h?ll*", includes_glob=True, escaped_glob_indices=(4,)),
        TokenDesc("foo[ld]", includes_glob=True),
        TokenDesc("o[]r", includes_glob=False, escaped_glob_indices=(1,)),
        TokenDesc("lo][l", includes_glob=False, escaped_glob_indices=(3,)),
        TokenDesc("2", includes_glob=False),
        TokenDesc("[[]", includes_glob=True),
        TokenDesc("[[yep]", includes_glob=True),
        TokenDesc("[nope!]", includes_glob=False, escaped_glob_indices=(0,)),
        TokenDesc("[still!*]", includes_glob=True, escaped_glob_indices=(0,)),
        TokenDesc("[ha", includes_glob=False, escaped_glob_indices=(0,)),
        TokenDesc("&", type=TokenType.CONTROL_OPERATOR, includes_glob=False),
        TokenDesc("]", includes_glob=False),
    )


def test_parse_arithmetic_expansions():
    tokens = tokenize(
        """
        echo "A$(( 1 + 1 ))B" '$(( 2**3 ))' A$((1&&1==( 1 || 1 )))B
        """
    )

    assert len(tokens) == 4
    assert_tokens(
        tokens,
        TokenDesc("echo"),
        TokenDesc(
            "A$(( 1 + 1 ))B",
            expansions=(
                TokenFeature(ARITHMETIC, " 1 + 1 ", start=1, end=13, quoted=True),
            ),
        ),
        TokenDesc("$(( 2**3 ))", escaped_glob_indices=(5, 6)),
        TokenDesc(
            "A$((1&&1==( 1 || 1 )))B",
            expansions=(
                TokenFeature(
                    ARITHMETIC,
                    "1&&1==( 1 || 1 )",
                    start=1,
                    end=22,
                    quoted=False,
                ),
            ),
        ),
    )


def test_parse_command_expansions():
    tokens = tokenize(
        r"""
        echo "A$(ls "foo")B" '$(ls foo)' A$(ls foo)B
        """
    )

    assert len(tokens) == 4
    assert_tokens(
        tokens,
        TokenDesc("echo"),
        TokenDesc(
            'A$(ls "foo")B',
            expansions=(
                TokenFeature(COMMAND, 'ls "foo"', start=1, end=12, quoted=True),
            ),
        ),
        TokenDesc("$(ls foo)"),
        TokenDesc(
            "A$(ls foo)B",
            expansions=(
                TokenFeature(COMMAND, "ls foo", start=1, end=10, quoted=False),
            ),
        ),
    )


def test_parse_operator_tokens():
    # NB we also parse `;;`, `;&`, `;;&` but we don't parse case statements so nevermind

    for op in ("&", "&&", "|", "||", "|&", ";"):
        tokens = tokenize(f"echo 1{op}echo 2 {op} echo 3")

        assert len(tokens) == 8
        assert_tokens(
            tokens,
            TokenDesc("echo"),
            TokenDesc("1"),
            TokenDesc(op, type=TokenType.CONTROL_OPERATOR),
            TokenDesc("echo"),
            TokenDesc("2"),
            TokenDesc(op, type=TokenType.CONTROL_OPERATOR),
            TokenDesc("echo"),
            TokenDesc("3"),
        )


def test_parse_tilde_tokens():
    tokens = tokenize(
        r"""
        echo ~ ~/ ~/foo ~user.name ~root/Library ~+ ~- ~12 ~+42 ~-11 ~-/foo ~+/foo x~ ~@ "~"
        """
    )

    assert len(tokens) == 16
    assert_tokens(
        tokens,
        TokenDesc("echo"),
        TokenDesc(
            "~",
            expansions=(TokenFeature(TILDE, "~", start=0, end=1, quoted=False),),
        ),
        TokenDesc(
            "~/",
            expansions=(TokenFeature(TILDE, "~", start=0, end=1, quoted=False),),
        ),
        TokenDesc(
            "~/foo",
            expansions=(TokenFeature(TILDE, "~", start=0, end=1, quoted=False),),
        ),
        TokenDesc(
            "~user.name",
            expansions=(
                TokenFeature(TILDE, "~user.name", start=0, end=10, quoted=False),
            ),
        ),
        TokenDesc(
            "~root/Library",
            expansions=(TokenFeature(TILDE, "~root", start=0, end=5, quoted=False),),
        ),
        TokenDesc(
            "~+",
            expansions=(TokenFeature(TILDE, "~+", start=0, end=2, quoted=False),),
        ),
        TokenDesc(
            "~-",
            expansions=(TokenFeature(TILDE, "~-", start=0, end=2, quoted=False),),
        ),
        TokenDesc(
            "~12",
            expansions=(TokenFeature(TILDE, "~12", start=0, end=3, quoted=False),),
        ),
        TokenDesc(
            "~+42",
            expansions=(TokenFeature(TILDE, "~+42", start=0, end=4, quoted=False),),
        ),
        TokenDesc(
            "~-11",
            expansions=(TokenFeature(TILDE, "~-11", start=0, end=4, quoted=False),),
        ),
        TokenDesc(
            "~-/foo",
            expansions=(TokenFeature(TILDE, "~-", start=0, end=2, quoted=False),),
        ),
        TokenDesc(
            "~+/foo",
            expansions=(TokenFeature(TILDE, "~+", start=0, end=2, quoted=False),),
        ),
        TokenDesc("x~"),
        TokenDesc("~@"),
        TokenDesc("~"),
    )


def test_parse_backticks():
    tokens = tokenize(
        r"""
        echo $WOO`goal or "${wut}" span`" and `more stuff` !" `yes`
        """
    )
    assert len(tokens) == 3
    assert_token(tokens[0], "echo")
    assert_token(
        tokens[1],
        '$WOO`goal or "${wut}" span` and `more stuff` !',
        expansions=(
            TokenFeature(
                type=VARIABLE,
                content="WOO",
                start=0,
                end=4,
                quoted=False,
            ),
            TokenFeature(
                type=COMMAND,
                content='goal or "${wut}" span',
                start=4,
                end=27,
                quoted=False,
            ),
            TokenFeature(
                type=COMMAND,
                content="more stuff",
                start=32,
                end=44,
                quoted=True,
            ),
        ),
    )
    assert_token(
        tokens[2],
        "`yes`",
        expansions=(
            TokenFeature(
                type=COMMAND,
                content="yes",
                start=0,
                end=5,
                quoted=False,
            ),
        ),
    )


def test_parse_isolated_variables():
    tokens = tokenize(
        """
        echo $NAKED_UPPER $naked_lower_1 ${CURLIES1} ${curlies2} "$Quoted" "${QuotedBrackets}" '$SQuoted' '${SQuotedBrackets}'
        """
    )
    assert len(tokens) == 9
    assert_token(tokens[0], "echo")
    assert_token(
        tokens[1],
        "$NAKED_UPPER",
        expansions=(TokenFeature(VARIABLE, "NAKED_UPPER", 0, 12, False),),
    )
    assert_token(
        tokens[2],
        "$naked_lower_1",
        expansions=(TokenFeature(VARIABLE, "naked_lower_1", 0, 14, False),),
    )
    assert_token(
        tokens[3],
        "${CURLIES1}",
        expansions=(TokenFeature(VARIABLE, "CURLIES1", 0, 11, False),),
    )
    assert_token(
        tokens[4],
        "${curlies2}",
        expansions=(TokenFeature(VARIABLE, "curlies2", 0, 11, False),),
    )
    assert_token(
        tokens[5],
        "$Quoted",
        expansions=(TokenFeature(VARIABLE, "Quoted", 0, 7, True),),
    )
    assert_token(
        tokens[6],
        "${QuotedBrackets}",
        expansions=(TokenFeature(VARIABLE, "QuotedBrackets", 0, 17, True),),
    )
    assert_token(tokens[7], "$SQuoted")
    assert_token(tokens[8], "${SQuotedBrackets}")

    tokens = tokenize("echo $")
    assert len(tokens) == 2
    assert_token(tokens[0], "echo")
    assert_token(tokens[1], "$")


def test_parse_escaped_variables():
    tokens = tokenize(
        r"""
        echo \$NAKED_UPPER \\\$naked_lower_1 \\\\\${CURLIES1} \${curlies2} "\$Quoted" "\${QuotedBrackets}" '\$SQuoted' '\${SQuotedBrackets}'
        """
    )
    assert len(tokens) == 9
    assert_token(tokens[0], "echo")
    assert_token(tokens[1], "$NAKED_UPPER")
    assert_token(tokens[2], "\\$naked_lower_1")
    assert_token(tokens[3], "\\\\${CURLIES1}")
    assert_token(tokens[4], "${curlies2}")
    assert_token(tokens[5], "$Quoted")
    assert_token(tokens[6], "${QuotedBrackets}")
    assert_token(tokens[7], "\\$SQuoted")
    assert_token(tokens[8], "\\${SQuotedBrackets}")


def test_parse_special_variables():
    tokens = tokenize(
        """
        echo $FOO$0A$1A$2A$3A$4A$5A$6A$7A$8A$9A$$A$!A$*A$?A$-A$#A$${LOL}
        """
    )
    assert len(tokens) == 2
    assert_token(tokens[0], "echo")
    assert_token(
        tokens[1],
        "$FOO$0A$1A$2A$3A$4A$5A$6A$7A$8A$9A$$A$!A$*A$?A$-A$#A$${LOL}",
        expansions=(
            TokenFeature(
                type=VARIABLE,
                content="FOO",
                start=0,
                end=4,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="0",
                start=4,
                end=6,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="1",
                start=7,
                end=9,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="2",
                start=10,
                end=12,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="3",
                start=13,
                end=15,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="4",
                start=16,
                end=18,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="5",
                start=19,
                end=21,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="6",
                start=22,
                end=24,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="7",
                start=25,
                end=27,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="8",
                start=28,
                end=30,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="9",
                start=31,
                end=33,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="$",
                start=34,
                end=36,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="!",
                start=37,
                end=39,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="*",
                start=40,
                end=42,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="?",
                start=43,
                end=45,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="-",
                start=46,
                end=48,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="#",
                start=49,
                end=51,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="$",
                start=52,
                end=54,
                quoted=False,
            ),
        ),
    )


def test_parse_jumbled_variables():
    tokens = tokenize(
        """
        echo go$NAKED_UPPER$naked_lower_1>${CURLIES1}"${curlies2} $Quoted"ok''"${QuotedBrackets}"'$SQuoted'99'${SQuotedBrackets}'
        """
    )
    assert_token(tokens[0], "echo")
    assert_token(
        tokens[1],
        "go$NAKED_UPPER$naked_lower_1",
        expansions=(
            TokenFeature(
                type=VARIABLE,
                content="NAKED_UPPER",
                start=2,
                end=14,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="naked_lower_1",
                start=14,
                end=28,
                quoted=False,
            ),
        ),
    )
    assert_token(tokens[2], ">", type=2)
    assert_token(
        tokens[3],
        "${CURLIES1}${curlies2} $Quotedok${QuotedBrackets}$SQuoted99${SQuotedBrackets}",
        expansions=(
            TokenFeature(
                type=VARIABLE,
                content="CURLIES1",
                start=0,
                end=11,
                quoted=False,
            ),
            TokenFeature(
                type=VARIABLE,
                content="curlies2",
                start=11,
                end=22,
                quoted=True,
            ),
            TokenFeature(
                type=VARIABLE,
                content="Quoted",
                start=23,
                end=30,
                quoted=True,
            ),
            TokenFeature(
                type=VARIABLE,
                content="QuotedBrackets",
                start=32,
                end=49,
                quoted=True,
            ),
        ),
    )


@pytest.mark.parametrize(
    "example",
    [
        ">",
        "<",
        ">>",
        ">&",
        "<&",  # semantically invalid without following digit
        "&>",
        "&>>",
        ">|",
        "<>",
        "1>",
        "1>>",
        "1>&",
        "1>|",
        "1<>",
        "1>&2",
        "1<&2",
        ">&2",
        "<&2",
        "1>&-",
        "1<&-",
        "1>&2-",
        "1<&2-",
        ">&2-",
        "<&2-",
        "11>",
        "11>>",
        "11>&",
        "11>|",
        "11<>",
        "11>&12",
        "11<&12",
        ">&12",
        "<&12",
        "11>&-",
        "11<&-",
        "11>&12-",
        "11<&12-",
        ">&12-",
        "<&12-",
    ],
)
def test_tokenize_redirects(example):
    tokens = tokenize(f"echo foo {example} bar")

    assert len(tokens) == 4
    assert_token(tokens[0], "echo")
    assert_token(tokens[1], "foo")
    assert_token(tokens[2], example, type=2)
    assert_token(tokens[3], "bar")


@pytest.mark.parametrize(
    "example, equivalent",
    [
        ("echo foo 1&> bar", "echo foo 1 &> bar"),
        ("echo foo 1&>bar", "echo foo 1 &> bar"),
        ("echo foo 1&>> bar", "echo foo 1 &>> bar"),
        ("echo foo 1&>>bar", "echo foo 1 &>> bar"),
        ("echo foo 1&>2 bar", "echo foo 1 &> 2 bar"),
        ("echo foo 1&>>2bar", "echo foo 1 &>> 2bar"),
        ("echo foo 1>2 bar", "echo foo 1> 2 bar"),
        ("echo foo 1>2bar", "echo foo 1> 2bar"),
        ("echo foo 1>>2 bar", "echo foo 1>> 2 bar"),
        ("echo foo 1>>2bar", "echo foo 1>> 2bar"),
        ("echo foo 1&>2 bar", "echo foo 1&> 2 bar"),
        ("echo foo 1&>2bar", "echo foo 1&> 2bar"),
        ("echo foo 1&>>2 bar", "echo foo 1&>> 2 bar"),
        ("echo foo 1&>>2bar", "echo foo 1&>> 2bar"),
        ("echo foo 1>|2 bar", "echo foo 1>| 2 bar"),
        ("echo foo 1>|2bar", "echo foo 1>| 2bar"),
        ("echo foo 1<>2 bar", "echo foo 1<> 2 bar"),
        ("echo foo 1<>2bar", "echo foo 1<> 2bar"),
        ("echo foo >2 bar", "echo foo > 2 bar"),
        ("echo foo >2bar", "echo foo > 2bar"),
        ("echo foo >>2 bar", "echo foo >> 2 bar"),
        ("echo foo >>2bar", "echo foo >> 2bar"),
        ("echo foo &>2 bar", "echo foo &> 2 bar"),
        ("echo foo &>2bar", "echo foo &> 2bar"),
        ("echo foo &>>2 bar", "echo foo &>> 2 bar"),
        ("echo foo &>>2bar", "echo foo &>> 2bar"),
        ("echo foo >|2 bar", "echo foo >| 2 bar"),
        ("echo foo >|2bar", "echo foo >| 2bar"),
        ("echo foo <>2 bar", "echo foo <> 2 bar"),
        ("echo foo <>2bar", "echo foo <> 2bar"),
        ("echo foo 2x> bar", "echo foo 2x > bar"),
        ("echo foo 2x>bar", "echo foo 2x > bar"),
        ("echo foo &< bar", "echo foo & < bar"),
        ("echo foo &<bar", "echo foo & < bar"),
    ],
)
def test_tokenize_redirect_boundaries(example, equivalent):
    assert tokenize(example) == tokenize(equivalent)


@pytest.mark.parametrize(
    "example, error_msg",
    [
        ("echo foo &>|   bar", "Syntax error near redirect operator '&>'"),
        ("echo foo &><   bar", "Syntax error near redirect operator '&>'"),
        ("echo foo &>&   bar", "Syntax error near redirect operator '&>'"),
    ],
)
def test_tokenize_redirect_errors(example, error_msg):
    with pytest.raises(CommandLineTokenizeError) as e_info:
        tokenize(example)

    assert e_info.value.args[0] == error_msg


def test_tokens_with_digits():
    tokens = tokenize(f"echo foo 888bar")
    assert_token(tokens[0], "echo")
    assert_token(tokens[1], "foo")
    assert_token(tokens[2], "888bar")

    tokens = tokenize(f"echo foo888 bar")
    assert_token(tokens[0], "echo")
    assert_token(tokens[1], "foo888")
    assert_token(tokens[2], "bar")

    tokens = tokenize(f"echo foo888bar")
    assert_token(tokens[0], "echo")
    assert_token(tokens[1], "foo888bar")

    tokens = tokenize(f"echo foo 888-bar")
    assert_token(tokens[0], "echo")
    assert_token(tokens[1], "foo")
    assert_token(tokens[2], "888-bar")

    tokens = tokenize(f"echo foo 888 bar")
    assert_token(tokens[0], "echo")
    assert_token(tokens[1], "foo")
    assert_token(tokens[2], "888")
    assert_token(tokens[3], "bar")


def assert_token(
    token,
    content: str,
    includes_glob=False,
    escaped_glob_indices=tuple(),
    expansions=tuple(),
    type=0,
):
    assert token.content == content
    assert token.includes_glob == includes_glob
    assert token.escaped_glob_indices == escaped_glob_indices
    assert token.expansions == expansions
    assert token.type == TokenType(type)


class TokenDesc:
    def __init__(
        self,
        content: str,
        includes_glob=False,
        escaped_glob_indices=tuple(),
        expansions=tuple(),
        type=0,
    ):
        self.content = content
        self.includes_glob = includes_glob
        self.escaped_glob_indices = escaped_glob_indices
        self.expansions = expansions
        self.type = type


def assert_tokens(tokens, *token_descriptions):
    for index, desc in enumerate(token_descriptions):
        assert_token(
            tokens[index],
            content=desc.content,
            includes_glob=desc.includes_glob,
            escaped_glob_indices=desc.escaped_glob_indices,
            expansions=desc.expansions,
            type=desc.type,
        )
