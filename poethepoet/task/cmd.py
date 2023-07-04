import os
import re
from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Type, Union

from .base import PoeTask

if TYPE_CHECKING:
    from ..config import PoeConfig
    from ..context import RunContext
    from ..env.manager import EnvVarsManager


_GLOBCHARS_PATTERN = re.compile(r".*[\*\?\[]")
_QUOTED_TOKEN_PATTERN = re.compile(r"(^\".*\"$|^'.*'$)")


class CmdTask(PoeTask):
    """
    A task consisting of a reference to a shell command
    """

    content: str

    __key__ = "cmd"
    __options__: Dict[str, Union[Type, Tuple[Type, ...]]] = {
        "use_exec": bool,
    }

    def _handle_run(
        self,
        context: "RunContext",
        extra_args: Sequence[str],
        env: "EnvVarsManager",
    ) -> int:
        named_arg_values = self.get_named_arg_values(env)
        env.update(named_arg_values)

        if named_arg_values:
            # If named arguments are defined then pass only arguments following a double
            # dash token: `--`
            try:
                split_index = extra_args.index("--")
                extra_args = extra_args[split_index + 1 :]
            except ValueError:
                extra_args = tuple()

        cmd = (*self._resolve_args(context, env), *extra_args)

        self._print_action(" ".join(cmd), context.dry)

        return context.get_executor(self.invocation, env, self.options).execute(
            cmd, use_exec=self.options.get("use_exec", False)
        )

    def _resolve_args(self, context: "RunContext", env: "EnvVarsManager"):
        if os.environ.get("POE_LEGACY_CMD_MODE"):
            # TODO: get flags from config instead!
            return self._resolve_args_legacy_mode(context, env)

        from ..helpers.shell_tokenizer import (
            CommandLineTokenizeError,
            tokenize,
            TokenFeatureType,
            includes_glob,
            TokenType,
        )

        try:
            raw_tokens = tokenize(self.content)
        except CommandLineTokenizeError as error:
            raise PoeException(error.message) from CommandLineTokenizeError

        # Filter tokens
        tokens = []
        for token in raw_tokens:
            if token.type in (TokenType.CONTROL_OPERATOR, TokenType.REDIRECT_OPERATOR):
                raise ExecutionError(
                    f"Cannot execute cmd task with unsupported operator {token.content!r}"
                )
            if token.type == TokenType.COMMENT:
                continue
            for feature in token.expansions:
                if feature.type == TokenFeatureType.COMMAND:
                    raise ExecutionError(
                        "Cannot execute cmd task with token including a command"
                        f" expansion {token.content!r}"
                    )
                if feature.type == TokenFeatureType.ARITHMETIC:
                    raise ExecutionError(
                        "Cannot execute cmd task with token including a arithmetic"
                        f" expansion {token.content!r}"
                    )
                if feature.type == TokenFeatureType.GROUP:
                    raise ExecutionError(
                        "Cannot execute cmd task with token including a command "
                        f" group {token.content!r}"
                    )

            tokens.append(token)

        # How to keep indices accurate? variable => tilde => glob (if there is any)
        # split into parts?
        #

        # some complexity emerges because of the requirement to split tokens on
        # whitespace that comes from unquoted parameter expansions

        expanded_tokens = []
        for token in tokens:
            token_parts = []
            cursor = 0
            token_includes_glob = token.includes_glob
            for feature in token.expansions:
                token_parts.append(token.content[cursor : feature.start])

                if feature.type == TokenFeatureType.VARIABLE:
                    value = env.get(feature.content, default="")
                    if includes_glob(value):
                        token_includes_glob = True

                    if not feature.quoted:
                        while whitespace := re.match("", value):
                            # split the resulting token on whitespace in the value
                            token_parts.append(value[: whitespace.start()])
                            expanded_tokens.append("".join(token_parts))
                            token_parts = []
                            value = value[whitespace.end() :]

                    token_parts.append(value)

                    # TODO: escape glob chars if they're quoted or escaped in
                    #       value and token has glob or if value has unescaped glob !!
                    # -

                elif feature.type == TokenFeatureType.TILDE:
                    home_dir = os.environ.get("HOME")
                    if feature.content == "~" and home_dir:
                        token_parts.append(home_dir)
                    else:
                        # ignore advanced tilde expansions
                        token_parts.append(feature.content)

                cursor = feature.end

                # MAYBE glob start chars should be features!?!
                #   any better way to iterate expansions and globs together?

    def _resolve_args_legacy_mode(self, context: "RunContext", env: "EnvVarsManager"):
        import shlex
        from glob import glob

        updated_content = env.fill_template(self.content.strip())
        # Parse shell command tokens and check if they're quoted
        if self._is_windows:
            cmd_tokens = (
                (compat_token, bool(_QUOTED_TOKEN_PATTERN.match(compat_token)))
                for compat_token in shlex.split(
                    updated_content,
                    posix=False,
                    comments=True,
                )
            )
        else:
            cmd_tokens = (
                (posix_token, bool(_QUOTED_TOKEN_PATTERN.match(compat_token)))
                for (posix_token, compat_token) in zip(
                    shlex.split(
                        updated_content,
                        posix=True,
                        comments=True,
                    ),
                    shlex.split(
                        updated_content,
                        posix=False,
                        comments=True,
                    ),
                )
            )
        # Resolve any unquoted glob pattern paths
        result = []
        for cmd_token, is_quoted in cmd_tokens:
            if not is_quoted and _GLOBCHARS_PATTERN.match(cmd_token):
                # looks like a glob path so resolve it
                result.extend(glob(cmd_token, recursive=True))
            elif is_quoted and self._is_windows:
                # In this case, the cmd_token will still be wrapped in double or single
                # quotes. We need to remove those otherwise they'll be pass into the
                # command.
                result.append(cmd_token[1:-1])
            else:
                result.append(cmd_token)
        return result
