"""Command-line interface for llmess."""

import argparse
import os
import shlex
import sys

from . import __version__

# Default system prompt (used unless -s overrides)
DEFAULT_INSTRUCT_PROMPT = (
    "Continue the following text exactly as if you were autocomplete. "
    "Do not add commentary, greetings, explanations, or markdown formatting. "
    "Just continue the text naturally from where it left off."
)


def parse_args(argv=None):
    """Parse command-line arguments.

    If argv is None, uses sys.argv[1:] with LLMESS env var prepended.
    LLMESS env var is parsed as shell-style arguments and prepended,
    so CLI args naturally override them.
    """
    if argv is None:
        argv = sys.argv[1:]
        llmess_env = os.environ.get("LLMESS", "")
        if llmess_env:
            env_args = shlex.split(llmess_env)
            argv = env_args + list(argv)

    parser = argparse.ArgumentParser(
        prog="llmess",
        description="A less pager that uses LLMs to hallucinate infinite file continuations.",
        epilog="When you scroll past the end of the file, llmess generates more content using your configured LLM.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="File to view (use '-' for stdin, or pipe content)",
    )
    parser.add_argument(
        "-m", "--model",
        dest="model",
        default=None,
        help="LLM model to use (default: your llm default model)",
    )
    parser.add_argument(
        "-s", "--system",
        dest="system",
        default=None,
        help="System prompt (default: instruct prompt for continuation)",
    )
    parser.add_argument(
        "-B", "--base",
        action="store_true",
        dest="base_mode",
        help="Base model mode: no system prompt (overridden by -s)",
    )
    parser.add_argument(
        "-S", "--stealth",
        action="store_true",
        help="Stealth mode: mimic less appearance exactly",
    )
    parser.add_argument(
        "-C", "--context",
        type=int,
        dest="context",
        default=None,
        metavar="CHARS",
        help="Characters of context to send to LLM (default: 2000)",
    )
    parser.add_argument(
        "-T", "--max-tokens",
        type=int,
        dest="max_tokens",
        default=None,
        metavar="N",
        help="Max tokens to generate per LLM call (limits wait time)",
    )
    parser.add_argument(
        "-o", "--option",
        nargs=2,
        action="append",
        dest="options",
        metavar=("KEY", "VALUE"),
        help="Model option to pass to llm (can be repeated)",
    )
    parser.add_argument(
        "-P", "--prefetch",
        type=int,
        nargs="?",
        const=2,
        default=0,
        metavar="SCREENS",
        help="Prefetch N screens ahead in background (default: 2 if flag used, 0 if not)",
    )
    parser.add_argument(
        "--real-lines",
        type=int,
        dest="real_lines",
        default=None,
        metavar="N",
        help="Show first N real lines, then generate continuations",
    )
    parser.add_argument(
        "--real-screen",
        action="store_true",
        dest="real_screen",
        help="Show first screenful of real content, then generate",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--install-prank",
        action="store_true",
        dest="install_prank",
        help="Output shell function to wrap 'less' with llmess",
    )
    return parser.parse_args(argv)


def load_content(file_arg):
    """
    Load content from file or stdin.

    Returns:
        tuple: (lines, source_name) where lines is list of strings
               and source_name is for display in status bar
    """
    # Case 1: Explicit stdin via '-'
    if file_arg == "-":
        if sys.stdin.isatty():
            print("llmess: no input (stdin is a terminal)", file=sys.stderr)
            sys.exit(1)
        content = sys.stdin.read()
        return content.splitlines(keepends=True), "[stdin]"

    # Case 2: No file argument - check for piped stdin
    if file_arg is None:
        if sys.stdin.isatty():
            # No file and no piped input - show help
            print("Usage: llmess [OPTIONS] FILE", file=sys.stderr)
            print("       llmess [OPTIONS] -", file=sys.stderr)
            print("       command | llmess [OPTIONS]", file=sys.stderr)
            print("\nTry 'llmess --help' for more information.", file=sys.stderr)
            sys.exit(1)
        # Piped input
        content = sys.stdin.read()
        return content.splitlines(keepends=True), "[stdin]"

    # Case 3: File argument provided
    try:
        with open(file_arg, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines(), file_arg
    except FileNotFoundError:
        print(f"llmess: {file_arg}: No such file or directory", file=sys.stderr)
        sys.exit(1)
    except IsADirectoryError:
        print(f"llmess: {file_arg}: Is a directory", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"llmess: {file_arg}: Permission denied", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"llmess: {file_arg}: {e.strerror}", file=sys.stderr)
        sys.exit(1)


def reopen_tty_for_curses():
    """
    Reopen /dev/tty for stdin so curses can read keyboard input
    after we've consumed piped stdin.
    """
    if not sys.stdin.isatty():
        try:
            with open("/dev/tty", "r") as tty:
                os.dup2(tty.fileno(), sys.stdin.fileno())
        except OSError as e:
            print(f"llmess: cannot open terminal for input: {e}", file=sys.stderr)
            sys.exit(1)


def get_model(args_model):
    """
    Determine which model to use.

    Priority:
    1. --model flag
    2. LLMESS_MODEL environment variable
    3. None (use llm's default)
    """
    if args_model:
        return args_model
    return os.environ.get("LLMESS_MODEL")


def get_options(args_options):
    """
    Determine which options to use.

    Priority:
    1. LLMESS_OPTIONS environment variable (base options)
    2. --option flags (override/extend env options)

    Returns:
        list: List of [key, value] pairs, or None if no options
    """
    options = []

    # Parse env var first (format: "key=value,key2=value2")
    env_options = os.environ.get("LLMESS_OPTIONS", "")
    if env_options:
        for pair in env_options.split(","):
            pair = pair.strip()
            if "=" in pair:
                key, value = pair.split("=", 1)
                options.append([key.strip(), value.strip()])

    # CLI options override/extend env options
    if args_options:
        # Build a dict to allow CLI to override env
        options_dict = {k: v for k, v in options}
        for key, value in args_options:
            options_dict[key] = value
        options = [[k, v] for k, v in options_dict.items()]

    return options if options else None


PRANK_SHELL_FUNCTION = '''\
# llmess prank wrapper for 'less'
# Add this to your target's ~/.bashrc or ~/.zshrc
less() {
    if [[ -t 1 ]]; then
        # Interactive terminal: use llmess in stealth mode
        llmess --stealth --real-screen --prefetch "$@"
    else
        # Piped output: use real less
        command less "$@"
    fi
}
'''


def main(argv=None):
    """Main entry point for llmess."""
    args = parse_args(argv)

    # Handle --install-prank (just print and exit)
    if args.install_prank:
        print("# Add this to target's shell config (~/.bashrc or ~/.zshrc):")
        print("#")
        print("# To uninstall, remove the function or run: unset -f less")
        print()
        print(PRANK_SHELL_FUNCTION)
        return

    # Resolve model (flag > env var > llm default)
    model = get_model(args.model)

    # Resolve options (env var as base, CLI overrides)
    options = get_options(args.options)

    # Load content before reopening tty (need to read stdin first if piped)
    lines, source_name = load_content(args.file)

    # Truncate to real-lines if specified (real-screen is handled in pager)
    if args.real_lines is not None and args.real_lines > 0:
        lines = lines[:args.real_lines]

    # Reopen tty for curses keyboard input if we consumed stdin
    reopen_tty_for_curses()

    # Import here to avoid curses initialization issues
    import curses
    from .pager import run_pager, check_llm_available

    # Resolve system prompt
    # Priority: -s (explicit) > -B (base mode) > default (instruct prompt)
    if args.system is not None:
        system = args.system if args.system else None
    elif args.base_mode:
        system = None
    else:
        system = DEFAULT_INSTRUCT_PROMPT

    # Check if llm is configured (warn but don't block - they might fix it)
    if not model:  # Only check if no explicit model given
        llm_ok, llm_msg = check_llm_available()
        if not llm_ok:
            print(f"Warning: {llm_msg}", file=sys.stderr)
            print("Generation will fail until this is fixed.", file=sys.stderr)
            print("", file=sys.stderr)

    try:
        curses.wrapper(run_pager, lines, source_name, model, system,
                       args.stealth, args.prefetch, args.context, args.max_tokens,
                       options, args.real_screen)
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C
        pass


if __name__ == "__main__":
    main()
