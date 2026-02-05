"""Tests for llmess."""

import os
import sys
import pytest

from llmess.cli import parse_args, get_model, get_options
from llmess.pager import (wrap_lines, build_status_bar, should_prefetch, PrefetchState,
                          DEFAULT_CONTEXT_LIMIT, find_matches, augment_system_for_search)


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_file_only(self):
        args = parse_args(["myfile.txt"])
        assert args.file == "myfile.txt"
        assert args.model is None

    def test_stdin_dash(self):
        args = parse_args(["-"])
        assert args.file == "-"
        assert args.model is None

    def test_no_args(self):
        args = parse_args([])
        assert args.file is None
        assert args.model is None

    def test_model_short(self):
        args = parse_args(["-m", "gpt-4", "file.txt"])
        assert args.file == "file.txt"
        assert args.model == "gpt-4"

    def test_model_long(self):
        args = parse_args(["--model", "openrouter/llama-3", "file.txt"])
        assert args.file == "file.txt"
        assert args.model == "openrouter/llama-3"

    def test_model_without_file(self):
        args = parse_args(["--model", "gpt-4"])
        assert args.file is None
        assert args.model == "gpt-4"

    def test_system_short(self):
        args = parse_args(["-s", "Continue this text", "file.txt"])
        assert args.file == "file.txt"
        assert args.system == "Continue this text"

    def test_system_long(self):
        args = parse_args(["--system", "Be a completion model", "file.txt"])
        assert args.file == "file.txt"
        assert args.system == "Be a completion model"

    def test_model_and_system(self):
        args = parse_args(["-m", "gpt-4", "-s", "Continue", "file.txt"])
        assert args.file == "file.txt"
        assert args.model == "gpt-4"
        assert args.system == "Continue"

    def test_stealth_short(self):
        args = parse_args(["-S", "file.txt"])
        assert args.file == "file.txt"
        assert args.stealth is True

    def test_stealth_long(self):
        args = parse_args(["--stealth", "file.txt"])
        assert args.stealth is True

    def test_stealth_default_false(self):
        args = parse_args(["file.txt"])
        assert args.stealth is False

    def test_prefetch_flag_only(self):
        # When -P is used without a number, file must come first or use --
        args = parse_args(["file.txt", "-P"])
        assert args.prefetch == 2  # Default when flag used without value

    def test_prefetch_with_value(self):
        args = parse_args(["-P", "5", "file.txt"])
        assert args.prefetch == 5

    def test_prefetch_long_form(self):
        args = parse_args(["--prefetch", "3", "file.txt"])
        assert args.prefetch == 3

    def test_prefetch_default_zero(self):
        args = parse_args(["file.txt"])
        assert args.prefetch == 0

    def test_context_short(self):
        args = parse_args(["-C", "4000", "file.txt"])
        assert args.context == 4000

    def test_context_long(self):
        args = parse_args(["--context", "8000", "file.txt"])
        assert args.context == 8000

    def test_context_default_none(self):
        args = parse_args(["file.txt"])
        assert args.context is None

    def test_base_mode_short(self):
        args = parse_args(["-B", "file.txt"])
        assert args.base_mode is True

    def test_base_mode_long(self):
        args = parse_args(["--base", "file.txt"])
        assert args.base_mode is True

    def test_base_mode_default_false(self):
        args = parse_args(["file.txt"])
        assert args.base_mode is False

    def test_system_default_none(self):
        args = parse_args(["file.txt"])
        assert args.system is None

    def test_max_tokens_short(self):
        args = parse_args(["-T", "500", "file.txt"])
        assert args.max_tokens == 500

    def test_max_tokens_long(self):
        args = parse_args(["--max-tokens", "1000", "file.txt"])
        assert args.max_tokens == 1000

    def test_max_tokens_default_none(self):
        args = parse_args(["file.txt"])
        assert args.max_tokens is None

    def test_option_single(self):
        args = parse_args(["-o", "temperature", "0.7", "file.txt"])
        assert args.options == [["temperature", "0.7"]]

    def test_option_multiple(self):
        args = parse_args(["-o", "temperature", "0.5", "-o", "thinking", "true", "file.txt"])
        assert args.options == [["temperature", "0.5"], ["thinking", "true"]]

    def test_option_long_form(self):
        args = parse_args(["--option", "top_p", "0.9", "file.txt"])
        assert args.options == [["top_p", "0.9"]]

    def test_option_default_none(self):
        args = parse_args(["file.txt"])
        assert args.options is None

    def test_real_lines(self):
        args = parse_args(["--real-lines", "50", "file.txt"])
        assert args.real_lines == 50

    def test_real_lines_default_none(self):
        args = parse_args(["file.txt"])
        assert args.real_lines is None

    def test_real_screen(self):
        args = parse_args(["--real-screen", "file.txt"])
        assert args.real_screen is True

    def test_real_screen_default_false(self):
        args = parse_args(["file.txt"])
        assert args.real_screen is False

    def test_install_prank(self):
        args = parse_args(["--install-prank"])
        assert args.install_prank is True

    def test_install_prank_default_false(self):
        args = parse_args(["file.txt"])
        assert args.install_prank is False


class TestLLMESSEnvVar:
    """Tests for LLMESS environment variable (like LESS)."""

    def test_llmess_env_prepends_args(self, monkeypatch):
        """LLMESS env var args are prepended to CLI args."""
        monkeypatch.setenv("LLMESS", "-S")
        monkeypatch.setattr(sys, "argv", ["llmess", "file.txt"])
        args = parse_args()
        assert args.stealth is True
        assert args.file == "file.txt"

    def test_cli_overrides_llmess_env(self, monkeypatch):
        """CLI args override LLMESS env var."""
        monkeypatch.setenv("LLMESS", "-m env-model")
        monkeypatch.setattr(sys, "argv", ["llmess", "-m", "cli-model", "file.txt"])
        args = parse_args()
        assert args.model == "cli-model"

    def test_llmess_env_multiple_flags(self, monkeypatch):
        """LLMESS env var can contain multiple flags."""
        monkeypatch.setenv("LLMESS", "-S -P 3")
        monkeypatch.setattr(sys, "argv", ["llmess", "file.txt"])
        args = parse_args()
        assert args.stealth is True
        assert args.prefetch == 3

    def test_llmess_env_empty(self, monkeypatch):
        """Empty LLMESS env var is handled."""
        monkeypatch.setenv("LLMESS", "")
        monkeypatch.setattr(sys, "argv", ["llmess", "file.txt"])
        args = parse_args()
        assert args.file == "file.txt"

    def test_llmess_env_not_used_with_explicit_argv(self, monkeypatch):
        """When argv is explicitly passed, LLMESS env is not used."""
        monkeypatch.setenv("LLMESS", "-S")
        args = parse_args(["file.txt"])
        assert args.stealth is False


class TestGetModel:
    """Tests for model selection logic."""

    def test_args_model_takes_precedence(self, monkeypatch):
        """--model flag takes precedence over env var."""
        monkeypatch.setenv("LLMESS_MODEL", "env-model")
        assert get_model("arg-model") == "arg-model"

    def test_env_var_used_when_no_arg(self, monkeypatch):
        """LLMESS_MODEL env var is used when no --model flag."""
        monkeypatch.setenv("LLMESS_MODEL", "env-model")
        assert get_model(None) == "env-model"

    def test_returns_none_when_no_config(self, monkeypatch):
        """Returns None when no --model flag and no env var (use llm default)."""
        monkeypatch.delenv("LLMESS_MODEL", raising=False)
        assert get_model(None) is None


class TestGetOptions:
    """Tests for options resolution logic."""

    def test_env_var_only(self, monkeypatch):
        """LLMESS_OPTIONS env var is parsed correctly."""
        monkeypatch.setenv("LLMESS_OPTIONS", "temperature=0.7,max_tokens=500")
        result = get_options(None)
        assert result == [["temperature", "0.7"], ["max_tokens", "500"]]

    def test_cli_only(self, monkeypatch):
        """CLI options work without env var."""
        monkeypatch.delenv("LLMESS_OPTIONS", raising=False)
        result = get_options([["temperature", "0.5"]])
        assert result == [["temperature", "0.5"]]

    def test_cli_overrides_env(self, monkeypatch):
        """CLI options override env var options with same key."""
        monkeypatch.setenv("LLMESS_OPTIONS", "temperature=0.7,max_tokens=500")
        result = get_options([["temperature", "0.9"]])
        # CLI temperature should override env temperature
        result_dict = {k: v for k, v in result}
        assert result_dict["temperature"] == "0.9"
        assert result_dict["max_tokens"] == "500"

    def test_cli_extends_env(self, monkeypatch):
        """CLI options extend env var options with different keys."""
        monkeypatch.setenv("LLMESS_OPTIONS", "temperature=0.7")
        result = get_options([["top_p", "0.9"]])
        result_dict = {k: v for k, v in result}
        assert result_dict["temperature"] == "0.7"
        assert result_dict["top_p"] == "0.9"

    def test_returns_none_when_no_options(self, monkeypatch):
        """Returns None when no env var and no CLI options."""
        monkeypatch.delenv("LLMESS_OPTIONS", raising=False)
        assert get_options(None) is None

    def test_handles_spaces_in_env(self, monkeypatch):
        """Handles spaces around delimiters in env var."""
        monkeypatch.setenv("LLMESS_OPTIONS", "temperature = 0.7 , max_tokens = 500")
        result = get_options(None)
        result_dict = {k: v for k, v in result}
        assert result_dict["temperature"] == "0.7"
        assert result_dict["max_tokens"] == "500"


class TestWrapLines:
    """Tests for line wrapping."""

    def test_empty_input(self):
        result = wrap_lines([], 80)
        assert result == []

    def test_short_lines(self):
        lines = ["hello\n", "world\n"]
        result = wrap_lines(lines, 80)
        assert len(result) == 2
        assert result[0] == ("hello", 0)
        assert result[1] == ("world", 1)

    def test_exact_width(self):
        lines = ["12345\n"]
        result = wrap_lines(lines, 5)
        assert len(result) == 1
        assert result[0] == ("12345", 0)

    def test_wrap_long_line(self):
        lines = ["1234567890\n"]
        result = wrap_lines(lines, 5)
        assert len(result) == 2
        assert result[0] == ("12345", 0)
        assert result[1] == ("67890", 0)

    def test_wrap_very_long_line(self):
        lines = ["123456789012345\n"]
        result = wrap_lines(lines, 5)
        assert len(result) == 3
        assert result[0] == ("12345", 0)
        assert result[1] == ("67890", 0)
        assert result[2] == ("12345", 0)

    def test_empty_line(self):
        lines = ["hello\n", "\n", "world\n"]
        result = wrap_lines(lines, 80)
        assert len(result) == 3
        assert result[0] == ("hello", 0)
        assert result[1] == ("", 1)
        assert result[2] == ("world", 2)

    def test_preserves_original_index(self):
        lines = ["short\n", "this is a longer line that will wrap\n", "end\n"]
        result = wrap_lines(lines, 10)
        # "this is a longer line that will wrap" wraps to 4 lines
        # Check that original indices are preserved
        assert result[0][1] == 0  # "short"
        assert result[1][1] == 1  # first part of long line
        assert result[2][1] == 1  # second part of long line
        # ... etc

    def test_no_newline_at_end(self):
        lines = ["hello"]  # No trailing newline
        result = wrap_lines(lines, 80)
        assert result == [("hello", 0)]

    def test_handles_zero_width(self):
        """Zero width should fallback to 80."""
        lines = ["hello\n"]
        result = wrap_lines(lines, 0)
        assert result == [("hello", 0)]

    def test_handles_negative_width(self):
        """Negative width should fallback to 80."""
        lines = ["hello\n"]
        result = wrap_lines(lines, -10)
        assert result == [("hello", 0)]


class TestContextLimit:
    """Tests for context handling."""

    def test_context_limit_constant(self):
        """Ensure default context limit is a reasonable value."""
        assert DEFAULT_CONTEXT_LIMIT > 0
        assert DEFAULT_CONTEXT_LIMIT <= 10000  # Sanity check


class TestStatusBar:
    """Tests for status bar building."""

    def test_normal_mode_includes_llmess(self):
        wrapped = [("line1", 0), ("line2", 1)]
        lines = ["line1\n", "line2\n"]
        status = build_status_bar("test.txt", 0, 24, wrapped, lines,
                                  False, None, False, stealth=False)
        assert "llmess" in status
        assert "test.txt" in status

    def test_stealth_mode_no_llmess(self):
        wrapped = [("line1", 0), ("line2", 1)]
        lines = ["line1\n", "line2\n"]
        status = build_status_bar("test.txt", 0, 24, wrapped, lines,
                                  False, None, False, stealth=True)
        assert "llmess" not in status
        assert "test.txt" in status
        assert "lines" in status

    def test_stealth_mode_at_bottom_shows_end(self):
        wrapped = [("line1", 0), ("line2", 1)]
        lines = ["line1\n", "line2\n"]
        status = build_status_bar("test.txt", 0, 24, wrapped, lines,
                                  False, None, True, stealth=True)
        assert "(END)" in status

    def test_normal_mode_at_bottom_shows_hint(self):
        wrapped = [("line1", 0), ("line2", 1)]
        lines = ["line1\n", "line2\n"]
        status = build_status_bar("test.txt", 0, 24, wrapped, lines,
                                  False, None, True, stealth=False)
        assert "END" in status
        assert "generates" in status

    def test_stealth_mode_generating_no_indicator(self):
        wrapped = [("line1", 0), ("line2", 1)]
        lines = ["line1\n", "line2\n"]
        status = build_status_bar("test.txt", 0, 24, wrapped, lines,
                                  True, None, False, stealth=True)
        assert "GENERATING" not in status

    def test_normal_mode_generating_shows_indicator(self):
        wrapped = [("line1", 0), ("line2", 1)]
        lines = ["line1\n", "line2\n"]
        status = build_status_bar("test.txt", 0, 24, wrapped, lines,
                                  True, None, False, stealth=False)
        assert "GENERATING" in status

    def test_prefetching_shows_buffering(self):
        wrapped = [("line1", 0), ("line2", 1)]
        lines = ["line1\n", "line2\n"]
        status = build_status_bar("test.txt", 0, 24, wrapped, lines,
                                  False, None, False, stealth=False, prefetching=True)
        assert "buffering" in status

    def test_stealth_prefetching_no_end(self):
        wrapped = [("line1", 0), ("line2", 1)]
        lines = ["line1\n", "line2\n"]
        status = build_status_bar("test.txt", 0, 24, wrapped, lines,
                                  False, None, True, stealth=True, prefetching=True)
        assert "(END)" not in status  # Don't show END while prefetching


class TestSearch:
    """Tests for search functionality."""

    def test_find_matches_found(self):
        wrapped = [("hello world", 0), ("password here", 1), ("goodbye", 2)]
        matches = find_matches(wrapped, "password")
        assert len(matches) == 1
        assert matches[0][0] == 1  # Line index
        assert matches[0][1] == 0  # Char position

    def test_find_matches_not_found(self):
        wrapped = [("hello world", 0), ("goodbye", 1)]
        matches = find_matches(wrapped, "password")
        assert matches == []

    def test_find_matches_case_insensitive(self):
        wrapped = [("PASSWORD here", 0), ("Hello WORLD", 1)]
        matches = find_matches(wrapped, "password")
        assert len(matches) == 1
        matches2 = find_matches(wrapped, "world")
        assert len(matches2) == 1

    def test_find_matches_multiple(self):
        wrapped = [("password one", 0), ("no match", 1), ("password two", 2)]
        matches = find_matches(wrapped, "password")
        assert len(matches) == 2
        assert matches[0][0] == 0
        assert matches[1][0] == 2

    def test_find_matches_char_position(self):
        wrapped = [("hello password world", 0)]
        matches = find_matches(wrapped, "password")
        assert len(matches) == 1
        assert matches[0][1] == 6  # "hello " is 6 chars

    def test_augment_system_for_search_with_existing(self):
        system = "Continue this text naturally."
        augmented = augment_system_for_search(system, "secret")
        assert "secret" in augmented
        assert "Continue this text naturally" in augmented
        assert "exact string" in augmented.lower()

    def test_augment_system_for_search_none(self):
        augmented = augment_system_for_search(None, "password")
        assert "password" in augmented
        assert "exact string" in augmented.lower()

    def test_augment_system_for_search_empty(self):
        augmented = augment_system_for_search("", "test")
        assert "test" in augmented


class TestPrefetch:
    """Tests for prefetch functionality."""

    def test_should_prefetch_disabled(self):
        assert should_prefetch(0, 24, 100, 0) is False

    def test_should_prefetch_plenty_of_buffer(self):
        # At line 0, 24 visible, 100 total, want 2 screens (48 lines) buffer
        # Buffer remaining = 100 - 24 = 76 > 48, no prefetch needed
        assert should_prefetch(0, 24, 100, 2) is False

    def test_should_prefetch_low_buffer(self):
        # At line 50, 24 visible, 60 total, want 2 screens (48 lines) buffer
        # Buffer remaining = 60 - 74 = -14 < 48, need prefetch
        assert should_prefetch(50, 24, 60, 2) is True

    def test_should_prefetch_at_bottom(self):
        # At line 76, 24 visible, 100 total
        # Buffer remaining = 100 - 100 = 0 < 48, need prefetch
        assert should_prefetch(76, 24, 100, 2) is True

    def test_prefetch_state_initial(self):
        state = PrefetchState()
        assert state.is_generating() is False
        assert state.get_error() is None

    def test_prefetch_state_start_and_finish(self):
        state = PrefetchState()
        assert state.start_generation() is True
        assert state.is_generating() is True
        assert state.start_generation() is False  # Already running
        state.finish_generation()
        assert state.is_generating() is False

    def test_prefetch_state_error(self):
        state = PrefetchState()
        state.start_generation()
        state.finish_generation("test error")
        assert state.is_generating() is False
        assert state.get_error() == "test error"
        assert state.get_error() is None  # Cleared after read
