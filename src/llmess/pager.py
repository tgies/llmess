"""Curses-based pager with LLM continuation."""

import curses
import subprocess
import threading

# Default context limit for LLM prompt (characters)
DEFAULT_CONTEXT_LIMIT = 2000


def check_llm_available():
    """
    Check if llm CLI is configured and ready.

    Returns:
        tuple: (ok, message) where ok is bool and message explains any issue
    """
    try:
        # Check if llm is installed and has a model available
        result = subprocess.run(
            ["llm", "models", "default"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, None
        # No default model set
        return False, "No default model set. Run: llm models default <model>"
    except FileNotFoundError:
        return False, "llm not found. Install with: pip install llm"
    except subprocess.TimeoutExpired:
        return True, None  # Assume OK if it's just slow
    except Exception:
        return True, None  # Assume OK, will fail later with better context


class PrefetchState:
    """Thread-safe state management for prefetch generation."""

    def __init__(self):
        self.lock = threading.Lock()
        self.generating = False
        self.error = None
        self.thread = None

    def start_generation(self):
        """Try to start generation. Returns True if started, False if already running."""
        with self.lock:
            if self.generating:
                return False
            self.generating = True
            self.error = None
            return True

    def finish_generation(self, error=None):
        """Mark generation as complete."""
        with self.lock:
            self.generating = False
            self.error = error

    def is_generating(self):
        """Check if generation is in progress."""
        with self.lock:
            return self.generating

    def get_error(self):
        """Get and clear any error from last generation."""
        with self.lock:
            error = self.error
            self.error = None
            return error


def get_continuation(context, model=None, system=None, context_limit=None,
                     max_tokens=None, options=None):
    """
    Call the llm CLI to generate continuation.

    Args:
        context: The text context to continue from
        model: Optional model name to pass to llm
        system: Optional system prompt to pass to llm
        context_limit: Max characters of context to send (default: DEFAULT_CONTEXT_LIMIT)
        max_tokens: Max tokens to generate (passed to llm -o max_tokens)
        options: List of [key, value] pairs to pass as llm -o options

    Returns:
        tuple: (lines, error) where lines is list of strings and error is str or None
    """
    try:
        # Limit context to avoid overly long prompts
        limit = context_limit if context_limit is not None else DEFAULT_CONTEXT_LIMIT
        prompt = context[-limit:]

        # Build command
        cmd = ["llm", "--no-stream"]
        if model:
            cmd.extend(["-m", model])
        if system:
            cmd.extend(["-s", system])
        if max_tokens:
            cmd.extend(["-o", "max_tokens", str(max_tokens)])
        if options:
            for key, value in options:
                cmd.extend(["-o", key, value])

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(input=prompt)

        if process.returncode != 0:
            # Collapse newlines to spaces for status bar display
            error_msg = " ".join(stderr.strip().split())
            # Provide helpful suggestions for common errors
            if "No key found" in error_msg or "API key" in error_msg:
                return [], "No LLM configured. Run: llm keys set <provider>"
            elif "No model" in error_msg:
                return [], "No default model. Run: llm models default <model>"
            return [], f"llm error: {error_msg}"

        return stdout.splitlines(keepends=True), None

    except FileNotFoundError:
        return [], "llm not found - install with: pip install llm"
    except Exception as e:
        return [], f"Error: {e}"


def find_matches(wrapped, term):
    """
    Find all occurrences of term in wrapped lines (case-insensitive).

    Args:
        wrapped: List of (display_line, original_line_index) tuples
        term: Search term

    Returns:
        list: List of (wrapped_line_idx, char_pos) tuples
    """
    matches = []
    term_lower = term.lower()
    for idx, (display_line, _) in enumerate(wrapped):
        pos = display_line.lower().find(term_lower)
        if pos != -1:
            matches.append((idx, pos))
    return matches


def augment_system_for_search(system, term):
    """
    Add search term requirement to system prompt for instruct models.

    Args:
        system: Original system prompt (may be None)
        term: Search term to include

    Returns:
        str: Augmented system prompt
    """
    base = system or ""
    addition = f" Include this exact string in your continuation: '{term}'"
    return (base + addition).strip()


def render_line_with_highlight(stdscr, row, line, width, search_term):
    """
    Render a line with search term highlighting (like less).

    Args:
        stdscr: Curses screen
        row: Row number to render at
        line: The line text to render
        width: Terminal width
        search_term: Term to highlight (case-insensitive), or None
    """
    line = line[:width]  # Truncate to screen width

    if not search_term:
        stdscr.addstr(row, 0, line)
        return

    # Find all matches in this line (case-insensitive)
    term_lower = search_term.lower()
    line_lower = line.lower()
    term_len = len(search_term)

    col = 0
    pos = 0
    while pos < len(line):
        match_pos = line_lower.find(term_lower, pos)
        if match_pos == -1:
            # No more matches - render rest of line normally
            stdscr.addstr(row, col, line[pos:])
            break
        else:
            # Render text before match normally
            if match_pos > pos:
                stdscr.addstr(row, col, line[pos:match_pos])
                col += match_pos - pos

            # Render match with standout (like less)
            stdscr.attron(curses.A_STANDOUT)
            stdscr.addstr(row, col, line[match_pos:match_pos + term_len])
            stdscr.attroff(curses.A_STANDOUT)
            col += term_len
            pos = match_pos + term_len
    else:
        # Loop completed without break - line ended exactly at/after last match
        pass


def wrap_lines(lines, width):
    """
    Wrap lines to fit within terminal width.

    Args:
        lines: List of content lines
        width: Terminal width

    Returns:
        list: List of (display_line, original_line_index) tuples
    """
    if width <= 0:
        width = 80  # Fallback

    wrapped = []
    for orig_idx, line in enumerate(lines):
        # Strip trailing newline for display
        line = line.rstrip("\n\r")

        if not line:
            wrapped.append(("", orig_idx))
            continue

        # Wrap long lines
        while len(line) > width:
            wrapped.append((line[:width], orig_idx))
            line = line[width:]

        wrapped.append((line, orig_idx))

    return wrapped


def build_status_bar(source_name, scroll_pos, content_height, wrapped, lines,
                     generating, status_error, at_bottom, stealth=False,
                     prefetching=False, search_term=None, search_matches=None,
                     search_match_idx=0):
    """
    Build the status bar string.

    Args:
        source_name: File name or [stdin]
        scroll_pos: Current scroll position in wrapped lines
        content_height: Number of visible content lines
        wrapped: List of wrapped display lines
        lines: Original content lines
        generating: Whether synchronous generation is in progress
        status_error: Error message to display, or None
        at_bottom: Whether scrolled to bottom
        stealth: Whether to use less-style status bar
        prefetching: Whether prefetch generation is running in background

    Returns:
        str: Status bar text
    """
    if stealth:
        # Mimic less status bar format
        if not wrapped:
            return f" {source_name} (empty)"

        # Calculate visible line range
        first_visible = scroll_pos + 1
        last_visible = min(scroll_pos + content_height, len(wrapped))

        # In stealth mode, don't reveal true line count - show ???
        # or a plausible fake that grows
        fake_total = max(len(lines), scroll_pos + content_height + 100)

        if generating:
            # In stealth mode, don't show generating - just show normal status
            # The pause is the only tell
            pct = min(99, int(100 * last_visible / fake_total))
            return f" {source_name} lines {first_visible}-{last_visible} {pct}%"
        elif status_error:
            # Still show errors even in stealth (user needs to know)
            return f" {source_name} [{status_error}]"
        elif at_bottom and not prefetching:
            # Mimic less (END) indicator
            return f" {source_name} lines {first_visible}-{last_visible} (END)"
        else:
            pct = min(99, int(100 * last_visible / fake_total))
            return f" {source_name} lines {first_visible}-{last_visible} {pct}%"
    else:
        # Normal llmess status bar
        if wrapped:
            _, orig_line = wrapped[min(scroll_pos, len(wrapped) - 1)]
            line_info = f"Line {orig_line + 1}/{len(lines)}"
        else:
            line_info = "Empty"

        status = f" llmess - {source_name} - {line_info}"

        if generating:
            if search_term:
                status += f" [searching '{search_term}'...]"
            else:
                status += " [GENERATING...]"
        elif status_error:
            status += f" [{status_error}]"
        elif search_term:
            if search_matches:
                status += f" ['{search_term}' {search_match_idx + 1}/{len(search_matches)}]"
            else:
                status += f" ['{search_term}' not found]"
        elif prefetching:
            status += " [buffering...]"
        elif at_bottom:
            status += " [END - ↓ generates more]"
        else:
            status += " [q:quit /search]"

        return status


def generate_sync(lines, model, system, context_limit=None, max_tokens=None, options=None):
    """
    Synchronously generate continuation and append to lines.

    Args:
        lines: Content lines list (modified in place)
        model: LLM model name
        system: System prompt
        context_limit: Max characters of context to send
        max_tokens: Max tokens to generate
        options: List of [key, value] pairs for llm options

    Returns:
        str or None: Error message if generation failed
    """
    context = "".join(lines)
    new_lines, error = get_continuation(context, model, system, context_limit, max_tokens, options)

    if error:
        return error
    elif new_lines:
        # Ensure last line has newline for continuity
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.extend(new_lines)
        return None
    else:
        return "LLM returned empty response"


def prefetch_worker(lines, model, system, prefetch_state, context_limit=None,
                    max_tokens=None, options=None):
    """
    Background worker for prefetch generation.

    Args:
        lines: Content lines list (modified in place - thread safe due to GIL)
        model: LLM model name
        system: System prompt
        prefetch_state: PrefetchState instance for coordination
        context_limit: Max characters of context to send
        max_tokens: Max tokens to generate
        options: List of [key, value] pairs for llm options
    """
    try:
        context = "".join(lines)
        new_lines, error = get_continuation(
            context, model, system, context_limit, max_tokens, options
        )

        if error:
            prefetch_state.finish_generation(error)
        elif new_lines:
            # Ensure last line has newline for continuity
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.extend(new_lines)
            prefetch_state.finish_generation(None)
        else:
            prefetch_state.finish_generation("LLM returned empty response")
    except Exception as e:
        prefetch_state.finish_generation(f"Prefetch error: {e}")


def should_prefetch(scroll_pos, content_height, wrapped_len, prefetch_screens):
    """
    Check if we should start prefetch generation.

    Args:
        scroll_pos: Current scroll position
        content_height: Visible content lines
        wrapped_len: Total wrapped lines
        prefetch_screens: Number of screens to buffer ahead

    Returns:
        bool: True if prefetch should be triggered
    """
    if prefetch_screens <= 0:
        return False

    visible_end = scroll_pos + content_height
    buffer_remaining = wrapped_len - visible_end
    buffer_target = content_height * prefetch_screens

    # Start prefetch when less than target buffer remains
    return buffer_remaining < buffer_target


def start_prefetch(lines, model, system, prefetch_state, context_limit=None,
                   max_tokens=None, options=None):
    """
    Start prefetch generation in background thread.

    Args:
        lines: Content lines list
        model: LLM model name
        system: System prompt
        prefetch_state: PrefetchState instance
        context_limit: Max characters of context to send
        max_tokens: Max tokens to generate
        options: List of [key, value] pairs for llm options

    Returns:
        bool: True if prefetch was started, False if already running
    """
    if not prefetch_state.start_generation():
        return False  # Already generating

    thread = threading.Thread(
        target=prefetch_worker,
        args=(lines, model, system, prefetch_state, context_limit, max_tokens, options),
        daemon=True,
    )
    prefetch_state.thread = thread
    thread.start()
    return True


def run_pager(stdscr, lines, source_name, model=None, system=None,
              stealth=False, prefetch_screens=0, context_limit=None,
              max_tokens=None, options=None, real_screen=False):
    """
    Run the interactive pager.

    Args:
        stdscr: curses window
        lines: List of content lines
        source_name: Display name for status bar
        model: Optional LLM model name
        system: Optional system prompt for LLM
        stealth: Whether to use less-compatible stealth mode
        prefetch_screens: Number of screens to buffer ahead (0 = disabled)
        context_limit: Max characters of context to send to LLM
        max_tokens: Max tokens to generate per LLM call
        options: List of [key, value] pairs for llm options
        real_screen: If True, truncate to first screenful then generate
    """
    curses.curs_set(0)  # Hide cursor

    # Handle real_screen: truncate to first screenful
    if real_screen:
        height, width = stdscr.getmaxyx()
        content_height = height - 1  # Reserve line for status bar
        # Truncate lines to fit one screen (accounting for wrapping)
        wrapped_count = 0
        truncate_at = 0
        for i, line in enumerate(lines):
            line_display = line.rstrip("\n\r")
            # Count wrapped lines this line will produce
            if not line_display:
                wrapped_count += 1
            else:
                wrapped_count += max(1, (len(line_display) + width - 1) // width)
            if wrapped_count >= content_height:
                truncate_at = i + 1
                break
            truncate_at = i + 1
        lines[:] = lines[:truncate_at]  # Truncate in place

    # Track state
    scroll_pos = 0
    generating = False  # Synchronous generation in progress
    status_error = None

    # Prefetch state (only used if prefetch_screens > 0)
    prefetch_state = PrefetchState() if prefetch_screens > 0 else None

    # Search state
    search_term = None
    search_matches = []
    search_match_idx = 0

    # Use non-blocking getch for prefetch mode to allow responsive UI
    if prefetch_screens > 0:
        stdscr.timeout(100)  # 100ms timeout for getch

    while True:
        height, width = stdscr.getmaxyx()
        content_height = height - 1  # Reserve last line for status bar

        # Wrap lines to current width
        wrapped = wrap_lines(lines, width)

        # Check for prefetch errors
        if prefetch_state:
            prefetch_error = prefetch_state.get_error()
            if prefetch_error:
                status_error = prefetch_error

        stdscr.clear()

        # Calculate scroll bounds
        max_scroll = max(0, len(wrapped) - content_height)
        at_bottom = scroll_pos >= max_scroll

        # Check if prefetch should be triggered
        prefetching = False
        if prefetch_state:
            prefetching = prefetch_state.is_generating()
            if not prefetching and should_prefetch(
                scroll_pos, content_height, len(wrapped), prefetch_screens
            ):
                start_prefetch(
                    lines, model, system, prefetch_state,
                    context_limit, max_tokens, options
                )
                prefetching = True

        # Display wrapped lines
        for i in range(content_height):
            display_idx = scroll_pos + i
            if display_idx < len(wrapped):
                display_line, _ = wrapped[display_idx]
                try:
                    render_line_with_highlight(stdscr, i, display_line, width, search_term)
                except curses.error:
                    pass
            else:
                # Show ~ for lines beyond content (like vim/less)
                stdscr.attron(curses.A_DIM)
                try:
                    stdscr.addstr(i, 0, "~")
                except curses.error:
                    pass
                finally:
                    stdscr.attroff(curses.A_DIM)

        # Build and display status bar
        status = build_status_bar(
            source_name, scroll_pos, content_height, wrapped, lines,
            generating, status_error, at_bottom, stealth, prefetching,
            search_term, search_matches, search_match_idx
        )

        stdscr.attron(curses.A_REVERSE)
        try:
            padded_status = status[:width].ljust(width)
            stdscr.addstr(height - 1, 0, padded_status[:width])
        except curses.error:
            pass
        finally:
            stdscr.attroff(curses.A_REVERSE)

        stdscr.refresh()

        # Input handling
        key = stdscr.getch()

        # In prefetch mode, -1 means timeout (no key pressed)
        if key == -1:
            continue  # Just redraw and check prefetch status

        # Clear transient error on any keypress
        if status_error:
            status_error = None

        if key == ord("q"):
            break

        elif key == curses.KEY_UP or key == ord("k"):
            if scroll_pos > 0:
                scroll_pos -= 1

        elif key == curses.KEY_DOWN or key == ord("j"):
            max_scroll = max(0, len(wrapped) - content_height)
            if scroll_pos < max_scroll:
                scroll_pos += 1
            elif scroll_pos >= max_scroll and not generating:
                # At bottom - trigger synchronous generation if no prefetch
                # or if prefetch buffer is exhausted
                if prefetch_state and prefetch_state.is_generating():
                    # Prefetch in progress - just wait (non-blocking)
                    pass
                elif prefetch_state and len(wrapped) > scroll_pos + content_height:
                    # Prefetch added content - can scroll
                    scroll_pos += 1
                else:
                    # No prefetch or buffer exhausted - sync generate
                    generating = True
                    if not stealth:
                        # Show generating status
                        if wrapped:
                            _, orig_line = wrapped[min(scroll_pos, len(wrapped) - 1)]
                            line_info = f"Line {orig_line + 1}/{len(lines)}"
                        else:
                            line_info = "Empty"
                        gen_status = f" llmess - {source_name} - {line_info} [GENERATING...]"
                        stdscr.attron(curses.A_REVERSE)
                        try:
                            stdscr.addstr(height - 1, 0, gen_status[:width].ljust(width))
                        except curses.error:
                            pass
                        finally:
                            stdscr.attroff(curses.A_REVERSE)
                        stdscr.refresh()

                    status_error = generate_sync(
                        lines, model, system, context_limit, max_tokens, options
                    )
                    generating = False

        elif key == curses.KEY_PPAGE or key == ord("b"):
            # Page up
            scroll_pos = max(0, scroll_pos - content_height)

        elif key == curses.KEY_NPAGE or key == ord("f") or key == ord(" "):
            # Page down
            max_scroll = max(0, len(wrapped) - content_height)
            if scroll_pos < max_scroll:
                scroll_pos = min(max_scroll, scroll_pos + content_height)
            elif not generating:
                # At bottom - same logic as down arrow
                if prefetch_state and prefetch_state.is_generating():
                    pass
                elif prefetch_state and len(wrapped) > scroll_pos + content_height:
                    scroll_pos = min(max_scroll, scroll_pos + content_height)
                else:
                    generating = True
                    if not stealth:
                        if wrapped:
                            _, orig_line = wrapped[min(scroll_pos, len(wrapped) - 1)]
                            line_info = f"Line {orig_line + 1}/{len(lines)}"
                        else:
                            line_info = "Empty"
                        gen_status = f" llmess - {source_name} - {line_info} [GENERATING...]"
                        stdscr.attron(curses.A_REVERSE)
                        try:
                            stdscr.addstr(height - 1, 0, gen_status[:width].ljust(width))
                        except curses.error:
                            pass
                        finally:
                            stdscr.attroff(curses.A_REVERSE)
                        stdscr.refresh()

                    status_error = generate_sync(
                        lines, model, system, context_limit, max_tokens, options
                    )
                    generating = False

        elif key == ord("g"):
            # Go to top
            scroll_pos = 0

        elif key == ord("G"):
            # Go to bottom
            scroll_pos = max(0, len(wrapped) - content_height)

        elif key == curses.KEY_HOME:
            scroll_pos = 0

        elif key == curses.KEY_END:
            scroll_pos = max(0, len(wrapped) - content_height)

        elif key == curses.KEY_RESIZE:
            # Terminal resized - just redraw
            pass

        elif key == ord("/"):
            # Search mode - switch to blocking input
            stdscr.timeout(-1)  # Blocking mode for search input
            curses.echo()
            curses.curs_set(1)
            # Clear status line (avoid writing to last cell - causes curses ERR)
            stdscr.addstr(height - 1, 0, "/" + " " * (width - 2))
            stdscr.move(height - 1, 1)
            stdscr.refresh()
            try:
                search_input = stdscr.getstr(height - 1, 1, width - 2).decode('utf-8')
            except curses.error:
                search_input = ""
            curses.noecho()
            curses.curs_set(0)
            stdscr.timeout(100)  # Restore non-blocking mode

            if search_input:
                search_term = search_input
                search_match_idx = 0
                search_matches = find_matches(wrapped, search_term)

                if search_matches:
                    # Found - jump to first match
                    scroll_pos = max(0, min(search_matches[0][0], len(wrapped) - content_height))
                elif system:
                    # Not found but in instruct mode - trigger search-aware generation
                    generating = True
                    augmented_system = augment_system_for_search(system, search_term)

                    # Show searching status
                    search_status = build_status_bar(
                        source_name, scroll_pos, content_height, wrapped, lines,
                        True, None, at_bottom, stealth, False,
                        search_term, [], 0
                    )
                    stdscr.attron(curses.A_REVERSE)
                    try:
                        stdscr.addstr(height - 1, 0, search_status[:width].ljust(width))
                    except curses.error:
                        pass
                    finally:
                        stdscr.attroff(curses.A_REVERSE)
                    stdscr.refresh()

                    # Generate with augmented prompt
                    status_error = generate_sync(lines, model, augmented_system,
                                                 context_limit, max_tokens, options)
                    generating = False

                    # Re-wrap and search again
                    wrapped = wrap_lines(lines, width)
                    search_matches = find_matches(wrapped, search_term)
                    if search_matches:
                        match_line = search_matches[0][0]
                        scroll_pos = max(0, min(match_line, len(wrapped) - content_height))
                # else: base mode, no match - status bar will show "not found"

        elif key == ord("n"):
            # Next search match
            if search_matches and len(search_matches) > 1:
                search_match_idx = (search_match_idx + 1) % len(search_matches)
                scroll_pos = max(0, min(search_matches[search_match_idx][0],
                                       len(wrapped) - content_height))

        elif key == ord("N"):
            # Previous search match
            if search_matches and len(search_matches) > 1:
                search_match_idx = (search_match_idx - 1) % len(search_matches)
                scroll_pos = max(0, min(search_matches[search_match_idx][0],
                                       len(wrapped) - content_height))

        elif key == 27:  # Escape
            # Clear search
            search_term = None
            search_matches = []
            search_match_idx = 0

        elif key == ord("s"):
            # Save to file (like less)
            stdscr.timeout(-1)  # Blocking mode for filename input
            curses.echo()
            curses.curs_set(1)
            # Prompt for filename (like less)
            stdscr.addstr(height - 1, 0, "log file: " + " " * (width - 11))
            stdscr.move(height - 1, 10)
            stdscr.refresh()
            try:
                filename = stdscr.getstr(height - 1, 10, width - 11).decode('utf-8').strip()
            except curses.error:
                filename = ""
            curses.noecho()
            curses.curs_set(0)
            stdscr.timeout(100)  # Restore non-blocking mode

            if filename:
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    status_error = f"Saved to {filename}"
                except OSError as e:
                    status_error = f"Save failed: {e.strerror}"
