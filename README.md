# llmess

Some files end too early. Some files don't contain what you're looking for.

`llmess` is a `less` pager that addresses both issues:

- When you scroll past the end of a file, an LLM generates more content.
- When you search for a term that doesn't exist, the LLM generates content containing it.

Works with any model supported by the [`llm`](https://llm.datasette.io/) CLI.

## Installation

### Prerequisites

You need the [`llm`](https://llm.datasette.io/) CLI tool installed and configured:

```bash
# Install llm
pipx install llm

# Install plugins for your preferred provider
llm install llm-openrouter  # or llm-claude, llm-ollama, etc.

# Configure API keys and default model
llm keys set openrouter
llm models default openrouter/meta-llama/llama-3.1-405b
```

### Install llmess

```bash
pipx install llmess
```

Or with pip:

```bash
pip install llmess
```

> **Note**: llmess calls `llm` via subprocess, so it uses whatever `llm` is on your PATH. Your plugins and configuration are preserved.

## Usage

```bash
llmess myfile.txt                    # View a file
llmess -m gpt-4o myfile.txt          # Use a specific model
llmess -B myfile.txt                 # Base mode: no system prompt
cat file.txt | llmess                # Read from stdin
```

### Options

| Flag | Description |
|------|-------------|
| `-m`, `--model` | LLM model to use (default: your `llm` default) |
| `-s`, `--system` | System prompt (default: instruct prompt) |
| `-B`, `--base` | Base model mode: no system prompt |
| `-S`, `--stealth` | Mimic `less` appearance exactly-ish probably |
| `-T`, `--max-tokens N` | Max tokens to generate per LLM call |
| `-C`, `--context CHARS` | Characters of context to send to LLM (default: 2000) |
| `-o`, `--option KEY VALUE` | Model option to pass to llm (can be repeated) |
| `-P`, `--prefetch [N]` | Prefetch N screens ahead in background (off by default; 2 if flag given without N) |
| `--real-lines N` | Show first N real lines, then generate continuations |
| `--real-screen` | Show first screenful of real content, then generate |
| `--install-prank` | Output shell function to wrap `less` with llmess |
| `-V`, `--version` | Show version |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LLMESS` | Default CLI flags (like `LESS` for less) |
| `LLMESS_MODEL` | Default model |
| `LLMESS_OPTIONS` | Default model options (comma-separated `key=value` pairs) |

```bash
# In ~/.bashrc or ~/.zshrc
export LLMESS="-S -P"          # stealth + prefetch by default
export LLMESS_MODEL="openrouter/meta-llama/llama-3.1-405b"
```

Priority: CLI flags > `LLMESS` > `LLMESS_MODEL`/`LLMESS_OPTIONS` > built-in defaults.

## Controls

| Key | Action |
|-----|--------|
| ↑ / k | Scroll up one line |
| ↓ / j | Scroll down one line |
| Page Up / b | Scroll up one page |
| Page Down / f / Space | Scroll down one page |
| g | Go to top |
| G | Go to bottom |
| / | Search forward |
| n | Next match |
| N | Previous match |
| Esc | Clear search |
| s | Save to file |
| q | Quit |

When you reach the bottom, scrolling down triggers LLM generation.

## Search

Press `/` to search. Matches are highlighted.

If the term isn't found, llmess generates content containing it:

```bash
echo "Hello world" | llmess
# Press /password<Enter>
# → Generates content containing "password", jumps to match
```

This requires a system prompt (the default). With `-s` (no system prompt), search behaves normally.

## Base Models vs. Instruct Models

By default, llmess sends a system prompt instructing the model to continue text without commentary. This works well with instruct models and is ignored by base models.

**Base models** continue text naturally. Use `-B` to skip the system prompt:

```bash
llmess -B -m openrouter/meta-llama/llama-3.1-405b myfile.txt
```

Search-triggered generation is not available with base models since they don't follow instructions.

**Instruct models** work out of the box. Use `-s "custom prompt"` to override the default:

```bash
llmess -s "Continue this text exactly. No commentary." myfile.txt
```

Any model supported by `llm` works, including local models via Ollama.

## Modes

### Stealth Mode

With `-S`, llmess mimics the appearance of `less`:
- Status bar shows `filename lines 1-24 50%` format
- No `[GENERATING...]` indicator
- Shows `(END)` at bottom

### Prefetch Mode

With `-P`, llmess generates content ahead of where you're reading:

```bash
llmess -P file.txt       # Prefetch 2 screens ahead
llmess -P 5 file.txt     # Prefetch 5 screens ahead
```

### Real-Then-Fake Mode

Show real file content first, then generate continuations:

```bash
llmess --real-lines 50 ~/.bashrc    # First 50 lines are real
llmess --real-screen config.yaml    # First screenful is real
```

## Prank Installation

```bash
llmess --install-prank
```

Outputs a shell function that wraps `less` with llmess. Add it to the target's shell config.

The wrapper uses stealth mode, shows real content first, prefetches in background, and falls back to real `less` for piped output.

## License

MIT
