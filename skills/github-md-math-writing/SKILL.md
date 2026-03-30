---
name: github-md-math-writing
description: Write or edit Markdown notes that must render correctly on GitHub, especially files containing math. Use for GitHub README/notes cleanup, KaTeX-compatible inline math, block math conversion, and distinguishing code identifiers from mathematical notation.
---

# GitHub Markdown Math Writing

Use this skill when editing Markdown that must render on GitHub.

## Goal

Produce Markdown that is readable in source form and renders correctly on GitHub, especially when prose mixes code identifiers and math notation.

## Rules

1. Treat GitHub rendering compatibility as the default target, not generic LaTeX compatibility.
2. Prefer the simplest math syntax that preserves meaning.
3. Distinguish code identifiers from mathematical symbols:
   - Keep real code/config/API names in backticks.
   - Write mathematical objects as math, not code.
4. Do not assume common LaTeX macros are supported just because they work elsewhere.

## Math Formatting

### Inline math

Prefer GitHub-safe inline math:

```md
$`\pi_{old}`$
$`A_t`$
$`r_t`$
$`-x \log x`$
```

Do not use plain `` `pi_old` `` when the text is referring to a mathematical object.

### Block math

Prefer fenced math blocks:

````md
```math
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
```
````

Avoid `$$ ... $$` blocks when editing for GitHub unless there is a strong reason to keep them.

## KaTeX/GitHub Compatibility

Prefer these commands:

- `\frac`
- `\log`
- `\exp`
- `\min`
- `\max`
- `\pi`
- `\theta`
- superscripts and subscripts like `x_t`, `A_t`, `\pi_{ref}`

Avoid or replace these unless you have verified GitHub accepts them:

- `\operatorname{...}`
- `\text{...}` inside math
- `\mathrm{...}`
- `\left` and `\right`
- spacing macros such as `\,` and `\!`

Safe replacements:

- `\operatorname{clip}(x)` -> `clip(x)`
- `\mathrm{KL}_t` -> `KL_t`
- `\pi_{\mathrm{old}}` -> `\pi_{old}`
- `\left( ... \right)` -> `( ... )`

## Prose Conventions

When the note explains formulas in prose:

- Use inline math for symbols: `$`r_t`$`, `$`A_t`$`, `$`\pi_{ref}`$`
- Use backticks only for code/config names: `ref_state_dict`, `old_logprobs`, `optimizer.step()`
- If adding your own interpretation, label it explicitly so it does not get confused with the source explanation

Recommended pattern:

```md
**我的理解：**
$`r_t`$ 表示当前策略和旧策略在同一个动作上的概率比值。
```

## Editing Workflow

1. Locate all math-like identifiers in prose.
2. Decide whether each one is math or code.
3. Convert math objects to GitHub-safe inline math.
4. Convert block formulas to fenced `math` blocks.
5. Remove unsupported or risky macros.
6. Re-read the surrounding sentence to ensure punctuation and Chinese text still read naturally.

## Quick Checks

After editing, run checks like:

```bash
rg -n '\\$\\$|\\\\operatorname|\\\\text\\{|\\\\mathrm\\{|\\\\left|\\\\right|\\\\!|\\\\,' path/to/file.md
rg -n '`pi_|`A_t`|`r_t`|`m_t`|`s_t`|`a_t`' path/to/file.md
```

If any of those remain, verify they are intentional.

## Default Judgment

If you are unsure whether something is code or math:

- If it appears in an equation or denotes a probability/symbol/variable, use math.
- If it is a literal field name, function name, config key, filename, or command, use backticks.
