import subprocess
import os
import re

def edit_marp_md(filepath='out/scripts/slides-1.2.md'):
    r"""
    Normalize MARP markdown:
    - Keep LaTeX intact; convert \( \) → $...$ (inline), \[ \] → $$...$$ (block).
    - Replace ONLY literal JSON-style '\n' with real newlines, but DO NOT touch LaTeX commands like \neq, \nabla.
    - Collapse multiple consecutive '---' separators to a single one.
    - Remove stray '## ---' headings that sometimes appear before separators.
    - Note: The file should NOT start with a slide delimiter.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1) Delimiter character replacements (safe, not using regex groups here)
    #    We only replace the tokens themselves; content remains intact.
    text = text.replace('\\(', '$').replace('\\)', '$')
    # Use $$ for display math blocks
    text = text.replace('\\[', '$$').replace('\\]', '$$')

    # 2) Replace literal backslash-n with a real newline ONLY when it's not starting a LaTeX command.
    #    Pattern: \n NOT followed by a letter (so \neq, \nabla, \newcommand, ... are preserved)
    text = re.sub(r'\\n(?![A-Za-z])', '\n', text)

    # 3) Remove accidental '## ---' headings
    text = text.replace('## ---', '')

    # 4) Collapse runs of slide separators to a single '---\n'
    #    Matches lines that consist of optional space + '---' + optional space, repeated.
    text = re.sub(r'(?:[ \t]*---[ \t]*\n){2,}', '---\n', text)

    # 4b) Normalize any single separator line to exactly '---\n'
    text = re.sub(r'^[ \t]*---[ \t]*\n', '---\n', text, flags=re.MULTILINE)

    # 4c) Ensure a BLANK LINE before each slide delimiter (except file start)
    #    Turn '...\n---\n' (no blank line) into '...\n\n---\n'
    text = re.sub(r'([^\n])\n---\n', r'\1\n\n---\n', text)

    # 4d) Fix LaTeX backslashes:
    #     - Many model outputs escape backslashes -> '\\\\command'; we want '\command'.
    #       Replace double-backslash before letters with single backslash.
    #       Keep '\\\\' linebreaks intact (they won't match because next char isn't a letter).
    text = re.sub(r'\\\\([A-Za-z]+)', r'\\\1', text)

    # 4e) Minimal rescue for commonly broken operator sequences that lost the backslash entirely
    #     e.g., 'operatorname{arccosh}' inside $$..$$ → '\operatorname{arccosh}'
    def _restore_op(m):
        inner = m.group(1)
        fixed = re.sub(r'(?<!\\\\)(operatorname|ln|cosh|sinh|tanh|sin|cos|tan|log)\\b', r'\\\1', inner)
        return f'$${fixed}$$'
    # Only attempt inside display-math where backslashes were likely stripped.
    text = re.sub(r'\$\$(.*?)\$\$', _restore_op, text, flags=re.DOTALL)

    # 4f) Fix \boxed blocks:
    #     a) Wrap bare \boxed{...} outside math with $...$
    text = re.sub(r'(?<!\$)\\boxed\{([^}]*)\}(?!\$)', r'$\\boxed{\1}$', text)
    #     b) If a line contains '$\\boxed{...}$' but '$' counts become odd due to stray $, we will fix in step 6.

    # 4g) If a line has common math commands but no $, we will wrap later in _wrap_display_math_if_needed.
    #     Here we only normalize spaces around '$$' to avoid '$$text$$\\' artifacts:
    text = re.sub(r"\$\$\s*\\\s*\n", "$$\n", text)

    # 6) Balance inline math per line (avoid raw LaTeX leaking):
    #    If a line has an odd number of single '$' (ignoring '$$...$$'), append a trailing '$'.
    def _balance_inline_math(line: str) -> str:
        if "$$" in line:
            return line
        tmp = re.sub(r'\$\$.*?\$\$', '', line)
        tmp = re.sub(r'\\\$', '', tmp)
        singles = tmp.count('$')
        if singles % 2 == 1:
            return line + '$'
        return line

    # New: If a line contains LaTeX commands but no $, wrap with $$...$$ (and fix boxed).
    def _wrap_display_math_if_needed(line: str) -> str:
        r"""
        If a line contains LaTeX commands (e.g., \\dfrac, \\boxed) but has no `$`:
        - If the line has a short prose prefix ending with ':' (e.g., "정답:"), keep the prefix and wrap only the math tail.
        - Otherwise wrap the WHOLE line with $$ ... $$. Also clean nested $ around \boxed.
        Always remove stray backslash after closing $$ at EOL.
        """
        if "$" in line or not re.search(r"\\[A-Za-z]+", line):
            return line

        # Split optional prose prefix like '정답:' or '해:'
        m = re.match(r"^(?P<prefix>[^$\\]*?:)\s*(?P<math>\\[A-Za-z].*)$", line)
        if m:
            prefix = m.group("prefix")
            mathpart = m.group("math")
            mathpart = re.sub(r"\$\\boxed\{([^}]*)\}\$", r"\\boxed{\1}", mathpart)
            return f"{prefix} $$ {mathpart} $$"

        # No explicit prefix; wrap entire line
        line = re.sub(r"\$\\boxed\{([^}]*)\}\$", r"\\boxed{\1}", line)
        return f"$${line}$$"

    def _wrap_outside_math_fragments(line: str) -> str:
        r"""
        Scan a line, preserving existing $...$ and $$...$$ math, and for any
        NON-math fragment that still contains TeX-like commands (e.g., \\quad, \\dfrac, \\\\, \\text{...}),
        wrap ONLY that fragment with inline math $...$.
        This prevents raw red/boxed text leaking into the slide.
        """
        if "\\" not in line:
            return line

        # Split by display math first
        parts = re.split(r"(\$\$.*?\$\$)", line)
        for i, part in enumerate(parts):
            if i % 2 == 1:  # this is a $$...$$ block, keep as is
                continue
            # Now split the non-display fragment by inline math
            subparts = re.split(r"(\$.*?\$)", part)
            for j, sp in enumerate(subparts):
                if j % 2 == 1:  # this is an inline $...$ block, keep as is
                    continue
                if re.search(r"\\[A-Za-z]+|\\{|\\}|\\\\", sp):
                    # Clean common artifacts before wrapping
                    sp_clean = sp
                    # Remove accidental spaces before backslashes that break commands
                    sp_clean = re.sub(r"\s+\\\s+", r" \\", sp_clean)
                    # If it's just whitespace, skip
                    if sp_clean.strip():
                        subparts[j] = f"${sp_clean}$"
            parts[i] = "".join(subparts)
        return "".join(parts)

    # 7) Split any slide that has more than 8 NON-EMPTY lines (to avoid overflow).
    #    Blank lines are preserved but do not count toward the limit.
    def _split_long_slide(block: str, title_suffix_idx: int = 1) -> list[str]:
        r"""
        Split a slide into chunks with at most 8 NON-EMPTY lines each.
        Blank lines are preserved but do not count toward the limit.
        Add a small '(이어서)' marker on subsequent chunks.
        Also balance inline math per line.
        """
        lines = block.strip("\n").split("\n")

        def flush_chunk(buf: list[str]) -> str:
            return "\n".join(_balance_inline_math(ln) for ln in buf)

        chunks, buf = [], []
        non_empty = 0
        for ln in lines:
            buf.append(ln)
            if ln.strip():
                non_empty += 1
            if non_empty >= 8:
                chunks.append(flush_chunk(buf))
                buf, non_empty = [], 0
        if buf:
            chunks.append(flush_chunk(buf))

        # Add continuation marker for 2nd+ chunks
        for i in range(1, len(chunks)):
            chunk_lines = chunks[i].split("\n")
            if chunk_lines and chunk_lines[0].startswith("#"):
                chunk_lines[0] = chunk_lines[0] + " (이어서)"
            else:
                chunk_lines.insert(0, "#### (이어서)")
            # keep at most 8 non-empty lines; remove trailing lines if needed
            kept, new_chunk = 0, []
            for ll in chunk_lines:
                if ll.strip():
                    kept += 1
                if kept <= 8:
                    new_chunk.append(ll)
            chunks[i] = "\n".join(new_chunk)

        return chunks

    # Process slide-by-slide: ensure blank line before '---', fix math, and split long slides.
    blocks = re.split(r'\n---\n', text.strip())
    processed_blocks = []
    for blk in blocks:
        # First, ensure math-heavy lines are wrapped in display mode if they lack $.
        fixed_lines = []
        for ln in blk.split("\n"):
            ln = _wrap_display_math_if_needed(ln)
            ln = _wrap_outside_math_fragments(ln)
            fixed_lines.append(ln)
        blk_fixed = "\n".join(fixed_lines)
        processed_blocks.extend(_split_long_slide(blk_fixed))

    # Re-join with required blank line before each delimiter
    text = ('\n\n---\n').join(processed_blocks)
    # If the result accidentally starts with a delimiter, drop it to avoid an empty first slide
    if text.startswith('---\n'):
        text = text[4:]

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    with open('temp.md', 'w', encoding='utf-8') as f:
        f.write(text)

def generate_and_rename(filepath='temp.md', outdir='out/slides', prefix='slides'):
    subprocess.run([
        "npx", "marp", filepath,
        "--images", "png",
        "--allow-local-files",
        "--no-stdin",
        "--output", outdir + "/" + prefix
    ], check=True)

    # Ensure all generated files have .png extension (defensive for older marp versions)
    if os.path.isdir(outdir):
        for name in os.listdir(outdir):
            if name.startswith(prefix + ".") and not name.lower().endswith(".png"):
                src = os.path.join(outdir, name)
                dst = src + ".png"
                os.rename(src, dst)

if __name__ == '__main__':
    edit_marp_md()
    generate_and_rename()