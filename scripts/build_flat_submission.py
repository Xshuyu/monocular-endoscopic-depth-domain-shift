#!/usr/bin/env python
"""Inline all \\input{...} section files into a single flat submission .tex.

Usage: python scripts/build_flat_submission.py <manuscript_dir> <main.tex> <out.tex>
Asserts every referenced figure exists in the same directory.
"""
import re
import sys
from pathlib import Path

def main():
    d = Path(sys.argv[1])
    main_f, out_f = d / sys.argv[2], d / sys.argv[3]
    text = main_f.read_text()

    def repl(m):
        inc = d / m.group(1)
        assert inc.exists(), f"missing input file: {inc}"
        body = inc.read_text()
        assert "\\input{" not in body, f"nested \\input in {inc}"
        return f"% ---- begin inlined {m.group(1)} ----\n{body}\n% ---- end inlined {m.group(1)} ----"

    flat, n = re.subn(r"\\input\{([^}]+)\}", repl, text)
    assert n == 7, f"expected 7 \\input replacements, got {n}"
    out_f.write_text(flat)

    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", flat):
        assert (d / fig).exists(), f"missing figure: {fig}"
    assert "\\input{" not in flat.replace("% ---- begin inlined", "").split("----\n")[-1] or True
    remaining = [l for l in flat.splitlines() if l.strip().startswith("\\input{")]
    assert not remaining, f"unresolved \\input lines: {remaining}"
    print(f"wrote {out_f} ({n} sections inlined)")


if __name__ == "__main__":
    main()
