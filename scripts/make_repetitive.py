"""A27 pilot filler: one Moby Dick paragraph tiled to filler length.

Lowest-complexity natural text at matched tokenizer statistics: same
vocabulary and style as the ordinary-prose control, near-zero novel
content per token past the first paragraph.
"""

text = open("data/mobydick.txt", encoding="utf-8", errors="ignore").read()
# a mid-book paragraph, well past the 30k-char header skip the prose
# control uses, so the two conditions share a source but not a prefix
start = text.find("Call me Ishmael")
if start < 0:
    start = 40000
para = text[start:start + 1200].strip() + "\n\n"
with open("data/repetitive.txt", "w", encoding="utf-8") as f:
    f.write(para * 400)
print(f"paragraph chars: {len(para)}, tiled x400")
