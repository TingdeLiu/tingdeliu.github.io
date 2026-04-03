import re
path = r"_posts/2026-04-03-Machine-Learning-Survey.md"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Add an empty line before <div align="center"> if there isn't one
text = re.sub(r'([^\n])\n<div align="center">', r'\1\n\n<div align="center">', text)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("Formatting fixed!")
