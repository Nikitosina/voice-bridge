import ast
import sys

with open('/Users/nikitarat/Projects/voice-bridge/server.py') as f:
    tree = ast.parse(f.read())
print('Syntax OK')

# Check that all required imports are present
imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            imports.add(alias.name)
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ''
        for alias in node.names:
            imports.add(f"{module}.{alias.name}")

print("Imports found:")
for i in sorted(imports):
    print(f"  - {i}")
