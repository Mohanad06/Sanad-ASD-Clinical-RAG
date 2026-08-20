"""Find and print all non-ASCII positions in the test file, then write a clean UTF-8 version."""
with open('backend/test_conversation_memory.py', 'rb') as f:
    data = f.read()

bad_positions = [i for i, c in enumerate(data) if c > 127]
print(f"Non-ASCII byte positions (first 30): {bad_positions[:30]}")

# Show context around each non-ASCII byte
for pos in bad_positions[:30]:
    snippet = data[max(0, pos-10):pos+10]
    print(f"  pos={pos}: {snippet}")
