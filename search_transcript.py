import json

path = r"C:\Users\MUTHURAMANRAMANATHAN\.gemini\antigravity\brain\532bd4ba-532c-4372-92de-f52d656a65ed\.system_generated\logs\transcript.jsonl"
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if '0.4237' in line or '0.42' in line:
            data = json.loads(line)
            content = data.get('content', '')
            if content:
                print(f"[{data.get('type')}] {content[:1000]}")
