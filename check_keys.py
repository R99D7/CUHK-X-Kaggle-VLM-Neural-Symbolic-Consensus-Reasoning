import os
print("API KEY:", "YES" if os.environ.get('GEMINI_API_KEY') else "NO")
print("OPENAI KEY:", "YES" if os.environ.get('OPENAI_API_KEY') else "NO")
