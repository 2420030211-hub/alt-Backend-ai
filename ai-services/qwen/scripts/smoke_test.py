import sys
import requests

def smoke_test(base_url="http://localhost:8002"):
    print(f"Testing Qwen Service at {base_url}...")
    try:
        r = requests.get(f"{base_url}/health", timeout=3)
        print("Health status:", r.json())
        
        chat_payload = {"message": "Hello, I was charged a mystery fee on my debit card."}
        c = requests.post(f"{base_url}/chat", json=chat_payload, timeout=5)
        print("Chat response:", c.json())
    except Exception as e:
        print("Smoke test failed (service may be offline):", e)

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
    smoke_test(url)
