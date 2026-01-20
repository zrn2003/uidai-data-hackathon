from pyngrok import ngrok
import time
import sys

# Terminate existing tunnels
ngrok.kill()

# Open a HTTP tunnel on port 8501
try:
    public_url = ngrok.connect(8501)
    print(f"\n=======================================================")
    print(f"🌍 PUBLIC URL GENERATED: {public_url}")
    print(f"=======================================================\n")
    sys.stdout.flush()
    
    # Keep alive
    while True:
        time.sleep(1)
except Exception as e:
    print(f"Error: {e}")
