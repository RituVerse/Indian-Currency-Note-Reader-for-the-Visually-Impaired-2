"""
serve.py  –  Local Development Web Server
==========================================
Why do we need this?
  When you double-click index.html, the browser uses the file:// protocol.
  TensorFlow.js CANNOT load model weights over file:// due to browser security
  (CORS policy).  You must use http://localhost instead.

  This tiny script starts Python's built-in HTTP server, which serves the
  current folder at http://localhost:8000.

Run:
  python serve.py

Then open:  http://localhost:8000
Press Ctrl+C to stop the server.
"""

import http.server
import socketserver
import webbrowser
import threading
import os

PORT = 8000

# Change to the project directory so all files are served correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class SilentHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Same as SimpleHTTPRequestHandler but quieter (no per-request log spam)."""
    def log_message(self, format, *args):
        # Only print errors (status ≥ 400), not every single GET request
        if args[1] >= "400":
            super().log_message(format, *args)


def open_browser():
    """Open the browser 1 second after the server starts."""
    import time
    time.sleep(1)
    webbrowser.open(f"http://localhost:{PORT}")


print(f"  Starting local server at http://localhost:{PORT}")
print(f"  Open that URL in your browser.")
print(f"  Press Ctrl+C to stop.\n")

# Open browser in a background thread so the server starts first
threading.Thread(target=open_browser, daemon=True).start()

with socketserver.TCPServer(("", PORT), SilentHTTPHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
