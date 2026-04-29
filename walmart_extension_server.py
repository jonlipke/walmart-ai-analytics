import http.server
import socketserver
import os
from pathlib import Path

CODE_DIR = Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Code")
PORT = 8765

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(CODE_DIR), **kwargs)
    
    def log_message(self, format, *args):
        pass  # Suppress access logs

print(f"Serving Walmart Extension at http://localhost:{PORT}")
print(f"Extension URL: http://localhost:{PORT}/walmart_extension.html")
print("Keep this running while using the Tableau Extension")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()