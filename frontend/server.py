#!/usr/bin/env python3
"""
简易开发服务器 —— 托管前端页面 + 反向代理 LangGraph API。
解决浏览器从 file:// 打开 HTML 时,跨域请求 localhost:8123 的 CORS 问题。

用法:
    python frontend/server.py

然后浏览器打开: http://localhost:9000
"""

import http.server
import json
import urllib.request
import urllib.error
import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent
LANGGRAPH_API = "http://localhost:8123"
LISTEN_PORT = 9000


class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    """GET / → index.html; 其他请求 → 反向代理到 LangGraph API"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            return super().do_GET()

        # 代理到 LangGraph
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")

    def do_OPTIONS(self):
        """CORS 预检"""
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _proxy(self, method):
        target_url = f"{LANGGRAPH_API}{self.path}"
        body = None
        content_type = "application/json"

        if method == "POST":
            length = int(self.headers.get("Content-Length", 0))
            if length > 0:
                body = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", content_type)

        req = urllib.request.Request(target_url, data=body, method=method)
        req.add_header("Content-Type", content_type)
        # 转发客户端的关键 headers
        for h in ("Authorization", "Accept", "X-Api-Key"):
            v = self.headers.get(h)
            if v:
                req.add_header(h, v)

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.send_response(resp.status)
                self._cors_headers()
                # 转发响应 headers
                for k, v in resp.headers.items():
                    if k.lower() in ("content-type", "content-length", "transfer-encoding"):
                        self.send_header(k, v)
                self.end_headers()
                # 流式写入响应体
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self._cors_headers()
            self.end_headers()
            err = e.read()
            if err:
                self.wfile.write(err)
        except urllib.error.URLError as e:
            self.send_response(502)
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "bad_gateway",
                "message": f"无法连接到 LangGraph API ({LANGGRAPH_API}): {e.reason}",
            }).encode())

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Api-Key")

    def log_message(self, format, *args):
        # 精简日志
        sys.stderr.write(f"[proxy] {args[0]}\n")


def main():
    print(f"""
╔══════════════════════════════════════════════════════╗
║  交易试算 Agent · 前端测试服务器                       ║
╠══════════════════════════════════════════════════════╣
║  前端地址:  http://localhost:{LISTEN_PORT}                  ║
║  代理后端:  {LANGGRAPH_API}                    ║
║                                                      ║
║  按 Ctrl+C 停止                                       ║
╚══════════════════════════════════════════════════════╝
""")
    server = http.server.HTTPServer(("0.0.0.0", LISTEN_PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[proxy] 已停止")
        server.server_close()


if __name__ == "__main__":
    main()
