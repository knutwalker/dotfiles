#!/usr/bin/env python3

# Python http.server that sets Access-Control-Allow-Origin header.
# https://gist.github.com/razor-x/9542707

import sys
import http.server
import socketserver
import ssl

PORT = 8000

class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        http.server.SimpleHTTPRequestHandler.end_headers(self)


def server(port):
    httpd = socketserver.TCPServer(('', port), HTTPRequestHandler)
    httpd.socket = ssl.wrap_socket(httpd.socket, server_side=True, certfile='localhost.pem', ssl_version=ssl.PROTOCOL_TLS)
    return httpd

if __name__ == "__main__":
    port = PORT
    httpd = server(port)
    try:
        print("\nserving at localhost:" + str(port))
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n...shutting down http server")
        httpd.shutdown()
        sys.exit()