"""Reach a host through a chosen Mac interface, without root or routing changes.

macOS sends traffic for a subnet down one interface. When the controller and the
Jetson share 192.168.123.0/24 on two adapters, the Jetson's link loses. A socket
bound to an interface (IP_BOUND_IF) bypasses the routing table, so:

    # ssh through en8
    ssh -o ProxyCommand=".venv/bin/python tools/via_iface.py en8 %h %p" unitree@192.168.123.164
    # forward localhost:8080 to the Jetson's camera page through en8 (Ctrl-C to stop)
    .venv/bin/python tools/via_iface.py en8 192.168.123.164 8080 --listen 8080

No sudo, no host routes. A proper switch or a host route makes this unnecessary.
"""
import argparse, select, socket, sys, threading

IP_BOUND_IF = 25   # macOS <netinet/in.h>


def connect(iface, host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_IP, IP_BOUND_IF, socket.if_nametoindex(iface))
    s.settimeout(6); s.connect((host, port)); s.settimeout(None)
    return s


def pump(a, b):
    try:
        while True:
            r, _, _ = select.select([a, b], [], [])
            for x in r:
                data = x.recv(65536)
                if not data: return
                (b if x is a else a).sendall(data)
    except OSError:
        return
    finally:
        for x in (a, b):
            try: x.close()
            except OSError: pass


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("iface"); ap.add_argument("host"); ap.add_argument("port", type=int)
ap.add_argument("--listen", type=int, help="local port to forward from; without it, act as an ssh ProxyCommand on stdin/stdout")
a = ap.parse_args()

if a.listen is None:                      # ProxyCommand mode: stdin/stdout <-> remote
    r = connect(a.iface, a.host, a.port)
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    def to_remote():
        while (d := inp.read1(65536)): r.sendall(d)
        r.shutdown(socket.SHUT_WR)
    threading.Thread(target=to_remote, daemon=True).start()
    while (d := r.recv(65536)):
        out.write(d); out.flush()
else:                                     # forwarder mode
    ls = socket.socket(); ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("127.0.0.1", a.listen)); ls.listen(16)
    print(f"localhost:{a.listen} -> {a.host}:{a.port} via {a.iface}", flush=True)
    while True:
        c, _ = ls.accept()
        try:
            threading.Thread(target=pump, args=(c, connect(a.iface, a.host, a.port)), daemon=True).start()
        except OSError as e:
            print("connect failed:", e, flush=True); c.close()
