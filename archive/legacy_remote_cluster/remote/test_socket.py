import sys
import socket
import time

sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from config import DASHBOARD_API, api_request, SERVER_IP
from manage_job import fetch_log

def main():
    script = """#!/bin/bash
python3 -c "
import socket
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', 9999))
s.listen(1)
print('Listening on 9999...')
s.settimeout(15)
try:
    conn, addr = s.accept()
    print('Accepted connection from', addr)
    conn.sendall(b'HELLO FROM SERVER')
    conn.close()
except Exception as e:
    print('Accept timeout or error:', e)
s.close()
"
"""
    payload = {"name": "test-socket", "gpu_count": 0, "time_limit_min": 5, "script": script}
    res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
    jid = res.get("id") or res.get("job_id")
    print(f"Submitted socket test job #{jid}")

    time.sleep(3)
    try:
        c = socket.socket()
        c.settimeout(5)
        c.connect((SERVER_IP, 9999))
        data = c.recv(1024)
        print("SUCCESS! RECEIVED FROM SERVER:", data.decode())
        c.close()
    except Exception as e:
        print("Connect error:", e)

    time.sleep(2)
    print(fetch_log(jid))

if __name__ == "__main__":
    main()

