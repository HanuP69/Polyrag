import subprocess
import os
import time
import sys

def wait_for_engine(url="http://localhost:8000/health", timeout=120):
    import urllib.request
    print(f"    Waiting for engine at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(url, timeout=3)
            if resp.status == 200:
                print(f"    Engine ready! ({int(time.time()-start)}s)")
                return True
        except Exception:
            pass
        time.sleep(2)
    print("    WARNING: Engine did not respond in time.")
    return False

def kill_port(port):
    import subprocess
    try:
        out = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
        for line in out.strip().splitlines():
            if "LISTENING" in line:
                pid = line.strip().split()[-1]
                subprocess.run(["taskkill", "/F", "/PID", pid, "/T"], capture_output=True)
    except Exception:
        pass

def run():
    print("=" * 60)
    print("  PolyRAG Full-Stack Restarter")
    print("=" * 60)

    my_pid = os.getpid()
    root_dir = os.path.dirname(os.path.abspath(__file__))

    print("[1/5] Cleaning up ports...")
    kill_port(8000)
    kill_port(3001)
    kill_port(5173)
    subprocess.run(["powershell", "-Command", "Stop-Process -Name node -Force -ErrorAction SilentlyContinue"], capture_output=True)
    time.sleep(2)

    print("[2/5] Starting Python Engine (port 8000)...")
    subprocess.Popen(
        [sys.executable, "-m", "engine.main"],
        cwd=root_dir,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )

    print("[3/5] Waiting for Engine to load models...")
    if not wait_for_engine():
        print("  Engine failed to start. Aborting.")
        return

    print("[4/5] Starting Node.js Orchestrator (port 3001)...")
    subprocess.Popen(
        ["node", "index.js"],
        cwd=os.path.join(root_dir, "server"),
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )

    print("[5/5] Starting React Frontend (port 5173)...")
    subprocess.Popen(
        "npm run dev",
        cwd=os.path.join(root_dir, "client"),
        shell=True,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )

    time.sleep(2)
    print()
    print("=" * 60)
    print("  All services running!")
    print("  Engine:       http://localhost:8000")
    print("  Orchestrator: http://localhost:3001")
    print("  Frontend:     http://localhost:5173")
    print("=" * 60)

if __name__ == "__main__":
    run()
