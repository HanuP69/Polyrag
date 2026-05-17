import subprocess
import os
import time
import sys

IS_WINDOWS = sys.platform == "win32"

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
    if IS_WINDOWS:
        try:
            out = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
            for line in out.strip().splitlines():
                if "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    subprocess.run(["taskkill", "/F", "/PID", pid, "/T"], capture_output=True)
        except Exception:
            pass
    else:
        # Linux / macOS
        try:
            subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
        except Exception:
            pass

def popen_new_console(args, cwd=None, env=None, shell=False):
    """Launch subprocess in a new console window (Windows) or background (Linux/macOS)."""
    kwargs = dict(cwd=cwd, env=env, shell=shell)
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    return subprocess.Popen(args, **kwargs)

def run():
    print("=" * 60)
    print("  PolyRAG Full-Stack Restarter")
    print("=" * 60)

    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Build env with PYTHONPATH set so `engine.*` always resolves
    engine_env = os.environ.copy()
    existing_pythonpath = engine_env.get("PYTHONPATH", "")
    engine_env["PYTHONPATH"] = (
        root_dir + os.pathsep + existing_pythonpath
        if existing_pythonpath else root_dir
    )

    print("[1/5] Cleaning up ports...")
    kill_port(8000)
    kill_port(3001)
    kill_port(5173)
    if IS_WINDOWS:
        subprocess.run(
            ["powershell", "-Command", "Stop-Process -Name node -Force -ErrorAction SilentlyContinue"],
            capture_output=True
        )
    time.sleep(2)

    print("[2/5] Starting Python Engine (port 8000)...")
    popen_new_console(
        [sys.executable, "-m", "engine.main"],
        cwd=root_dir,
        env=engine_env,
    )

    print("[3/5] Waiting for Engine to load models...")
    if not wait_for_engine():
        print("  Engine failed to start. Aborting.")
        return

    print("[4/5] Starting Node.js Orchestrator (port 3001)...")
    popen_new_console(
        ["node", "index.js"],
        cwd=os.path.join(root_dir, "server"),
    )

    print("[5/5] Starting React Frontend (port 5173)...")
    popen_new_console(
        "npm run dev",
        cwd=os.path.join(root_dir, "client"),
        shell=True,
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
