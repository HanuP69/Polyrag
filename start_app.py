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

def clear_blocked_memory():
    """Clear legacy python/node processes and unload Ollama models to free up CPU & GPU memory."""
    print("    Clearing blocked CPU & GPU memory...")
    current_pid = os.getpid()

    # 1. Kill other Python processes (Windows/Unix) to release BGE-M3 or reranker GPU VRAM / RAM
    if IS_WINDOWS:
        try:
            cmd = f"powershell -Command \"Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {{ $_.Id -ne {current_pid} }} | Stop-Process -Force\""
            subprocess.run(cmd, shell=True, capture_output=True)
            print("      Terminated legacy Python background processes.")
        except Exception as e:
            print(f"      Error killing python processes: {e}")
    else:
        try:
            cmd = f"pgrep -f python | grep -v {current_pid} | xargs kill -9 2>/dev/null"
            subprocess.run(cmd, shell=True)
            print("      Terminated legacy Python background processes.")
        except Exception:
            pass

    # 2. Unload Ollama models
    import urllib.request
    import json
    try:
        url = "http://localhost:11434"
        req = urllib.request.Request(f"{url}/api/ps")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
        
        for model in models:
            req = urllib.request.Request(
                f"{url}/api/generate",
                data=json.dumps({"model": model, "keep_alive": 0, "prompt": ""}).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                pass
            print(f"      Unloaded Ollama model: {model}")
    except Exception:
        pass

    # 3. Clear torch/gc cache locally if imported
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("      Local PyTorch VRAM cache cleared.")
    except Exception:
        pass

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

    print("[1/5] Cleaning up ports and memory...")
    kill_port(8000)
    kill_port(3001)
    kill_port(5173)
    if IS_WINDOWS:
        subprocess.run(
            ["powershell", "-Command", "Stop-Process -Name node -Force -ErrorAction SilentlyContinue"],
            capture_output=True
        )
    clear_blocked_memory()
    time.sleep(2)

    print("[2/5] Starting Python Engine (port 8000)...")
    popen_new_console(
        [sys.executable, "-m", "engine_v4.main"],
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
