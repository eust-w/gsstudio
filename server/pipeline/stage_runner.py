"""Generic stage executor with subprocess management and log streaming."""
from __future__ import annotations

import asyncio
from collections import deque
import os
import signal
from pathlib import Path
from typing import AsyncGenerator, Optional


async def run_stage(
    stage_name: str,
    cmd: list[str],
    env: Optional[dict[str, str]] = None,
    log_file: Optional[Path] = None,
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> tuple[bool, str]:
    """Run a pipeline stage as a subprocess.

    Returns (success, error_message).
    """
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    log_lines: deque[str] = deque(maxlen=20)

    try:
        log_fh = open(log_file, "a") if log_file else None
    except OSError as exc:
        return False, f"Cannot open stage log: {exc}"
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=merged_env,
            cwd=str(cwd) if cwd else None,
            limit=10 * 1024 * 1024,
            start_new_session=os.name != "nt",
        )
    except OSError as e:
        if log_fh:
            log_fh.close()
        return False, f"Command not found: {cmd[0]}: {e}"

    async def _read_output():
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip()
            log_lines.append(decoded)
            if log_fh:
                log_fh.write(decoded + "\n")
                log_fh.flush()

    reader_task = asyncio.create_task(_read_output())

    async def stop_process_tree():
        # torchrun/model tools launch children which also hold GPU memory and
        # stdout pipes. Signal the session, then reap the subprocess before
        # returning cancellation or timeout to the worker.
        def send_signal(value):
            try:
                if os.name != "nt":
                    os.killpg(process.pid, value)
                elif process.returncode is None:
                    process.kill() if value == signal.SIGKILL else process.terminate()
            except ProcessLookupError:
                pass

        send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except asyncio.TimeoutError:
            pass
        finally:
            send_signal(signal.SIGKILL)
        await process.wait()

    wait_task = asyncio.create_task(process.wait())
    cancel_task = asyncio.create_task(cancel_event.wait()) if cancel_event else None
    try:
        watched = [wait_task, reader_task]
        if cancel_task:
            watched.append(cancel_task)
        # Reader completion is also watched to surface decoding/pipe failures;
        # never leave a GPU subprocess running behind a failed log reader.
        deadline = asyncio.get_running_loop().time() + timeout if timeout is not None else None
        while True:
            remaining = max(0, deadline - asyncio.get_running_loop().time()) if deadline is not None else None
            done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED, timeout=remaining)
            if not done:
                await stop_process_tree()
                await reader_task
                return False, f"Timeout after {timeout}s"
            if cancel_task and cancel_task in done:
                await stop_process_tree()
                await reader_task
                return False, "Cancelled by user"
            if reader_task in done:
                await reader_task
                watched.remove(reader_task)
            if wait_task in done:
                exit_code = wait_task.result()
                await reader_task
                break
        if exit_code == 0:
            return True, ""
        tail = "\n".join(log_lines) if log_lines else "no output"
        return False, f"Exit code {exit_code}. Last output: {tail}"
    except asyncio.CancelledError:
        await stop_process_tree()
        raise
    except Exception as exc:
        await stop_process_tree()
        return False, f"Stage execution failed: {exc}"
    finally:
        for task in (wait_task, cancel_task, reader_task):
            if task and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (wait_task, cancel_task, reader_task) if task), return_exceptions=True)
        if log_fh:
            log_fh.close()


def read_log_tail(log_file: Path, lines: int = 100) -> list[str]:
    """Read the last N lines from a log file."""
    if not log_file.exists():
        return []
    try:
        content = log_file.read_text(errors="replace")
        all_lines = content.splitlines()
        return all_lines[-lines:]
    except Exception:
        return []


async def stream_log(log_file: Path) -> AsyncGenerator[str, None]:
    """Async generator that yields new lines as they are appended to a log file."""
    if not log_file.exists():
        return

    with open(log_file, "r", errors="replace") as f:
        # Yield existing content first
        for line in f:
            yield line.rstrip()

        # Then follow new lines
        while True:
            line = f.readline()
            if line:
                yield line.rstrip()
            else:
                await asyncio.sleep(0.5)
