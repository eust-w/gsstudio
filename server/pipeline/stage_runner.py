"""Generic stage executor with subprocess management and log streaming."""
from __future__ import annotations

import asyncio
import os
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
) -> Tuple[bool, str]:
    """Run a pipeline stage as a subprocess.

    Returns (success, error_message).
    """
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    log_lines: list[str] = []

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=merged_env,
            cwd=str(cwd) if cwd else None,
            limit=10 * 1024 * 1024,
        )
    except FileNotFoundError as e:
        return False, f"Command not found: {cmd[0]}: {e}"

    log_fh = open(log_file, "a") if log_file else None

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

    try:
        if cancel_event:
            # Wait for either process completion or cancellation
            wait_task = asyncio.create_task(process.wait())
            cancel_task = asyncio.create_task(cancel_event.wait())
            done, pending = await asyncio.wait(
                [wait_task, cancel_task],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout,
            )
            for t in pending:
                t.cancel()

            if cancel_task in done:
                process.terminate()
                await process.wait()
                await reader_task
                if log_fh:
                    log_fh.close()
                return False, "Cancelled by user"

            exit_code = wait_task.result()
        else:
            exit_code = await asyncio.wait_for(process.wait(), timeout=timeout)

        await reader_task

    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        await reader_task
        if log_fh:
            log_fh.close()
        return False, f"Timeout after {timeout}s"

    if log_fh:
        log_fh.close()

    if exit_code == 0:
        return True, ""
    else:
        tail = "\n".join(log_lines[-20:]) if log_lines else "no output"
        return False, f"Exit code {exit_code}. Last output: {tail}"


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
