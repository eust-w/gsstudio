"""API transport and subprocess lifecycle invariants, without GPU execution."""
import asyncio
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from server.config import Config
from server.db import create_task, get_task, init_db, list_tasks, update_task
from server.delivery_manifest import build_delivery_manifest
from server.pipeline.stage_runner import run_stage
from server.pipeline.worker import PipelineWorker
from server.routes import tasks as routes


def uploads(count=12, content=None):
    if content is None:
        buffer = BytesIO()
        Image.new('RGB', (20, 20), 'orange').save(buffer, format='PNG')
        content = buffer.getvalue()
    return [('images', (f'view-{i:02d}.png', content, 'image/png')) for i in range(count)]


@pytest.fixture
def isolated_config(tmp_path):
    cfg = Config()
    cfg.DATA_DIR = tmp_path
    cfg.DB_PATH = tmp_path / 'test.db'
    cfg.MOCK_PIPELINE = True
    cfg.MANAGE_VLLM = False
    init_db(cfg.DB_PATH)
    return cfg


@pytest.fixture
def api(isolated_config, monkeypatch):
    monkeypatch.setattr(routes, '_config', isolated_config)
    monkeypatch.setattr(routes, '_worker', None)
    app = FastAPI()
    app.include_router(routes.router, prefix='/api')
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize('media', [[], uploads(11), uploads(49), uploads(12, b'not-an-image')])
def test_rejects_missing_incomplete_excessive_or_corrupt_reconstruction(api, isolated_config, media):
    response = api.post('/api/tasks', data={'type': 'worldrecon', 'name': 'capture'}, files=media)
    assert response.status_code == 400
    assert list_tasks(isolated_config.DB_PATH)[1] == 0
    assert not list(isolated_config.tasks_dir.glob('.incoming/*'))


def test_rejects_ambiguous_reconstruction_media(api, isolated_config):
    response = api.post('/api/tasks', data={'type': 'worldrecon', 'name': 'capture'},
                        files=uploads() + [('video', ('clip.mp4', b'video', 'video/mp4'))])
    assert response.status_code == 400
    assert list_tasks(isolated_config.DB_PATH)[1] == 0


@pytest.mark.parametrize('config', [{'seed': True}, {'seed': 1.7}, {'seed': -1}, {'seed': 2**32},
                                   {'seed': None}, {'recon_gpus': False}, {'recon_gpus': 1.5},
                                   {'quality_preset': []}, {'recovery_policy': {}}, {'prompt': []}])
def test_rejects_malformed_public_configuration(api, config):
    response = api.post('/api/tasks', data={'type': 'worldrecon', 'name': 'capture', 'config': json.dumps(config)}, files=uploads())
    assert response.status_code == 400


@pytest.mark.parametrize('name', [' ', 'x' * 121])
def test_rejects_invalid_task_names(api, name):
    assert api.post('/api/tasks', data={'type': 'worldrecon', 'name': name}, files=uploads()).status_code == 400


def test_normalizes_name_and_persists_demo_provenance(api, isolated_config):
    response = api.post('/api/tasks', data={'type': 'worldrecon', 'name': '  room  '}, files=uploads())
    assert response.status_code == 201
    task_id = response.json()['task_id']
    task = api.get(f'/api/tasks/{task_id}').json()
    assert task['name'] == 'room'
    assert task['execution_mode'] == task['config']['execution_mode'] == 'demo'
    assert len(list(isolated_config.task_input_dir(task_id).glob('image_*.png'))) == 12


def test_image_and_total_size_limits_leave_no_queued_task(api, isolated_config, monkeypatch):
    monkeypatch.setattr(routes, 'MAX_IMAGE_BYTES', 10)
    response = api.post('/api/tasks', data={'type': 'worldrecon', 'name': 'oversize'}, files=uploads())
    assert response.status_code == 400
    assert list_tasks(isolated_config.DB_PATH)[1] == 0
    assert not list(isolated_config.tasks_dir.glob('.incoming/*'))


def test_readiness_failure_does_not_publish_input_or_task(api, isolated_config, monkeypatch):
    from server import runtime
    def reject(*args):
        raise HTTPException(503, {'code': 'runtime_not_ready', 'message': 'GPU unavailable'})
    monkeypatch.setattr(runtime, 'require_task_capability', reject)
    response = api.post('/api/tasks', data={'type': 'worldrecon', 'name': 'no-gpu'}, files=uploads())
    assert response.status_code == 503
    assert list_tasks(isolated_config.DB_PATH)[1] == 0
    assert not list(isolated_config.tasks_dir.glob('.incoming/*'))


def test_demo_generation_does_not_require_gated_model_access(api, monkeypatch):
    monkeypatch.delenv('HF_TOKEN', raising=False)
    monkeypatch.setattr(routes, '_sam3_is_cached', lambda path: False)
    response = api.post('/api/tasks', data={'type': 'worldgen', 'name': 'demo', 'prompt': 'a room'})
    assert response.status_code == 201


def test_demo_never_exports_fake_models_or_real_stage_checkpoints(isolated_config, monkeypatch):
    import server.pipeline.worker as module
    task_id = create_task(isolated_config.DB_PATH, name='demo', task_type='worldrecon', config={'execution_mode': 'demo'})
    monkeypatch.setattr(module, 'get_stage_names', lambda _: ['worldrecon'])
    asyncio.run(PipelineWorker(isolated_config).execute_task(task_id))
    task = get_task(isolated_config.DB_PATH, task_id)
    root = isolated_config.task_dir(task_id)
    assert task['status'] == 'completed'
    assert all(stage['status'] == 'skipped' for stage in task['stages'])
    assert not list(root.glob('.stage_*_done'))
    assert list(root.glob('.demo_stage_*_done'))
    assert not list(root.rglob('*.ply')) and not list(root.rglob('*.spz'))
    manifest = build_delivery_manifest(root, task)
    assert manifest['execution_mode'] == 'demo'
    assert manifest['verified'] is False
    assert manifest['free_view_eligible'] is False
    assert manifest['product_state'] == 'demo'
    assert any(artifact['name'] == 'execution_report.json' for artifact in task['artifacts'])


def test_real_worker_rejects_demo_task_before_execution(isolated_config):
    task_id = create_task(isolated_config.DB_PATH, name='demo', task_type='worldrecon', config={'execution_mode': 'demo'})
    isolated_config.MOCK_PIPELINE = False
    asyncio.run(PipelineWorker(isolated_config).execute_task(task_id))
    task = get_task(isolated_config.DB_PATH, task_id)
    assert task['status'] == 'failed'
    assert task['error_code'] == 'execution_mode_mismatch'
    assert not isolated_config.task_dir(task_id).exists()


def test_cancelled_task_is_not_restarted_from_stale_queue_snapshot(isolated_config):
    task_id = create_task(isolated_config.DB_PATH, name='cancelled', task_type='worldrecon')
    update_task(isolated_config.DB_PATH, task_id, status='cancelled')
    asyncio.run(PipelineWorker(isolated_config).execute_task(task_id))
    assert get_task(isolated_config.DB_PATH, task_id)['status'] == 'cancelled'
    assert not isolated_config.task_dir(task_id).exists()


def test_cancelling_worker_blocks_delete_and_retry_until_process_stops(api, isolated_config, monkeypatch):
    task_id = create_task(isolated_config.DB_PATH, name='stopping', task_type='worldrecon')
    update_task(isolated_config.DB_PATH, task_id, status='cancelled')
    worker = PipelineWorker(isolated_config)
    worker._cancel_events[task_id] = asyncio.Event()
    monkeypatch.setattr(routes, '_worker', worker)
    assert api.delete(f'/api/tasks/{task_id}').status_code == 409
    assert api.post(f'/api/tasks/{task_id}/retry').status_code == 409
    assert get_task(isolated_config.DB_PATH, task_id)['status'] == 'cancelled'


@pytest.mark.skipif(os.name == 'nt', reason='POSIX process group lifecycle')
@pytest.mark.parametrize('stop_kind', ['timeout', 'event', 'shutdown'])
def test_subprocess_tree_is_reaped_on_all_stop_paths(tmp_path, stop_kind):
    pid_file = tmp_path / 'child.pid'
    child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    program = f"import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c',{child!r}]); pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); print('started',flush=True); time.sleep(60)"
    async def scenario():
        cancel = asyncio.Event()
        task = asyncio.create_task(run_stage('sleep', [sys.executable, '-c', program],
                                            log_file=tmp_path / 'stage.log', timeout=0.3 if stop_kind == 'timeout' else 60,
                                            cancel_event=cancel))
        if stop_kind != 'timeout':
            for _ in range(100):
                if pid_file.exists():
                    break
                await asyncio.sleep(0.01)
            assert pid_file.exists()
            if stop_kind == 'event':
                cancel.set()
            else:
                task.cancel()
        if stop_kind == 'shutdown':
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            ok, reason = await task
            assert ok is False
            assert ('Timeout' if stop_kind == 'timeout' else 'Cancelled') in reason
    asyncio.run(scenario())
    assert pid_file.exists()
    result = subprocess.run(['ps', '-p', pid_file.read_text(), '-o', 'stat='], capture_output=True, text=True)
    assert result.returncode != 0 or result.stdout.strip().startswith('Z'), result.stdout


def test_completed_log_stream_crosses_stages_and_resumes_without_replay(api, isolated_config):
    task_id = create_task(isolated_config.DB_PATH, name='logs', task_type='worldrecon')
    logs = isolated_config.task_logs_dir(task_id)
    logs.mkdir(parents=True)
    (logs / 'input_preflight.log').write_text('first\nsecond\n')
    (logs / 'worldrecon.log').write_text('third\n')
    update_task(isolated_config.DB_PATH, task_id, status='completed')
    with api.websocket_connect(f'/api/tasks/{task_id}/logs') as socket:
        first = socket.receive_json()
        second = socket.receive_json()
        third = socket.receive_json()
        complete = socket.receive_json()
    assert [first['line'], second['line'], third['line']] == ['first', 'second', 'third']
    assert third['stage'] == 'worldrecon'
    assert first['cursor'] != second['cursor']
    assert complete['type'] == 'complete'
    with api.websocket_connect(f'/api/tasks/{task_id}/logs?cursor={second["cursor"]}') as socket:
        assert socket.receive_json()['line'] == 'third'
        assert socket.receive_json()['type'] == 'complete'


def test_queued_log_stream_follows_new_stage_and_terminal_empty_logs(api, isolated_config):
    task_id = create_task(isolated_config.DB_PATH, name='queued-logs', task_type='worldrecon')
    with api.websocket_connect(f'/api/tasks/{task_id}/logs') as socket:
        logs = isolated_config.task_logs_dir(task_id)
        logs.mkdir(parents=True)
        (logs / 'worldrecon.log').write_text('new stage\n')
        update_task(isolated_config.DB_PATH, task_id, status='completed')
        assert socket.receive_json()['line'] == 'new stage'
        assert socket.receive_json()['type'] == 'complete'
    empty_id = create_task(isolated_config.DB_PATH, name='empty-logs', task_type='worldrecon')
    update_task(isolated_config.DB_PATH, empty_id, status='cancelled')
    with api.websocket_connect(f'/api/tasks/{empty_id}/logs') as socket:
        assert socket.receive_json()['type'] == 'complete'


def test_recovery_scene_artifacts_keep_task_relative_paths(isolated_config, tmp_path):
    root = isolated_config.task_dir('recovery')
    scene = root / 'recovery_attempts' / 'attempt-001' / 'scene'
    scene.mkdir(parents=True)
    (scene / 'panorama.png').write_bytes(b'png')
    external = tmp_path / 'external.png'
    external.write_bytes(b'secret')
    (scene / 'escaped.png').symlink_to(external)
    artifacts = PipelineWorker(isolated_config)._collect_artifacts(root / 'output', scene, root)
    assert artifacts == [{'name': 'panorama.png', 'path': 'recovery_attempts/attempt-001/scene/panorama.png', 'size': 3, 'type': 'image'}]


def test_restarting_demo_server_preserves_explicit_demo_completion(isolated_config):
    task_id = create_task(isolated_config.DB_PATH, name='demo', task_type='worldgen', config={'execution_mode': 'demo'})
    update_task(isolated_config.DB_PATH, task_id, status='completed')
    PipelineWorker(isolated_config)._recover_interrupted_tasks()
    assert get_task(isolated_config.DB_PATH, task_id)['status'] == 'completed'


def test_existing_idempotent_request_replays_while_runtime_is_unavailable(api, isolated_config, monkeypatch):
    from server import runtime
    headers = {'Idempotency-Key': 'replay-while-busy'}
    data = {'type': 'worldrecon', 'name': 'capture'}
    first = api.post('/api/tasks', data=data, files=uploads(), headers=headers)
    assert first.status_code == 201
    def unavailable(*args):
        raise HTTPException(503, {'code': 'runtime_not_ready', 'message': 'GPU busy'})
    monkeypatch.setattr(runtime, 'require_task_capability', unavailable)
    replay = api.post('/api/tasks', data=data, files=uploads(), headers=headers)
    assert replay.status_code == 200
    assert replay.json()['task_id'] == first.json()['task_id']
    conflict = api.post('/api/tasks', data={**data, 'name': 'different'}, files=uploads(), headers=headers)
    assert conflict.status_code == 409
    assert list_tasks(isolated_config.DB_PATH)[1] == 1


def test_unavailable_worker_leaves_preexisting_task_pending_with_diagnostic(isolated_config, monkeypatch):
    from server import runtime
    task_id = create_task(isolated_config.DB_PATH, name='legacy-pending', task_type='worldrecon')
    def unavailable(*args):
        raise HTTPException(503, {'code': 'runtime_not_ready', 'message': 'CUDA unavailable'})
    monkeypatch.setattr(runtime, 'require_task_capability', unavailable)
    worker = PipelineWorker(isolated_config)
    assert asyncio.run(worker._task_runtime_ready(get_task(isolated_config.DB_PATH, task_id))) is False
    task = get_task(isolated_config.DB_PATH, task_id)
    assert task['status'] == 'pending'
    assert task['error_code'] == 'runtime_not_ready'
    assert task['error_summary'] == 'CUDA unavailable'
    assert not isolated_config.task_dir(task_id).exists()


def test_repair_only_admission_requires_all_upstream_checkpoints(isolated_config, monkeypatch):
    from server import runtime
    called = []
    monkeypatch.setattr(runtime, 'require_task_capability', lambda *args: called.append('full'))
    monkeypatch.setattr(runtime, 'require_repair_capability', lambda *args: called.append('repair'))
    task_id = create_task(isolated_config.DB_PATH, name='repair', task_type='worldrecon')
    root = isolated_config.task_dir(task_id)
    root.mkdir(parents=True)
    (root / 'gaussian_repair_request.json').write_text('{}')
    worker = PipelineWorker(isolated_config)
    task = get_task(isolated_config.DB_PATH, task_id)
    assert asyncio.run(worker._task_runtime_ready(task))
    (root / '.stage_input_preflight_done').touch()
    (root / '.stage_worldrecon_done').touch()
    assert asyncio.run(worker._task_runtime_ready(task))
    assert called == ['full', 'repair']


def test_only_managed_vlm_blocker_restores_stopped_service_then_rechecks(isolated_config, monkeypatch):
    from server import runtime
    isolated_config.MANAGE_VLLM = True
    task_id = create_task(isolated_config.DB_PATH, name='restore-vlm', task_type='worldrecon')
    worker = PipelineWorker(isolated_config)
    calls = []
    def admission(*args):
        calls.append('admit')
        if calls.count('admit') == 1:
            raise HTTPException(503, {'message': 'VLM unavailable', 'blockers': [{'code': 'vlm_service'}]})
    async def supervisor(*args):
        assert args == ('status', 'vllm')
        return False, 'vllm STOPPED Not started'
    async def restore():
        assert worker._vllm_stopped is True
        calls.append('restore')
    monkeypatch.setattr(runtime, 'require_task_capability', admission)
    monkeypatch.setattr(runtime, 'clear_runtime_cache', lambda: calls.append('clear-cache'))
    monkeypatch.setattr(worker, '_supervisorctl', supervisor)
    monkeypatch.setattr(worker, '_start_vllm', restore)
    assert asyncio.run(worker._task_runtime_ready(get_task(isolated_config.DB_PATH, task_id))) is True
    assert calls == ['admit', 'restore', 'clear-cache', 'admit']


@pytest.mark.parametrize('managed,codes', [(False, ['vlm_service']), (True, ['vlm_service', 'gpu_capacity'])])
def test_admission_never_starts_vlm_for_unmanaged_or_other_missing_dependencies(isolated_config, monkeypatch, managed, codes):
    from server import runtime
    isolated_config.MANAGE_VLLM = managed
    task_id = create_task(isolated_config.DB_PATH, name='stay-pending', task_type='worldrecon')
    worker = PipelineWorker(isolated_config)
    def admission(*args):
        raise HTTPException(503, {'message': 'Not ready', 'blockers': [{'code': code} for code in codes]})
    async def unexpected(*args):
        raise AssertionError('Must not control VLM without sole managed VLM blocker')
    monkeypatch.setattr(runtime, 'require_task_capability', admission)
    monkeypatch.setattr(worker, '_supervisorctl', unexpected)
    assert asyncio.run(worker._task_runtime_ready(get_task(isolated_config.DB_PATH, task_id))) is False
    assert get_task(isolated_config.DB_PATH, task_id)['status'] == 'pending'


def test_failed_managed_vlm_recovery_is_rate_limited(isolated_config, monkeypatch):
    from server import runtime
    isolated_config.MANAGE_VLLM = True
    task_id = create_task(isolated_config.DB_PATH, name='backoff', task_type='worldrecon')
    worker = PipelineWorker(isolated_config)
    calls = []
    def admission(*args):
        raise HTTPException(503, {'message': 'VLM unavailable', 'blockers': [{'code': 'vlm_service'}]})
    async def supervisor(*args):
        calls.append('inspect')
        return False, 'vllm FATAL failed too quickly'
    async def restore():
        calls.append('restore')
        raise RuntimeError('Could not start VLM')
    monkeypatch.setattr(runtime, 'require_task_capability', admission)
    monkeypatch.setattr(worker, '_supervisorctl', supervisor)
    monkeypatch.setattr(worker, '_start_vllm', restore)
    task = get_task(isolated_config.DB_PATH, task_id)
    assert asyncio.run(worker._task_runtime_ready(task)) is False
    assert asyncio.run(worker._task_runtime_ready(task)) is False
    assert calls == ['inspect', 'restore']
