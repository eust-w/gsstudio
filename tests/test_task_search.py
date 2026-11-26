from server.db import create_task, init_db, list_tasks


def test_search_is_literal_and_applies_before_pagination(tmp_path):
    path = tmp_path / "tasks.db"
    init_db(path)
    for name in ("Scene_01", "SceneX01", "Scene 100%", "Scene 100A"):
        create_task(path, name, "worldrecon")
    tasks, total = list_tasks(path, search="_01")
    assert total == 1 and tasks[0]["name"] == "Scene_01"
    tasks, total = list_tasks(path, search="100%")
    assert total == 1 and tasks[0]["name"] == "Scene 100%"
    tasks, total = list_tasks(path, search="scene", limit=1, offset=1)
    assert total == 4 and len(tasks) == 1
    assert list_tasks(path, search=tasks[0]["id"])[1] == 1
