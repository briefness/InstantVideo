import pytest
from pipeline.run_state import RunWorkspace
from pipeline.models import RunOptions, RunStatus


def prepared(tmp_path):
    from test_planning_citations import recorded
    w = RunWorkspace.create(tmp_path, RunOptions(request='offline', paid_take_budget=0, take_policy='single_pass'))
    w.save_storyboard(recorded()['storyboard.json']['value'])
    w.checkpoint('planning_ready', RunStatus.interrupted)
    return w


def test_explicit_authorization_preserves_plan_and_is_idempotent(tmp_path):
    w = prepared(tmp_path)
    original = (w.path / 'storyboard.json').read_bytes()
    w.authorize_planned_generation(3)
    w.authorize_planned_generation(3)
    assert RunWorkspace.resume(w.path).options.paid_take_budget == 3
    assert w.options.take_policy == 'single_pass'
    assert (w.path / 'storyboard.json').read_bytes() == original
    assert not w.manifest.paid_take_budget.reservations
    with pytest.raises(ValueError):
        w.authorize_planned_generation(6)
    w.reserve_paid_take(1)
    with pytest.raises(ValueError):
        w.authorize_planned_generation(3)


@pytest.mark.parametrize('limit', [0, -1, True, '3', None])
def test_invalid_authorization_leaves_manifest_unchanged(tmp_path, limit):
    w = prepared(tmp_path)
    original = w.manifest_path.read_bytes()
    with pytest.raises(ValueError):
        w.authorize_planned_generation(limit)
    assert w.manifest_path.read_bytes() == original


def test_unready_or_active_workspaces_cannot_be_authorized(tmp_path):
    w = prepared(tmp_path)
    with w.execution(), pytest.raises(RuntimeError):
        RunWorkspace.resume(w.path).authorize_planned_generation(3)
    w.checkpoint('storyboard_ready', RunStatus.failed)
    with pytest.raises(ValueError):
        w.authorize_planned_generation(3)


def test_cli_authorizes_then_resumes_existing_plan(tmp_path, monkeypatch):
    import main
    w = prepared(tmp_path)
    called = []
    async def run(self, request):
        called.append(self.paid_take_budget)
        assert self.workspace == w.path and not self.plan_only
        return 'offline.mp4'
    monkeypatch.setattr(main.VideoPipeline, 'run', run)
    monkeypatch.setattr('sys.argv', ['main.py', '--resume', str(w.path), '--authorize-paid-takes', '3'])
    main.main()
    assert called == [3]
