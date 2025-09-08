import pytest

def pytest_collectstart(collector):
    if hasattr(collector, 'skip_compare'):
        collector.skip_compare += 'stderr',

# cf. https://stackoverflow.com/questions/62044541/change-pytest-working-directory-to-test-case-directory
@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)
