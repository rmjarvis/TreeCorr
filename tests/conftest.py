def pytest_collectstart(collector):
    if hasattr(collector, 'skip_compare'):
        collector.skip_compare += 'stderr',
