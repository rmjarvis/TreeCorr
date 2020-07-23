def pytest_collectstart(collector):
    collector.skip_compare += 'stderr',
