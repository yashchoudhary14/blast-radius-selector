"""LPT using the scorer's integer milliseconds and per-worker charge."""


def schedule(selected, millis, forced=None):
    if not selected:
        return [], 0
    order = sorted(selected, key=lambda n: (-millis[n], n))
    choices = [forced] if forced else range(1, min(8, len(order)) + 1)
    best_workers, best_cost = None, None
    for k in choices:
        workers, loads = [[] for _ in range(k)], [0] * k
        for n in order:
            w = min(range(k), key=loads.__getitem__)
            workers[w].append(n)
            loads[w] += millis[n]
        workers = [w for w in workers if w]
        cost = max(loads) + 2000 * len(workers)
        if best_cost is None or cost < best_cost:
            best_workers, best_cost = workers, cost
    return best_workers, best_cost
