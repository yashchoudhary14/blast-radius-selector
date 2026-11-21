"""Balance scorer milliseconds, including the charge for every occupied worker."""

WORKER_MS = 2000


def _split_two(order, millis, total):
    # Bit i records a reachable sum. Retain the prefix masks for reconstruction.
    half = total // 2
    limit = (1 << (half + 1)) - 1
    prefix = []
    reachable = 1
    for n in order:
        prefix.append(reachable)
        reachable |= (reachable << millis[n]) & limit
        if reachable.bit_length() > half:
            break
    target = reachable.bit_length() - 1
    cost = total - target + 2 * WORKER_MS
    left = set()
    for n, prior in zip(reversed(order[:len(prefix)]), reversed(prefix)):
        if not ((prior >> target) & 1):
            left.add(n)
            target -= millis[n]
    return [[n for n in order if n in left], [n for n in order if n not in left]], cost


def schedule(selected, millis, forced=None):
    order = sorted(selected, key=lambda n: (-millis[n], n))
    if not order:
        return [], 0
    total = sum(millis[n] for n in order)
    best, cost = [order], total + WORKER_MS
    counts = (min(forced, len(order)),) if forced else range(2, min(8, len(order)) + 1)
    for k in counts:
        lower = max(millis[order[0]], (total + k - 1) // k) + WORKER_MS * k
        if not forced and lower >= cost:
            continue
        workers, loads = [[] for _ in range(k)], [0] * k
        for i, n in enumerate(order):
            w = i if i < k else loads.index(min(loads))
            workers[w].append(n)
            loads[w] += millis[n]
        score = max(loads) + k * WORKER_MS
        if k == 2 and score > lower and total * len(order) < 8_000_000:
            workers, score = _split_two(order, millis, total)
        if forced or score < cost:
            best, cost = workers, score
    return best, cost
