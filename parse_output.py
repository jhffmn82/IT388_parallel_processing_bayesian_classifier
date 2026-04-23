import re
import os
from collections import defaultdict


def format_size(rows):
    sizes = {"253680": "heart", "20000": "20k", "100000": "100k", "500000": "500k", "1000000": "1m"}
    return sizes.get(rows, rows)


def parse_label(label):
    """Parse the old '-- label --' format used in aspen/local runs.
    Returns (binary, procs, threads) or (None, 1, 1) if unrecognized."""
    label = label.strip().strip('-').strip()
    procs = 1
    threads = 1

    # Phase 1 and 2 labels look like: "Serial | heart | 1 thread"
    if '|' in label:
        parts = [p.strip() for p in label.split('|')]
        first = parts[0].lower()
        if any(b in first for b in ['serial', 'omp', 'mpi', 'hybrid']):
            if   'serial' in first: binary = 'serial'
            elif 'hybrid' in first: binary = 'hybrid'
            elif 'omp'    in first: binary = 'omp'
            else:                   binary = 'mpi'
            config = parts[2] if len(parts) > 2 else ''
            m = re.search(r'(\d+)\s*proc',   config)
            if m: procs = int(m.group(1))
            m = re.search(r'(\d+)\s*thread', config)
            if m: threads = int(m.group(1))
            return binary, procs, threads

    # Phase 3 labels look like: "Run 1"
    if re.match(r'^Run\s+\d+$', label, re.IGNORECASE):
        return 'serial', 1, 1

    # Phase 6 labels look like: "1 procs x 2 threads | Run 1"
    m = re.match(r'^(\d+)\s+procs?\s*x\s*(\d+)\s*threads?\s*\|', label, re.IGNORECASE)
    if m:
        return 'hybrid', int(m.group(1)), int(m.group(2))

    # Phase 4 labels look like: "4 threads | Run 1"
    m = re.match(r'^(\d+)\s+threads?\s*\|', label, re.IGNORECASE)
    if m:
        return 'omp', 1, int(m.group(1))

    # Phase 5 labels look like: "4 procs | Run 1"
    m = re.match(r'^(\d+)\s+procs?\s*\|', label, re.IGNORECASE)
    if m:
        return 'mpi', int(m.group(1)), 1

    return None, procs, threads


def infer_binary_from_context(phase, block_info, phase2_position):
    """For Expanse format, infer the binary type from:
    - current phase header
    - whether the block has 'Processes:' or 'Threads:' line
    - how many runs we've seen in this phase so far

    Phase 1 (heart correctness): serial, omp, mpi, hybrid in that order
    Phase 2 (data-size): for each size (20k, 100k, 500k): serial, omp, mpi, hybrid
    Phase 3: all serial at 1m
    Phase 4: all omp at 1m
    Phase 5: all mpi at 1m
    Phase 6: all hybrid at 1m
    """
    has_threads = block_info.get('threads_line') is not None
    has_processes = block_info.get('processes_line') is not None
    threads_val = block_info.get('threads_line', 1)
    processes_val = block_info.get('processes_line', 1)

    if phase == 1:
        # 4 runs in fixed order: serial, omp, mpi, hybrid
        if phase2_position == 0:
            return 'serial', 1, 1
        elif phase2_position == 1:
            return 'omp', 1, threads_val
        elif phase2_position == 2:
            return 'mpi', processes_val, 1
        elif phase2_position == 3:
            return 'hybrid', processes_val, 8  # Phase 1 hybrid is 4p x 8t but Processes line shows 4
        return None, 1, 1

    if phase == 2:
        # 12 runs: (serial, omp, mpi, hybrid) x (20k, 100k, 500k)
        binary_idx = phase2_position % 4
        if binary_idx == 0:
            return 'serial', 1, 1
        elif binary_idx == 1:
            return 'omp', 1, threads_val
        elif binary_idx == 2:
            return 'mpi', processes_val, 1
        elif binary_idx == 3:
            # Phase 2 hybrid is 2p x 4t; Processes line shows 2
            return 'hybrid', processes_val, 4

    if phase == 3:
        return 'serial', 1, 1

    if phase == 4:
        return 'omp', 1, threads_val

    if phase == 5:
        return 'mpi', processes_val, 1

    if phase == 6:
        # hybrid runs - we need to figure out the thread count
        # Expanse hybrid only prints "Processes: N" not threads
        # But OMP runtime prints "OpenMP running with X threads" lines before the block
        # We track threads separately from the OpenMP announcement lines
        threads = block_info.get('omp_threads', 1)
        return 'hybrid', processes_val, threads

    return None, 1, 1


def parse_output(filename):
    """Parse both label-based format and Expanse phase-based format."""
    runs = []
    current_binary  = None
    current_procs   = 1
    current_threads = 1

    # Phase tracking for Expanse format
    current_phase = 0
    phase_position = 0  # Position within current phase

    # Track recent OpenMP thread announcements (for hybrid where we can't see threads in block)
    recent_omp_threads = None

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Old-format label: "-- Serial | heart | 1 thread --"
        m = re.match(r'^--\s+(.+?)\s+--\s*$', line)
        if m:
            current_binary, current_procs, current_threads = parse_label(m.group(1))
            i += 1
            continue

        # Expanse-format phase header: "=== PHASE N: ... ==="
        m = re.match(r'^=== PHASE (\d+):', line)
        if m:
            current_phase = int(m.group(1))
            phase_position = 0
            recent_omp_threads = None
            i += 1
            continue

        # Track OpenMP thread announcement so we can use it for hybrid blocks
        m = re.match(r'^OpenMP running with (\d+) threads?', line)
        if m:
            recent_omp_threads = int(m.group(1))
            i += 1
            continue

        if '=== Naive Bayesian Classification Results ===' in line:
            run = {
                'binary':        current_binary,
                'size':          None,
                'procs':         current_procs,
                'threads':       current_threads,
                'train_time':    None,
                'classify_time': None,
                'cv_time':       None,
                'total_time':    None,
            }

            block_info = {
                'processes_line': None,
                'threads_line':   None,
                'omp_threads':    recent_omp_threads,
            }

            i += 1
            while i < len(lines):
                l = lines[i].rstrip()
                m = re.match(r'Training rows:\s+(\d+)', l)
                if m: run['size'] = format_size(m.group(1))
                m = re.match(r'Processes:\s+(\d+)', l)
                if m: block_info['processes_line'] = int(m.group(1))
                m = re.match(r'Threads:\s+(\d+)', l)
                if m: block_info['threads_line'] = int(m.group(1))
                m = re.match(r'Threads/proc:\s+(\d+)', l)
                if m: block_info['threads_line'] = int(m.group(1))
                m = re.match(r'Train time:\s+([\d.]+)', l)
                if m: run['train_time'] = float(m.group(1))
                m = re.match(r'Classify time:\s+([\d.]+)', l)
                if m: run['classify_time'] = float(m.group(1))
                m = re.match(r'CV time:\s+([\d.]+)', l)
                if m: run['cv_time'] = float(m.group(1))
                m = re.match(r'Total time:\s+([\d.]+)', l)
                if m:
                    run['total_time'] = float(m.group(1))

                    # If Expanse-format phase is active, derive binary/procs/threads from context
                    if current_phase > 0:
                        binary, procs, threads = infer_binary_from_context(
                            current_phase, block_info, phase_position
                        )
                        if binary:
                            run['binary'] = binary
                            run['procs'] = procs
                            run['threads'] = threads

                        # Override size for phases 3-6 (always 1m)
                        if current_phase >= 3:
                            run['size'] = '1m'

                        phase_position += 1
                        # Reset recent_omp_threads after we use it
                        recent_omp_threads = None

                    runs.append(run)
                    i += 1
                    break
                i += 1
            continue

        i += 1

    return runs


def avg(values):
    return sum(values) / len(values) if values else 0.0


def print_all_runs(runs):
    groups = defaultdict(list)
    for run in runs:
        key = (run['binary'], run['size'], run['procs'], run['threads'])
        groups[key].append(run)

    # Sort groups for readable output
    def sort_key(key):
        binary, size, procs, threads = key
        binary_order = {'serial': 0, 'omp': 1, 'mpi': 2, 'hybrid': 3, None: 99}.get(binary, 99)
        size_order = {'20k': 0, '100k': 1, '500k': 2, '1m': 3, 'heart': 4}.get(size, 99)
        return (binary_order, size_order, procs, threads)

    for key in sorted(groups.keys(), key=sort_key):
        binary, size, procs, threads = key
        group = groups[key]
        config = f"{procs}p x {threads}t" if binary == 'hybrid' else \
                 f"{procs} procs"          if binary == 'mpi'    else \
                 f"{threads} threads"      if binary == 'omp'    else \
                 "1 thread"
        print(f"\n{binary} | {size} | {config}")
        print(f"  {'':4} {'train':>10} {'classify':>10} {'cv':>10} {'total':>10}")
        for idx, r in enumerate(group, 1):
            print(f"  Run {idx:<2} {r['train_time']:>10.6f} {r['classify_time']:>10.6f} "
                  f"{r['cv_time']:>10.6f} {r['total_time']:>10.6f}")
        if len(group) > 1:
            print(f"  avg    {avg([r['train_time'] for r in group]):>10.6f} "
                  f"{avg([r['classify_time'] for r in group]):>10.6f} "
                  f"{avg([r['cv_time'] for r in group]):>10.6f} "
                  f"{avg([r['total_time'] for r in group]):>10.6f}")


def build_tables(runs):
    groups = defaultdict(list)
    for run in runs:
        key = (run['binary'], run['size'], run['procs'], run['threads'])
        groups[key].append(run)

    averages = {}
    for key, group in groups.items():
        averages[key] = {
            'train':    avg([r['train_time']    for r in group]),
            'classify': avg([r['classify_time'] for r in group]),
            'cv':       avg([r['cv_time']       for r in group]),
            'total':    avg([r['total_time']    for r in group]),
        }

    # Each binary uses its own 1-worker run as the speedup baseline.
    # Hybrid uses 1p x 1t as the baseline for all its tables.
    serial_1m  = averages.get(('serial', '1m', 1, 1), {}).get('total', 0)
    omp_base   = averages.get(('omp',    '1m', 1, 1), {}).get('total', 0)
    mpi_base   = averages.get(('mpi',    '1m', 1, 1), {}).get('total', 0)
    hybrid_base = averages.get(('hybrid','1m', 1, 1), {}).get('total', 0)

    tables = []

    def make_table(title, rows, baseline):
        col_w = 16
        header  = f"\n{title}\n"
        header += f"  {'config':<{col_w}} {'train':>10} {'classify':>10} {'cv':>10} {'total':>10} {'speedup':>9} {'efficiency':>11}\n"
        header += f"  {'-' * col_w} {'----------':>10} {'----------':>10} {'----------':>10} {'----------':>10} {'---------':>9} {'-----------':>11}\n"
        lines = [header]
        for (label, a, total_workers) in rows:
            speedup    = baseline / a['total'] if baseline > 0 and a['total'] > 0 else 0.0
            efficiency = speedup / total_workers if total_workers > 0 else 0.0
            lines.append(
                f"  {label:<{col_w}} {a['train']:>10.6f} {a['classify']:>10.6f} "
                f"{a['cv']:>10.6f} {a['total']:>10.6f} {speedup:>9.3f} {efficiency:>11.3f}\n"
            )
        return ''.join(lines)

    # Serial table — speedup vs serial 1m baseline
    serial_rows = []
    size_order = {'20k': 0, '100k': 1, '500k': 2, '1m': 3, 'heart': 4}
    serial_items = [(k, a) for k, a in averages.items() if k[0] == 'serial']
    serial_items.sort(key=lambda x: size_order.get(x[0][1], 99))
    for (binary, size, p, t), a in serial_items:
        serial_rows.append((size, a, 1))
    tables.append(make_table("Serial (baseline = serial 1m)", serial_rows, serial_1m))

    # OMP table — speedup vs OMP 1 thread
    omp_rows = []
    for (binary, size, p, t), a in averages.items():
        if binary == 'omp' and size == '1m':
            omp_rows.append((f"{t} threads", a, t))
    tables.append(make_table(
        "OMP (1m dataset, baseline = 1 thread)",
        sorted(omp_rows, key=lambda x: int(x[0].split()[0])),
        omp_base
    ))

    # MPI table — speedup vs MPI 1 proc
    mpi_rows = []
    for (binary, size, p, t), a in averages.items():
        if binary == 'mpi' and size == '1m':
            mpi_rows.append((f"{p} procs", a, p))
    tables.append(make_table(
        "MPI (1m dataset, baseline = 1 proc)",
        sorted(mpi_rows, key=lambda x: int(x[0].split()[0])),
        mpi_base
    ))

    # Hybrid tables — one per process count, all vs 1p x 1t baseline
    hybrid_procs = sorted(set(p for (binary, size, p, t) in averages if binary == 'hybrid' and size == '1m'))
    for proc_count in hybrid_procs:
        hybrid_rows = []
        for (binary, size, p, t), a in averages.items():
            if binary == 'hybrid' and size == '1m' and p == proc_count:
                hybrid_rows.append((f"{p}p x {t}t", a, p * t))
        tables.append(make_table(
            f"Hybrid (1m dataset, {proc_count} procs, baseline = 1p x 1t)",
            sorted(hybrid_rows, key=lambda x: int(x[0].split('x')[1].strip().rstrip('t'))),
            hybrid_base
        ))

    # Data-size scaling table — fixed config per binary, total time and % of serial at each size.
    # Configs used in Phase 2: serial=1t, omp=8t, mpi=8p, hybrid=2px4t
    sizes_ordered = ['20k', '100k', '500k', '1m']
    scaling_binaries = [
        ('serial', 1, 1, 'serial (1t)'),
        ('omp',    1, 8, 'omp (8t)'),
        ('mpi',    8, 1, 'mpi (8p)'),
        ('hybrid', 2, 4, 'hybrid (2p x 4t)'),
    ]

    col_w = 16
    size_header = f"  {'binary':<{col_w}}"
    for s in sizes_ordered:
        size_header += f"   {'time (s)':>8}  {'% serial':>8}  ({s})"
    size_header += "\n"
    sep = f"  {'-' * col_w}"
    for _ in sizes_ordered:
        sep += f"   {'--------':>8}  {'--------':>8}  {'------'}"
    sep += "\n"

    lines = ["\nData-size scaling (total time and % of serial)\n", size_header, sep]

    for (binary, p, t, label) in scaling_binaries:
        line = f"  {label:<{col_w}}"
        for s in sizes_ordered:
            a = averages.get((binary, s, p, t))
            serial_a = averages.get(('serial', s, 1, 1))
            if a and serial_a and serial_a['total'] > 0:
                pct = (a['total'] / serial_a['total']) * 100
                line += f"   {a['total']:>8.4f}  {pct:>7.1f}%  "
            else:
                line += f"   {'N/A':>8}  {'N/A':>8}  "
        lines.append(line + "\n")

    tables.append(''.join(lines))

    return tables


if __name__ == '__main__':
    filename = input("Enter the output filename: ").strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, filename)

    runs = parse_output(output_file)
    print(f"\nParsed {len(runs)} total runs.")

    print_all_runs(runs)

    tables = build_tables(runs)
    base = os.path.splitext(filename)[0]
    parsed_file = os.path.join(script_dir, base + '_parsed.txt')

    with open(parsed_file, 'w') as f:
        for table in tables:
            f.write(table)
            f.write('\n')

    print(f"\nSummary tables written to {parsed_file}")