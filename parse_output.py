import re
import os
from collections import defaultdict


def format_size(rows):
    sizes = {"253680": "heart", "20000": "20k", "100000": "100k", "500000": "500k", "1000000": "1m"}
    return sizes.get(rows, rows)


def parse_label(label):
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


def parse_output(filename):
    runs = []
    current_binary  = None
    current_procs   = 1
    current_threads = 1

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        m = re.match(r'^--\s+(.+?)\s+--\s*$', line)
        if m:
            current_binary, current_procs, current_threads = parse_label(m.group(1))
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
            i += 1
            while i < len(lines):
                l = lines[i].rstrip()
                m = re.match(r'Training rows:\s+(\d+)', l)
                if m: run['size'] = format_size(m.group(1))
                m = re.match(r'Train time:\s+([\d.]+)', l)
                if m: run['train_time'] = float(m.group(1))
                m = re.match(r'Classify time:\s+([\d.]+)', l)
                if m: run['classify_time'] = float(m.group(1))
                m = re.match(r'CV time:\s+([\d.]+)', l)
                if m: run['cv_time'] = float(m.group(1))
                m = re.match(r'Total time:\s+([\d.]+)', l)
                if m:
                    run['total_time'] = float(m.group(1))
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

    for (binary, size, procs, threads), group in groups.items():
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

    # Serial table uses serial 1m as baseline across all sizes
    serial_baselines = {
        size: averages[('serial', size, 1, 1)]['total']
        for (binary, size, p, t) in averages
        if binary == 'serial'
    }

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
    for (binary, size, p, t), a in averages.items():
        if binary == 'serial':
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