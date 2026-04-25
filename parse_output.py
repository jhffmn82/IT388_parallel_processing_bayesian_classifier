import re
import os
from collections import defaultdict


def format_size(rows):
    sizes = {"253680": "heart", "20000": "20k", "100000": "100k",
             "500000": "500k", "1000000": "1m"}
    return sizes.get(rows, rows)


def parse_label(label):
    """Parse a -- label -- line from cluster format.
    Returns (binary, procs, threads) or (None, 1, 1).

    Two label styles exist:
      Phase 1/2 (named):  "Serial | heart | 1 thread"
                          "OMP | 500000 | 8 threads"
                          "Hybrid | heart | 4 procs x 2 threads"
      Phase 3-6 (positional):
                          "Run N"                         -> serial
                          "N threads | Run K"             -> omp
                          "N procs | Run K"               -> mpi
                          "N procs x M threads | Run K"   -> hybrid
    """
    label = label.strip().strip('-').strip()
    procs   = 1
    threads = 1

    # ── Phase 3-6 positional patterns (check before pipe branch) ─────────────
    # "Run N"  (serial baseline)
    if re.match(r'^Run\s+\d+$', label, re.IGNORECASE):
        return 'serial', 1, 1

    # "N procs x M threads | Run K"  (hybrid — must come before mpi/omp)
    m = re.match(r'^(\d+)\s+procs?\s*x\s*(\d+)\s*threads?\s*\|', label, re.IGNORECASE)
    if m:
        return 'hybrid', int(m.group(1)), int(m.group(2))

    # "N threads | Run K"  (omp)
    m = re.match(r'^(\d+)\s+threads?\s*\|', label, re.IGNORECASE)
    if m:
        return 'omp', 1, int(m.group(1))

    # "N procs | Run K"  (mpi)
    m = re.match(r'^(\d+)\s+procs?\s*\|', label, re.IGNORECASE)
    if m:
        return 'mpi', int(m.group(1)), 1

    # ── Phase 1/2 named patterns ──────────────────────────────────────────────
    # "Serial | heart | 1 thread"  /  "Hybrid | heart | 4 procs x 2 threads"
    if '|' in label:
        parts = [p.strip() for p in label.split('|')]
        first = parts[0].lower()
        if   'serial' in first: binary = 'serial'
        elif 'hybrid' in first: binary = 'hybrid'
        elif 'omp'    in first: binary = 'omp'
        elif 'mpi'    in first: binary = 'mpi'
        else: return None, 1, 1

        config = parts[2] if len(parts) > 2 else ''
        m = re.search(r'(\d+)\s*proc',   config)
        if m: procs = int(m.group(1))
        m = re.search(r'(\d+)\s*thread', config)
        if m: threads = int(m.group(1))
        return binary, procs, threads

    return None, procs, threads


def detect_format(lines):
    """Return 'expanse' if we see === PHASE N: headers, else 'cluster'."""
    for line in lines[:50]:
        if re.match(r'^=== PHASE \d+:', line):
            return 'expanse'
    return 'cluster'


def parse_output(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    fmt = detect_format(lines)

    if fmt == 'expanse':
        return parse_expanse(lines)
    else:
        return parse_cluster(lines)


# ── Expanse parser ─────────────────────────────────────────────────────────────
def parse_expanse(lines):
    runs = []
    current_phase = 0
    phase_position = 0
    recent_omp_threads = None

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        m = re.match(r'^=== PHASE (\d+):', line)
        if m:
            current_phase = int(m.group(1))
            phase_position = 0
            recent_omp_threads = None
            i += 1
            continue

        m = re.match(r'^OpenMP running with (\d+) threads?', line)
        if m:
            recent_omp_threads = int(m.group(1))
            i += 1
            continue

        if 'Naive Bayesian Classification Results' in line:
            run = {'binary': None, 'size': None, 'procs': 1, 'threads': 1,
                   'train_time': None, 'classify_time': None,
                   'cv_time': None, 'total_time': None}
            block = {'processes_line': None, 'threads_line': None,
                     'omp_threads': recent_omp_threads}
            i += 1
            while i < len(lines):
                l = lines[i].rstrip()
                m = re.match(r'Training rows:\s+(\d+)', l)
                if m: run['size'] = format_size(m.group(1))
                m = re.match(r'Processes:\s+(\d+)', l)
                if m: block['processes_line'] = int(m.group(1))
                m = re.match(r'Threads:\s+(\d+)', l)
                if m: block['threads_line'] = int(m.group(1))
                m = re.match(r'Threads/proc:\s+(\d+)', l)
                if m: block['threads_line'] = int(m.group(1))
                m = re.match(r'Train time:\s+([\d.]+)', l)
                if m: run['train_time'] = float(m.group(1))
                m = re.match(r'Classify time:\s+([\d.]+)', l)
                if m: run['classify_time'] = float(m.group(1))
                m = re.match(r'CV time:\s+([\d.]+)', l)
                if m: run['cv_time'] = float(m.group(1))
                m = re.match(r'Total time:\s+([\d.]+)', l)
                if m:
                    run['total_time'] = float(m.group(1))
                    p = block['processes_line'] or 1
                    t = block['threads_line'] or 1

                    if current_phase == 1:
                        order = ['serial', 'omp', 'mpi', 'hybrid']
                        run['binary'] = order[phase_position % 4]
                        run['procs'] = p; run['threads'] = t
                        if run['binary'] == 'hybrid': run['threads'] = 8
                    elif current_phase == 2:
                        order = ['serial', 'omp', 'mpi', 'hybrid']
                        run['binary'] = order[phase_position % 4]
                        run['procs'] = p; run['threads'] = t
                        if run['binary'] == 'hybrid': run['threads'] = 4
                    elif current_phase == 3:
                        run['binary'] = 'serial'; run['procs'] = 1; run['threads'] = 1
                    elif current_phase == 4:
                        run['binary'] = 'omp'; run['procs'] = 1; run['threads'] = t
                    elif current_phase == 5:
                        run['binary'] = 'mpi'; run['procs'] = p; run['threads'] = 1
                    elif current_phase == 6:
                        run['binary'] = 'hybrid'; run['procs'] = p
                        run['threads'] = block['threads_line'] or block['omp_threads'] or 1

                    if current_phase >= 3:
                        run['size'] = '1m'

                    phase_position += 1
                    recent_omp_threads = None
                    runs.append(run)
                    i += 1
                    break
                i += 1
            continue
        i += 1
    return runs


# ── Cluster parser ─────────────────────────────────────────────────────────────
def parse_cluster(lines):
    """Cluster format uses -- label -- lines to identify each run."""
    runs = []
    current_binary  = None
    current_procs   = 1
    current_threads = 1
    recent_omp_threads = None

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Phase headers in cluster format (mixed case) — only used for context
        m = re.match(r'^=== Phase \d+:', line, re.IGNORECASE)
        if m:
            i += 1
            continue

        # Label line: -- OMP | 500000 | 8 threads --
        m = re.match(r'^--\s+(.+?)\s+--\s*$', line)
        if m:
            current_binary, current_procs, current_threads = parse_label(m.group(1))
            recent_omp_threads = None
            i += 1
            continue

        m = re.match(r'^OpenMP running with (\d+) threads?', line)
        if m:
            recent_omp_threads = int(m.group(1))
            i += 1
            continue

        if 'Naive Bayesian Classification Results' in line:
            run = {'binary': current_binary, 'size': None,
                   'procs': current_procs, 'threads': current_threads,
                   'train_time': None, 'classify_time': None,
                   'cv_time': None, 'total_time': None}

            # For hybrid, the label gives procs but not always threads —
            # fall back to the OMP thread announcement if needed
            if current_binary == 'hybrid' and current_threads == 1 and recent_omp_threads:
                run['threads'] = recent_omp_threads

            block = {'processes_line': None, 'threads_line': None}
            i += 1
            while i < len(lines):
                l = lines[i].rstrip()
                m = re.match(r'Training rows:\s+(\d+)', l)
                if m: run['size'] = format_size(m.group(1))
                m = re.match(r'Processes:\s+(\d+)', l)
                if m: block['processes_line'] = int(m.group(1))
                m = re.match(r'Threads:\s+(\d+)', l)
                if m: block['threads_line'] = int(m.group(1))
                m = re.match(r'Threads/proc:\s+(\d+)', l)
                if m: block['threads_line'] = int(m.group(1))
                m = re.match(r'Train time:\s+([\d.]+)', l)
                if m: run['train_time'] = float(m.group(1))
                m = re.match(r'Classify time:\s+([\d.]+)', l)
                if m: run['classify_time'] = float(m.group(1))
                m = re.match(r'CV time:\s+([\d.]+)', l)
                if m: run['cv_time'] = float(m.group(1))
                m = re.match(r'Total time:\s+([\d.]+)', l)
                if m:
                    run['total_time'] = float(m.group(1))
                    # Override procs/threads from block if available
                    if block['processes_line']: run['procs'] = block['processes_line']
                    if block['threads_line']:   run['threads'] = block['threads_line']
                    runs.append(run)
                    i += 1
                    break
                i += 1
            continue
        i += 1
    return runs


# ── Averaging and table building ───────────────────────────────────────────────
def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


def build_tables(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[(r['binary'], r['size'], r['procs'], r['threads'])].append(r)

    avgs = {}
    for k, g in groups.items():
        avgs[k] = {f: avg([r[f + '_time'] for r in g])
                   for f in ['train', 'classify', 'cv', 'total']}

    serial_1m = avgs.get(('serial', '1m', 1, 1), {}).get('total', 0)
    omp_base  = avgs.get(('omp',   '1m', 1, 1), {}).get('total', 0)
    mpi_base  = avgs.get(('mpi',   '1m', 1, 1), {}).get('total', 0)
    hyb_base  = avgs.get(('hybrid','1m', 1, 1), {}).get('total', 0)

    out = []

    def table(title, rows, base):
        s = f"\n{title}\n"
        s += f"  {'config':<16} {'train':>10} {'classify':>10} {'cv':>10} {'total':>10} {'speedup':>9} {'efficiency':>11}\n"
        s += f"  {'-'*16} {'----------':>10} {'----------':>10} {'----------':>10} {'----------':>10} {'---------':>9} {'-----------':>11}\n"
        for label, a, w in rows:
            sp = base / a['total'] if base > 0 and a['total'] > 0 else 0.0
            ef = sp / w if w > 0 else 0.0
            s += (f"  {label:<16} {a['train']:>10.6f} {a['classify']:>10.6f} "
                  f"{a['cv']:>10.6f} {a['total']:>10.6f} {sp:>9.3f} {ef:>11.3f}\n")
        return s

    # Serial
    size_ord = {'20k': 0, '100k': 1, '500k': 2, '1m': 3, 'heart': 4}
    serial_rows = sorted(
        [(k, a) for k, a in avgs.items() if k[0] == 'serial'],
        key=lambda x: size_ord.get(x[0][1], 99)
    )
    out.append(table("Serial (baseline = serial 1m)",
                     [(k[1], a, 1) for k, a in serial_rows], serial_1m))

    # OMP
    omp_rows = sorted(
        [(f"{t} threads", a, t)
         for (b, s, p, t), a in avgs.items() if b == 'omp' and s == '1m'],
        key=lambda x: int(x[0].split()[0])
    )
    out.append(table("OMP (1m dataset, baseline = 1 thread)", omp_rows, omp_base))

    # MPI
    mpi_rows = sorted(
        [(f"{p} procs", a, p)
         for (b, s, p, t), a in avgs.items() if b == 'mpi' and s == '1m'],
        key=lambda x: int(x[0].split()[0])
    )
    out.append(table("MPI (1m dataset, baseline = 1 proc)", mpi_rows, mpi_base))

    # Hybrid — one table per proc count
    hybrid_procs = sorted(set(p for (b, s, p, t) in avgs
                               if b == 'hybrid' and s == '1m'))
    for pc in hybrid_procs:
        rows = sorted(
            [(f"{p}p x {t}t", a, p * t)
             for (b, s, p, t), a in avgs.items()
             if b == 'hybrid' and s == '1m' and p == pc],
            key=lambda x: int(x[0].split('x')[1].strip().rstrip('t'))
        )
        out.append(table(
            f"Hybrid (1m dataset, {pc} procs, baseline = 1p x 1t)", rows, hyb_base
        ))

    # Data-size scaling
    sizes = ['20k', '100k', '500k', '1m']
    binaries = [
        ('serial', 1, 1, 'serial (1t)'),
        ('omp',    1, 8, 'omp (8t)'),
        ('mpi',    8, 1, 'mpi (8p)'),
        ('hybrid', 2, 4, 'hybrid (2p x 4t)'),
    ]
    s = "\nData-size scaling (total time and % of serial)\n"
    s += (f"  {'binary':<16}"
          + "".join(f"   {'time (s)':>8}  {'% serial':>8}  ({sz})" for sz in sizes)
          + "\n")
    s += (f"  {'-'*16}"
          + "".join(f"   {'--------':>8}  {'--------':>8}  {'------'}" for _ in sizes)
          + "\n")
    for (b, p, t, lbl) in binaries:
        line = f"  {lbl:<16}"
        for sz in sizes:
            a  = avgs.get((b, sz, p, t))
            sa = avgs.get(('serial', sz, 1, 1))
            if a and sa and sa['total'] > 0:
                line += f"   {a['total']:>8.4f}  {a['total']/sa['total']*100:>7.1f}%  "
            else:
                line += f"   {'N/A':>8}  {'N/A':>8}  "
        s += line + "\n"
    out.append(s)

    return out


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    filename = input("Enter the output filename: ").strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    runs = parse_output(filepath)
    print(f"\nParsed {len(runs)} total runs.")

    tables = build_tables(runs)
    base = os.path.splitext(filename)[0]
    parsed_file = os.path.join(script_dir, base + '_parsed.txt')

    with open(parsed_file, 'w') as f:
        for t in tables:
            f.write(t)
            f.write('\n')

    print(f"Summary tables written to {parsed_file}")