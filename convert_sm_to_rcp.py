"""
convert_sm_to_rcp.py
--------------------
Zet PSPLIB-bestanden (.sm) om naar het Patterson-formaat (.rcp) dat ProPy kan inlezen
met read_patterson_file().

Gebruik (in de VS Code-terminal, vanuit de map DDPM_ProPy_V1.3):
    py convert_sm_to_rcp.py
Standaard: leest Example_Files/j30/*.sm en schrijft Example_Files/j30_rcp/*.rcp
Andere map:
    py convert_sm_to_rcp.py j60
"""

import os
import sys
import glob


def read_sm(path):
    """Lees een PSPLIB .sm-bestand (single-mode RCPSP)."""
    with open(path, "r") as f:
        lines = [line.rstrip() for line in f]

    n_jobs = None
    n_res = None
    successors = {}
    durations = {}
    requests = {}
    availability = None

    section = None
    for line in lines:
        s = line.strip()
        if not s or s.startswith("***") or s.startswith("---"):
            continue

        # Algemene info
        if s.startswith("jobs (incl. supersource/sink"):
            n_jobs = int(s.split(":")[1])
            continue
        if s.startswith("- renewable"):
            n_res = int(s.split(":")[1].split()[0])
            continue

        # Secties herkennen
        if s.startswith("PRECEDENCE RELATIONS"):
            section = "prec"
            continue
        if s.startswith("REQUESTS/DURATIONS"):
            section = "req"
            continue
        if s.startswith("RESOURCEAVAILABILITIES"):
            section = "avail"
            continue
        if s.startswith("PROJECT INFORMATION") or s.startswith("RESOURCES"):
            section = None
            continue

        # Kopregels van tabellen overslaan
        if s.startswith("jobnr.") or s.startswith("pronr.") or s.startswith("R 1") or s.startswith("R1"):
            continue

        parts = s.split()
        if section == "prec":
            job = int(parts[0])
            n_succ = int(parts[2])
            successors[job] = [int(x) for x in parts[3:3 + n_succ]]
        elif section == "req":
            job = int(parts[0])
            durations[job] = int(parts[2])
            requests[job] = [int(x) for x in parts[3:3 + n_res]]
        elif section == "avail":
            availability = [int(x) for x in parts[:n_res]]
            section = None

    # Controles
    if n_jobs is None or n_res is None or availability is None:
        raise ValueError("Bestand onvolledig of onbekend formaat")
    if len(successors) != n_jobs or len(durations) != n_jobs:
        raise ValueError(f"Aantal activiteiten klopt niet ({len(successors)}/{len(durations)} i.p.v. {n_jobs})")

    return n_jobs, n_res, availability, durations, requests, successors


def write_rcp(path, n_jobs, n_res, availability, durations, requests, successors):
    """Schrijf in Patterson-formaat:
       regel 1: #activiteiten  #resources
       regel 2: beschikbaarheid per resource
       daarna per activiteit: duur  vraag_R1 ... vraag_Rk  #opvolgers  opvolgers...
    """
    with open(path, "w") as f:
        f.write(f"{n_jobs} {n_res}\n")
        f.write(" ".join(str(a) for a in availability) + "\n\n")
        for job in range(1, n_jobs + 1):
            succ = successors[job]
            row = [durations[job]] + requests[job] + [len(succ)] + succ
            f.write(" ".join(str(x) for x in row) + "\n")


def main():
    subset = sys.argv[1] if len(sys.argv) > 1 else "j30"
    base = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base, "Example_Files", subset)
    dst_dir = os.path.join(base, "Example_Files", subset + "_rcp")

    files = sorted(glob.glob(os.path.join(src_dir, "*.sm")))
    if not files:
        print(f"Geen .sm-bestanden gevonden in {src_dir}")
        return

    os.makedirs(dst_dir, exist_ok=True)
    ok, failed = 0, []
    for src in files:
        name = os.path.splitext(os.path.basename(src))[0] + ".rcp"
        try:
            data = read_sm(src)
            write_rcp(os.path.join(dst_dir, name), *data)
            ok += 1
        except Exception as e:
            failed.append((os.path.basename(src), str(e)))

    print(f"{ok} bestanden omgezet naar {dst_dir}")
    if failed:
        print(f"{len(failed)} bestanden mislukt:")
        for name, err in failed[:10]:
            print(f"   {name}: {err}")


if __name__ == "__main__":
    main()
