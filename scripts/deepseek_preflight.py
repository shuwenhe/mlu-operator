"""Fail closed unless CNMON reports the eight expected idle cards."""
import csv
import json
import re
import sys
from pathlib import Path


def inventory(snapshot):
    cards = {}
    current = None
    for line in snapshot.splitlines():
        card = re.match(r"^\|\s*(\d+)\s+/\s+(MLU\S+)\s+", line)
        if card:
            if current is not None:
                raise ValueError("CNMON card is missing a memory row")
            current = int(card[1])
            if current in cards or card[2] != "MLU590-M9":
                raise ValueError(f"Unexpected CNMON device row: {line}")
            continue
        memory = re.search(r"\|\s*(\d+)\s*MiB/\s*(\d+)\s*MiB\s*\|", line)
        if current is not None and memory:
            cards[current] = (int(memory[1]), int(memory[2]))
            current = None
    if current is not None or set(cards) != set(range(8)):
        raise ValueError(f"Expected complete CNMON inventory for MLU590-M9 cards 0–7, got {sorted(cards)}")
    lines = snapshot.split("Processes:", 1)
    idle = len(lines) == 2 and any(
        re.fullmatch(r"\|\s*No running processes found\s*\|", row) for row in lines[1].splitlines()
    )
    return [
        dict(card=card, card_memory_used_mib=used, card_memory_total_mib=total,
             process_table_empty=idle, occupied=bool(used) or not idle)
        for card, (used, total) in sorted(cards.items())
    ]


if __name__ == "__main__":
    snapshot, directory = map(Path, sys.argv[1:])
    try:
        rows = inventory(snapshot.read_text())
        with (directory / "card_occupancy.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (directory / "card_occupancy.json").write_text(json.dumps(rows, indent=2) + "\n")
        if any(card["occupied"] or card["card_memory_total_mib"] <= 0 for card in rows):
            raise ValueError("All eight cards must be idle with zero allocated device memory")
    except (ValueError, OSError) as exc:
        sys.exit(f"DeepSeek device preflight refused: {exc}")
