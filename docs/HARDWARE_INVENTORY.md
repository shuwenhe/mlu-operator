# Hardware and allocation

## Historical host snapshot

Host `tza-0a09-ai01-em9`. Read-only CNMON exited 0 and reported eight MLU590-M9 cards, driver v6.5.52 and firmware field v1.3.1. Each card had a vLLM worker using 93,556 MiB in the host snapshot. The separate workspace sandbox exposes no Cambricon nodes. Docker reported the existing GLM and two team containers.

| Card | PCI address | Allocation at capture |
| --- | --- | --- |
| 0 | 0000:4f:00.0 | Occupied |
| 1 | 0000:53:00.0 | Occupied |
| 2 | 0000:6f:00.0 | Occupied |
| 3 | 0000:73:00.0 | Occupied |
| 4 | 0000:af:00.0 | Occupied |
| 5 | 0000:b3:00.0 | Occupied |
| 6 | 0000:cf:00.0 | Occupied |
| 7 | 0000:d3:00.0 | Occupied |

No exclusive reservation was available at the time of this snapshot.

## Single-card readiness

Confirm current card allocation on the host. Obtain an exclusive single-card reservation and run the CNRT selection, queue, transfer and cleanup probe in compatible userspace. The probe enumerates all visible cards, so expose only the assigned device before starting it.
