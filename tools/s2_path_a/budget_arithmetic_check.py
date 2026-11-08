#!/usr/bin/env python3
"""S2 go/no-go arithmetic check (design doc §4 item 2b).

Pure arithmetic, no vllm imports. Anchors the KV accounting for the
C8-mixed target geometry to LOG-MEASURED constants and cross-checks three
views (design slot model / all-groups-per-window / amortized ceiling).

Measured anchors (A 机 run-3 + merge bf16 smoke, 2026-11-08):
- C8 kernel block [512] hard constraint -> C8 page 135,168 B = 512 tok x 264 B
  (the 33,792 B "264x128" page model is DEAD).
- Upstream skip-layers platform path pads every layer to shared page 270,336 B
  (block 512->1024); run-3 budget line `0.2841 (6758400/23789568)` = 25/88,
  capacity 448,133 tokens = 5.3x LOSS vs bf16 2,387,389 -> REJECTED form.
- V1 run-3 groups [1024, 128, 128, 128] prove the scheduler supports mixed
  group block sizes via lcm.

Layer split: 25 full + 63 sliding (no MTP); production adds 1 MTP full layer.
"""
import math

BF16_PAGE_128 = 65_536          # B, 128-token bf16 page (512 B/token)
C8_PAGE_512 = 135_168           # B, 512-token C8 page (264 B/token)
C8_PAGE_1024_PADDED = 270_336   # B, upstream-padded shared page
N_FULL, N_SLIDING = 25, 63      # no-MTP split (26/89 with MTP)
SLIDING_GROUPS = 3
MTP_EXTRA_FULL = 1
BF16_B_TOK, C8_B_TOK = 512, 264
WINDOW = 512                    # lcm window: full kernel 512, sliding 128
ISL, SLID_WIN = 65_536, 4_096   # request profile

failures = []


def check(name, got, want, tol=0.0):
    ok = abs(got - want) <= tol if tol else got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got:,} want={want:,}")
    if not ok:
        failures.append(name)


# --- 1. page models ----------------------------------------------------------
check("bf16 page @128 tok = 512 B/token", BF16_PAGE_128 // 128, BF16_B_TOK)
check("C8 page @512 tok = 264 B/token", C8_PAGE_512 // 512, C8_B_TOK)

# --- 2. run-3 measured anchors (private per-layer path, REJECTED form) -------
check("run-3 budget bpb = 25 x padded page", 25 * C8_PAGE_1024_PADDED, 6_758_400)
check("run-3 total = 88 x padded page", 88 * C8_PAGE_1024_PADDED, 23_789_568)
check("run-3 scale = 25/88", round(6_758_400 / 23_789_568, 4), 0.2841)
check("run-3 sliding density = padded page/(128 tok x bf16) = 4x waste",
      C8_PAGE_1024_PADDED // (128 * BF16_B_TOK), 4)

# --- 3. design slot model (production MTP form, 26 full layers) --------------
nf = N_FULL + MTP_EXTRA_FULL
full_c8_win = nf * C8_PAGE_512                    # full group slot content/window
sliding_win = (N_SLIDING // SLIDING_GROUPS) * (WINDOW // 128) * BF16_PAGE_128
bpb_c8 = max(full_c8_win, sliding_win)
bpb_bf16 = nf * (WINDOW // 128) * BF16_PAGE_128   # bf16: full group pins
check("full C8 slot /window (MTP)", full_c8_win, 3_514_368)
check("sliding slot /window", sliding_win, 5_505_024)
check("bpb(C8) sliding-pinned /window", bpb_c8, 5_505_024)
check("bpb(bf16) full-pinned /window", bpb_bf16, 6_815_744)
check("B/token C8-mixed (design slot model)", bpb_c8 // WINDOW, 10_752)
check("B/token bf16 (design slot model)", bpb_bf16 // WINDOW, 13_312)
check("design pool ratio", round(bpb_bf16 / bpb_c8, 3), 1.238)
# no-MTP form (25 full): ratio drops to 25/21 shape
bpb_bf16_25 = N_FULL * 4 * BF16_PAGE_128
check("no-MTP pool ratio (25/21 shape)", round(bpb_bf16_25 / bpb_c8, 3), 1.190)
waste = 1 - full_c8_win / bpb_c8
check("full-group slot waste ~36% (MTP)", round(waste, 2), 0.36, tol=0.01)

# --- 4. cross-views ----------------------------------------------------------
# all-groups-per-window (no cross-request sliding reuse):
all_c8 = full_c8_win + N_SLIDING * (WINDOW // 128) * BF16_PAGE_128
all_bf16 = (nf + N_SLIDING) * (WINDOW // 128) * BF16_PAGE_128
check("all-groups/window B/token C8 (MTP)", all_c8 // WINDOW, 39_120)
check("all-groups/window B/token bf16", all_bf16 // WINDOW, 45_568)
check("all-groups/window ratio ~1.17", round(all_bf16 / all_c8, 3), 1.165)
# amortized per-request ceiling (sliding stores only its 4,096 window):
req_c8 = ISL * nf * C8_B_TOK + SLID_WIN * N_SLIDING * BF16_B_TOK
req_bf16 = ISL * nf * BF16_B_TOK + SLID_WIN * N_SLIDING * BF16_B_TOK
print(f"INFO  per-request: C8-mixed {req_c8:,} B vs bf16 {req_bf16:,} B "
      f"-> ceiling x{req_bf16 / req_c8:.2f}")
check("amortized ceiling ~1.7", round(req_bf16 / req_c8, 2), 1.73, tol=0.02)
check("consistency: design ratio <= ceiling", bpb_bf16 / bpb_c8 <= req_bf16 / req_c8, True)
check("consistency: all-groups ratio <= ceiling", all_bf16 / all_c8 <= req_bf16 / req_c8, True)

print()
if failures:
    print(f"FAILED: {len(failures)}: {failures}")
    raise SystemExit(1)
print("ALL GREEN — token-normalized accounting: design slot model 10,752 "
      "B/token (x1.238 vs bf16 13,312, MTP form; x1.19 no-MTP); floor "
      "(no window reuse) x1.17; hard ceiling x1.73. Rejected forms: 33,792 "
      "page model; skip-layers padded path (4x sliding waste, 5.3x loss).")
