# selfdoubt RunPod key + account pre-flight, sourced by pod_launch.sh, pod_wait.sh and pod_preflight.sh.
#   KEY_VAR   name of the key variable in ENV_FILE (default RUNPOD_API_KEY = Berkeley account; RUNPOD_API_KEY_2 = gmail)
#   ENV_FILE  default: the main checkout's .env
# The key is never printed and never put on a command line: it lives only in a 600-mode header file that curl reads
# with -H @file. Every API call of one run uses that one file, so the launcher, its DELETE timer and the waiter all
# use the same account. (The in-pod self-stop watchdog uses the pod's own RUNPOD_API_KEY, which RunPod injects for
# the account that created the pod, so it is the same account by construction.)
KEY_VAR="${KEY_VAR:-RUNPOD_API_KEY}"
ENV_FILE="${ENV_FILE:-/home/qamil-mirza/Code/llm-memctl/.env}"
API="${RUNPOD_REST:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GQL:-https://api.runpod.io/graphql}"

# rp_write_hdr <path>: write "Authorization: Bearer <key>" (mode 600) from $KEY_VAR in $ENV_FILE.
rp_write_hdr() {
  [[ "$KEY_VAR" =~ ^RUNPOD_API_KEY(_[A-Z0-9]+)?$ ]] || { echo "ABORT: KEY_VAR '$KEY_VAR' is not RUNPOD_API_KEY[_X]" >&2; return 2; }
  local k
  k=$(grep -E "^(export )?${KEY_VAR}=" "$ENV_FILE" 2>/dev/null | head -1 \
      | sed -E "s/^(export )?${KEY_VAR}=//; s/^\"//; s/\"\$//; s/^'//; s/'\$//" || true)
  [ -n "$k" ] || { echo "ABORT: $KEY_VAR is not set in $ENV_FILE" >&2; return 2; }
  ( umask 077; printf 'Authorization: Bearer %s\n' "$k" > "$1" )
}

# rp_preflight <hdr> <cap_usd>: needs EXPECT_EMAIL. Prints the account email, balance and pods (never the key).
# Returns non-zero unless the email matches, the balance covers the cap, and the account has 0 pods (pods are listed).
# Sibling: RP_ALLOW_IDS (comma-separated pod ids our own launcher recorded) may be running; any other pod aborts.
# RP_EXTRA_CAP (USD) is the siblings' remaining worst case, added to the cap the balance must cover.
rp_preflight() {
  [ -n "${EXPECT_EMAIL:-}" ] || { echo "ABORT: EXPECT_EMAIL not set" >&2; return 2; }
  local gq pods
  gq=$(curl -s -H @"$1" -H 'Content-Type: application/json' "$GQL" -d '{"query":"{ myself { email clientBalance } }"}')
  pods=$(curl -s -H @"$1" "$API/pods")
  RP_GQ="$gq" RP_PODS="$pods" python3 - "$EXPECT_EMAIL" "$2" "$KEY_VAR" "${RP_ALLOW_IDS:-}" "${RP_EXTRA_CAP:-0}" <<'PY'
import json, os, sys
want, cap, var = sys.argv[1].strip().lower(), float(sys.argv[2]), sys.argv[3]
allow, extra = {x for x in sys.argv[4].split(",") if x}, float(sys.argv[5])
try:
    me = json.loads(os.environ["RP_GQ"])["data"]["myself"]
    email, bal = (me["email"] or "").strip().lower(), float(me["clientBalance"])
except Exception as e:
    print(f"ABORT: account query failed for {var} ({type(e).__name__})"); sys.exit(3)
try:
    pods = json.loads(os.environ["RP_PODS"])
    assert isinstance(pods, list)
except Exception as e:
    print(f"ABORT: pod list failed for {var} ({type(e).__name__})"); sys.exit(3)
need = cap + extra
print(f"account ({var}): {email}, balance ${bal:.4f}; cap ${cap:.3f}"
      + (f" + sibling ${extra:.3f}" if extra else "") + f"; pods {len(pods)}")
for p in pods:
    tag = " (allowed sibling)" if p.get("id") in allow else ""
    print(f"  pod {p.get('id')} name={p.get('name')} status={p.get('desiredStatus')} $/h={p.get('costPerHr')}{tag}")
bad = []
if email != want: bad.append(f"email is {email}, expected {want}")
if bal < need: bad.append(f"balance ${bal:.4f} below cap ${need:.3f}")
other = [p for p in pods if p.get("id") not in allow]
if other: bad.append(f"{len(other)} pod(s) already on this account")
if bad:
    print("ABORT: " + "; ".join(bad)); sys.exit(4)
print("pre-flight OK")
PY
}
