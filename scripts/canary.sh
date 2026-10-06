#!/usr/bin/env bash
#
# Bounded cluster inference canary. Runs on the head (rank 0).
#
# A healthy /health endpoint does not prove that distributed generation works:
# the API answers before the ranks have finished establishing a usable
# communicator. This sends one tiny deterministic completion and requires that
# the server reports generating at least one completion token.
#
# A non-empty reply body is not sufficient evidence — short quoted JSON fields
# such as "role":"assistant" or "finish_reason":"stop" would pass a naive
# quoted-string match even with an empty completion — so the check is on
# usage.completion_tokens.
#
# Usage: bash canary.sh BASE_URL MODEL

set -uo pipefail

BASE="${1:?base URL such as http://10.20.0.11:8889}"
MODEL="${2:?served model name}"

MAX_TOKENS="${CANARY_MAX_TOKENS:-8}"
TIMEOUT="${CANARY_TIMEOUT:-300}"
RETRIES="${CANARY_RETRIES:-3}"
RETRY_DELAY="${CANARY_RETRY_DELAY:-10}"

payload="$(mktemp --tmpdir canary.XXXXXX.json)"
trap 'rm -f -- "${payload}"' EXIT

printf '{"model":"%s","messages":[{"role":"user","content":"ping"}],"max_tokens":%s,"stream":false}\n' \
  "${MODEL}" "${MAX_TOKENS}" > "${payload}"

attempt=0
while [ "${attempt}" -lt "${RETRIES}" ]; do
  attempt=$((attempt + 1))
  reply="$(curl -fsS --max-time "${TIMEOUT}" \
    -H 'Content-Type: application/json' \
    --data-binary "@${payload}" \
    "${BASE}/v1/chat/completions" 2>/dev/null)" || reply=""
  if printf '%s' "${reply}" | grep -Eq '"completion_tokens"[: ]*[1-9][0-9]*'; then
    echo "CANARY_PASS cluster inference works (attempt ${attempt}/${RETRIES})"
    exit 0
  fi
  echo "canary not yet satisfied (attempt ${attempt}/${RETRIES})"
  [ "${attempt}" -lt "${RETRIES}" ] && sleep "${RETRY_DELAY}"
done

echo "CANARY_FAIL no bounded completion produced a generated token in ${RETRIES} attempts" >&2
exit 1
