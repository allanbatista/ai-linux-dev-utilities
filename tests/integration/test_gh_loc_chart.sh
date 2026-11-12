#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="$ROOT_DIR/scripts/gh-loc-chart"
TODAY="$(date -u +%Y-%m-%d)"
YESTERDAY="$(date -u -d '1 day ago' +%Y-%m-%d)"
FAKE_DIR=""
RUN_OUTPUT=""
RUN_STATUS=0

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_status() {
    local expected="$1"
    [ "$RUN_STATUS" -eq "$expected" ] || fail "expected exit $expected, got $RUN_STATUS"
}

assert_contains() {
    local needle="$1"
    printf '%s' "$RUN_OUTPUT" | grep -Fq -- "$needle" || fail "missing: $needle"
}

assert_not_contains() {
    local needle="$1"
    if printf '%s' "$RUN_OUTPUT" | grep -Fq -- "$needle"; then
        fail "unexpected: $needle"
    fi
}

make_fake_gh() {
    FAKE_DIR="$(mktemp -d)"
    mkdir -p "$FAKE_DIR/bin"
    cat >"$FAKE_DIR/bin/gh" <<'EOF'
#!/bin/bash
set -euo pipefail

debug_log() {
    [ -n "${GH_SMOKE_DEBUG:-}" ] || return 0
    printf '%s\n' "$*" >>"$GH_SMOKE_DEBUG"
}

emit_commit() {
    cat <<JSON
{"files":[{"filename":"src/main.py","additions":$1,"deletions":$2},{"filename":"graphify-out/graph.json","additions":100,"deletions":50},{"filename":".features/spec.md","additions":100,"deletions":50},{"filename":".sdd/plan.md","additions":100,"deletions":50},{"filename":".changelog/change.md","additions":100,"deletions":50}]}
JSON
}

emit_history() {
    local date="$1"
    local additions="$2"
    local deletions="$3"
    debug_log "emit_history date=$date add=$additions del=$deletions"
    cat <<JSON
{"data":{"repository":{"defaultBranchRef":{"target":{"history":{"pageInfo":{"hasNextPage":false,"endCursor":null},"nodes":[{"committedDate":"${date}T10:00:00Z","additions":${additions},"deletions":${deletions}}]}}}}}}
JSON
}

emit_empty_history() {
    cat <<'JSON'
{"data":{"repository":{"defaultBranchRef":{"target":{"history":{"pageInfo":{"hasNextPage":false,"endCursor":null},"nodes":[]}}}}}}
JSON
}

emit_org_page1() {
    cat <<'JSON'
{"data":{"repositoryOwner":{"login":"Acme","repositories":{"pageInfo":{"hasNextPage":true,"endCursor":"C1"},"nodes":[{"nameWithOwner":"Acme/repo-one","owner":{"login":"Acme"}},{"nameWithOwner":"Other/repo-x","owner":{"login":"Other"}}]}}}}
JSON
}

emit_org_page2() {
    cat <<'JSON'
{"data":{"repositoryOwner":{"login":"Acme","repositories":{"pageInfo":{"hasNextPage":false,"endCursor":null},"nodes":[{"nameWithOwner":"Acme/repo-two","owner":{"login":"Acme"}}]}}}}
JSON
}

emit_org_empty() {
    cat <<'JSON'
{"data":{"repositoryOwner":{"login":"Empty","repositories":{"pageInfo":{"hasNextPage":false,"endCursor":null},"nodes":[]}}}}
JSON
}

emit_org_error() {
    cat <<'JSON'
{"data":{"repositoryOwner":null},"errors":[{"message":"not found"}]}
JSON
}

emit_history_error() {
    cat <<'JSON'
{"data":{"repository":{"defaultBranchRef":null}},"errors":[{"message":"boom"}]}
JSON
}

case "${1:-}" in
    auth)
        [ "${2:-}" = "status" ] || exit 1
        exit 0
        ;;
    api)
        if [ "${2:-}" = "user" ]; then
            printf '{"login":"tester","node_id":"USER_NODE"}\n'
            exit 0
        fi
        case "${2:-}" in
            repos/acme/base-repo/commits/base) emit_commit 3 1; exit 0 ;;
            repos/acme/extra-repo/commits/extra) emit_commit 7 2; exit 0 ;;
            repos/Acme/repo-one/commits/one) emit_commit 10 2; exit 0 ;;
            repos/Acme/repo-two/commits/two) emit_commit 5 1; exit 0 ;;
        esac
        [ "${2:-}" = "graphql" ] || exit 1
        shift 2

        login=""
        owner=""
        name=""
        cursor=""
        author_id=""
        since=""
        query_blob=""
        saw_q=0

        while [ $# -gt 0 ]; do
            case "$1" in
                -q)
                    saw_q=1
                    shift 2
                    ;;
                -F|-f)
                    key="${2%%=*}"
                    val="${2#*=}"
                    case "$1:$key" in
                        -F:login) login="$val" ;;
                        -F:owner) owner="$val" ;;
                        -F:name) name="$val" ;;
                        -F:cursor) cursor="$val" ;;
                        -F:authorId) author_id="$val" ;;
                        -F:since) since="$val" ;;
                        -f:query) query_blob="$val" ;;
                    esac
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done

        debug_log "scenario=${SCENARIO:-} saw_q=$saw_q login=$login owner=$owner name=$name cursor=$cursor author_id=$author_id since=$since query_len=${#query_blob}"

        if [ "$saw_q" -eq 1 ]; then
            debug_log "branch=default"
            case "${SCENARIO:-}" in
                default-regression)
                    printf '%s\n' "acme/base-repo" "acme/extra-repo"
                    exit 0
                    ;;
                *)
                    echo "unexpected -q usage for scenario ${SCENARIO:-}" >&2
                    exit 1
                    ;;
            esac
        fi

        if [ -n "$login" ] && [ -z "$author_id" ] && [ -z "$since" ]; then
            debug_log "branch=org"
            case "${SCENARIO:-}" in
                org-discovery|org-added|org-liquid|org-zero-commits|org-partial-error)
                    if [ "$login" = "Acme" ]; then
                        if [ "$cursor" = "C1" ]; then
                            emit_org_page2
                            exit 0
                        fi
                        emit_org_page1
                        exit 0
                    fi
                    ;;
                org-empty)
                    if [ "$login" = "Empty" ]; then
                        emit_org_empty
                        exit 0
                    fi
                    ;;
                org-error)
                    if [ "$login" = "Missing" ]; then
                        emit_org_error
                        exit 0
                    fi
                    ;;
            esac
            echo "unexpected org discovery args: $*" >&2
            exit 1
        fi

        if [ -n "$author_id" ] || [ -n "$since" ]; then
            debug_log "branch=history"
            debug_log "history_values owner=$(printf '%q' "$owner") name=$(printf '%q' "$name") author_id=$(printf '%q' "$author_id") since=$(printf '%q' "$since")"
            case "${SCENARIO:-}" in
                default-regression|repo-regression)
                    if [ "$owner" = "acme" ] && [ "$name" = "base-repo" ]; then
                        debug_log "match=base"
                        debug_log "before_emit_base"
                        cat <<JSON
{"data":{"repository":{"defaultBranchRef":{"target":{"history":{"pageInfo":{"hasNextPage":false,"endCursor":null},"nodes":[{"oid":"base","committedDate":"${FAKE_TODAY}T10:00:00Z"}]}}}}}}
JSON
                        debug_log "after_emit_base"
                        exit 0
                    fi
                    if [ "$owner" = "acme" ] && [ "$name" = "extra-repo" ]; then
                        debug_log "match=extra"
                        debug_log "before_emit_extra"
                        cat <<JSON
{"data":{"repository":{"defaultBranchRef":{"target":{"history":{"pageInfo":{"hasNextPage":false,"endCursor":null},"nodes":[{"oid":"extra","committedDate":"${FAKE_YESTERDAY}T10:00:00Z"}]}}}}}}
JSON
                        debug_log "after_emit_extra"
                        exit 0
                    fi
                    debug_log "match=none"
                    ;;
                org-added|org-liquid|org-discovery|org-partial-error|org-zero-commits)
                    if [ "$owner" = "Acme" ] && [ "$name" = "repo-one" ]; then
                        debug_log "match=repo-one"
                        case "${SCENARIO:-}" in
                            org-zero-commits|org-discovery)
                                emit_empty_history
                                exit 0
                                ;;
                            *)
                                cat <<JSON
{"data":{"repository":{"defaultBranchRef":{"target":{"history":{"pageInfo":{"hasNextPage":false,"endCursor":null},"nodes":[{"oid":"one","committedDate":"${FAKE_TODAY}T10:00:00Z"}]}}}}}}
JSON
                                exit 0
                                ;;
                        esac
                    fi
                    if [ "$owner" = "Acme" ] && [ "$name" = "repo-two" ]; then
                        debug_log "match=repo-two"
                        case "${SCENARIO:-}" in
                            org-partial-error)
                                emit_history_error
                                exit 0
                                ;;
                            org-zero-commits|org-discovery)
                                emit_empty_history
                                exit 0
                                ;;
                            *)
                                cat <<JSON
{"data":{"repository":{"defaultBranchRef":{"target":{"history":{"pageInfo":{"hasNextPage":false,"endCursor":null},"nodes":[{"oid":"two","committedDate":"${FAKE_YESTERDAY}T10:00:00Z"}]}}}}}}
JSON
                                exit 0
                                ;;
                        esac
                    fi
                    ;;
            esac
            echo "unexpected history args: $*" >&2
            exit 1
        fi

        echo "unexpected graphql args: $*" >&2
        exit 1
        ;;
esac

echo "unexpected gh command: $*" >&2
exit 1
EOF
    chmod +x "$FAKE_DIR/bin/gh"
}

run_cli() {
    local scenario="$1"
    shift
    local out_file status
    out_file="$(mktemp)"
    set +e
    env \
        PATH="$FAKE_DIR/bin:$PATH" \
        SCENARIO="$scenario" \
        FAKE_TODAY="$TODAY" \
        FAKE_YESTERDAY="$YESTERDAY" \
        FORCE_COLOR="${FORCE_COLOR:-}" \
        NO_COLOR="${NO_COLOR:-}" \
        "$TARGET" "$@" >"$out_file" 2>&1
    status=$?
    set -e
    RUN_STATUS="$status"
    RUN_OUTPUT="$(cat "$out_file")"
    rm -f "$out_file"
}

run_cli_color() {
    local scenario="$1"
    shift
    local out_file status
    out_file="$(mktemp)"
    set +e
    env \
        PATH="$FAKE_DIR/bin:$PATH" \
        SCENARIO="$scenario" \
        FAKE_TODAY="$TODAY" \
        FAKE_YESTERDAY="$YESTERDAY" \
        FORCE_COLOR=1 \
        "$TARGET" "$@" >"$out_file" 2>&1
    status=$?
    set -e
    RUN_STATUS="$status"
    RUN_OUTPUT="$(cat "$out_file")"
    rm -f "$out_file"
}

test_default_regression() {
    run_cli default-regression --dias 2
    assert_status 0
    assert_contains "Listando repositórios com contribuições..."
    assert_contains "Commits processados: 2"
    assert_contains "Por repositório:"
    assert_contains "acme/extra-repo: 7"
    assert_contains "acme/base-repo: 3"
}

test_repo_regression() {
    run_cli repo-regression --repo acme/base-repo --dias 2
    assert_status 0
    assert_contains "Repositório:   acme/base-repo"
    assert_contains "Commits processados: 1"
    assert_not_contains "Por repositório:"
}

test_org_discovery() {
    run_cli org-discovery --org Acme --dias 2
    assert_status 0
    assert_contains "Organização:   Acme"
    assert_contains "Listando repositórios da organização..."
    assert_contains "Acme/repo-one"
    assert_contains "Acme/repo-two"
    assert_not_contains "Other/repo-x"
}

test_org_empty() {
    run_cli org-empty --org Empty --dias 2
    assert_status 0
    assert_contains "Nenhum repositório acessível encontrado para a organização 'Empty'."
    assert_not_contains "Commits processados:"
}

test_org_error() {
    run_cli org-error --org Missing --dias 2
    assert_status 1
    assert_contains "Erro: organização 'Missing' inexistente, inacessível ou inválida."
}

test_org_added() {
    run_cli org-added --org Acme --dias 2
    assert_status 0
    assert_contains "Organização:   Acme"
    assert_contains "Commits processados: 2"
    assert_contains "Linhas adicionadas por dia (máx=10, total=15):"
    assert_contains "Acme/repo-one: 10"
    assert_contains "Acme/repo-two: 5"
    assert_not_contains "Other/repo-x"

    chart_width=$(printf '%s\n' "$RUN_OUTPUT" | awk -F '│' '/^  [0-9][0-9][0-9][0-9]-/ { print length($2); exit }')
    [ "$chart_width" = "52" ] || fail "expected chart body width 52, got ${chart_width:-unset}"
}

test_org_colored() {
    run_cli_color org-added --org Acme --dias 2
    assert_status 0
    assert_contains $'\033[38;5;'
    assert_contains "Acme/repo-one: 10"
    assert_contains "Acme/repo-two: 5"
}

test_org_liquid() {
    run_cli org-liquid --org Acme --dias 2 --liquido
    assert_status 0
    assert_contains "Modo:          líquido (additions - deletions)"
    assert_contains "Linhas líquidas por dia (máx=8, total=12):"
    assert_contains "Commits processados: 2"
}

test_org_zero_commits() {
    run_cli org-zero-commits --org Acme --dias 2
    assert_status 0
    assert_contains "Commits processados: 0"
    assert_contains "Linhas adicionadas por dia (máx=0, total=0):"
}

test_org_partial_error() {
    run_cli org-partial-error --org Acme --dias 2
    assert_status 0
    assert_contains "Aviso: alguns repositórios foram ignorados por erro:"
    assert_contains "Acme/repo-two"
    assert_contains "Acme/repo-one: 10"
    assert_not_contains "Acme/repo-two: 5"
}

test_invalid_days() {
    run_cli org-zero-commits --dias 0
    assert_status 1
    assert_contains "Erro: --dias precisa ser inteiro positivo."

    run_cli repo-regression --repo acme/base-repo --dias 0
    assert_status 1
    assert_contains "Erro: --dias precisa ser inteiro positivo."

    run_cli org-added --org Acme --dias 0
    assert_status 1
    assert_contains "Erro: --dias precisa ser inteiro positivo."
}

main() {
    [ -x "$TARGET" ] || fail "missing executable: $TARGET"
    make_fake_gh
    trap 'rm -rf "$FAKE_DIR"' EXIT

    case "${1:-all}" in
        all)
            test_default_regression
            test_repo_regression
            test_org_discovery
            test_org_empty
            test_org_error
            test_org_added
            test_org_colored
            test_org_liquid
            test_org_zero_commits
            test_org_partial_error
            test_invalid_days
            ;;
        default-regression) test_default_regression ;;
        repo-regression) test_repo_regression ;;
        org-discovery) test_org_discovery ;;
        org-empty) test_org_empty ;;
        org-error) test_org_error ;;
        org-added) test_org_added ;;
        org-colored) test_org_colored ;;
        org-liquid) test_org_liquid ;;
        org-zero-commits) test_org_zero_commits ;;
        org-partial-error) test_org_partial_error ;;
        invalid-dias) test_invalid_days ;;
        *)
            fail "usage: $0 [all|default-regression|repo-regression|org-discovery|org-empty|org-error|org-added|org-colored|org-liquid|org-zero-commits|org-partial-error|invalid-dias]"
            ;;
    esac
}

main "$@"
