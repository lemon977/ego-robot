#!/usr/bin/env bash
# Commit the whole working tree onto refs/heads/takeover-20261004 without touching HEAD/index/worktree.
# Usage: bash scripts/takeover/commit_snapshot.sh "message"
set -euo pipefail
cd /path/to/chaoyang
MSG=${1:?message required}
BRANCH=refs/heads/takeover-20261004
PARENT=$(git rev-parse "$BRANCH")
export GIT_INDEX_FILE=.git/takeover_snapshot_index.tmp
rm -f "$GIT_INDEX_FILE"
git read-tree HEAD
git add -A
TREE=$(git write-tree)
rm -f "$GIT_INDEX_FILE"; unset GIT_INDEX_FILE
if [ "$TREE" = "$(git rev-parse "$PARENT^{tree}")" ]; then echo "NO_CHANGE $PARENT"; exit 0; fi
COMMIT=$(git commit-tree "$TREE" -p "$PARENT" -m "$MSG")
git update-ref "$BRANCH" "$COMMIT" "$PARENT"
echo "COMMITTED $COMMIT tree=$TREE parent=$PARENT"
