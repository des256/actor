#!/bin/bash
set -euo pipefail
EXCLUDES=(
    --exclude=target/
    --exclude=.git/
    --exclude=.worktrees/
    --exclude='mutants.out*'
)
echo "pushing to murdock"
rsync -az --delete "${EXCLUDES[@]}" ./ "murdock:/home/desmond/actor/"
