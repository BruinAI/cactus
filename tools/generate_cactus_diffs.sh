#!/usr/bin/env bash

set -euo pipefail

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is not installed or not in PATH" >&2
  exit 1
fi

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Error: could not determine git repository root" >&2
  exit 1
fi

OUTPUT_DIR="${REPO_ROOT}/diffs"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

mapfile -t CHANGED_FILES < <(git diff --name-only origin/main -- 'cactus/**/*.cpp' 'cactus/**/*.h')

if [[ ${#CHANGED_FILES[@]} -eq 0 ]]; then
  echo "No .cpp or .h changes detected under cactus/ relative to origin/main"
  exit 0
fi

for file in "${CHANGED_FILES[@]}"; do
  diff_output=$(git diff origin/main -- "${file}")
  if [[ -z "${diff_output}" ]]; then
    continue
  fi
  output_path="${OUTPUT_DIR}/${file}.diff"
  mkdir -p "$(dirname "${output_path}")"
  printf '%s\n' "${diff_output}" > "${output_path}"
  echo "Wrote diff for ${file} -> ${output_path}"
done

echo "All diffs written under ${OUTPUT_DIR}"
