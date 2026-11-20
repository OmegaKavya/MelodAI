#!/usr/bin/env bash
# Publish to a Hugging Face Space.
#
# The Space is its own git remote, so this assembles a clean tree rather than
# pushing the research repo: the app, the four runtime assets, and the Space
# front matter as README.md. The training scripts, the experiment logs and the
# LFS checkpoints stay in GitHub where they belong.
#
# Usage: scripts/deploy_space.sh <hf-username> [space-name]
set -euo pipefail

USER="${1:?usage: deploy_space.sh <hf-username> [space-name]}"
NAME="${2:-MelodAI}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Assembling $USER/$NAME in $WORK"
cd "$WORK"
git clone "https://huggingface.co/spaces/$USER/$NAME" repo 2>/dev/null || {
  echo "Space not found. Create it first at:"
  echo "  https://huggingface.co/new-space  (SDK: Docker, name: $NAME)"
  exit 1
}
cd repo

# Replace tracked content rather than layering onto whatever was there before.
git rm -rq --ignore-unmatch . 2>/dev/null || true

cp "$ROOT/Dockerfile" .
cp "$ROOT/space/README.md" README.md
mkdir -p melodai server models web
cp "$ROOT"/melodai/*.py melodai/
cp "$ROOT"/server/*.py server/
cp "$ROOT/server/requirements.txt" server/
cp "$ROOT"/web/package.json "$ROOT"/web/package-lock.json "$ROOT"/web/index.html "$ROOT"/web/vite.config.js web/
cp -R "$ROOT/web/src" web/src
cp -R "$ROOT/web/public" web/public
for asset in energy_head.npz corpus.npz yamnet.tflite yamnet_class_map.csv; do
  cp "$ROOT/models/$asset" models/
done

git add -A
git commit -q -m "Deploy MelodAI" || { echo "Nothing changed."; exit 0; }
git push
echo "Live shortly at https://huggingface.co/spaces/$USER/$NAME"
