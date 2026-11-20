"""Write the corpus map into the client bundle.

The map is the project's finding, not a side effect of an upload, so it
should not wait on a sleeping free tier API to appear. It is 131 songs and
four fields each, so baking it in costs a few kilobytes and removes the only
request the page makes before a visitor does anything.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from melodai.runtime import Runtime  # noqa: E402


def main():
    songs = Runtime(ROOT).corpus()
    out = os.path.join(ROOT, "web", "src", "corpus.json")
    with open(out, "w") as handle:
        json.dump(songs, handle, separators=(",", ":"))
    print(f"wrote {len(songs)} songs to {out} ({os.path.getsize(out) / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
