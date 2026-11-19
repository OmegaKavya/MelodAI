"""
The dataset assembled from raw audio rather than rendered spectrograms.

The five source folders collapse into three classes, and the collapse is not
a rename: eleven songs sit in more than one folder because the source CSV
lists them under more than one category, and the same YouTube URL was
downloaded twice. Seven of those pairs land inside one class, where the song
would simply be counted twice. Five cross class boundaries, where the same
audio carries two contradictory labels.

The original pipeline deduplicated by accident: identical filenames
overwrote each other during a folder merge, which is why 148 files became
141 songs. Depending on that is depending on filename luck, so this module
deduplicates on content hash and states what it dropped.
"""
import collections
import hashlib
import os

CLASSES = ["emotional", "energetic", "peaceful"]

SOURCE_TO_CLASS = {
    "calm": "peaceful",
    "drive": "peaceful",
    "party": "energetic",
    "romantic": "emotional",
    "sad": "emotional",
}


def digest(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def scan(root="data"):
    """One record per distinct piece of audio.

    Keyed by content, so a song present in two source folders appears once
    with both of its labels attached, and the caller can see the conflict
    rather than silently taking whichever was read last.
    """
    found = collections.defaultdict(lambda: {"paths": [], "sources": set()})
    for folder in sorted(os.listdir(root)):
        directory = os.path.join(root, folder)
        if not os.path.isdir(directory) or folder not in SOURCE_TO_CLASS:
            continue
        for name in sorted(os.listdir(directory)):
            if not name.lower().endswith((".mp3", ".wav", ".m4a")):
                continue
            path = os.path.join(directory, name)
            record = found[digest(path)]
            record["paths"].append(path)
            record["sources"].add(folder)

    items = []
    conflicted = []
    for key, record in found.items():
        classes = {SOURCE_TO_CLASS[s] for s in record["sources"]}
        entry = {
            "key": key,
            "path": record["paths"][0],
            "name": os.path.basename(record["paths"][0]),
            "sources": sorted(record["sources"]),
            "classes": sorted(classes),
        }
        if len(classes) > 1:
            conflicted.append(entry)
        else:
            entry["mood"] = classes.pop()
            entry["label"] = CLASSES.index(entry["mood"])
            items.append(entry)
    return items, conflicted


def split_by_song(items, validation_fraction=0.2, seed=42):
    """Stratified holdout over distinct songs.

    Songs, not clips: every window a song produces stays on one side.
    """
    import random

    by_class = collections.defaultdict(list)
    for item in items:
        by_class[item["mood"]].append(item)

    rng = random.Random(seed)
    train, validation = [], []
    for mood, group in by_class.items():
        ordered = sorted(group, key=lambda i: i["key"])
        rng.shuffle(ordered)
        take = max(1, round(len(ordered) * validation_fraction))
        validation += ordered[:take]
        train += ordered[take:]
    return train, validation


def summarise(items, conflicted, train=None, validation=None):
    lines = [
        f"distinct audio: {len(items) + len(conflicted)}",
        f"usable songs:   {len(items)}",
        f"dropped (contradictory labels): {len(conflicted)}",
    ]
    for entry in sorted(conflicted, key=lambda e: e["name"]):
        lines.append(f"    {entry['sources']} -> {entry['classes']}  {entry['name'][:44]}")
    counts = collections.Counter(i["mood"] for i in items)
    lines.append("songs per class: " + "  ".join(f"{c}={counts[c]}" for c in CLASSES))
    if train is not None:
        tc = collections.Counter(i["mood"] for i in train)
        vc = collections.Counter(i["mood"] for i in validation)
        lines.append("train: " + "  ".join(f"{c}={tc[c]}" for c in CLASSES))
        lines.append("val:   " + "  ".join(f"{c}={vc[c]}" for c in CLASSES))
    return "\n".join(lines)
