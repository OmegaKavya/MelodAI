"""
Dataset assembly: song-level splits, with contradictory labels removed.

Three problems with how the data was previously fed to the model, all of
which this module fixes.

1. The split was alphabetical over filenames. Songs stayed whole by accident
   of the naming scheme, but the holdout was whichever songs sorted last,
   which is arbitrary and not reproducible under any renaming.

2. Five songs exist as byte-identical images under two different class
   folders, because the source CSV lists eleven songs under more than one
   category and the same YouTube URL was downloaded twice. The model was
   trained on an image labelled peaceful and the same image labelled
   emotional. Every one of those five involves `peaceful`, which is the
   class that scores 9% per song.

3. Nothing enforced that a song's eight augmented variants stay on one side
   of the split.

Groups are songs. A song is in training or in validation, never both, and a
song whose label is contradictory is dropped rather than guessed at.
"""
import collections
import hashlib
import os
import random

CLASSES = ["emotional", "energetic", "peaceful"]

# The five source categories collapse into three classes. This mapping was
# never written down: it existed only as a folder reorganisation, so the
# only record of it was the arrangement of the output directories. Recovered
# by matching 127 CSV rows to the class folder their spectrograms landed in.
#
# DRIVE joining CALM rather than PARTY is worth noticing. It puts mid-tempo
# road songs beside genuinely quiet ones, and `peaceful` is the class the
# models struggle with most. It also leaves `energetic` as PARTY alone, which
# is why that class has 29 songs against emotional's 60.
SOURCE_TO_CLASS = {
    "calm": "peaceful",
    "drive": "peaceful",
    "party": "energetic",
    "romantic": "emotional",
    "sad": "emotional",
}


def song_of(filename):
    """The song a variant belongs to: everything before the segment marker."""
    return os.path.basename(filename).split("_seg")[0]


def scan(root):
    """Every image, with its class and song."""
    items = []
    for label, mood in enumerate(CLASSES):
        directory = os.path.join(root, mood)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.endswith(".png"):
                items.append({
                    "path": os.path.join(directory, name),
                    "mood": mood,
                    "label": label,
                    "song": song_of(name),
                })
    return items


def contradictory_songs(items):
    """Songs whose images appear byte-identical under two different classes.

    Hashed rather than matched by name: the same song can be downloaded under
    two titles, and two different songs can share a title. Identical bytes is
    the only claim that cannot be argued with.
    """
    by_hash = collections.defaultdict(set)
    hash_songs = collections.defaultdict(set)
    for item in items:
        with open(item["path"], "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        by_hash[digest].add(item["mood"])
        hash_songs[digest].add(item["song"])

    bad = set()
    for digest, moods in by_hash.items():
        if len(moods) > 1:
            bad |= hash_songs[digest]
    return bad


def split_by_song(items, validation_fraction=0.2, seed=42, drop_contradictions=True):
    """Partition into training and validation, grouping by song.

    Stratified per class, so a 20% holdout takes 20% of each class's songs
    rather than 20% of all songs, which on this corpus would leave the
    smallest class barely represented.
    """
    dropped = contradictory_songs(items) if drop_contradictions else set()
    kept = [i for i in items if i["song"] not in dropped]

    songs_by_class = collections.defaultdict(set)
    for item in kept:
        songs_by_class[item["mood"]].add(item["song"])

    rng = random.Random(seed)
    validation_songs = set()
    for mood, songs in songs_by_class.items():
        ordered = sorted(songs)
        rng.shuffle(ordered)
        take = max(1, round(len(ordered) * validation_fraction))
        validation_songs |= set(ordered[:take])

    train = [i for i in kept if i["song"] not in validation_songs]
    validation = [i for i in kept if i["song"] in validation_songs]
    return train, validation, dropped


def summarise(train, validation, dropped):
    lines = [f"dropped {len(dropped)} songs with contradictory labels"]
    for name, part in (("train", train), ("val", validation)):
        counts = collections.Counter(i["mood"] for i in part)
        songs = collections.Counter()
        for i in part:
            songs[i["mood"]] = songs[i["mood"]]
        distinct = {m: len({i["song"] for i in part if i["mood"] == m}) for m in CLASSES}
        lines.append(
            f"{name:5} {len(part):5d} images  " +
            "  ".join(f"{m}={counts[m]:4d}({distinct[m]:2d} songs)" for m in CLASSES)
        )
    overlap = {i["song"] for i in train} & {i["song"] for i in validation}
    lines.append(f"songs in both splits: {len(overlap)}")
    return "\n".join(lines)
