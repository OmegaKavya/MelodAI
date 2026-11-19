"""Do the five source categories separate better than the merged three?

peaceful = calm + drive. Drive songs are mid-tempo road music; calm songs are
quiet. If that class is heterogeneous, a model asked to learn it will fail at
exactly the rate observed, and the fault is the taxonomy rather than the
model. Cheap to test: the embeddings are already cached, only the labels change.
"""
import sys, os, collections, math
sys.path.insert(0, 'scripts')
import numpy as np, tensorflow as tf
from audio_dataset import SOURCE_TO_CLASS, digest
from train_yamnet import load_cache, build, wilson

SOURCES = ["calm", "drive", "party", "romantic", "sad"]

# Rebuild the manifest keeping the original five labels.
found = collections.defaultdict(lambda: {"paths": [], "sources": set()})
for folder in SOURCES:
    d = os.path.join("data", folder)
    for name in sorted(os.listdir(d)):
        if name.lower().endswith(".mp3"):
            p = os.path.join(d, name)
            r = found[digest(p)]; r["paths"].append(p); r["sources"].add(folder)

items = []
for key, r in found.items():
    if len(r["sources"]) > 1:     # ambiguous at source: drop, as before
        continue
    src = next(iter(r["sources"]))
    items.append({"key": key, "label": SOURCES.index(src), "mood": src})

cache, _ = load_cache()
items = [i for i in items if i["key"] in cache]
print("songs per source category:", dict(collections.Counter(i["mood"] for i in items)))

import random
rng = random.Random(42)
by = collections.defaultdict(list)
for i in items: by[i["mood"]].append(i)
train, val = [], []
for mood, g in by.items():
    g = sorted(g, key=lambda i: i["key"]); rng.shuffle(g)
    take = max(1, round(len(g) * 0.2)); val += g[:take]; train += g[take:]

def frames(subset):
    x = np.concatenate([cache[i["key"]] for i in subset])
    y = np.concatenate([np.full(len(cache[i["key"]]), i["label"]) for i in subset])
    return x, y

xtr, ytr = frames(train); xva, yva = frames(val)
counts = np.bincount(ytr, minlength=5)
weights = {i: len(ytr)/(5*c) for i, c in enumerate(counts) if c}

model = build(xtr.shape[1], 5)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
model.fit(xtr, ytr, validation_data=(xva, yva), epochs=150, batch_size=256,
          class_weight=weights, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping("val_accuracy", patience=30,
                                                      restore_best_weights=True)])

correct = 0; per = {s: [0, 0] for s in SOURCES}
conf = np.zeros((5, 5), int)
for i in val:
    p = int(model.predict(cache[i["key"]], verbose=0).mean(0).argmax())
    conf[i["label"]][p] += 1; per[i["mood"]][1] += 1
    if p == i["label"]: correct += 1; per[i["mood"]][0] += 1
n = len(val); lo, hi = wilson(correct, n)
print(f"\nFIVE-CLASS per song: {correct}/{n} = {correct/n:.1%}  95% CI {lo:.0%}-{hi:.0%}  (chance 20%)")
for s, (c, t) in per.items():
    if t: print(f"   {s:9} {c}/{t} = {c/t:5.1%}")
print("\n   confusion (rows true, cols predicted):")
print("     " + "".join(f"{s[:8]:>10}" for s in SOURCES))
for i, s in enumerate(SOURCES):
    print(f"     {s[:8]:<8}" + "".join(f"{v:>10}" for v in conf[i]))
