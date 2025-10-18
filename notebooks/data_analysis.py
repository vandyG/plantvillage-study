# %%
# Check if cuda is available in tensorflow.
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict

# %%
print("Num GPUs Available: ", tf.config.list_physical_devices("GPU"))

# %%
plant_village_data, info = tfds.load(
    "plant_village", with_info=True, as_supervised=True
)
print(info)

# %%
# Plot some sample images from `plant_village_data` as subplots in a grid.
# Reduce the size of the label text.
fig = plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(plant_village_data["train"].take(9)):
    ax = fig.add_subplot(3, 3, i + 1)
    ax.imshow(image.numpy())
    ax.set_title(info.features["label"].int2str(label), fontsize=8)
    ax.axis("off")

# Aggregate counts of healthy vs diseased per plant type
train_ds = plant_village_data["train"]
counts = defaultdict(lambda: {"healthy": 0, "diseased": 0})

for _, label in tfds.as_numpy(train_ds):
    # label may be a numpy scalar; ensure it's an int for int2str
    label_str = info.features["label"].int2str(int(label))
    # Split into plant type and disease name
    plant_type, disease_name = label_str.split("___", 1)
    if disease_name.lower() == "healthy":
        counts[plant_type]["healthy"] += 1
    else:
        counts[plant_type]["diseased"] += 1

# Prepare data for plotting
plants = sorted(counts.keys())
healthy_counts = [counts[p]["healthy"] for p in plants]
diseased_counts = [counts[p]["diseased"] for p in plants]

# Plot grouped bar chart
x = np.arange(len(plants))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width / 2, healthy_counts, width, label="healthy", color="green")
ax.bar(x + width / 2, diseased_counts, width, label="diseased", color="red")

ax.set_xticks(x)
ax.set_xticklabels(plants, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Count")
ax.set_title("Healthy vs Diseased counts by plant type")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# New: total healthy vs unhealthy across all plant types
total_healthy = sum(healthy_counts)
total_diseased = sum(diseased_counts)
totals = [total_healthy, total_diseased]
labels = ["Healthy", "Unhealthy"]
colors = ["green", "red"]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, totals, color=colors)
ax.set_ylabel("Total count")
ax.set_title("Total Healthy vs Unhealthy (all plant types)")

# Annotate bars with counts and percentages (inside bars)
total_all = (
    total_healthy + total_diseased if (total_healthy + total_diseased) > 0 else 1
)
for bar, value in zip(bars, totals):
    height = int(value)
    pct = value / total_all * 100
    # Place the label at the center of the bar (vertical center) and center-align
    ax.annotate(
        f"{height}\n{pct:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height / 2),
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        fontweight="bold",
    )

plt.tight_layout()
plt.show()
# %%

# Relabel dataset with binary targets: 0 = healthy, 1 = diseased.
label_names = info.features["label"].names
binary_lookup = np.array(
    [
        0 if name.split("___", 1)[-1].lower() == "healthy" else 1
        for name in label_names
    ],
    dtype=np.int32,
)
binary_lookup_tf = tf.constant(binary_lookup)


def to_binary_label(image, label):
    label = tf.cast(label, tf.int32)
    binary_label = tf.gather(binary_lookup_tf, label)
    return image, binary_label


binary_plant_village_data = {
    split: ds.map(to_binary_label, num_parallel_calls=tf.data.AUTOTUNE)
    for split, ds in plant_village_data.items()
}
binary_label_names = {0: "healthy", 1: "diseased"}

# Preview a few binary-labeled examples to confirm mapping.
for image, label in binary_plant_village_data["train"].take(3):
    print("Binary label:", int(label.numpy()), "->", binary_label_names[int(label.numpy())])

# %%
# Augment healthy and diseased samples to rebalance without cropping.
healthy_label = 0
diseased_label = 1

train_binary_ds = binary_plant_village_data["train"]
healthy_ds = train_binary_ds.filter(lambda _, lbl: tf.equal(lbl, healthy_label))
diseased_ds = train_binary_ds.filter(lambda _, lbl: tf.equal(lbl, diseased_label))


def augment_healthy(image, label):
    image_f = tf.image.convert_image_dtype(image, tf.float32)
    image_f = tf.image.random_flip_left_right(image_f)
    image_f = tf.image.random_flip_up_down(image_f)
    image_f = tf.image.rot90(
        image_f, tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    )
    image_f = tf.image.random_saturation(image_f, 0.8, 1.25)
    image_f = tf.image.random_hue(image_f, 0.05)
    image_f = tf.image.random_brightness(image_f, 0.12)
    image_f = tf.image.random_contrast(image_f, 0.8, 1.25)
    image_f = tf.clip_by_value(image_f, 0.0, 1.0)
    image_aug = tf.image.convert_image_dtype(image_f, tf.uint8)
    return image_aug, label


# Apply augmentation to only a small fraction of diseased images (10%)
diseased_aug_prob = 0.1  # 10%

def augment_diseased_with_replacement(image, label):
    image_f = tf.image.convert_image_dtype(image, tf.float32)

    def augmented():
        aug = tf.image.random_flip_left_right(image_f)
        aug = tf.image.rot90(
            aug, tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        )
        aug = tf.image.random_contrast(aug, 0.9, 1.1)
        aug = tf.image.random_brightness(aug, 0.08)
        aug = tf.image.random_hue(aug, 0.03)
        aug = tf.clip_by_value(aug, 0.0, 1.0)
        return tf.image.convert_image_dtype(aug, tf.uint8)

    def original():
        return image

    choice = tf.random.uniform([], 0.0, 1.0)
    # augment only when random value < diseased_aug_prob (≈10%); otherwise keep original
    return tf.cond(tf.less(choice, diseased_aug_prob), augmented, original), label


healthy_multiplier = 0
if total_healthy > 0:
    healthy_multiplier = max(1, math.ceil(total_diseased / total_healthy) - 1)

augmented_healthy_datasets = [healthy_ds]
for _ in range(healthy_multiplier):
    augmented_healthy_datasets.append(
        healthy_ds.map(augment_healthy, num_parallel_calls=tf.data.AUTOTUNE)
    )

healthy_augmented_ds = augmented_healthy_datasets[0]
for ds in augmented_healthy_datasets[1:]:
    healthy_augmented_ds = healthy_augmented_ds.concatenate(ds)

healthy_augmented_ds = healthy_augmented_ds.shuffle(4096)

diseased_augmented_ds = diseased_ds.map(
    augment_diseased_with_replacement, num_parallel_calls=tf.data.AUTOTUNE
)

augmented_train_ds = healthy_augmented_ds.concatenate(diseased_augmented_ds)
augmented_train_ds = augmented_train_ds.shuffle(8192).prefetch(tf.data.AUTOTUNE)

binary_plant_village_data["train"] = augmented_train_ds

approx_total_healthy = (
    total_healthy * (healthy_multiplier + 1) if total_healthy > 0 else 0
)
print(
    f"Augmented train dataset prepared. Healthy≈{approx_total_healthy}, "
    f"Diseased≈{total_diseased}."
)
# %%

# Visualize example augmentations (original + deterministic variants)
# Select an example image: prefer a healthy sample, otherwise take any train sample.
example_img = None
for img, lbl in healthy_ds.take(1):
    example_img = img.numpy()
if example_img is None:
    for img, lbl in train_binary_ds.take(1):
        example_img = img.numpy()

if example_img is not None:
    # Convert to tensor float in [0,1] for consistent transforms
    example_tf = tf.image.convert_image_dtype(tf.convert_to_tensor(example_img), tf.float32)

    # Define deterministic augmentations (operate on float images in [0,1])
    aug_fns = [
        ("original", lambda x: x),
        ("flip_lr", lambda x: tf.image.flip_left_right(x)),
        ("flip_ud", lambda x: tf.image.flip_up_down(x)),
        ("rot90", lambda x: tf.image.rot90(x, k=1)),
        ("rot180", lambda x: tf.image.rot90(x, k=2)),
        ("rot270", lambda x: tf.image.rot90(x, k=3)),
        ("sat_low (0.7)", lambda x: tf.image.adjust_saturation(x, 0.7)),
        ("sat_high (1.4)", lambda x: tf.image.adjust_saturation(x, 1.4)),
        ("hue +0.05", lambda x: tf.image.adjust_hue(x, 0.05)),
        ("bright +0.12", lambda x: tf.image.adjust_brightness(x, 0.12)),
        ("bright -0.12", lambda x: tf.image.adjust_brightness(x, -0.12)),
        ("contrast_low (0.8)", lambda x: tf.image.adjust_contrast(x, 0.8)),
        ("contrast_high (1.25)", lambda x: tf.image.adjust_contrast(x, 1.25)),
        ("combined aug 1", lambda x: tf.clip_by_value(
            tf.image.adjust_contrast(
                tf.image.adjust_brightness(
                    tf.image.adjust_hue(
                        tf.image.adjust_saturation(tf.image.rot90(tf.image.flip_left_right(x), k=1), 1.1),
                        0.03,
                    ),
                    0.08,
                ),
                1.05,
            ),
            0.0,
            1.0,
        )),
        ("diseased-style aug", lambda x: tf.clip_by_value(
            tf.image.adjust_brightness(
                tf.image.adjust_hue(
                    tf.image.adjust_contrast(tf.image.rot90(tf.image.flip_left_right(x), k=1), 0.95),
                    0.03,
                ),
                0.06,
            ),
            0.0,
            1.0,
        )),
    ]

    # Apply augmentations and convert to uint8 numpy for plotting
    aug_images = []
    for title, fn in aug_fns:
        out = fn(example_tf)
        out = tf.clip_by_value(out, 0.0, 1.0)
        out_uint8 = tf.image.convert_image_dtype(out, tf.uint8).numpy()
        aug_images.append((title, out_uint8))

    # Plot grid
    n = len(aug_images)
    cols = 5
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.axis("off")

    for i, (title, img) in enumerate(aug_images):
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("No example image found for augmentation visualization.")
# %% 
# Plot counts of examples per class in the (augmented) training dataset
from collections import Counter

train_for_counts = binary_plant_village_data["train"]
label_counts = Counter()
# Use tfds.as_numpy to iterate efficiently
for _, lbl in tfds.as_numpy(train_for_counts):
    label_counts[int(lbl)] += 1

healthy_count = label_counts.get(0, 0)
diseased_count = label_counts.get(1, 0)
labels = ["healthy", "diseased"]
counts = [healthy_count, diseased_count]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, counts, color=["green", "red"])
ax.set_ylabel("Count")
ax.set_title("Counts per class in training dataset (after augmentation)")

total = max(1, healthy_count + diseased_count)
for bar, val in zip(bars, counts):
    pct = val / total * 100
    # Place the label centered inside the bar (vertical center) with white text
    x = bar.get_x() + bar.get_width() / 2
    y = val / 2
    ax.annotate(
        f"{val}\n{pct:.1f}%",
        xy=(x, y),
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        fontweight="bold",
    )

plt.tight_layout()
plt.show()

# %%
