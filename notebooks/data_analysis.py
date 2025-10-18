# %%
# Check if cuda is available in tensorflow.
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# %%
print("Num GPUs Available: ", tf.config.list_physical_devices("GPU"))

# %%
plant_village_data, info = tfds.load("plant_village", with_info=True, as_supervised=True)
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
ax.bar(x - width/2, healthy_counts, width, label="healthy", color="green")
ax.bar(x + width/2, diseased_counts, width, label="diseased", color="red")

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
total_all = total_healthy + total_diseased if (total_healthy + total_diseased) > 0 else 1
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
