import re
import matplotlib.pyplot as plt

log_path = "results/001-DiT-S-4/log.txt"   # path to your log file

# Regex to catch common loss patterns
# Matches things like: loss=0.123, loss: 0.123, train_loss 0.123
loss_pattern = re.compile(
    r"(?:train[_ ]?loss|loss)\s*[:=]\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE
)

losses = []

with open(log_path, "r") as f:
    for line in f:
        match = loss_pattern.search(line)
        if match:
            losses.append(float(match.group(1)))

if not losses:
    raise ValueError("No loss values found. Check the regex or log format.")

losses = losses[int(0.2*len(losses)):]

plt.figure()
plt.plot(losses)
plt.xlabel("Training step")
plt.ylabel("Train loss")
plt.yscale("log")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.savefig(f"train_loss.png")

