
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# Use Times New Roman for all text in the figure
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']

data = [
    ["1D CNN Global", 0.4032, 0.9590, 0.2505, 0.3975],
    ["LSTM Global", 0.4990, 0.4360, 0.4030, 0.3820],
    ["RF Global", 0.7457, 0.2275, 0.0766, 0.0799],
    ["MLP Per Nurse", 0.5115, 0.5411, 0.8789, 0.5856],
    ["LSTM Per Nurse", 0.4990, 0.4360, 0.4030, 0.3820],
    ["RF Per Nurse", 0.5228, 0.5873, 0.7717, 0.5545],
    ["Idealized RF Per Nurse", 0.8105, 0.8510, 0.8983, 0.8593],
]

columns = ["Model", "Accuracy", "Precision", "Recall", "F1"]

df = pd.DataFrame(data, columns=columns)

# Format numeric columns to 3 decimal places for display
cell_text = df.copy()
for col in columns[1:]:
    cell_text[col] = cell_text[col].map(lambda x: f"{x:.3f}")

# Keep the canvas compact so the saved TIFF does not contain large empty margins
ncols = df.shape[1]
fig, ax = plt.subplots(figsize=(8.5, 2.9))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')

# Title above the table
fig.suptitle("Model Evaluation Metrics", fontname='Times New Roman', fontsize=16, y=0.985)

table = ax.table(
    cellText=cell_text.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    bbox=[0.0, 0.0, 1.0, 0.90],
)

# Let matplotlib auto-adjust each column width to the content (if supported)
try:
    table.auto_set_column_width(col=list(range(ncols)))
except Exception:
    # Fallback: scale the table if the matplotlib version doesn't support auto column widths
    table.scale(1.05, 1.8)

# Set font size to 15 as requested
table.auto_set_font_size(False)
table.set_fontsize(15)

# Slight additional scaling for row height
table.scale(1.04, 1.7)

plt.savefig("results_table.tiff", dpi=300, bbox_inches='tight', pad_inches=0.0)