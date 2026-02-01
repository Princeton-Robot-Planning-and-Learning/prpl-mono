import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
import os
import urllib.request
import zipfile
import tempfile

# Download and register Inter font
def setup_inter_font():
    font_dir = os.path.expanduser("~/.local/share/fonts")
    os.makedirs(font_dir, exist_ok=True)
    inter_path = os.path.join(font_dir, "Inter-Regular.ttf")

    if not os.path.exists(inter_path):
        print("Downloading Inter font...")
        url = "https://github.com/rsms/inter/releases/download/v4.0/Inter-4.0.zip"
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "inter.zip")
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                for name in z.namelist():
                    if name.endswith("Inter-Regular.ttf"):
                        z.extract(name, tmpdir)
                        extracted = os.path.join(tmpdir, name)
                        os.rename(extracted, inter_path)
                        break

    fm.fontManager.addfont(inter_path)
    plt.rcParams['font.family'] = 'Inter'

setup_inter_font()

# 1. Setup the data
data = """Description    Category    Core Challenges
ClutteredRetrieval2D    Kinematic2D    Combinatorial Geometric Constraints
Motion2D    Kinematic2D    Basic Spatial Relations
Obstruction2D    Kinematic2D    Combinatorial Geometric Constraints
PushPullHook2D    Kinematic2D    Tool Use
ClutteredStorage2D    Kinematic2D    Combinatorial Geometric Constraints
StickButton2D    Kinematic2D    Tool Use
DynObstruction2D    Dynamic2D    Combinatorial Geometric Constraints, Nonprehensile Multi-Object Manipulation
DynPushPullHook2D    Dynamic2D    Nonprehensile Multi-Object Manipulation, Tool Use
DynPushT2D    Dynamic2D    Basic Spatial Relations
DynScoopPour2D    Dynamic2D    Nonprehensile Multi-Object Manipulation, Tool Use, Dynamic Constraints
Obstruction3D    Kinematic3D    Combinatorial Geometric Constraints
Packing3D    Kinematic3D    Combinatorial Geometric Constraints
Table3D    Kinematic3D    Basic Spatial Relations
Transport3D    Kinematic3D    Combinatorial Geometric Constraints, Tool Use
BaseMotion3D    Kinematic3D    Basic Spatial Relations
ConstrainedCupboard3D    Dynamic3D    Combinatorial Geometric Constraints
Shelf3D    Dynamic3D    Basic Spatial Relations, Combinatorial Geometric Constraints
SortClutteredBlocks3D    Dynamic3D    Basic Spatial Relations
Rearrange3D    Dynamic3D    Basic Spatial Relations
SweepSimple3D    Dynamic3D    Tool Use, Nonprehensile Multi-Object Manipulation
Dynamo3D    Dynamic3D    Nonprehensile Multi-Object Manipulation
Tossing3D    Dynamic3D    Dynamic Constraints
ScoopPour3D    Dynamic3D    Dynamic Constraints, Tool Use, Nonprehensile Multi-Object Manipulation
BalanceBeam3D    Dynamic3D    Tool Use, Dynamic Constraints
SweepIntoDrawer3D    Dynamic3D    Dynamic Constraints, Tool Use, Nonprehensile Multi-Object Manipulation"""

# 2. Parse and Process Data
# Read data
df = pd.read_csv(io.StringIO(data), sep=r'\s{2,}', engine='python')

# Explode the 'Core Challenges' so we have one row per challenge
df['Core Challenges'] = df['Core Challenges'].str.split(', ')
df_exploded = df.explode('Core Challenges')
df_exploded['Core Challenges'] = df_exploded['Core Challenges'].str.strip()

# Create a Binary Pivot Table
# Index (Rows) = Challenges
# Columns = Tasks (Descriptions)
pivot_df = pd.crosstab(df_exploded['Core Challenges'], df_exploded['Description'])

# 3. Sort Columns by Category to keep the groups together
# We create a lookup dictionary for Task -> Category
task_to_cat = dict(zip(df['Description'], df['Category']))

# Get list of tasks sorted by Category first, then by Name
category_order = ["Kinematic2D", "Dynamic2D", "Kinematic3D", "Dynamic3D"]
sorted_tasks = sorted(pivot_df.columns, key=lambda x: (category_order.index(task_to_cat[x]), x))

# Reorder the pivot table columns
pivot_df = pivot_df[sorted_tasks]

# Reorder rows
row_order = [
    "Basic Spatial Relations",
    "Nonprehensile Multi-Object Manipulation",
    "Tool Use",
    "Combinatorial Geometric Constraints",
    "Dynamic Constraints",
]
pivot_df = pivot_df.reindex(row_order)

# Rename row labels with line breaks
row_labels = {
    "Basic Spatial Relations": "Basic Spatial\nRelations",
    "Nonprehensile Multi-Object Manipulation": "Nonprehensile\nMulti-Object",
    "Tool Use": "Tool Use",
    "Combinatorial Geometric Constraints": "Combinatorial\nGeometric",
    "Dynamic Constraints": "Dynamic\nConstraints",
}
pivot_df = pivot_df.rename(index=row_labels)

# 4. Create Category Colors for the Top Bar
# Generate a color palette for the unique categories
unique_categories = ["Kinematic2D", "Dynamic2D", "Kinematic3D", "Dynamic3D"]
palette = sns.color_palette("Set2", len(unique_categories))
cat_color_map = dict(zip(unique_categories, palette))

# Map the sorted columns (tasks) to their category colors
col_colors = pivot_df.columns.map(lambda x: cat_color_map[task_to_cat[x]])

# 5. Generate the Plot
# We use clustermap because it easily handles the "row colors" (or col_colors in this case)
# row_cluster=True: Groups similar challenges together (e.g., Tool Use and Nonprehensile often appear together)
# col_cluster=False: KEEPS your sorting so categories stay grouped
g = sns.clustermap(
    pivot_df,
    figsize=(16, 5),       # Wide aspect ratio
    col_cluster=False,     # Do not shuffle columns (keeps Categories grouped)
    row_cluster=False,     # Keep custom row order
    col_colors=col_colors, # Adds the colored strip at the top
    cmap="Blues",          # Color scheme (White to Blue)
    linewidths=0.5,        # Grid lines
    linecolor='lightgray',
    cbar_pos=None,         # Hide the color bar (since it's just binary 0/1)
    dendrogram_ratio=(0.1, 0.05) # Adjust size of the tree diagram on the left
)

# 6. Formatting
# Hide the row dendrogram (tree on the left)
g.ax_row_dendrogram.set_visible(False)

# Rotate the task names on the bottom X-axis
from matplotlib.transforms import ScaledTranslation
dx, dy = 10/72., 0  # 10 points to the right
offset = ScaledTranslation(dx, dy, g.fig.dpi_scale_trans)
for label in g.ax_heatmap.get_xticklabels():
    label.set_transform(label.get_transform() + offset)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=12)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=13, fontweight='bold')

# Add left padding for the legend
g.fig.subplots_adjust(left=0.08)

# Add a Custom Legend for the Categories
from matplotlib.patches import Patch
handles = [Patch(facecolor=cat_color_map[key], edgecolor='w', label=key) for key in unique_categories]
legend = g.ax_heatmap.legend(
    handles=handles,
    title="Category",
    bbox_to_anchor=(-0.02, 0.5),
    loc='center right',
    borderaxespad=0,
    fontsize=12,
    title_fontsize=13
)

# Cleanup
g.ax_heatmap.set_xlabel("")
g.ax_heatmap.set_ylabel("")

# Save and show plot
plt.savefig("prbench/scripts/core_challenge_coverage.png", dpi=300, bbox_inches='tight')
plt.savefig("prbench/scripts/core_challenge_coverage.pdf", bbox_inches='tight')
plt.show()