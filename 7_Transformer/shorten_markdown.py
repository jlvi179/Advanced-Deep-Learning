import json

with open("Submission7.ipynb", "r") as f:
    nb = json.load(f)

# Ensure the last cell is the markdown cell we want to shorten
if nb["cells"][-1]["cell_type"] == "markdown" and "**Evaluation Plots" in nb["cells"][-1]["source"][0]:
    short_explanation = """**Evaluation Plots:**
- **Learning Curves:** Decreasing MSE shows the model is learning successfully.
- **True vs Predicted:** Points along the diagonal indicate accurate predictions.
- **Resolution Distribution:** Peak near zero confirms very small localization errors."""
    
    nb["cells"][-1]["source"] = [line + "\n" for line in short_explanation.split("\n")]
    nb["cells"][-1]["source"][-1] = nb["cells"][-1]["source"][-1][:-1]

with open("Submission7.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Shortened explanation cell.")
