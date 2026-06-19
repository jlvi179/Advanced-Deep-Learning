import json

with open("Submission7.ipynb", "r") as f:
    nb = json.load(f)

explanation = """**Evaluation Plots Explanation:**
- **Learning Curves (Left):** Shows the training and validation Mean Squared Error (MSE) over epochs. A decreasing and stabilizing trend indicates that the model is successfully learning without overfitting.
- **True vs Predicted (Middle):** A scatter plot comparing the model's predicted $(x, y)$ coordinates against the actual ground truth coordinates. Points tightly clustered around the diagonal dashed line indicate highly accurate predictions.
- **Resolution Distribution (Right):** A histogram displaying the Euclidean distance (error) between the predicted and true interaction positions. A strong peak near zero means the majority of predictions are very close to the true locations, demonstrating the Transformer's localization accuracy."""

new_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [line + "\n" for line in explanation.split("\n")]
}
# Remove the trailing newline from the very last line
new_cell["source"][-1] = new_cell["source"][-1][:-1]

nb["cells"].append(new_cell)

with open("Submission7.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Added explanation cell.")
