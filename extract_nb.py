import json

with open(
    "c:\\Users\\Usuario\\Documents\\Data Science\\Repositorios\\mlektic\\Bocetos_Lib_RLin.ipynb", "r", encoding="utf-8"
) as f:
    nb = json.load(f)

with open(
    "c:\\Users\\Usuario\\Documents\\Data Science\\Repositorios\\mlektic\\src\\mlektic\\core.py", "w", encoding="utf-8"
) as out:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            # filter out the user example parts (like 'np.random.seed(7)')
            if "np.random.seed(" in source or "model.fit(X, y)" in source:
                continue
            if (
                "build_multivar_lr_figure" in source
                or "fit_history" in source
                or "build_plane_lr_figure" in source
                or "build_simple_lr_figure" in source
                or "visualize_lr" in source
            ):
                out.write(source)
                out.write("\n\n")

print("Extraction completed!")
