# SSG-Vul: Structural-Sequential Graph for Line-Level Code Vulnerability Detection

SSG-Vul is a structural-sequential graph learning framework for software vulnerability detection.  
It supports both **function-level vulnerability classification** and **line-level vulnerability localization**.

The model constructs a lightweight line-level graph from source code and combines sequential and structural information for vulnerability detection.

---

## Repository Structure

```text
SSG-Vul/
├── Entry/
│   ├── util.py
│   │   # Data preprocessing, line segmentation, graph construction,
│   │   # valid-line mask generation, and utility functions.
│   │
│   └── line_vul_graph.py
│       # Main training and evaluation script.
│
├── Models/
│   └── line_vul_graph.py
│       # Structural-sequential graph model.
│       # Includes GCN/GGNN branch, BiGRU branch, gated fusion,
│       # function-level head, and line-level head.
│
├── resource/
│   └── dataset/
│       └── big-vul/
│           ├── train.csv
│           ├── valid.csv
│           └── test.csv
│
└── README.md
```

---

## Dataset Format

### Big-Vul

For Big-Vul, each CSV file should contain function-level labels and, when available, line-level vulnerability labels.

Expected columns:

```text
func_before
target
flaw_line_index
flaw_line
```

Column descriptions:

- `func_before`: source code of the function.
- `target`: function-level vulnerability label.
  - `1`: vulnerable
  - `0`: non-vulnerable
- `flaw_line_index`: vulnerable line indices.
- `flaw_line`: vulnerable source-code lines.

---

## Training

Run the following command to train SSG-Vul:

```bash
python -m Entry.line_vul_graph
```


## License

This repository is released for academic research purposes only.
