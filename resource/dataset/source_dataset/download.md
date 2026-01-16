## Dataset

The dataset used in this study is derived from the **BigVul** dataset released by Fan et al.
We focus on the following three fields to conduct our experiments:

- `func_before` (string): the original function written in C/C++.
- `target` (int): the function-level label indicating whether the function is vulnerable.
- `flaw_line_index` (string): the index of the labeled vulnerable statement.

The training, validation, and test splits can be downloaded from the following links:

- Training set: https://drive.google.com/uc?id=1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw
- Validation set: https://drive.google.com/uc?id=1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ 
- Test set:  https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V

The complete dataset without splitting is available at:

- https://drive.google.com/uc?id=1WqvMoALIbL3V1KNQpGvvTIuc3TL5v5Q8
