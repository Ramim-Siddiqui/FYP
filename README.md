# Detecting Source Code Vulnerabilities in C/C++ using Graph Neural Networks

Final Year Project (FYP). A GraphSAGE classifier, built with PyTorch Geometric, that labels C/C++ functions from the Devign dataset as *vulnerable* or *not vulnerable* using features derived from their libclang abstract syntax trees.

## Abstract

Memory-unsafe languages such as C and C++ remain the source of a large share of exploitable software vulnerabilities, and rule-based static analysers trade recall for precision. This project treats vulnerability detection as graph classification: each function is parsed with libclang into an abstract syntax tree (AST), the tree is linearised into a sequence of node-kind and identifier tokens, and the sequence is turned into a PyTorch Geometric graph with one-hot node features. A three-layer GraphSAGE network with global mean pooling is trained on 24,000 functions from the Devign dataset (FFmpeg and QEMU) in chunks of 3,000 and evaluated on the remaining 3,318 functions. On that held-out slice the model reaches 93.0% accuracy with a macro F1-score of 0.93.

## The problem

- Vulnerabilities such as buffer overflows, use-after-free and integer errors are common in C/C++ and expensive to find by code review.
- Classical static analysis is pattern-driven and either misses novel bugs or floods developers with false positives.
- Real-world labelled data exists: Devign pairs functions from open-source projects with labels derived from vulnerability-fixing commits. The question this project asks is whether a GNN over the syntactic structure of a function can learn to separate vulnerable from non-vulnerable code on that data.

## Approach and pipeline

Everything lives in `final code.ipynb`. The pipeline, as implemented:

```
output.xlsx  (Devign: project, commit_id, target, func)
     |
     |  pandas.read_excel -> rows [start:end], chunks of 3,000 functions
     v
[1] Parse        libclang (clang.cindex) parses each function in isolation
                 (written to temp.c, parsed with -std=c99)
     v
[2] Linearise    pre-order walk of the cursor tree -> one line per node
                 "CursorKind.<KIND> <spelling>" -> whitespace-split into tokens
     v
[3] Encode       node_encoder.pt maps each token to an index (54,823-entry
                 vocabulary); each node becomes a one-hot vector of that size
     v
[4] Build graph  torch_geometric.data.Data: nodes = tokens, bidirectional edges
                 between consecutive tokens, y = target (0/1)
     v
[5] Train        GraphSAGE (3 x SAGEConv) + global mean pool, NLL loss, Adam,
                 batch 16; weights saved to gnn_trained.pth and carried over
                 from chunk to chunk (8 chunks, rows 0-24,000)
     v
[6] Evaluate     rows 24,000-27,318 -> accuracy + classification report
                 (scikit-learn)
```

Notes on the design:

- **Parser.** `clang.cindex` (the Python bindings to libclang). Each function is written to `temp.c` and parsed on its own, without project headers, so unresolved types and identifiers are tolerated through clang's error recovery.
- **Node vocabulary.** `node_encoder.pt` is a plain `{token: index}` dictionary with 54,823 entries: 53 libclang `CursorKind.*` node types plus roughly 54,770 identifier and literal spellings. Tokens not in the vocabulary fall back to index 0. The dictionary was produced by an earlier preprocessing run; the code that built it is left commented out in `load_dataset_chunk`.
- **Graph structure.** Edges connect consecutive tokens of the linearised AST in both directions, so the graph is a chain over the pre-order traversal rather than a tree with parent-child edges. Control-flow edges are not used.
- **Chunked training.** With a 54,823-dimensional one-hot feature per node, holding all 27,318 graphs in memory is impractical, so the dataset is processed in slices of 3,000 rows. Each slice is parsed, converted to graphs and trained on; the checkpoint from the previous slice is loaded first.

## Model architecture

```
Input   one-hot node features, dim = |vocab| = 54,823
        SAGEConv(54,823 -> 256)  -> ReLU -> Dropout(0.3)
        SAGEConv(256 -> 128)     -> ReLU -> Dropout(0.3)
        SAGEConv(128 -> 64)      -> ReLU
        global_mean_pool (graph-level readout)
        Linear(64 -> 2)          -> log_softmax
```

| Setting | Value |
| --- | --- |
| Library | PyTorch + PyTorch Geometric (`SAGEConv`, `global_mean_pool`) |
| Parameters | about 28.2 M, over 99% of them in the first layer because of the one-hot input |
| Loss / optimiser | negative log-likelihood (`F.nll_loss`) / Adam, lr = 1e-3 |
| Batch size | 16 graphs |
| Regularisation | dropout 0.3 after the first two convolutions |
| Early stopping | on *training* loss, patience 2; `max_epochs` defaults to 5 (the first four chunks were run with 10) |
| Device | CPU (`device = torch.device('cpu')` in the notebook) |

## Dataset

The data is the **Devign** dataset (Zhou et al., NeurIPS 2019): C functions from the FFmpeg and QEMU projects, labelled 1 if the function was changed by a vulnerability-fixing commit and 0 otherwise.

- Project page and download: <https://sites.google.com/view/devign>
- Also distributed as the CodeXGLUE *Defect Detection* task: <https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection>
- Paper: Y. Zhou, S. Liu, J. Siow, X. Du, Y. Liu, *Devign: Effective Vulnerability Identification by Learning Comprehensive Semantic Representations from Code*, NeurIPS 2019. <https://arxiv.org/abs/1909.03496>

`output.xlsx` is that dataset as a single Excel sheet with the original columns `project`, `commit_id`, `target`, `func`:

| | Count |
| --- | ---: |
| Functions | 27,318 |
| QEMU / FFmpeg | 17,549 / 9,769 |
| Not vulnerable (0) / vulnerable (1) | 14,858 / 12,460 |
| Training rows (0 to 23,999, eight chunks of 3,000) | 24,000 |
| Test rows (24,000 to 27,317) | 3,318 (2,083 QEMU, 1,235 FFmpeg) |

The split is positional (first 24,000 rows train, remaining rows test); it is not the official CodeXGLUE train/valid/test split, so the results below are not directly comparable with published Devign numbers.

## Results

Numbers below are copied from the saved notebook outputs.

**Held-out test slice (3,318 functions): accuracy 0.9301.**

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Not Vulnerable | 0.96 | 0.90 | 0.93 | 1,759 |
| Vulnerable | 0.90 | 0.96 | 0.93 | 1,559 |
| Macro average | 0.93 | 0.93 | 0.93 | 3,318 |
| Weighted average | 0.93 | 0.93 | 0.93 | 3,318 |

Training loss and accuracy at the last epoch of each chunk (training set only, no validation split):

| Chunk (rows) | Epochs | Final loss | Final train acc. |
| --- | ---: | ---: | ---: |
| 0 - 3,000 | 10 | 0.0399 | 0.9787 |
| 3,000 - 6,000 | 10 | 0.0622 | 0.9740 |
| 6,000 - 9,000 | 10 | 0.0588 | 0.9697 |
| 9,000 - 12,000 | 10 | 0.0462 | 0.9763 |
| 12,000 - 15,000 | 5 | 0.2702 | 0.8957 |
| 15,000 - 18,000 | 5 | 0.1509 | 0.9450 |
| 18,000 - 21,000 | 5 | 0.1428 | 0.9463 |
| 21,000 - 24,000 | 5 | 0.1292 | 0.9540 |

Each new chunk starts with loss around 0.8 and accuracy around 0.53 before recovering, which indicates the network adapts to the vocabulary of each slice rather than transferring cleanly between slices. See the notebook for the full epoch-by-epoch logs.

## Repository contents

| File | Purpose |
| --- | --- |
| `final code.ipynb` | The complete pipeline: parsing, graph construction, the `GNN` model, chunked training, and evaluation. Outputs (training logs, classification report) are saved in the notebook. |
| `output.xlsx` | The Devign dataset as an Excel sheet (27,318 functions, 15 MB). Read with `pandas.read_excel` in slices. |
| `node_encoder.pt` | The token-to-index vocabulary (54,823 entries) saved with `torch.save`. Required to reproduce the exact feature space and to load the trained model; loaded with `torch.load`. |
| `temp.c` | Scratch file that `generate_ast_clang()` overwrites before every libclang parse. After a run it simply holds the last function parsed (`rv34_pred_mv` from FFmpeg). It is not an input and carries no information of its own. |
| `requirements.txt` | Python dependencies actually imported by the notebook. |
| `gnn_trained.pth` | *Generated, not committed.* Model weights saved after each chunk (about 113 MB, above GitHub's file-size limit). Produced by running the notebook. |

## How to run

Tested with Python 3.11 (the notebook was last executed with 3.11.3).

1. **Install LLVM/Clang.** The notebook points libclang at a Windows install:

   ```python
   cindex.Config.set_library_file(r"C:\Program Files\LLVM\bin\libclang.dll")
   ```

   Change the path for your machine (for example `/usr/lib/llvm-14/lib/libclang.so` on Debian/Ubuntu, or the Homebrew LLVM path on macOS). If you install the `libclang` wheel from PyPI (listed in `requirements.txt`) it bundles the shared library, and this line can be removed.

2. **Create an environment and install dependencies.**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # Linux / macOS
   pip install -r requirements.txt
   ```

   PyTorch and PyTorch Geometric wheels are platform specific; if the default install fails, follow the selectors at <https://pytorch.org/get-started/locally/> and <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>.

3. **Open the notebook and run it top to bottom.**

   ```bash
   jupyter notebook "final code.ipynb"
   ```

   The first cell defines the pipeline. The next eight cells each parse one 3,000-row chunk, load `gnn_trained.pth` from the previous chunk, train, and save again. The last two cells build the test graphs from rows 24,000 to 27,318 and print the classification report. `output.xlsx` and `node_encoder.pt` must be in the working directory.

4. **GPU.** The notebook pins `device = torch.device('cpu')`. To use a GPU, change it to

   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

   and install a CUDA build of PyTorch. Note that the one-hot input makes the first layer large (54,823 x 256 weights, twice over inside `SAGEConv`), so keep the chunk size at 3,000 unless you have memory to spare.

5. **Runtime.** On CPU, parsing one chunk takes about 1 to 2 minutes and graph construction about 1 minute (from the progress bars in the notebook); training time per chunk is not logged.

## Limitations and future work

- **Chain graphs instead of trees.** Edges join consecutive tokens in the linearised AST, so the GNN sees sequence structure rather than genuine parent-child relations. Adding real AST edges from the cursor tree, and control- and data-flow edges (for example from a Joern code property graph, as in the Devign paper), is the most direct next step.
- **One-hot identifiers.** The 54,823-token vocabulary is dominated by identifier and literal spellings, which inflates the first layer to roughly 28 M parameters and gives unseen identifiers no representation (they map to index 0). Learned embeddings (`nn.Embedding`), subword tokenisation, or restricting node features to node kinds plus type information would shrink the model and generalise better.
- **No validation set, positional split.** Early stopping monitors training loss, the vocabulary was built in an earlier run rather than strictly on the training rows, and the split is positional rather than the official CodeXGLUE split. Devign also contains the same function at several commits, so near-duplicates can appear on both sides of the split. The 93% figure should therefore be read as an in-distribution estimate; a cross-project evaluation (train on FFmpeg, test on QEMU) would be a sterner test.
- **Sequential chunk training.** Each chunk is seen once, in order, with the previous checkpoint as the starting point. The loss resets to roughly 0.8 at the start of every chunk, which suggests forgetting across chunks. Streaming the graphs from disk and shuffling across the whole training set would address this.
- **Parsing without context.** Functions are parsed in isolation with `-std=c99`, so macros, project types and called functions are unresolved; some identifiers are dropped from the AST as a result.
- **Excel as a data format.** The sheet is re-read for every chunk, and 66 functions are cut at Excel's 32,767-character cell limit. A CSV/Parquet file or the original Devign JSON would avoid both issues.
- **Evaluation depth.** Only accuracy, precision, recall and F1 are reported. ROC-AUC, calibration, per-project breakdowns, and explanation methods (for example GNNExplainer) that point to the statements driving a prediction would make the results more useful to developers.

## Citation

If you use the dataset, please cite the Devign paper:

```bibtex
@inproceedings{zhou2019devign,
  title     = {Devign: Effective Vulnerability Identification by Learning Comprehensive Semantic Representations from Code},
  author    = {Zhou, Yaqin and Liu, Shangqing and Siow, Jingkai and Du, Xiaoning and Liu, Yang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019}
}
```
