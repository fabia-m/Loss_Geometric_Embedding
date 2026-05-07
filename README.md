# Loss_Geometric_Embedding_of_Ontologies

This project experiments with learning geometric box embeddings for OWL ontologies. Classes are represented as axis-aligned boxes, and ontology relationships such as subclass and disjointness constraints are encoded as geometric losses.

The project includes utilities for loading OWL files, training box embeddings, sweeping embedding dimensions, evaluating constraint violations, and visualising results.

## Project Structure
```
text
PythonProject/
├── data/
│   └── OWL ontology files
├── functions.py
├── penalty_model.ipynb
└── README.md
```
## Features

- Load OWL ontologies and extract:
  - named classes
  - subclass relationships
  - equivalent-class relationships
  - disjointness axioms
  - all-disjoint-class collections

- Learn box embeddings for ontology classes.

- Support training with:
  - standard optimisation
  - curriculum-based optimisation
  - noisy ontology variants

- Evaluate learned embeddings using:
  - subclass violation counts
  - disjointness violation counts
  - average box size
  - sibling distance
  - final loss

- Run dimension sweeps across multiple embedding sizes.

- Save and reload trained sweep results.

- Plot and compare evaluation results.

## Installation

Create and activate a Python virtual environment, then install the required packages:
```
bash
pip install --upgrade pip
pip install rdflib pykeen pandas owlready2 torch owlrl matplotlib numpy seaborn
```
## Data

Place OWL ontology files in the `data/` directory.

Example expected paths:
```
text
data/oeo-full.owl
data/go-plus.owl
```
You can use any OWL ontology file, as long as it is readable by `rdflib`.

## Basic Usage

Open `penalty_model.ipynb` and run the setup cells, or use the helper functions directly from Python.

Example:
```
python
from functions import *

OWL_PATH = "data/oeo-full.owl"

cfg = BoxConfig(
    dim=6,
    steps=10000,
    seed=42,
    size_weight=0.1,
    subclass_weight=2.0,
    disjoint_weight=1.0,
    big_box_weight=0.1,
    depth_scale=0.5,
    distance_weight=0.1,
)

model, df, edges, final_loss = learn_boxes_from_owl(
    owl_path=OWL_PATH,
    cfg=cfg,
)
```
The returned values are:

- `model`: trained box embedding model
- `df`: dataframe containing class URIs and learned box bounds
- `edges`: ontology edge helper object with evaluation utilities
- `final_loss`: final optimisation loss

## Curriculum Training

Curriculum training allows different loss terms to start at different stages of optimisation.
```
python
from functions import *

OWL_PATH = "data/oeo-full.owl"

cfg = BoxConfig(
    dim=6,
    steps=10000,
    seed=42,
)

schedule = CurriculumSchedule(
    subclass_start=0.0,
    disjoint_start=0.0,
    sibling_start=0.7,
    big_box_start=0.7,
    ramp=False,
)

model, df, edges, final_loss = learn_boxes_with_curriculum(
    owl_path=OWL_PATH,
    cfg=cfg,
    schedule=schedule,
)
```
## Dimension Sweeps

You can train models across several embedding dimensions and compare their performance.
```
python
from functions import *

OWL_PATH = "data/oeo-full.owl"

cfg = BoxConfig(
    dim=6,
    steps=10000,
    seed=42,
)

results = sweep_dimensions(
    owl_path=OWL_PATH,
    learn_fn=learn_boxes_from_owl,
    dims=range(2, 11),
    cfg=cfg,
    path="saved/plain_oeo",
)
```
Each dimension result contains:

- trained model
- learned box dataframe
- ontology edge object
- subclass violations
- disjoint violations
- average sibling distance
- final loss

## Loading Saved Results

Saved sweep results can be loaded later for evaluation or plotting.
```
python
from functions import *

OWL_PATH = "data/oeo-full.owl"

classes, subclass_of, disjoint_pairs = load_owl(OWL_PATH)

results = load_sweep_results(
    path="saved/plain_oeo",
    classes=classes,
    subclass_of=subclass_of,
    disjoint_pairs=disjoint_pairs,
)
```
If results were trained with injected ontology noise, load the ontology using the noisy loader before reloading the saved results:
```
python
classes, subclass_of, disjoint_pairs = load_owl_with_errors(OWL_PATH)

results = load_sweep_results(
    path="saved/plain_oeo_noise",
    classes=classes,
    subclass_of=subclass_of,
    disjoint_pairs=disjoint_pairs,
)
```
## Evaluation

Evaluate a sweep result dictionary:
```
python
eval_df = evaluate_models(results)
print(eval_df)
```
Plot evaluation metrics:
```
python
plot_evaluation(
    eval_df,
    title="Ontology Box Evaluation",
    save_path="evaluation.pdf",
    fmt="pdf",
)
```
Compare plain and curriculum-trained models:
```
python
plot_sweep_comparison(
    results_plain,
    results_curriculum,
    onto="OEO",
)

table_sweep_comparison(
    results_plain,
    results_curriculum,
    onto="OEO",
)
```
## Noisy Ontology Experiments

The project supports injecting controlled noise into an ontology before training.

Noise options include:

- dropping subclass axioms
- flipping subclass directions
- injecting disjointness axioms
- dropping classes

Example:
```
python
classes, subclass_of, disjoint_pairs = load_owl_with_errors(
    owl_path="data/oeo-full.owl",
    subclass_drop_rate=0.1,
    subclass_flip_rate=0.0,
    disjoint_inject_rate=0.05,
    class_drop_rate=0.0,
    seed=40,
)
```
You can also pass `noise=True` into sweep training:
```
python
results_noise = sweep_dimensions(
    owl_path="data/oeo-full.owl",
    learn_fn=learn_boxes_from_owl,
    dims=range(2, 11),
    cfg=cfg,
    noise=True,
    path="saved/plain_oeo_noise",
)
```
## Outputs

Depending on the functions used, the project may generate:

- trained PyTorch model checkpoints
- pickled sweep metadata
- evaluation plots
- comparison plots
- tabular metric summaries

Saved sweep directories follow this structure:
```
text
saved/
└── plain_oeo/
    ├── dim_2/
    │   ├── model.pt
    │   └── data.pkl
    ├── dim_3/
    │   ├── model.pt
    │   └── data.pkl
    └── ...
```
## Notes

- Training can be computationally expensive for large ontologies.
- The code automatically uses Apple Metal Performance Shaders (`mps`) when available; otherwise it falls back to CPU.
- For CPU training, dimension sweeps may use multiprocessing.
- Saved noisy results should be reloaded with the same ontology/noise setup used during training.
- Higher embedding dimensions may reduce constraint violations but can increase training time.

## License

This project is licensed under the MIT License.

```
