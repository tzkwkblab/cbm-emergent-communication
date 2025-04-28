Concept Bottleneck Model with Emergent Communication Framework
===============================================================

This repository provides code for training a Concept Bottleneck Model integrated with an emergent communication framework using reinforcement learning (PPO).
The project is associated with the research paper:  
**"Concept Bottleneck Model with Emergent Communication Framework for Explainable AI"**  
by **Farnoosh Javar** and **Kei Wakabayashi**,  
presented at **The 3rd World Conference on eXplainable Artificial Intelligence (XAI-2025)**.


Installation
------------

Install the required packages:

    pip install -r requirements.txt

Usage
-----

To start training, run:

    python main.py

Paths to datasets and training parameters can be adjusted in `src/config.py`.

The training script expects:
- Pre-extracted feature files (`.npz`) for train/val/test sets
- A HOC annotation CSV file

If needed, preprocessing scripts are provided to generate these files.

Project Structure
-----------------
    main.py
    requirements.txt
    LICENSE         # <-- MIT License for code
    src/
      ├── config.py
      ├── models.py
      ├── environment.py
      ├── utils.py
      ├── train.py
      └── extract_resnet_features.py
    Data/
      ├── Generate_Subset.py
      ├── HOC_annotations.csv
      ├── HOC_list.txt              
      ├── HOC_list.txt
      ├── LICENSE    # <-- CC BY-SA 4.0 License for dataset
      └── README.md # <-- dataset-specific README
License
-------

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.