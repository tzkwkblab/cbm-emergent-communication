## 🧾 Attribution

This work is based on image data from the **Oxford-IIIT Pet Dataset**, created by:
- Omkar M. Parkhi
- Andrea Vedaldi
- Andrew Zisserman
- C. V. Jawahar

Original dataset: [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/)


## 📚 Context

The original Oxford-IIIT Pet Dataset is a 37-category pet dataset with roughly 200 images per class. The images exhibit high variation in scale, pose, and lighting. Each image is annotated with breed labels, head region-of-interest (ROI), and pixel-level trimap segmentation masks.

# Human-Observable Concept Annotations for Oxford-IIIT Pet Subset

This dataset provides manually annotated human-understandable concept labels (HOCs) for a filtered subset of the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). These labels represent visual attributes like fur texture, eye shape, and body type to support research in Explainable AI (XAI), particularly Concept Bottleneck Models.

## 📂 Files Included
- `HOC_list.txt` — List of concepts (HOC1–HOC26) with their descriptions.
- `hoc_annotations.csv` — Binary annotation matrix where rows are image filenames and columns are visual concepts (1 = present, 0 = not present).

## 📊 Subset Details
- **Classes**: Persian, Sphynx, Russian Blue, Ragdoll
- **Images**: 50 per class × 4 classes = 200 total

## ⚖️ License
The original Oxford-IIIT Pet Dataset is released under the [CC BY-SA 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/). These annotations and subsets follow the same license.

## 🔍 Suggested Citation

If you use these annotations in your research, please cite the original Oxford-IIIT Pet paper and link back to this dataset.

---

For questions or collaboration, please contact the dataset contributor.
