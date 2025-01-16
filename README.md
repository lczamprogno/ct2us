# ct2us
Tool intended to automate the generation of simulated ultrasound image and label map pairs from ct images (.nii/.nii.gz).

## WIP:
- there will be a visualizer for the end (slices) and intermediary (segmentation) results 
- progress bar will be improved
- code for two alternate optimized segmentation pipelines is still being developed
    - one focusing on avoiding internal totalsegmentator steps being saved to memory
    - another further optimizes by properly using gpu and cpu acceleration. 

## Notebooks

| Notebook      | Link |
| ----------- | ----------- |
| 0 | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lczamprogno/ct2us/blob/main/CT2US.ipynb)|
