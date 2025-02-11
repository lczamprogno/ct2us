# ct2us

This tool is intended to automate the generation of simulated ultrasound image and label pairs from ct volumes (.nii/.nii.gz).

---

### Purpose
Intended to be capable of supplementing datasets for ultrasound image labeling.

## Expandability
Image generation process is very dependant on tissue attenuation, so specialized US renderers would be necessary/ideal to expand this tool to work on other body parts. For this purpose, much of the following code has hence been designed with modularity as a core goal, so that new methods can be added/replaced, as for example the segmentation quality or speed could have a significant impact on overall results. 

---

## Current use:
- ![example](../assets/Full%20Demo.gif)
  

## Further goals:
- code for two alternate optimized segmentation pipelines is still being developed
  - one focusing on avoiding internal totalsegmentator steps being saved to memory
  - another further optimizes by properly using gpu and cpu acceleration.

- Throughout the process of acquiring the necessary information for the rendering - and even for visualization - could be a useful resource. With that in mind, enabling saving these intermediary results is a goal, albeit one that has yet to be fully achieved. The download tab is intended to serve the purpose of adjusting saved information. [ ]

- Improved version of the totalsegmentator nnunet is still WIP. Once that is taken care of, pluging this in the pipeline with the stacked assemble should yield a significant speed up. [ ]

- Currently there is some support for cpu, but this script is mainly designed with gpu acceleration in mind. For cpu optimization, the label_map dict needs to be used to gather all the label names which are contained. This list then can be used by just plugging it in the totalsegmentator function as a parameter, and the built in ROI should yield faster results. [ ]

- There seemed to be no significant improvement through using the ROI feature in the gpu version. Here, it is important to note that allocation of gpu memory and copying data over has a significant memory cost, which is actually one of the main improvements of the new totalsegmentator pipeline implemented. [ ]

---

## Link to colab

| Notebook      | Link |
| ----------- | ----------- |
| 0 | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lczamprogno/ct2us/blob/main/CT2US.ipynb)|
