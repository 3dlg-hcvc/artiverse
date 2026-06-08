# Artiverse: A Diverse and Physically Grounded Dataset for Articulated 3D Objects

[Denys Iliash](https://diliash.github.io)<sup>1</sup>,
[Jiayi Liu](https://sevenljy.github.io/)<sup>1</sup>,
[Egor Fokin](https://www.linkedin.com/in/edfokin/)<sup>1</sup>,
[Qirui Wu](https://qiruiw.github.io/)<sup>1</sup>,
[Ali Mahdavi-Amiri](https://www.sfu.ca/~amahdavi/)<sup>1</sup>,
[Manolis Savva](https://msavva.github.io/)<sup>1</sup>,
[Angel X. Chang](https://angelxuanchang.github.io)<sup>1,2</sup>

<sup>1</sup>Simon Fraser University, <sup>2</sup>Canada-CIFAR AI Chair, Amii

### [Project Page](https://3dlg-hcvc.github.io/artiverse/) | [Dataset](https://huggingface.co/datasets/3dlg-hcvc/artiverse) | [Data Viewer](https://aspis.cmpt.sfu.ca/artiverse_viewer/#/)

We present **Artiverse**, a diverse and physically grounded dataset of high-quality articulated 3D objects designed for realistic functional modeling and simulation.

Artiverse contains **5.4K** human-authored objects across **88 categories**, aggregated from multiple 3D static repositories. Objects are annotated with functional parts, interior structures, realistic kinematic relationships and articulated joints (including multi-DoF joints), and physical attributes such as metric scale, material, and mass.

<img src="docs/visuals/teaser_v3.png" alt="Artiverse teaser"/>

## Data

The Artiverse dataset is available on [Hugging Face](https://huggingface.co/datasets/3dlg-hcvc/artiverse).

Please follow the dataset card for access and terms. After approval, authenticate with the Hugging Face CLI and download with Git LFS:

```bash
git lfs install
git clone https://huggingface.co/datasets/3dlg-hcvc/artiverse
```

## TODO

- [3.5k/5.4k] Release data.
- [] Release cleanup pass.
- [] Release functional USD.
- [X] Deploy data viewer.
- [ ] Release pipeline code.

## Acknowledgements

This work was funded in part by a CIFAR AI Chair, a Canada Research Chair, and NSERC Discovery grants, and supported by the Digital Research Alliance of Canada. We thank our annotators for their dedication in ensuring the data quality.

## Citation

Please cite our work if you use Artiverse.

```bibtex
@inproceedings{iliash2026artiverse,
  title={Artiverse: A Diverse and Physically Grounded Dataset for Articulated Objects},
  author={Iliash, Denys and Liu, Jiayi and Fokin, Egor and Wu, Qirui and Mahdavi-Amiri, Ali and Savva, Manolis and Chang, Angel X.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
