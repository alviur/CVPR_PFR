# Continually Learning Self-Supervised Representations with Projected Functional Regularization

This is the official repository for the paper:
> **[Continually Learning Self-Supervised Representations with Projected Functional Regularization](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/html/Gomez-Villa_Continually_Learning_Self-Supervised_Representations_With_Projected_Functional_Regularization_CVPRW_2022_paper.html)**<br>
> [Alex Gomez-Villa](https://scholar.google.com/citations?user=A2dhwNgAAAAJ&hl=en), [Bartlomiej Twardowski](https://scholar.google.com/citations?user=8yywECgAAAAJ&hl), [Lu Yu](https://scholar.google.com/citations?user=8KhrWbYAAAAJ&hl=en&authuser=1), [Andrew D. Bagdanov](https://scholar.google.com/citations?user=_Fk4YUcAAAAJ&hl=en&authuser=1), [Joost van de Weijer](https://scholar.google.com/citations?user=Gsw2iUEAAAAJ&hl)<br>
> **CVPR 2022**

> **Abstract:** *Recent self-supervised learning methods are able to learn high-quality image representations and are closing the gap with supervised approaches. However, these methods are unable to acquire new knowledge incrementally -- they are, in fact, mostly used only as a pre-training phase over IID data. In this work we investigate self-supervised methods in continual learning regimes without any replay mechanism. We show that naive functional regularization, also known as feature distillation, leads to lower plasticity and limits continual learning performance. Instead, we propose Projected Functional Regularization in which a separate temporal projection network ensures that the newly learned feature space preserves information of the previous one, while at the same time allowing for the learning of new features. This prevents forgetting while maintaining the plasticity of the learner. Comparison with other incremental learning approaches applied to self-supervision demonstrates that our method obtains competitive performance in different scenarios and on multiple datasets.*
<br>

<p align="center" float="left">
    <img src="./figs/pfr.jpeg"/ width=40%> 
    
</p>



# Citation
If you like our work, please cite our [paper](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/html/Gomez-Villa_Continually_Learning_Self-Supervised_Representations_With_Projected_Functional_Regularization_CVPRW_2022_paper.html):
```
@inproceedings{gomezvilla2022,
  title={Continually Learning Self-Supervised Representations with Projected Functional Regularization},
  author={Gomez-Villa, Alex and Twardowski, Bartlomiej and Yu, Lu and Bagdanov, Andrew and van de Weijer, Joost},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}