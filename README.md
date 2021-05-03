# AdelaiDepth
AdelaiDepth is an open source toolbox for monocular depth prediction. Relevant work from our group is open-sourced here.

AdelaiDepth contains the following algorithms:
* 3DSceneShape: [Code](3DSceneShape), [Learning to Recover 3D Scene Shape from a Single Image](https://arxiv.org/abs/2012.09365)
* DiverseDepth: [Code](https://github.com/YvanYin/DiverseDepth), [DiverseDepth: Affine-invariant Depth Prediction Using Diverse Data](https://arxiv.org/abs/2002.00569)
* Virtual Normal: [Code](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction),  [Enforcing geometric constraints of virtual normal for depth prediction](https://arxiv.org/abs/1907.12209)


## Results:
* 3D Scene Shape
  
You may want to check [this video](http://www.youtube.com/watch?v=UuT5_GK_TWk) which provides a very brief introduction to the work:

<table>
  <tr>
    <td>RGB</td>
     <td>Depth</td>
     <td>Point Cloud</td>
  </tr>
  <tr>
    <td><img src="examples/2-rgb.jpg" width=400 height=300></td>  
    <td><img src="examples/2.jpg" width=400 height=300></td>
    <td><img src="examples/2.gif" width=400 height=300></td>
  </tr>
 </table>

![Depth](./examples/depth.png)

## BibTeX

```BibTeX
@inproceedings{Yin2019enforcing,
  title={Enforcing geometric constraints of virtual normal for depth prediction},
  author={Yin, Wei and Liu, Yifan and Shen, Chunhua and Yan, Youliang},
  booktitle= {The IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}

@article{yin2020diversedepth,
  title={Diversedepth: Affine-invariant depth prediction using diverse data},
  author={Yin, Wei and Wang, Xinlong and Shen, Chunhua and Liu, Yifan and Tian, Zhi and Xu, Songcen and Sun, Changming and Renyin, Dou},
  journal={arXiv preprint arXiv:2002.00569},
  year={2020}
}

@inproceedings{Wei2021CVPR,
  title     =  {Learning to Recover 3D Scene Shape from a Single Image},
  author    =  {Wei Yin and Jianming Zhang and Oliver Wang and Simon Niklaus and Long Mai and Simon Chen and Chunhua Shen},
  booktitle =  {Proc. IEEE Conf. Comp. Vis. Patt. Recogn. (CVPR)},
  year      =  {2021}
}
```

## License

The *3D Scene Shape* code is under a non-commercial license from *Adobe Research*. See the [LICENSE file](LeReS/LICENSE) for details.

Other depth prediction projects are licensed under the 2-clause BSD License - see the [LICENSE file](LICENSE) for details. For commercial use, please contact [Chunhua Shen](https://git.io/shen).
