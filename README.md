# AdelaiDepth
AdelaiDepth is an open source toolbox for monocular depth prediction. All works from our group are open-sourced here.

AdelaiDepth contains the following algorithms:
* LeReS: [CODE](LeReS), [Learning to Recover 3D Scene Shape from a Single Image](https://arxiv.org/abs/2012.09365)
* DiverseDepth: [CODE](DiverseDepth), [DiverseDepth: Affine-invariant Depth Prediction Using Diverse Data](https://arxiv.org/abs/2002.00569)
* Virtual Normal: [CODE](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction),  [Enforcing geometric constraints of virtual normal for depth prediction](https://arxiv.org/abs/1907.12209)


## Results:
* LeReS
  
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

@inproceedings{Wei2021CVPR,
  title     =  {Learning to Recover 3D Scene Shape from a Single Image},
  author    =  {Wei Yin and Jianming Zhang and Oliver Wang and Simon Niklaus and Long Mai and Simon Chen and Chunhua Shen},
  booktitle =  {Proc. IEEE Conf. Comp. Vis. Patt. Recogn. (CVPR)},
  year      =  {2021}
}
```

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).