#Description

These are only experiments and messing around before I started work on [ATRA, my Affine Transformation Autoencoder project.](https://github.com/DarkElement75/atra)

I messed around with torch a bit because [these guys looked pretty professional and were from MIT](https://github.com/mrkulk/Unsupervised-Capsule-Network), and they were also the only implementation I could find so far of transforming autoencoders. However, I realized that while torch definitely has its advantages, it does not have enough to be superior to Keras/Tensorflow for my uses. Because of this, and because I don't want to learn how to program in Lua or an entirely new library, I went back to Keras. torch_tests/ has the torch tests, and affine_tests/ has mostly Keras stuff.

Most of this was my work on how to apply full affine transformations (i.e. transformation matrices drawn from the standard normal distribution) to images in a completely differentiable way. This means I didn't want to use loops unless I absolutely had to, so I spent a good deal of time working on an equation for bilinear interpolation sampling of a source image and meshgrid to get a destination image and meshgrid, in the hopes I could make one that worked for a source image of shape (B, H, W, C), a 4-tensor. 

While I eventually realised this was infeasible, and if it was feasible, it was likely far more space-complex than my computer could handle for all but the smallest test cases, I did manage to optimize in a way I didn't see anyone else doing. Maybe that's because my way has a flaw, but so far it's proven to work great.

In the [Spatial Transformer Network Paper](https://arxiv.org/abs/1506.02025), section 3.3, they go over a way to do differentiable image sampling with both integer sampling and bilinear sampling. From those equations, I was able to derive a seemingly more optimal equation for each output pixel, which I will describe in a blog series soon to come. When I make that, you will be able to find it [here.](https://dark-element.com/)  Otherwise, look at the most recent interpolation file in affine_tests/.

I also found an implementation of what I was doing with the professional/MIT guys from above, [which can be found here.](https://github.com/qassemoquab/stnbhwd/blob/master/generic/BilinearSamplerBHWD.c) They used torch because it seems you can add in cuda backend code quite easily, and this allowed them to take a more iterative approach to bilinear interpolation with the c file I have referenced. 

To summarize, it was because of the following that I went with my solution:

  1. Likely High Space complexity for other simultaneous linear algebraic solutions
  2. Everyone seemed to do it in an iterative manner, because of the same reasons
  3. Spatial Transformers and MIT guys also did it this way.
  4. Far harder to implement a more complex solution for likely minimal gains

This is probably (hopefully) the hardest part of my [ATRA project](https://github.com/DarkElement75/atra), as it is the only part of it where I could not see a way to code it with symbolic math operations. Thankfully, it is complete, and I am now working on ATRA. I will be writing a blog series detailing everything I can about the implementation and inner workings of both transforming autoencoders and ATRA as soon as I am finished, so I hope you find these repositories of use.
