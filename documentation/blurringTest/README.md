# Blurring Test

When blurring an image, we make the colour transition from one side of an edge in the image to another smooth rather than sudden. 
The effect is to average out rapid changes in pixel intensity.
`Blurring` is to make something less clear or distinct. 
This could be interpreted quite broadly in the context of image analysis 
- anything that reduces or distorts the detail of an image might apply.

A `Gaussian blur` is a filter that makes use of a `Gaussian kernel`.
A kernel, in this context, is a small matrix which is combined with the image using a mathematical technique: convolution. 
Different sizes, shapes and contents of kernel produce different effects. 
The kernel can be thought of as a little image in itself, and will favour features of a similar size 
and shape in the main image. On convolution with an image, a big, blobby kernel will retain big, blobby, low spatial frequency features.

An example explanation on this can be found at [The Carpentries and Data Carpentry](https://datacarpentry.org/image-processing/06-blurring/).

Based on the pictures 
`rotated_by_15_Screen Shot 2018-06-08 at 4.59.36 PM.png`,
`rotated_by_15_Screen Shot 2018-06-07 at 2.15.20 PM.png` and 
`rotated_by_15_Screen Shot 2018-06-07 at 2.15.34 PM.png`,
different blurring levels have been tested for experimentation.

Some impressions can be seen in the following.

<table>
  <tr>
     <td>Sigma</td>
     <td>Picture</td>
  </tr>
  <tr>
    <td>None</td>
    <td><img src="./Figure_apple_ok_1_nonBlurred.png" height="100" /></td>
  </tr>
 </table>

The expert panel decided for `sigma=6`resulting in pictures 
that hardly can be used to distinguish fresh and rotten apples.
At least, that's the human perspective.
Let's see inhowfar ANNs are challenged by this.

![alt-text-1](Figure_apple_ok_1_nonBlurred.png "title-1") ![alt-text-2](Figure_apple_ok_1_blurredWithSigma6.png "title-2")

<img src="./Figure_apple_ok_1_nonBlurred.png" height="100" /> <img src="./Figure_apple_ok_1_blurredWithSigma6.png" height="100" />

Please remark: Larger sigma values may remove more noise, which is beneficial.
But larger sigma values will also remove detail from an image, which is a challenge.