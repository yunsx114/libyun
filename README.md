# A Simple Image Processing Libaray

### 1 用户使用手册

库中的类和函数均处于名为`yun`的命名空间中，想要调用库中的类和函数需要预先加上`yun::`的命名空间前缀，或在文件开头部分写好`using namespace yun`

#### 1.1 基本的图片读写

```c++
yun::Mat image = yun::imread("/Path/to/picture");
```

利用imread函数，填入图片的路径，这样就将图片的信息存储到了Mat类的实例image中。由于函数内部会自动识别图片的类型，所以不用手动指定。

```c++
yun::imwrite(image, PNG,"/Path/test.png");
```

利用imwrite函数，传入参数：Mat类实例，指定的图片类型，和路径。图片类型是头文件中定义的宏，包括PNG, BMP, JPG, JPEG。如果指定的类型和路径中的图片后缀不一致，可以输出图片，但图片可能损坏。

此外，由于jpg(jpeg)图片支持有损压缩，因此可以指定图片的输出质量(0到100)：

```C++
yun::imwrite(image, JPG,"/Path/test.jpg", 90); //保持90%的质量
```

#### 1.2 亮度调整

库中提供了Mat类的加减乘除的函数重载，可以通过简单的编写实现图像处理：

```c++
yun::Mat image = yun::imread("/Path/to/picture");
image = image - 20;
image += 50;
image *= 0.7;
yun::addLight(image,20);
```

可以通过对image加减整数改变每个像素每个通道的值，也可以通过和浮点数的乘法实现每个像素每个通道值的等比例缩放，从而实现不同效果的亮度调整。此外，库也提供了一个标准接口，通过调用addLight进行亮度调节。

#### 1.3 图片混合

```c++
yun::Mat image1 = yun::imread("/Path/to/picture1");
yun::Mat image2 = yun::imread("/Path/to/picture2");

float coeff = 0.3;
//method 1
yun::Mat image = image1 * coeff  + image2 * (1-coeff);
//method 2
yun::blendLinear(iamge1,image1,coeff,image2,1-coeff);
```

设置系数并通过让Mat乘以系数进行图片混合，以上代码将两个图片以3:7的比例进行混合。此外，可以调用blendLinear来实现相同效果

#### 1.4 图片放缩

```c++
yun::Mat image1 = yun::imread("/Path/to/picture1"); // assume is 1024 x 1024
yun::Mat image2 = yun::imread("/Path/to/picture2"); // assmue is 512 x 512

yun::Mat image3 = rescale(image1,256,2048);
yun::Mat image4 = rescale(image2,1024,1024) * 0.5 + image2 * 0.5;
```

通过调用Mat rescale(Mat&src, size_t new_width, size_t new_height)来实现图片的放缩，函数返回一个新的Mat类，因此可以直接参与计算，比如两张图片大小不一样的时候可以放缩到同一大小再进行混合。

#### 1.5 图片裁剪

```c++
yun::Mat image = yun::imread("/Path/to/picture1"); // assume is 1024 x 1024
yun::Mat small = ROI(image,512,512,1024,1024);
```

通过调用Mat ROI(const Mat& src, int x1, int y1, int x2, int y2)得到从(x1,y1)到(x2,y2)对角线所在矩形部分的区域。

1.3.6 卷积操作

```c++
yun::Mat image = yun::imread("/Path/to/picture1");
filter2D(image,GBlur,1,5);
```

可以通过调用`void filter2D(Mat& dst, const Mat& src, int type,size_t size);`进行有限制的卷积核调用，其中type表明卷积核的类型，如G_Blur代表高斯模糊，size代表卷积核的宽度

```c++
yun::Mat image = yun::imread("/Path/to/picture1");
yun::Kernel kernel({0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111});
image *= kernel;
```

此外，允许通过kenel类自定义卷积核，长度和内容任意，可以和Mat类相乘来实现卷积操作。