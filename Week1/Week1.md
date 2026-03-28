# Computer Vision Leaning Notes : Week1



> 学习材料：[2 A Simple Vision System – Foundations of Computer Vision](https://visionbook.mit.edu/simplesystem.html) 、 [6.S058 Schedule](https://introtocv.github.io/schedule.html)



## <center>第一部分、CV理论学习



### 学习总结

- 当阅读英文文献书籍速度过慢的时候，可使用大模型辅助阅读。但要注意不能完全依赖于大模型，逐步提升自己的英文文献书籍阅读能力。





### Chapter 2 阅读笔记

1. <img src="file:///C:\Users\MIRROR~1\AppData\Local\Temp\QQ_1774703879108.png" alt="img" style="zoom:50%;" />

这张图片对比了 **Perspective Projection(透视投影法)**和**Parallel Projection(平行投影法)**的区别，透视投影法近大远小，平行关系改变。平行投影法不改变平行关系。我们常见的相机都是采用了**Perspective Projection**。

那么便产生思考，我们在已经得到照片的前提下，我们该怎么让计算机知道两条线是平行的呢？



2. <img src="file:///C:\Users\MIRROR~1\AppData\Local\Temp\QQ_1774704229601.png" alt="img" style="zoom:50%;" />

   关于实际中的一个点(X,Y,Z)如何投影到照片里面。解决的是三维如何向照片里面的二维投影的问题。

   <img src="C:\Users\MirrorMoon\AppData\Roaming\Typora\typora-user-images\image-20260328213116571.png" alt="image-20260328213116571" style="zoom:50%;" />

   转换关系可以由（2.1）的公式得到。

   因此呢，一个平行于y轴移动的点和一个平行于z轴移动的点是不能区分的。但是这个公式把真实世界的Y和Z混合在一起了，然后我们要做的就是已知x,y，还原X,Y,Z。作者在这章节主要实现坐标的相互转换，先不考虑颜色等复杂问题的解决。

   

3. 可以把观测到的照片看成是一个函数l(x,y)，输入是坐标(x,y)，输出是亮度<img src="C:\Users\MirrorMoon\AppData\Roaming\Typora\typora-user-images\image-20260328213945524.png" alt="image-20260328213945524" style="zoom:50%;" />

   因此，一张照片就可以由上面形式的二元函数表示。

   这是我们人眼对于一张照片的初始方式，即，一个坐标对应一个亮度，但是要是把这个二维的照片在三维空间里面解读，去切割图片，去寻找图片中，物体的边界。那么就需要一个新的图片初始方式。

   然后就有一种动物，他的眼睛看到的和人的眼睛不一样，每个x,y对应的位置还储存的距离信息。

   

4. 物体的边界信息：图像的边界信息成因有很多，我们要做的第一个任务就是分类各个边界产生的原因。我们考虑的边界有：**物体边界、表面方向变化产生的边界、阴影边界、接触边界、遮挡边界（后面两个是两个物体相互作用形成的）**。这个工作看起来可能简单，但是实际执行起来却很困难，



5. 提取物体的边界：我们把图像看成一个在x,y平面上连续的函数，那么我们可以用**梯度Gradient**来刻画图像颜色的变化。

   $$ \grad l = (\frac{\part{l}}{\part{x}},\frac{\part{l}}{\part{y}}) $$。梯度的方向是变化最大的方向。这里和微积分的定义是一样子的。<img src="file:///C:\Users\MIRROR~1\AppData\Local\Temp\QQ_1774707048308.png" alt="img" style="zoom:50%;" />



6. 近似计算图像上的偏导数的值：

   <img src="file:///C:\Users\MIRROR~1\AppData\Local\Temp\QQ_1774707131512.png" alt="img" style="zoom:50%;" />

然而还有加权的事情，将在后续的学习中学到。

我们从梯度中得到两个概念，梯度的模和梯度的方向。前者叫做边缘强度，后者叫做边缘方向。



7. 然后我们就可以通过简单的给边缘设置个阈值，然后找到梯度突变的地方，作为物体的边界，就像下面，我们找到了物体的边界。

<img src="file:///C:\Users\MIRROR~1\AppData\Local\Temp\QQ_1774707440882.png" alt="img" style="zoom:50%;" />



8. Figure/Ground分割：我们可以通过像素的亮度和饱和度来判断是主物体还是背景。要是我们班识别到地面，我们可以直接把Y(x,y)这个函数值设置成0.



9. 遮挡边界、接触边界：我们做垂直线扫描的时候，从下到上，ground到figure是接触边界，figure到ground是垂直边界，然后呢，这种启发式的方法将会在物体相互遮挡的时候失效。



10. 从真实世界转换到图像世界会损失很多东西，下面是一些保留的性质（2.6.4）<img src="file:///C:\Users\MIRROR~1\AppData\Local\Temp\QQ_1774708771559.png" alt="img" style="zoom:50%;" />

这些性质只能在三维->二维成立，但是在逆过程中未必成立



11. 非偶然特性：非偶然特性只有在特定角度照相才能实现，我们可以初步认为照片里面的性质与三维里面的性质相对应。就是尽管逆过程不总成立，但是失效条件比较苛刻，我们可以初步认为成立。

12. <img src="file:///C:\Users\MIRROR~1\AppData\Local\Temp\QQ_1774709297302.png" alt="img" style="zoom:50%;" />

    这里还是有一些很特殊的情况的，但是在Chapter2我们只先考虑一般情况。

13. 我们现在只知道边界的信息，平坦表面的信息还暂时不知道。我们现在需要把边界的信息传播到表面

    平坦表面需要满足一下约束条件：

    <img src="C:\Users\MirrorMoon\Desktop\QQ_1774709922584.png" alt="img" style="zoom:50%;" />

    二阶导数的近似可以用类似于一阶导数的方法。

14. 我们可以把之前提到的所有约束条件，都改写成aY=b这种约束形式。我们把所有的约束系数a写成矩阵的形式A,可以表示为AY=b的形式，其中矩阵A是**高度稀疏矩阵**，这样就变成计算机能处理巅峰矩阵问题了。

15.总结：我们虽然初步实现了把二维照片还原到了三维世界，但是我们还存在一个巨大问题，就是怎么知道哪些点构成一个整体呢？而且目前这种还原，需要很多很多的限制条件，而且只是简单的假设。







## <center> 第二部分、初探大模型系统+无人机



相关学习资料整理：

[OpenClaw + Dimensional OS：使用自然语言掌控任意机器人 | 工业智能算网](https://www.gyznsw.cn/2026/03/25/2026-03-25-openclaw-dimensional-os-robotics/)

[dimensionalOS/dimos: Dimensional is the agentic operating system for physical space. Vibecode humanoids, quadrupeds, drones, and other hardware platforms in natural language and build multi-agent systems that work seamlessly with physical input (cameras, lidar, actuators).](https://github.com/dimensionalOS/dimos)



### 一、DimOS

Dimensional 是面向物理空间的智能体操作系统。它让你可以使用自然语言对人形机器人、四足机器人、无人机及其他硬件平台进行编程（Vibecode），并构建能够与物理输入（摄像头、激光雷达、执行器）无缝协作的多智能体系统。

**<img src="file:///C:\Users\MIRROR~1\AppData\Local\Temp\QQ_1774712530527.png" alt="img" style="zoom:50%;" />**



### 二、通过dimOS集成OpenClaw

<img src="file:///C:\Users\MIRROR~1\AppData\Local\Temp\QQ_1774712580621.png" alt="img" style="zoom:50%;" />