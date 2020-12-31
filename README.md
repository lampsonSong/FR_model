# 人脸识别模型使用说明

这个项目中只有人脸识别模型的inference，没有训练部分的代码。第一步先用centerface做人脸检测，然后用本项目中的模型提取一个256维的特征，最后比较不同人脸的特征相似度。

### 依赖
- tensorflow 1.15.4 (其他版本应该也可以，只是没有测试过)

### CPP版本
同样功能完善的CPP版本在`http://tools.ulsee.com/SDK/gpu_sdks/FaceRecognitionGPU.git`， 可以找杭州同事添加，CPP中的模型已经加密。

### 使用
> `python3 demo.py image1_path image2_path`

对demo.py输入俩张图片，每张图片应该只包含一张人脸，程序会输出俩张图片的相似度。

### 例子
> `python3 demo.py ./imgs/zhao1.png ./imgs/zhao2.jpg`


### TF转ONNX
1. 使用以下git中的工具可以将tensorflow的PB模型转换成onnx：
> https://github.com/onnx/tensorflow-onnx

命令为： 
> python -m tf2onnx.convert --input ./FR_small_FBN.pb --output ./FR_small_batch1.onnx --inputs input0:0[1,112,112,3] --outputs 1238_classifier.0/BiasAdd:0

请根据相关git中的说明文档使用

2. 转出来的onnx还可以经过一步简化:
> https://github.com/daquexian/onnx-simplifier

命令为:
> python -m onnxsim FR_small_batch1.onnx FR_small_batch1_sim.onnx

也请根据相关git中的说明文档使用
