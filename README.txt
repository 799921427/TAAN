有可能发生错误，很多地方使用了绝对路径

直接通过sh run4SYSU.sh运行
完成训练之后，使用python extract_feature.py提取特征
然后用evaluation/demo.m进行测试

环境：
python3
torch_0.4.0

log中有训练好的模型，在sh run4SYSU.sh中添加
--resume ./logs/model_best.pth.tar --evaluate 来直接测试
