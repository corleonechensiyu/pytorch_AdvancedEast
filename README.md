# pytorch_AdvancedEast
pytorch实现AdvancedEast+mobilenetv3

参考https://github.com/huoyijie/AdvancedEAST 
training
1.modify config params in cfg.py, see default values.
2.python preprocess.py, resize image to 256256,384384,512512,640640,736*736, and train respectively could speed up training process.
3.python label.py
python train.py
python predict.py

