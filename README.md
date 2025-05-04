<h2>Tensorflow-Image-Segmentation-Pancreas-T1W-MRI (2025/05/04)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>

This is the first experiment of Image Segmentation for Pancreas-T1W MRI Images based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/15sTp6iWcmsKeBetI0p60GYcwqfel-ekR/view?usp=sharing">Pancreas-T1-ImageMaskDataset.zip</a>, 
which is a subset of T1W (t1.zip) in the original Pancreas_MRI_Dataset of OSF Storage <a href="https://osf.io/kysnj/">
<b>PanSegData.</b></a><br><br>

Please see also our experiment.<br>
<li> <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Pancreas-T2W-MRI">Tensorflow-Image-Segmentation-Pancreas-T2W-MRI<a/></li>

<br>

<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
 The inferred colorized masks predicted by our segmentation model trained on the T1-ImageMaskDataset appear 
 similar to the ground truth masks, but lack precision in some areas. To improve segmentation accuracy, 
 we could consider using a different segmentation model better suited for this task, 
 or explore online data augmentation strategies.
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/images/10010_12.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/masks/10010_12.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test_output/10010_12.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/images/10006_21.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/masks/10006_21.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test_output/10006_21.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/images/10078_15.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/masks/10078_15.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test_output/10078_15.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Pancreas-T1 Segmentation.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other more advanced TensorFlow UNet Models to get better segmentation models:<br>
<br>
<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>
<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been taken from OSF HOME <a href="https://osf.io/kysnj/"><b>PanSegData</b></a><br>
Contributors: Ulas Bagci Debesh Jha Zheyuan Zhang Elif Keles<br>
Date created: 2024-04-28 02:14 PM | Last Updated: 2024-07-08 11:41 PM<br>
Identifier: DOI 10.17605/OSF.IO/KYSNJ<br>
Category:  Data<br>
Description: <i>The dataset consists of 767 MRI scans (385 TIW) and 382 T1W scans from five 
different institutions.</i><br>
License: <i>GNU General Public License (GPL) 3.0</i> <br>
<br>

<h3>
<a id="2">
2 Pancreas-T1 ImageMask Dataset
</a>
</h3>
 If you would like to train this Pancreas-T1 Segmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/15sTp6iWcmsKeBetI0p60GYcwqfel-ekR/view?usp=sharing">Pancreas-T1-ImageMaskDataset.zip</a>, 
<br>
Please expand the downloaded ImageMaskDataset and place it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Pancreas-T1
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
On the derivation of this dataset, please refer to the following Python scripts in <a href="./generator">generator</a>.<br>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>


<b>Pancreas-T1 Dataset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/Pancreas-T1_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid dataset is enough large for use our segmentation model.<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
4 Train TensorflowUNet Model
</h3>
 We trained Pancreas-T1 TensorflowUNet Model by using the 
<a href="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/Pancreas-T1 and run the following bat file for Python script <a href="./src/TensorflowUNetTrainer.py">TensorflowUNetTrainer.py</a>.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>

<h3>The summary of train_eval_infer.config</h3>.
<b>Model parameters</b><br>
Defined small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and large num_layers (including a bridge).
<pre>
[model]
model         = "TensorflowUNet"
base_filters   = 16 
base_kernels   = (9,9)
num_layers     = 8
dilation       = (3,3)
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation tool. To enable ir, set generator parameter to True.  
<pre>
[model]
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback. 
<pre> 
[train]
learning_rate_reducer = True
reducer_factor        = 0.4
reducer_patience      = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callback</b><br>
Enabled EpochChange infer callback.<br>
<pre>
[train]
epoch_change_infer     = True
epoch_change_infer_dir =  "./epoch_change_infer"
num_infer_images       = 6
</pre>

By using this EpochChangeInference callback, on every epoch_change, the inference procedure can be called
 for 6 bimages in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes at each epoch during your training process.<br> 
<br>

<b>Epoch_change_inference output at start (1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/asset/epoch_change_infer_start.png" width="1024" height="auto"><br>
<br>
<br>
<b>Epoch_change_inference output at end (98,99,100)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/asset/epoch_change_infer_end.png" width="1024" height="auto"><br>
<br>
<br>


In this case, the training process terminated at epoch 100 as shown below.<br>
<b>Training console output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/asset/train_console_output_at_epoch_100.png" width="720" height="auto"><br>
<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
5 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Pancreas-T1.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/asset/evaluate_console_output_at_epoch_100.png" width="720" height="auto">
<br><br>

The loss (bce_dice_loss) score for this "Pancreas-T1/test" dataset is not low, but dice_coef not high as shown below.<br>
<pre>
loss,0.2368
dice_coef,0.5769
</pre>


<h3>
6 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Pancreas-T1.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>



<b>Enlarged images and masks (512x512 pixels)</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/images/10003_21.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/masks/10003_21.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test_output/10003_21.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/images/10021_7.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/masks/10021_7.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test_output/10021_7.jpg" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/images/10078_15.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/masks/10078_15.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test_output/10078_15.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/images/10210_54.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/masks/10210_54.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test_output/10210_54.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/images/10253_65.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/masks/10253_65.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test_output/10253_65.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/images/10276_41.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test/masks/10276_41.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pancreas-T1/mini_test_output/10276_41.jpg" width="320" height="auto"></td>
</tr>

</table>

<br>
<br>
<!--
  -->

<h3>
References
</h3>
<b>1. Large-Scale Multi-Center CT and MRI Segmentation of Pancreas with Deep Learning </b><br>
 Zheyuan Zhanga, Elif Kelesa, Gorkem Duraka, Yavuz Taktakb, Onkar Susladkara, Vandan Goradea, Debesh Jhaa,<br> 
 Asli C. Ormecib, Alpay Medetalibeyoglua, Lanhong Yaoa, Bin Wanga, Ilkin Sevgi Islera, Linkai Penga, <br>
 Hongyi Pana, Camila Lopes Vendramia, Amir Bourhania, Yury Velichkoa, Boqing Gongd, Concetto Spampinatoe, <br>
 Ayis Pyrrosf, Pallavi Tiwarig, Derk C F Klatteh, Megan Engelsh, Sanne Hoogenboomh, Candice W. Bolani, <br>
 Emil Agarunovj, Nassier Harfouchk, Chenchan Huangk, Marco J Brunol, Ivo Schootsm, Rajesh N Keswanin, <br>
 Frank H Millera, Tamas Gondaj, Cemal Yazicio, Temel Tirkesp, Baris Turkbeyq, Michael B Wallacer, Ulas Bagcia,<br>

<a href="https://arxiv.org/pdf/2405.12367">https://arxiv.org/pdf/2405.12367</a>
<br>
<a href="https://github.com/NUBagciLab/PANSegNet">Large-Scale Multi-Center CT and MRI Segmentation of Pancreas with Deep Learning</a>
<br>
<br>
<b>2. Pancreas Segmentation in MRI using Graph-Based Decision Fusion on Convolutional Neural Networks</b><br>
Jinzheng Cai, Le Lu, Zizhao Zhang, Fuyong Xing, Lin Yang, Qian Yin
<br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5223591/">https://pmc.ncbi.nlm.nih.gov/articles/PMC5223591/</a>
<br>
<br>
<b>2. ImageMask-Dataset-Pancreas-T2</b><br>
Toshiyuki Arai antillia.com<br>
<a href="https://github.com/sarah-antillia/ImageMask-Dataset-Pancreas-T2">https://github.com/sarah-antillia/ImageMask-Dataset-Pancreas-T2
</a>
<br>
<br>


<b>3. Tensorflow-Image-Segmentation-Pancreas-T2W-MRI</b><br>
Toshiyuki Arai antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Pancreas-T2W-MRI">
Tensorflow-Image-Segmentation-Pancreas-T2W-MRI<a/><br>


