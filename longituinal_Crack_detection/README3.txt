1) TRANSFER LEARN A MODEL:

Start Anaconda Prompt (tensorflow)

CD to top level folder that contains "Annotations" "data" "ImageSets"... etc.

Legacy Method (Supports multi gpu):
python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=ssd_mobilenet_v2_quant_sean.config

New Method (Only single gpu supported):
python ..\models-master\research\object_detection/model_main.py --alsologtostderrz --pipeline_config_path=ssd_mobilenet_v2_quant_sean.config --model_dir=./models/train

2) Launch tensorboard to watch progress

Launch Tensorboard In "/models/train" directory

tensorboard --logdir ./ --port 6008

View Tensorboard by going to webpage displayed in consel with FireFox (best)

3) Export a frozen file

Option 1 for desktop use:
python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./ssd_mobilenet_v2_quant_sean.config --trained_checkpoint_prefix ./models/train/model.ckpt-0 --output_directory ./fine_tuned_model

Option 2 as first step to TFLITE (only works on a SSD):
python export_tflite_ssd_graph.py --input_type image_tensor --pipeline_config_path ./ssd_mobilenet_v2_quant_sean.config --trained_checkpoint_prefix ./models/train/model.ckpt-0 --output_directory ./fine_tuned_model

4) CONVERT TO TFLITE (must be in linux)

tflite_convert \
  --input_shapes=1,300,300,3 \
  --output_file=optimizedQ.tflite \
  --graph_def_file=tflite_graph.pb \
  --output_format=TFLITE \
  --input_arrays=normalized_input_image_tensor \
  --output_array='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
  --inference_type=QUANTIZED_UINT8 \
  --allow_custom_ops \
  --mean_values=128 \
  --std_dev_values=127


NOTES:

DESIGNED TO WORK WITH THE TENSORFLOW MODELS-MASTER REPOSITORY AS OF 4/1/2019