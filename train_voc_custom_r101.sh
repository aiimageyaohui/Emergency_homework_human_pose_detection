#
#
#
#  train a detection model
#  on custom VOC like dataset




python3 main.py ctdet --exp_id pascal_custom_res101_trafficlight \
--dataset pascal_custom \
--arch res_101 \
--batch_size 4 \
--num_epochs 900 \
--master_batch 5 \
--val_intervals -1 \
--lr 3.75e-4 \
--gpus 0
