python \
./cross_reid_wo_tri_train.py \
-d sysu \
-b 64 \
--epochs 2000 \
--num-instances 4 \
-j 4 \
--att_mode 1 \
--lr 1e-4 \
-a two_pipe \
--margin 2.4 \
--use_adam \
--features 2048 \
--combine-trainval \
--logs-dir ./tri_256_128_2000epoch_instance_4_64_0.9_ir_1.0_rgb \
--start_save 800
