CUDA_VISIBLE_DEVICES=1 \
python \
./cross_reid_wo_dis.py \
-d sysu \
-b 32 \
--epochs 7000 \
--num-instances 4 \
-j 4 \
--att_mode 1 \
--lr 1e-4 \
-a two_pipe \
--margin 2.4 \
--use_adam \
--features 2048 \
--combine-trainval \
--logs-dir ./mode11_wo_dis_att_256_128_7000epoch_instance_4_0.5_new_source_fea14_wi_norm_0.1_[300,400] \
--start_save 800
