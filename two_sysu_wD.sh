CUDA_VISIBLE_DEVICES=0 \
python \
./cross_reid_wi_dis.py \
-d sysu \
-b 16 \
--epochs 2000 \
--num-instances 2 \
-j 4 \
--att_mode 1 \
--lr 1e-4 \
-a two_pipe_wD \
--margin 2.4 \
--use_adam \
--features 2048 \
--combine-trainval \
--logs-dir ./mode1_wi_dis_att_1e-4_256_128_wo_g_1000000 \
--start_save 800
