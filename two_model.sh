CUDA_VISIBLE_DEVICES=3 \
python \
./cross_reid_wo_two_train.py \
-d sysu \
-b 4 \
--epochs 1200 \
--num-instances 4 \
-j 4 \
--att_mode 1 \
--lr 1e-4 \
-a two_pipe \
--margin 2.4 \
--use_adam \
--features 2048 \
--combine-trainval \
--logs-dir ./models/1mode_1_512_256_32_4_rgb_1000_a \
--start_save 800
