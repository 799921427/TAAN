CUDA_VISIBLE_DEVICES=3 \
python \
./cross_reid_wo_dis.py \
-d sysu \
-b 32 \
--epochs 1000 \
--num-instances 4 \
-j 4 \
--lr 1e-4 \
-a baseline_wo_D \
--att_mode 1 \
--margin 2.4 \
--use_adam \
--features 2048 \
--combine-trainval \
--logs-dir ./baseline_256_128_insance_4_wo_D \
--start_save 800
