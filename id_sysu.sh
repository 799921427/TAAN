CUDA_VISIBLE_DEVICES=3 \
python \
./train_s.py \
-d sysu_id \
-b 16 \
--epochs 100 \
--num-instances 4 \
-j 4 \
--att_mode 1 \
--lr 1e-4 \
-a baseline_wo_D \
--margin 2.4 \
--features 2048 \
--logs-dir ./1sysu_logs_5000_mode1_attloss \
--start_save 800
