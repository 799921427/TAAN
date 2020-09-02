python \
./cross_reid_tri.py \
-d sysu \
-b 64 \
--epochs 1000 \
--num-instances 4 \
-j 4 \
--att_mode 1 \
--lr 1e-4 \
-a two_pipe \
--margin 2.4 \
--use_adam \
--features 2048 \
--combine-trainval \
--rgb_w 0.5 \
--ir_w 1.3000000000000003 \
--logs-dir ./0.5_rgb_1.3000000000000003_ir \
--start_save 800
