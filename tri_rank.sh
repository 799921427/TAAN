python \
./cross_reid_tri_rank_loss.py \
-d sysu \
-b 64 \
--epochs 200 \
--num-instances 8 \
-j 4 \
--att_mode 1 \
--lr 1e-4 \
-a two_pipe \
--margin 2.4 \
--use_adam \
--features 2048 \
--combine-trainval \
--logs-dir ./rank_loss \
--start_save 800
