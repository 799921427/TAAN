CUDA_VISIBLE_DEVICES=0 \
python \
./triplet_loss4SYSU.py \
-d sysu \
-b 64 \
--epochs 1500 \
--num-instances 4 \
-j 4 \
-a baseline \
--margin 2.4 \
--features 2048 \
--combine-trainval \
--logs-dir ./logs \
--start_save 800
