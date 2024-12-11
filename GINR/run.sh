# true: gc m4 to m5
# input: graphcast mesh size 4 predictions on 2022-01-01T00:00:00
# target: graphcast mesh size 5 predictions on 2022-01-01T00:00:00
# train
python train_ginr_e300.py --dataset_dir dataset/gcm4to5 --lr 0.001 --n_layers 8 --skip=True
python train_ginr_e1k.py --dataset_dir dataset/gcm4to5 --lr 0.001 --n_layers 8 --skip=True
# eval on 5 - self
python eval_ginr_gc.py lightning_logs/GINR/45e300/checkpoints/best.ckpt --dataset_dir dataset/gcm5 --variable 4 --filename 45e300_on5_precip
python eval_ginr_gc.py lightning_logs/GINR/45e300/checkpoints/best.ckpt --dataset_dir dataset/gcm5 --variable 7 --filename 45e300_on5_temp
python eval_ginr_gc.py lightning_logs/GINR/45e1k/checkpoints/best.ckpt --dataset_dir dataset/gcm5 --variable 4 --filename 45e1k_on5_precip
python eval_ginr_gc.py lightning_logs/GINR/45e1k/checkpoints/best.ckpt --dataset_dir dataset/gcm5 --variable 7 --filename 45e1k_on5_temp
# eval on 6 - real
python eval_ginr_gc.py lightning_logs/GINR/45e300/checkpoints/best.ckpt --dataset_dir dataset/gcm6 --variable 4 --filename 45e300_on6_precip
python eval_ginr_gc.py lightning_logs/GINR/45e300/checkpoints/best.ckpt --dataset_dir dataset/gcm6 --variable 7 --filename 45e300_on6_temp
python eval_ginr_gc.py lightning_logs/GINR/45e1k/checkpoints/best.ckpt --dataset_dir dataset/gcm6 --variable 4 --filename 45e1k_on6_precip
python eval_ginr_gc.py lightning_logs/GINR/45e1k/checkpoints/best.ckpt --dataset_dir dataset/gcm6 --variable 7 --filename 45e1k_on6_temp


# true: gc m5 to m6
# input: graphcast mesh size 5 predictions on 2022-01-01T00:00:00
# target: graphcast mesh size 6 predictions on 2022-01-01T00:00:00
# train
python train_ginr_e300.py --dataset_dir dataset/gcm5to6 --lr 0.001 --n_layers 8 --skip=True
python train_ginr_e1k.py --dataset_dir dataset/gcm5to6 --lr 0.001 --n_layers 8 --skip=True
# eval on 6 - self
python eval_ginr_gc.py lightning_logs/GINR/56e300/checkpoints/best.ckpt --dataset_dir dataset/gcm6 --variable 4 --filename 56e300_on6_precip
python eval_ginr_gc.py lightning_logs/GINR/56e300/checkpoints/best.ckpt --dataset_dir dataset/gcm6 --variable 7 --filename 56e300_on6_temp
python eval_ginr_gc.py lightning_logs/GINR/56e1k/checkpoints/best.ckpt --dataset_dir dataset/gcm6 --variable 4 --filename 56e1k_on6_precip
python eval_ginr_gc.py lightning_logs/GINR/56e1k/checkpoints/best.ckpt --dataset_dir dataset/gcm6 --variable 7 --filename 56e1k_on6_temp
# eval on 5 - reversability
python eval_ginr_gc.py lightning_logs/GINR/56e300/checkpoints/best.ckpt --dataset_dir dataset/gcm5 --variable 4 --filename 56e300_on5_precip
python eval_ginr_gc.py lightning_logs/GINR/56e300/checkpoints/best.ckpt --dataset_dir dataset/gcm5 --variable 7 --filename 56e300_on5_temp
python eval_ginr_gc.py lightning_logs/GINR/56e1k/checkpoints/best.ckpt --dataset_dir dataset/gcm5 --variable 4 --filename 56e1k_on5_precip
python eval_ginr_gc.py lightning_logs/GINR/56e1k/checkpoints/best.ckpt --dataset_dir dataset/gcm5 --variable 7 --filename 56e1k_on5_temp

# true: gc m4 to m6
# input: graphcast mesh size 4 predictions on 2022-01-01T00:00:00
# target: graphcast mesh size 6 predictions on 2022-01-01T00:00:00
# train
python train_ginr_e300.py --dataset_dir dataset/gcm4to6 --lr 0.001 --n_layers 8 --skip=True
python train_ginr_e1k.py --dataset_dir dataset/gcm4to6 --lr 0.001 --n_layers 8 --skip=True
# eval on 6 - self
python eval_ginr_gc.py lightning_logs/GINR/46e300/checkpoints/best.ckpt --dataset_dir dataset/gcm6 --variable 4 --filename 46e300_on6_precip
python eval_ginr_gc.py lightning_logs/GINR/46e300/checkpoints/best.ckpt --dataset_dir dataset/gcm6 --variable 7 --filename 46e300_on6_temp
python eval_ginr_gc.py lightning_logs/GINR/46e1k/checkpoints/best.ckpt --dataset_dir dataset/gcm6 --variable 4 --filename 46e1k_on6_precip
python eval_ginr_gc.py lightning_logs/GINR/46e1k/checkpoints/best.ckpt --dataset_dir dataset/gcm6 --variable 7 --filename 46e1k_on6_temp
# eval on 5 - versatility
python eval_ginr_gc.py lightning_logs/GINR/46e300/checkpoints/best.ckpt --dataset_dir dataset/gcm5 --variable 4 --filename 46e300_on5_precip
python eval_ginr_gc.py lightning_logs/GINR/46e300/checkpoints/best.ckpt --dataset_dir dataset/gcm5 --variable 7 --filename 46e300_on5_temp
python eval_ginr_gc.py lightning_logs/GINR/46e1k/checkpoints/best.ckpt --dataset_dir dataset/gcm5 --variable 4 --filename 46e1k_on5_precip
python eval_ginr_gc.py lightning_logs/GINR/46e1k/checkpoints/best.ckpt --dataset_dir dataset/gcm5 --variable 7 --filename 46e1k_on5_temp
