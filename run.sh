python train.py \
    --dataset_path tokenized_tinystories \
    --num_epochs 1 \
    --pad_sequence \
    --batch_size 64 \
    --val_loss_every 512 \
    --learning_rate 0.0003 \
    --weight_decay 0.1 \
    --num_token_permutations 0 \
    --project_name ts-pt \
