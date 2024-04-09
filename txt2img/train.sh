# lambdalabs/pokemon-blip-captions
CUDA_VISIBLE_DEVICES='0,1' python main.py \
    -t \
    --base v1_binarydm.yaml \
    --gpus 0,1 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 10 \
    --finetune_from epoch=000142.ckpt?download=true 2>&1 | tee binarydm.log

# m1guelpf/nouns
CUDA_VISIBLE_DEVICES='0,1' python main.py \
    -t \
    --base v1_binarydm.yaml \
    --gpus 0,1 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 10 \
    --finetune_from nouns-pretrain-fp.ckpt 2>&1 | tee binarydm_nouns.log
