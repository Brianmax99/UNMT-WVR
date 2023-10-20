export NGPU=4;
PYTHONIOENCODING=utf-8 python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
--exp_name unsupMT_enfr_tcma_m30_random_fp_18_co \
--dump_path './dumped/' \
--data_path './data_m30_random/processed/en-fr.18co/' \
--reload_checkpoint './dumped/unsupMT_enfr_tcma/t03iwun0e0/checkpoint.pth' \
--lgs 'en-fr' \
--ae_steps 'en,fr' \
--bt_steps 'en-fr-en,fr-en-fr' \
--word_shuffle 3 \
--word_dropout 0.1 \
--word_blank 0.1 \
--add_image true \
--image_id 3 \
--image_embed_dropout_prob 0.1 \
--max_align_image 2 \
--codes_path 'codes_enfr' \
--vocab_path 'vocab_enfr' \
--lambda_ae '0:1,100000:0.1,300000:0' \
--encoder_only false \
--emb_dim 1024 \
--n_layers 6 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--tokens_per_batch 2000 \
--batch_size 32 \
--bptt 256 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 10000 \
--max_epoch 70 \
--eval_bleu true \
--eval_add_image true \
--validation_metrics 'valid_en-fr_mt_bleu,valid_en-fr_mt_bleu'

# --stopping_criterion 'valid_en-fr_mt_bleu,10' \
# --reload_model 'mlm_enfr_1024.pth,mlm_enfr_1024.pth' \

# --reload_checkpoint './dumped/unsupMT_enfr_tcma/t03iwun0e0/checkpoint.pth' \

