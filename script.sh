SILVER_LEN=200
MAX_SILVER_LEN=400
BRONZE_LEN=0
MAX_BRONZE_LEN=4000
EPOCHS=5
BATCHSIZE=16

MODEL='distilbert-base-uncased'
# echo 'wandb login {api for wandb}' >> out.sh

for j in $( eval echo {0..$MAX_SILVER_LEN..$SILVER_LEN} )
do
    echo "running $j silver labels"
    echo "python train_new.py --model=$MODEL --use_silver=$j --use_bronze=$BRONZE_LEN --epochs=$EPOCHS --batch_size=$BATCHSIZE --seed=$RANDOM" >> out.sh
done

chmod 755 out.sh
sbatch -G 1 --cpus-per-task=1 --mem-per-cpu=16g --gpus=rtx_2080:1 --wrap='./out.sh' 