SILVER_LEN=200
MAX_SILVER_LEN=4000
BRONZE_LEN=0
MAX_BRONZE_LEN=4000
EPOCHS=5
BATCHSIZE=32

MODEL='distilbert-base-uncased'
# echo 'wandb login {api for wandb}' >> out.sh
module load gcc/8.2.0 cuda/11.6.2 python_gpu/3.11.2

for j in $( eval echo {600..$MAX_SILVER_LEN..$SILVER_LEN} )
do
    echo "running $j silver labels"
    sbatch --time=30 -G 1 --cpus-per-task=1 --mem-per-cpu=16g --gres=gpumem:8g --wrap="python train_new.py --model=$MODEL --use_silver=$j --use_bronze=$BRONZE_LEN --epochs=$EPOCHS --batch_size=$BATCHSIZE --seed=$RANDOM"
done
