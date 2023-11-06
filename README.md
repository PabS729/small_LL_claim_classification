Hierarchy of data folder:
-data
--gold
--silver+bronze

module load gcc/8.2.0 cuda/11.6.2 python_gpu/3.11.2

pip install -r requirements.txt -U

chmod 755 script.sh

Insert your wandb key in script.sh

./script.sh