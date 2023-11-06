wandb login ab526b074d1667944fd0e6f990aef5dc2c28a93a
python train_new.py --model=distilbert-base-uncased --use_silver=0 --use_bronze=0 --epochs=5 --batch_size=16--seed=25415
python train_new.py --model=distilbert-base-uncased --use_silver=200 --use_bronze=0 --epochs=5 --batch_size=16--seed=27331
python train_new.py --model=distilbert-base-uncased --use_silver=400 --use_bronze=0 --epochs=5 --batch_size=16--seed=26448
