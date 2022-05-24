import os

for run in [0, 2, 1, 3, 4]:
    os.system("python train.py --optimizer=AdaSpider --run_id={0} --eta=1 --dataset=MNIST".format(str(run)))
