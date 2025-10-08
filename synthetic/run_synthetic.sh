
nohup bash -c '
               python -u main.py --dataset synthetic --alpha 5.0 --beta 5 --gamma 0.5 --user_nums 1000 --social_nums 10000 --item_nums 5000 --cate_nums 10 --name "run_synthetic" --gpu 5 >> synthetic/run_synthetic.out' &> run_synthetic.out &