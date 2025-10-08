nohup bash -c '
               python -u main_mitigate.py --dataset ciao --alpha 5 --beta 5 --gamma 0.5 --epsilon 0.0 --s1 1 --sigma 10 --name "ciao_run_mitigate_s1_10" --gpu 6 >> ciao/ciao_run_mitigate_s1_10.out' &> ciao_run_mitigate.out &
