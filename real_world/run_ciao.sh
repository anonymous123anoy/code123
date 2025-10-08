nohup bash -c '
               python -u main.py --dataset ciao --alpha 5 --beta 5 --gamma 0.5 --epsilon 0.0 --name "ciao_run_alpha_5.0" --gpu 7 >> ciao/ciao_run_alpha_5.0.out && \
               python -u main.py --dataset ciao --alpha 5 --beta 5 --gamma 0.5 --epsilon 0.0 --seed 42 --name "ciao_run_alpha_5.0_42" --gpu 7 >> ciao/ciao_run_alpha_5.0_42.out' &> ciao_run_alpha_5.0.out &
