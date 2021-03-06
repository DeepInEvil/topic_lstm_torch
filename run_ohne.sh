#!/usr/bin/env bash
#grid search through parameters
batch_sizes=(32)
lstm_hidden=(32)
emb_drop=(0.5 0.6)
fc_size=(100 200 300)
early_stop=(9 11 13 15)
for batch in "${batch_sizes[@]}"
do
  for lstm_h in "${lstm_hidden[@]}"
    do
      for emb_d in "${emb_drop[@]}"
        do
          for fc_s in "${fc_size[@]}"
            do
              for early_s in "${early_stop[@]}"
                do
                  #python ./main.py --early-stopping $early_s --batch-size $batch --hidden-size $lstm_h --fc-layer $fc_s --emb-drop $emb_d --mit-topic true >./mit_topic/$batch_$lstm_h_$fc_s$emb_d_$early_s.txt
                  python ./main.py --early-stopping $early_s --batch-size $batch --hidden-size $lstm_h --fc-layer $fc_s --emb-drop $emb_d >./ohne_topic/$batch'_'$lstm_h'_'$fc_s'_'$emb_d'_'$early_s.txt
                done
            done
        done
    done
done