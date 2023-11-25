for noise in 0.1; do 
  for param in GPwall_safeb_base
  do
    for i in {1..10}
    do
      # if [ $param = sparse_bandit_double ]
      # then
      #   python main_no_visu.py --env $i --param $param --noise_sigma $noise --generate False
      # else
      python main_no_visu_walls.py --env $i --param $param --noise_sigma $noise --iter 200 # --generate False
      # fi
    done
  done
done