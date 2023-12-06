for noise in 0.01 ; do 
  for param in GP_base_voronoi GP_bandit_voronoi random_base_voronoi random_bandit_voronoi # sparse_base_voronoi sparse_bandit_voronoi
  do
    for i in {1..10}
    do
      python main_no_visu_voronoi.py --env $i --param $param --noise_sigma $noise --iter 200 --generate False
      # fi
    done
  done
done