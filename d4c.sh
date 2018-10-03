#!/bin/bash
#SBATCH --time=8:05:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=4

img_classes=("n02389026" "n03888257" "n03584829" "n02607072" "n03297495" "n03063599" "n03792782" "n04086273" "n02510455" "n11939491" "n02951358" "n02281787" "n02106662" "n04120489" "n03590841" "n02992529" "n03445777" "n03180011" "n02906734" "n07873807" "n03773504" "n02492035" "n03982430" "n03709823" "n03100240" "n03376595" "n03877472" "n03775071" "n03272010" "n04069434" "n03452741" "n03792972" "n07753592" "n13054560" "n03197337" "n02504458" "n02690373" "n03272562" "n04044716" "n02124075")

for i in "${img_classes[@]}"
do
	wget "http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid="$i
	# python2 ./downloadutils.py --downloadImages --wnid $i
done