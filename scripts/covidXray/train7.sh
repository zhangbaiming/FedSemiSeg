<<<<<<< HEAD
python main.py --task covidXray_segmentation_cnum10_dist2_skew0.5_seed0 --model unet34 --algorithm fedsemi --optimizer Adam --num_rounds 50 --num_epochs 10 --learning_rate 0.0002 --proportion 1 --batch_size 8 --eval_interval 1 --sample full --aggregate weighted_entropy --gpu 0 --labeledclients --aggregate_labeled model_sim_max --numlabeled 10
=======
python main.py --task covidXray_segmentation_cnum10_dist2_skew0.5_seed0 --model unet34 --algorithm fedsemi --optimizer Adam --num_rounds 50 --num_epochs 10 --learning_rate 0.0002 --proportion 1 --batch_size 8 --eval_interval 1 --sample full --aggregate weighted_entropy --gpu 2 --labeledclients --aggregate_labeled model_sim_max --numlabeled 10
>>>>>>> 94698ad529b1960202245285235a857048f5a9be
