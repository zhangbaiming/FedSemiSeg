<<<<<<< HEAD
python main.py --task covidXray_segmentation_cnum10_dist2_skew0.5_seed0 --model unet34 --algorithm fedsemi --optimizer Adam --num_rounds 50 --num_epochs 5 --learning_rate 0.0002 --proportion 1 --batch_size 8 --eval_interval 5 --sample full --aggregate weighted_entropy --gpu 2 --labeledclients --aggregate_labeled model_sim_max  --numlabeled 10
=======
python main.py --task covidXray_segmentation_cnum10_dist2_skew0.81_seed0 --model unet34 --algorithm fedsemi --optimizer Adam --logger seg_logger --num_rounds 50 --num_epochs 5 --learning_rate 0.0002 --proportion 1 --batch_size 8 --eval_interval 1 --sample full --aggregate weighted_consistency --gpu 3 --labeledclients --contrastive --aggregate_labeled proto_sim --numlabeled 5
>>>>>>> 560c8eb28a37012e7b134dc05a9feac0483a082f
