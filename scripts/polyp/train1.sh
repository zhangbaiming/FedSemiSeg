python main.py --task polyp_segmentation_cnum4_dist0_skew0_seed0 --model unet34 --algorithm fedsemi --optimizer Adam --logger seg_logger --num_rounds 50 --num_epochs 5 --learning_rate 0.0002 --proportion 1 --batch_size 8 --eval_interval 1 --sample labeled --aggregate weighted_scale --gpu 0 --labeledclients 0 1 2 --aggregate_labeled proto_sim --numlabeled 3

# 2 3 4 
# --num_threads 4
# python main.py --task polyp_segmentation_cnum4_dist0_skew0_seed0 --model unet34 --algorithm fedsemi --optimizer Adam --logger seg_logger --num_rounds 50 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 8 --eval_interval 1 --sample full --aggregate weighted_scale --gpu 6 --labeledclients 1 2 3
# python main.py --task polyp_segmentation_cnum4_dist0_skew0_seed0 --model unet34 --algorithm fedsemi --optimizer Adam --logger seg_logger --num_rounds 50 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 8 --eval_interval 1 --sample full --aggregate weighted_scale --gpu 7 --labeledclients 0 1 2 3
