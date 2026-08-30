export NUMEXPR_MAX_THREADS=16 && export OMP_NUM_THREADS=16 && export MKL_NUM_THREADS=16 && export CUDA_VISIBLE_DEVICES='6' && python3 run.py --input-dir data/kinect_rosbags/processed/distilled_bag --output-dir outputs/distilledbag_new && python3 run.py --input-dir data/kinect_rosbags/processed/distilled_bag2 --output-dir outputs/distilledbag2_new && python3 run.py --input-dir data/kinect_rosbags/processed/distilled_bag3 --output-dir outputs/distilledbag3_new

export NUMEXPR_MAX_THREADS=16 && export OMP_NUM_THREADS=16 && export MKL_NUM_THREADS=16 && export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9' && python3 scripts/scannet_vipe_bench_evaluator.py --work-dir ./workspace/evaluation_scannet_default --input-root data/scannet --raw-root /robodata/smodak/datasets/scannet_v2/scans --do-final-eval

export NUMEXPR_MAX_THREADS=16 && export OMP_NUM_THREADS=16 && export MKL_NUM_THREADS=16 && export CUDA_VISIBLE_DEVICES='0' && python3 scripts/scannet_vipe_bench_evaluator.py --scenes scene0000_00 --work-dir ./workspace/evaluation_scannet_scene0000_00 --input-root data/scannet --raw-root /robodata/smodak/datasets/scannet_v2/scans --do-final-eval

export NUMEXPR_MAX_THREADS=16 && export OMP_NUM_THREADS=16 && export MKL_NUM_THREADS=16 && export CUDA_VISIBLE_DEVICES='4,6,7' && python3 scripts/replica_vipe_bench_evaluator.py --work-dir ./workspace/evaluation_replica_default --input-root data/replica --raw-root /robodata/smodak/datasets/Replica_full

python3 scripts/data_extract/scannet_to_vipe.py --scans-root /robodata/smodak/datasets/scannet_v2/scans --output-root data/scannet --vipe-res 1280 --vipe-fps 5 --num-workers 4

python3 scripts/data_extract/replica_niceslam_to_vipe.py --niceslam-root /robodata/smodak/datasets/Replica --full-root /robodata/smodak/datasets/Replica_full --output-root data/replica --vipe-res 1280 --vipe-fps 5 --num-workers 4

python3 scripts/data_extract/rosbag_to_vipe.py data/kinect_rosbags/raw/distilled_bag/distilled_bag_0.mcap --output-dir data/kinect_rosbags/processed/distilled_bag --vipe-res 1280 --vipe-fps 5

python3 scripts/data_extract/rosbag_to_vipe.py data/kinect_rosbags/raw/distilled_bag2/distilled_bag2_0.mcap --output-dir data/kinect_rosbags/processed/distilled_bag2 --vipe-res 1280 --vipe-fps 2.5

python3 scripts/data_extract/rosbag_to_vipe.py data/kinect_rosbags/raw/distilled_bag3/distilled_bag3_0.mcap --output-dir data/kinect_rosbags/processed/distilled_bag3 --vipe-res 1280 --vipe-fps 2.5

python3 scripts/scannet_eval_dashboard.py --before-root workspace/evaluation_scannet_default_old --after-root workspace/evaluation_scannet_default_new --input-root data/scannet --host 127.0.0.1 --port 18799
