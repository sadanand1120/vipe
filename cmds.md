export NUMEXPR_MAX_THREADS=16 && export OMP_NUM_THREADS=16 && export MKL_NUM_THREADS=16 && export CUDA_VISIBLE_DEVICES='6' && python3 run.py --input-dir data/kinect_rosbags/processed/distilled_bag --output-dir outputs/distilledbag_new && python3 run.py --input-dir data/kinect_rosbags/processed/distilled_bag2 --output-dir outputs/distilledbag2_new && python3 run.py --input-dir data/kinect_rosbags/processed/distilled_bag3 --output-dir outputs/distilledbag3_new

export NUMEXPR_MAX_THREADS=16 && export OMP_NUM_THREADS=16 && export MKL_NUM_THREADS=16 && export CUDA_VISIBLE_DEVICES='4,6,7' && python3 scripts/scannet_vipe_bench_evaluator.py --scenes scene0000_00 scene0011_00 scene0378_00 --work-dir ./workspace/evaluation_scannet_default --input-root data/scannet --raw-root /robodata/smodak/datasets/scannet_v2/scans

export NUMEXPR_MAX_THREADS=16 && export OMP_NUM_THREADS=16 && export MKL_NUM_THREADS=16 && export CUDA_VISIBLE_DEVICES='4,6,7' && python3 scripts/replica_vipe_bench_evaluator.py --work-dir ./workspace/evaluation_replica_default --input-root data/replica --raw-root /robodata/smodak/datasets/Replica_full

python3 scripts/data_extract/rosbag_to_vipe.py data/kinect_rosbags/raw/distilled_bag2/distilled_bag2_0.mcap --output-dir data/kinect_rosbags/processed/distilled_bag2

python3 scripts/data_extract/scannet_to_vipe.py --scans-root /robodata/smodak/datasets/scannet_v2/scans --output-root data/scannet --scenes scene0000_00 scene0011_00 scene0378_00 --frame-skip 1 --num-workers 4

python3 scripts/data_extract/replica_niceslam_to_vipe.py --niceslam-root /robodata/smodak/datasets/Replica --full-root /robodata/smodak/datasets/Replica_full --output-root data/replica --num-workers 4
