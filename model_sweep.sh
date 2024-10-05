# Prepare a dataset with only major classes: led, particle, flip, Particle_Big, marked
python prepare_dataset.py --coco_json ./data/annotations/instance.json --images_dir ./data/images --yolo_dir ./yolo_dataset --train_ratio 0.8 --included_classes led particle flip Particle_Big marked

# Run model sweep, including: yolov5xu (97.2M), yolov8x (68.2M), yolov9e (58.1M), yolov10x (29.5M), yolo11x (56.9M)
python main.py --data_config ./yolo_dataset/data.yaml --model yolov5xu --epochs 500 --batch_size 16 --run_name major-cls-yolov5xu
python main.py --data_config ./yolo_dataset/data.yaml --model yolov8x --epochs 500 --batch_size 16 --run_name major-cls-yolov8x
python main.py --data_config ./yolo_dataset/data.yaml --model yolov9e --epochs 500 --batch_size 16 --run_name major-cls-yolov9e
python main.py --data_config ./yolo_dataset/data.yaml --model yolov10x --epochs 500 --batch_size 16 --run_name major-cls-yolov10x
python main.py --data_config ./yolo_dataset/data.yaml --model yolo11x --epochs 500 --batch_size 16 --run_name major-cls-yolo11x
python main.py --data_config ./yolo_dataset/data.yaml --model rtdetr-x --epochs 500 --batch_size 16 --run_name major-cls-rtdetr-x