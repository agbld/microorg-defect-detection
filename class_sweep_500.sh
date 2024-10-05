# Separately train models for each class: 
for class in led particle flip Particle_Big marked tilt led_ng; do
    python prepare_dataset.py --coco_json ./data/annotations/instance.json --images_dir ./data/images --yolo_dir ./yolo_dataset --train_ratio 0.8 --included_classes $class
    python main.py --data_config ./yolo_dataset/data.yaml --model yolo11x --epochs 500 --batch_size 16 --run_name ${class}-yolo11x
done