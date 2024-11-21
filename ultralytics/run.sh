python prepare_dataset.py --coco_json ../data/original/annotations/instance.json --images_dir ../data/original/images --yolo_dir ./yolo_dataset --train_ratio 0.8
python train.py --data_config ./yolo_dataset/data.yaml --model yolo11x --epochs 1000 --batch_size 16 --run_name all-yolo11x

# Separately train models for each class: 
for class in Melosira Peranema Scaridium Opercularia Paramecium Chaetonotus Philodina Amoeba Beggiatoa Colurella Monostyla Litonotus Daphnia Macrobiotus Trachelophyllum Spirostomum Euplotes Prorodontida Aspidisca Arcella Chilodonella Acineta Podophrya Stylonychia Tokophrya Aeolosoma Lecane Spirochaeta Lepadella Vorticella; do
    python prepare_dataset.py --coco_json ../data/original/annotations/instance.json --images_dir ../data/original/images --yolo_dir ./yolo_dataset --train_ratio 0.8 --included_classes $class
    python train.py --data_config ./yolo_dataset/data.yaml --model yolo11x --epochs 500 --batch_size 16 --run_name ${class}-yolo11x
done