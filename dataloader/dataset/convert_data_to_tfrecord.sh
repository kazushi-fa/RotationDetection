python /home/ubuntu/work/RotationDetection/dataloader/dataset/convert_data_to_tfrecord.py \
    --root_dir='/home/ubuntu/work/RotationDetection/dataloader/dataset/DOTA/DOTA1.0/' \
    --xml_dir='train/labeltxt' \
    --image_dir='train/images' \
    --save_name='train' \
    --img_format='.png' \
    --dataset='DOTA1.0'