from demo import single

config_path = "configs/voc12.yaml"
model_path = "data/models/voc12/deeplabv2_resnet101_msc/train_aug/checkpoint_final.pth"
image_root = "/home/kuangbixia/projects/awesome-semantic-segmentation-pytorch/datasets"
save_root = "/home/kuangbixia/projects/deeplab-pytorch/data/features/test_results/val"

val_dir = image_root
voc_dir = image_root + "/voc/VOC2012/JPEGImages"
voc_val_txt = image_root + "/voc/VOC2012/ImageSets/Segmentation/val.txt"


def voc_vals():
    with open(voc_val_txt, 'r') as fin:
        vals = fin.readlines()

    vals = [val.strip() for val in vals]

    for val in vals:
        save_path = save_root + "/voc12/{}.png".format(val)
        image_path = voc_dir + "/{}.jpg".format(val)
        single(config_path=config_path, model_path=model_path, image_path=image_path, save_path=save_path)

def my_vals():
    image_path = val_dir + "/walkers.jpeg"
    save_path = save_root + "/walkers/walkers.png"
    single(config_path=config_path,model_path=model_path,image_path=image_path,save_path=save_path)
    for i in range(5):
        save_path = save_root + "/walkers/walkers{}.png".format(str(i+2))
        image_path = val_dir + "/walkers{}.jpeg".format(str(i+2))
        single(config_path=config_path,model_path=model_path,image_path=image_path,save_path=save_path)

if __name__ == '__main__':
    my_vals()
    voc_vals()
