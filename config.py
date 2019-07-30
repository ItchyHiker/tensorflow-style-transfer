# config parameters
MODEL_NAME = 'msgnet'
STYLE_IMG = 'imgs/style/1/2.jpg'
CONTENT_IMG = './imgs/content/content2.jpg'
STYLE_IMG_PATH = 'imgs/style'
CONTENT_IMG_PATH = '/home/ubuntu/dataset/COCO_2017'

epochs = 1
batch_size = 4
learning_rate = 1e-3
content_img_size = 256
style_img_size = 256

style_loss_weights = [x*4e2 for x in [.35, .35, .35, .35]]
# style_loss_weights = [5, 5, 5, 5]
content_loss_weights = [0, 1, 0, 0]
reg_loss_weight = 1e-7

ckpt_dir='./ckpt/metanet'
log_dir='./logs/metanet'
